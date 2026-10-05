from langgraph.graph import StateGraph, END

from nodes import (
    GraphState,
    retrieve,
    grade_documents,
    rewrite_query,
    generate,
    check_grounding,
    give_up,
)

MAX_REWRITES = 2
MAX_RETRIES = 2
DEFAULT_K = 5 


def increment_rewrites(state: GraphState) -> GraphState:
    return {
        **state,
        "rewrite_attempts": state["rewrite_attempts"] + 1,
    }


def increment_retries(state: GraphState) -> GraphState:
    """Small bookkeeping node: bump the retry counter before looping back to generate."""
    return {**state, "retries": state["retries"] + 1}


def route_after_grading(state: GraphState) -> str:
    if state["documents"]:
        return "generate"

    if state["rewrite_attempts"] >= MAX_REWRITES:
        return "give_up"

    return "rewrite"


def route_after_grounding(state: GraphState) -> str:
    if state["grounded"] or state["retries"] >= MAX_RETRIES:
        return "end"
    return "regenerate"


def build_graph(vectorstore, k: int = DEFAULT_K):
    """Assemble and compile the graph.

    `vectorstore` is the raw PineconeVectorStore (not a pre-built LangChain
    retriever) -- the retrieve node needs it directly so it can pass a
    per-question ticker filter to similarity_search(), which a static
    .as_retriever() object can't do. It's injected via lambda since LangGraph
    only ever calls a node with a single argument (state).
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", lambda s: retrieve(s, vectorstore, k))
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("check_grounding", check_grounding)
    workflow.add_node("increment_retries", increment_retries)
    workflow.add_node("increment_rewrites", increment_rewrites)
    workflow.add_node("give_up", give_up)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        route_after_grading,
        {"generate": "generate", "rewrite": "rewrite", "give_up": "give_up"},
    )
    workflow.add_edge("rewrite", "increment_rewrites")
    workflow.add_edge("increment_rewrites", "retrieve")  # the loop-back arrow
    workflow.add_edge("give_up", END)

    workflow.add_edge("generate", "check_grounding")
    workflow.add_conditional_edges(
        "check_grounding",
        route_after_grounding,
        {"end": END, "regenerate": "increment_retries"},
    )
    workflow.add_edge("increment_retries", "generate")  # the other loop-back arrow

    return workflow.compile()


def run_query(question: str, vectorstore, k: int = DEFAULT_K) -> dict:
    """Convenience wrapper: run one question through the compiled graph."""
    graph = build_graph(vectorstore, k)
    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
        "retries": 0,
        "grounded": False,
        "rewrite_attempts": 0,
    }
    return graph.invoke(initial_state)
