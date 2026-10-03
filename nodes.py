"""
nodes.py
Each function here is one node in the LangGraph state machine.
Every node takes the current `state` (our shared notebook) and returns
an updated `state`.
"""

import os
import time
from typing import TypedDict, List

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# temperature=0 -> deterministic, factual behavior. We're not writing poetry here.
# The model name lives in .env (GROQ_MODEL) because Groq retires models every few
# months -- when that happens, change the .env value instead of editing code.
# See https://console.groq.com/docs/models for the current list.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
llm = ChatGroq(model=GROQ_MODEL, temperature=0)


def _is_rate_limit(e: Exception) -> bool:
    # Exact class name or explicit 429 code -- not a loose substring match on the
    # message. Groq/OpenAI-style clients raise RateLimitError (.status_code == 429);
    # ResourceExhausted / .code == 429 are kept for other providers. A *daily*
    # token-limit 429 (TPD) won't clear by waiting; the retries run out and re-raise.
    if type(e).__name__ in ("RateLimitError", "ResourceExhausted"):
        return True
    return 429 in (getattr(e, "status_code", None), getattr(e, "code", None))


def invoke_with_retry(chain, inputs, max_retries=6, base_delay=15):
    """Call chain.invoke(inputs), retrying with exponential backoff on rate limits.

    Groq's free tier caps requests per minute, and our graph makes several calls
    per question (one grading call per retrieved chunk, plus rewrite/generate/
    grounding) -- easy to trip during an eval run. base_delay=15 is deliberately
    generous: it's better to wait out the per-minute reset than to fail the
    question and waste every token already spent on it.
    Non-rate-limit errors are re-raised immediately -- no point retrying a bug.
    """
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if not _is_rate_limit(e) or attempt == max_retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            print(f"    [rate limited] waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)


class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    retries: int
    grounded: bool
    rewrite_attempts: int


# Company name/ticker aliases -> the ticker used in ingest.py's chunk metadata.
# Add a row here for every company you actually ingest in COMPANIES (ingest.py).
TICKER_ALIASES = {
    "apple": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT",
    "tesla": "TSLA", "tsla": "TSLA",
    "nvidia": "NVDA", "nvda": "NVDA",
    "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL",
}


def detect_tickers(question: str) -> list:
    """Find every company mentioned in the question, by simple keyword match.

    Returns a list of tickers (possibly empty, one, or several -- e.g. a
    comparison question naming two companies). Word-boundary matching avoids
    partial-word false hits (e.g. "apple" inside some unrelated word).
    """
    import re

    found = []
    q = question.lower()
    for alias, ticker in TICKER_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", q) and ticker not in found:
            found.append(ticker)
    return found


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------
TICKER_NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "TSLA": "Tesla", "NVDA": "NVIDIA", "GOOGL": "Alphabet"}


def _per_company_queries(question: str, tickers: list) -> list:
    """Split a comparison question into one standalone lookup query per company.
    Falls back to the original question for every company if the model's reply
    doesn't have exactly one line per ticker."""
    prompt = ChatPromptTemplate.from_template(
        "Rewrite this comparison question as one standalone lookup question per company, "
        "using financial-filing wording (e.g. 'total revenue', 'research and development expenses') "
        "and the company's latest fiscal year. Output exactly {n} lines, in this order, nothing else:\n"
        "{companies}\n\nQuestion: {question}"
    )
    try:
        reply = invoke_with_retry(prompt | llm | StrOutputParser(), {
            "n": len(tickers),
            "companies": "\n".join(TICKER_NAMES.get(t, t) for t in tickers),
            "question": question,
        })
    except Exception:  # noqa: BLE001
        return [question] * len(tickers)
    lines = [l.strip() for l in reply.splitlines() if l.strip()]
    return lines if len(lines) == len(tickers) else [question] * len(tickers)


def retrieve_chunks(vectorstore, question: str, k: int = 3) -> list:
    """Shared retrieval logic used by both the graph's retrieve node and
    eval.py's baseline, so the two stay consistent.

    Filters to the ticker(s) actually named in the question, so "Apple's R&D"
    can't retrieve a similarly-worded chunk from Microsoft/Tesla/NVIDIA/
    Google's filing instead. Falls back to an unfiltered search when no known
    company is named (e.g. a genuinely unanswerable question like "what's the
    capital of France").

    For a COMPARISON question naming multiple companies (e.g. "Apple vs
    Microsoft R&D"), a single combined search for k chunks across both
    tickers can let one company's chunks crowd out the other's entirely if
    they rank slightly closer to the query embedding -- this is exactly what
    caused every comparison question to fail with "the context doesn't
    mention Microsoft" even though Microsoft's data was in the index the
    whole time. So when 2+ tickers are named, we search k chunks PER
    ticker separately and concatenate, guaranteeing every named company gets
    a fair shot at the context instead of competing for the same k slots.
    """
    tickers = detect_tickers(question)

    if len(tickers) <= 1:
        filter_dict = {"ticker": {"$in": tickers}} if tickers else None
        docs = vectorstore.similarity_search(question, k=k, filter=filter_dict)
        return [d.page_content for d in docs]

    # Vague comparison wording ("most recent fiscal year", "which had higher...")
    # retrieves generic MD&A text rather than the income statement, so each
    # company gets its own standalone, filing-style lookup query.
    queries = _per_company_queries(question, tickers)
    all_docs = []
    for ticker, q in zip(tickers, queries):
        docs = vectorstore.similarity_search(q, k=k, filter={"ticker": {"$in": [ticker]}})
        all_docs.extend(d.page_content for d in docs)
    return all_docs


def retrieve(state: GraphState, vectorstore, k: int = 3) -> GraphState:
    """Embed the question, pull the nearest chunks from Pinecone (see
    retrieve_chunks for the per-company-vs-shared-k logic)."""
    return {**state, "documents": retrieve_chunks(vectorstore, state["question"], k)}


# ---------------------------------------------------------------------------
# Node: give_up
# ---------------------------------------------------------------------------
def give_up(state: GraphState) -> GraphState:
    """Reached when rewriting the query MAX_REWRITES times still found no
    relevant documents. Without this node, the graph used to jump straight to
    END from here -- state["generation"] was left at its initial empty
    string "", so the system silently returned nothing instead of telling the
    user it couldn't find an answer. This makes that give-up explicit."""
    return {
        **state,
        "generation": (
            "I couldn't find information relevant to this question in the "
            "ingested filings, even after rewriting the query."
        ),
        "grounded": True,  # an explicit "I don't know" makes no unsupported claims
    }


# ---------------------------------------------------------------------------
# Helper: turn a one-word LLM reply into a boolean
# ---------------------------------------------------------------------------
def _parse_yes_no(text: str) -> bool:
    """Interpret an LLM reply as a boolean by looking at its first word.

    We deliberately use plain-text yes/no answers instead of
    llm.with_structured_output(...) for the two graders. Structured output
    relies on forced tool-calling, which depends on both the model and the
    library version supporting it -- and it broke when we moved to gpt-oss.
    A one-word answer works with any model.
    """
    words = (text or "").strip().lower().split()
    if not words:
        return False  # empty reply -> treat as "no" (fails safe)
    return words[0].strip(".,!:;\"'*") in ("yes", "true")


# ---------------------------------------------------------------------------
# Helper: parse a numbered yes/no verdict list back into booleans, in order
# ---------------------------------------------------------------------------
def _parse_numbered_verdicts(text: str, expected_count: int):
    """Parse lines like '1: yes' / '2. no' into {index: bool}, 1-indexed.

    Returns None (instead of a partial dict) if we can't confidently recover
    a verdict for every document -- the caller decides the fallback, we don't
    guess silently here.
    """
    import re

    verdicts = {}
    for line in (text or "").splitlines():
        m = re.match(r"\s*(\d+)\s*[.:)]\s*(yes|no|true|false)\b", line.strip(), re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            verdicts[idx] = m.group(2).lower() in ("yes", "true")

    if len(verdicts) != expected_count or set(verdicts) != set(range(1, expected_count + 1)):
        return None
    return verdicts


# ---------------------------------------------------------------------------
# Node: grade_documents
# ---------------------------------------------------------------------------
def grade_documents(state: GraphState) -> GraphState:
    """Check all retrieved chunks in ONE call instead of one call per chunk.

    Grading k chunks used to cost k separate LLM calls -- the single biggest
    source of API traffic in the graph (up to 5 calls just for grading one
    question). Asking for all k verdicts in one prompt cuts that to 1 call.

    Fallback: if the model's reply can't be cleanly parsed back into exactly
    one verdict per document (wrong count, malformed lines), we fail OPEN --
    keep every document as "relevant" rather than silently dropping some or
    re-querying per-document (which would defeat the whole point of batching).
    This trades a little precision for a system that degrades safely instead
    of being fragile to the model's exact formatting.
    """
    docs = state["documents"]
    if not docs:
        return {**state, "documents": []}

    numbered = "\n\n".join(f"{i + 1}. {doc}" for i, doc in enumerate(docs))
    prompt = ChatPromptTemplate.from_template(
        "Question: {question}\n\n"
        "Below are {n} numbered documents. For EACH one, decide whether it contains "
        "information relevant to answering the question.\n\n"
        "{numbered_docs}\n\n"
        "Reply with exactly {n} lines, one per document, in this exact format "
        "(no other text):\n"
        "1: yes\n2: no\n...(continue for all {n})"
    )
    chain = prompt | llm | StrOutputParser()
    reply = invoke_with_retry(
        chain, {"question": state["question"], "numbered_docs": numbered, "n": len(docs)}
    )

    verdicts = _parse_numbered_verdicts(reply, expected_count=len(docs))
    if verdicts is None:
        print(f"    [grade_documents] couldn't parse {len(docs)} verdicts from batched reply, "
              f"keeping all {len(docs)} documents (fail-open).")
        return {**state, "documents": docs}

    relevant_docs = [doc for i, doc in enumerate(docs, start=1) if verdicts[i]]
    return {**state, "documents": relevant_docs}


# ---------------------------------------------------------------------------
# Node: rewrite_query
# ---------------------------------------------------------------------------
def rewrite_query(state: GraphState) -> GraphState:
    """No relevant docs were found -- reword the question and try retrieval again."""
    prompt = ChatPromptTemplate.from_template(
        "The question below retrieved no relevant results from a set of SEC 10-K filings.\n"
        "Rewrite it to be more specific and to use language likely to appear in a financial filing "
        "(e.g. prefer 'research and development expenses' over 'R&D spending').\n\n"
        "Original question: {question}\n\n"
        "Rewritten question (respond with ONLY the rewritten question, nothing else):"
    )
    new_question = invoke_with_retry(prompt | llm | StrOutputParser(), {"question": state["question"]})
    return {**state, "question": new_question.strip(), "rewrite_attempts": state.get("rewrite_attempts", 0) + 1}


# ---------------------------------------------------------------------------
# Node: generate
# ---------------------------------------------------------------------------
def _strip_context_leak(text: str) -> str:
    """Belt-and-suspenders cleanup: the generation prompt TELLS the model not
    to paste raw context/table fragments into its answer, but gpt-oss-120b
    doesn't reliably follow that instruction -- it sometimes appends a
    bracketed context dump anyway (e.g. "...was $29,915[ Context] Research
    and development | $ | 34,550 | ..."). A normal, clean answer sentence
    has no reason to contain a literal "[" character, so we just cut there.
    This can't fix a WRONG number (that's a retrieval problem), only the
    formatting leak."""
    text = (text or "").strip()
    cut = text.find("[")
    if cut != -1:
        text = text[:cut].strip()
    return text.rstrip(" .") + "." if text and not text.endswith((".", "!", "?")) else text


def generate(state: GraphState) -> GraphState:
    """Write an answer using ONLY the surviving (relevant) chunks as context."""
    prompt = ChatPromptTemplate.from_template(
        "Answer the question using ONLY the context below. Cite specific figures exactly as written. "
        "If the context does not contain the answer, say so explicitly.\n\n"
        "Write your answer as a single, plain, natural-language sentence for a human reader. "
        "Do NOT paste, quote, or reference the raw context text, table formatting, pipe characters (|), "
        "brackets, or the word 'Context' -- state the fact directly, e.g. "
        "\"Apple's R&D expense in fiscal 2023 was $29,915 million.\"\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    context = "\n\n".join(state["documents"])
    answer = invoke_with_retry(
        prompt | llm | StrOutputParser(), {"context": context, "question": state["question"]}
    )
    return {**state, "generation": _strip_context_leak(answer)}


# ---------------------------------------------------------------------------
# Node: check_grounding
# ---------------------------------------------------------------------------
def check_grounding(state: GraphState) -> GraphState:
    """Hallucination check: does the generated answer only say things the context supports?"""
    prompt = ChatPromptTemplate.from_template(
        "Context:\n{context}\n\nAnswer:\n{answer}\n\n"
        "Is every fact and number in the answer directly supported by the context, "
        "with nothing invented? Reply with exactly one word: yes or no."
    )
    reply = invoke_with_retry(
        prompt | llm | StrOutputParser(),
        {"context": "\n\n".join(state["documents"]), "answer": state["generation"]},
    )
    return {**state, "grounded": _parse_yes_no(reply)}
