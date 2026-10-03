"""
eval.py - Evaluate the 10-K RAG system: self-correcting LangGraph vs. naive baseline.

Usage:
    python eval.py               # full run (resumes automatically if interrupted)
    python eval.py --limit 4     # quick smoke test on the first 4 questions
    python eval.py --fresh       # ignore any saved progress, start completely over

Resuming: every (question, system) result is appended to results/progress.jsonl
the moment it's judged -- not held in memory until the end. If the terminal
closes or the run crashes partway through, rerunning `python eval.py` picks up
right after the last completed question instead of redoing (and re-paying for)
everything. Use --fresh to discard progress.jsonl and start clean.

Needs env vars: GROQ_API_KEY, COHERE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
Optional:       GROQ_MODEL   (generator, default openai/gpt-oss-120b)
                JUDGE_MODEL  (default openai/gpt-oss-120b -- ideally a different model,
                              which also gets its own Groq daily token quota)
                EVAL_PAUSE_SECONDS (extra pause between questions, default 3)
Flags:          --no-ragas skips RAGAS (the biggest spender of requests)

What it measures
  RAGAS (answerable questions only):
      context precision, context recall, faithfulness, answer relevancy
  Custom:
      accuracy         - LLM judge: does the answer match the ground truth?
      abstention rate  - on unanswerable questions, did the system say "I don't know"?
      latency, retries, rewrites, crashes

Why the judge model is different from the generator: a model grading its own
output is biased toward itself. Generator = your nodes.py model, judge = JUDGE_MODEL.
Try to keep them different.

NOTE on fairness: for the graph, "contexts" are the chunks that SURVIVED grading
(what the generator actually saw). For the baseline they are the raw top-k.
So compare context precision with that in mind - the graph's filter is part of
what you're testing.
"""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # must run BEFORE importing graph/nodes (they create the LLM client)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from graph import build_graph
from nodes import invoke_with_retry, retrieve_chunks, _strip_context_leak, _parse_yes_no
from testset import TESTSET

TOP_K = 5  # was 3. At k=3, nvda_rev_fy26 failed (income-statement chunk likely ranked 4th+),
           # so we're back at 5. Trade-off: bigger grading/generation prompts -> more tokens per call.
           # Confirm with: python debug_retrieval.py "<question>" --k 8
RECURSION_LIMIT = 25
RESULTS_DIR = Path("results")

# Keep this identical to the generate prompt in nodes.py, otherwise the
# baseline-vs-graph comparison is unfair.
GENERATION_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below. Cite specific figures.\n"
    'If the context does not contain the answer, say "I don\'t know based on the '
    'provided documents."\n\n'
    "Write your answer as a single, plain, natural-language sentence for a human reader. "
    "Do NOT paste, quote, or reference the raw context text, table formatting, pipe characters (|), "
    "brackets, or the word 'Context' -- state the fact directly.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


# ------------------------------------------------------------------ judges
# Plain-text yes/no judges (not with_structured_output): forced tool-calling is
# fragile across models/library versions; a one-word answer works everywhere.
CORRECT_PROMPT = ChatPromptTemplate.from_template(
    "Question: {question}\n"
    "Ground truth: {truth}\n"
    "Candidate answer: {answer}\n\n"
    "Does the candidate answer state the same key fact(s) as the ground truth? "
    "Accept equivalent formatting or units (e.g. $29.9 billion = $29,915 million). "
    "10-K tables report in millions, so a bare '$34,550' matches a ground truth of '$34,550 million'. "
    "Extra detail is fine. A wrong or missing key number is not.\n"
    "Reply with exactly one word: yes or no."
)

ABSTAIN_PROMPT = ChatPromptTemplate.from_template(
    "Question: {question}\n"
    "Answer: {answer}\n\n"
    "This question cannot be answered from SEC 10-K filings. Did the answer decline "
    "or say it doesn't know, WITHOUT asserting a specific factual answer?\n"
    "Reply with exactly one word: yes or no."
)


# ---------------------------------------------------------------- systems
def get_embeddings():
    """Use the exact same embedding model used during ingestion."""
    from langchain_cohere import CohereEmbeddings

    return CohereEmbeddings(
        model="embed-english-v3.0"
    )


def get_vectorstore():
    """Returns the raw PineconeVectorStore, not a static .as_retriever().

    We need the raw store (not a pre-built retriever) so both the graph and
    the baseline can pass a per-question ticker filter to similarity_search()
    -- a .as_retriever() object bakes its search_kwargs in at construction
    time and can't vary the filter per call.
    """
    embeddings = get_embeddings()
    index_name = os.getenv("PINECONE_INDEX_NAME", "10k-filings")

    # Guard: a mismatched embedding model gives garbage retrieval with no error.
    index_dim = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).describe_index(index_name).dimension
    embed_dim = len(embeddings.embed_query("dimension check"))
    if index_dim != embed_dim:
        raise SystemExit(
            f"Embedding mismatch: Pinecone index '{index_name}' has dimension {index_dim}, "
            f"but the eval embedder outputs {embed_dim}. Set EMBEDDING_MODEL to the model "
            f"ingest.py used (or edit get_embeddings())."
        )

    return PineconeVectorStore(index_name=index_name, embedding=embeddings)


def run_baseline(question, vectorstore, llm, k=TOP_K):
    """Naive RAG: retrieve top-k, generate once. No grading, no grounding check.

    Uses the SAME retrieve_chunks() helper as the graph's retrieve node (see
    nodes.py) -- including its per-company retrieval for comparison questions
    -- so the baseline-vs-graph comparison isolates the effect of the
    self-correction loop, not a retrieval-quality difference between the two
    systems.
    """
    start = time.perf_counter()
    error = ""
    contexts, answer = [], ""
    try:
        contexts = retrieve_chunks(vectorstore, question, k)
        answer = _strip_context_leak(invoke_with_retry(
            GENERATION_PROMPT | llm | StrOutputParser(), {"context": "\n\n".join(contexts), "question": question}
        ))
    except Exception as e:  # noqa: BLE001
        error = type(e).__name__
    return {
        "answer": answer,
        "contexts": contexts,
        "latency_s": time.perf_counter() - start,
        "retries": 0,
        "rewrites": 0,
        "error": error,
    }


def run_graph(question, graph):
    """Self-correcting LangGraph. Streams updates so we can count node visits."""
    state = {
    "question": question,
    "documents": [],
    "generation": "",
    "retries": 0,
    "grounded": False,
    "rewrite_attempts": 0,
    }
    visits = Counter()
    error = ""
    start = time.perf_counter()
    try:
        for update in graph.stream(state, config={"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"):
            for node, delta in update.items():
                visits[node] += 1
                if delta:
                    state.update(delta)
    except Exception as e:  # e.g. GraphRecursionError from an endless rewrite loop
        error = type(e).__name__
    return {
        "answer": state.get("generation", ""),
        "contexts": state.get("documents", []),
        "latency_s": time.perf_counter() - start,
        "retries": state.get("retries", 0),
        "rewrites": visits["rewrite"],
        "error": error,
    }


PROGRESS_PATH = RESULTS_DIR / "progress.jsonl"


def load_progress():
    """Read whatever question/system pairs already finished in a previous,
    interrupted run. A pair that errored (e.g. RateLimitError) is NOT counted
    as done -- it gets retried on the next run instead of being skipped forever."""
    if not PROGRESS_PATH.exists():
        return [], set()
    records = [json.loads(line) for line in PROGRESS_PATH.read_text().splitlines() if line.strip()]
    ok_records = [r for r in records if not r.get("error")]
    done = {(r["id"], r["system"]) for r in ok_records}
    return ok_records, done


def append_progress(record):
    """Write one finished (item, system) result immediately, so a closed
    terminal or crash only costs the question in flight, not the whole run."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(PROGRESS_PATH, "a") as f:
        f.write(json.dumps(record, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------- scoring
def judge(item, out, correct_judge, abstain_judge):
    """True = correct, False = incorrect, None = the judge itself failed (rate limit etc.)."""
    if out["error"] or not out["answer"].strip():
        return False
    try:
        if item["type"] == "unanswerable":
            reply = invoke_with_retry(abstain_judge, {"question": item["question"], "answer": out["answer"]})
        else:
            reply = invoke_with_retry(
                correct_judge,
                {"question": item["question"], "truth": item["ground_truth"], "answer": out["answer"]},
            )
        return _parse_yes_no(reply)
    except Exception as e:  # a judge failure shouldn't throw away the finished answer
        print(f"    [judge error] {type(e).__name__}: {e}")
        return None  # NOT False: caller records it as an error so resume retries it


def ragas_scores(rows, judge_llm, judge_emb):
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "response": r["answer"] or "(no answer)",
                "retrieved_contexts": r["contexts"] or ["(no context retrieved)"],
                "reference": r["ground_truth"],
            }
            for r in rows
        ]
    )
    result = evaluate(
        dataset,
        metrics=[
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            Faithfulness(),
            ResponseRelevancy(strictness=1),  # Groq doesn't support n>1 completions
        ],
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=RunConfig(max_workers=2, timeout=180, max_retries=6),  # gentle on Groq rate limits
    )
    return result.to_pandas().select_dtypes("number")


# ------------------------------------------------------------------- main
def main(limit=None, fresh=False, no_ragas=False):
    items = TESTSET[:limit] if limit else TESTSET

    unverified = [t["id"] for t in items if not t.get("verified", False)]
    if unverified:
        print(f"WARNING: {len(unverified)} test items have verified=False: {unverified}")
        print("         Check their ground truth against the real filings before trusting scores.\n")

    if fresh and PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
        print("--fresh: cleared previous progress.jsonl, starting from scratch.\n")

    records, done = load_progress()
    if done:
        print(f"Resuming: {len(done)} (question, system) results already completed, skipping those.\n")

    vectorstore = get_vectorstore()
    graph = build_graph(vectorstore, k=TOP_K)
    from nodes import llm as gen_llm  # same generator the graph uses

    judge_llm = ChatGroq(model=os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b"), temperature=0)
    correct_judge = CORRECT_PROMPT | judge_llm | StrOutputParser()
    abstain_judge = ABSTAIN_PROMPT | judge_llm | StrOutputParser()

    systems = {
        "baseline": lambda q: run_baseline(q, vectorstore, gen_llm, k=TOP_K),
        "graph": lambda q: run_graph(q, graph),
    }

    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['id']}")
        for name, runner in systems.items():
            if (item["id"], name) in done:
                print(f"  {name}: already done, skipping")
                continue
            out = runner(item["question"])
            out.update(
                id=item["id"],
                type=item["type"],
                system=name,
                question=item["question"],
                ground_truth=item["ground_truth"],
            )
            verdict = judge(item, out, correct_judge, abstain_judge)
            if verdict is None:
                out["error"] = "JudgeError"  # load_progress skips errored rows, so the next run retries it
            out["passed"] = bool(verdict)
            append_progress(out)  # saved to disk NOW, before moving to the next question
            records.append(out)
            time.sleep(float(os.getenv("EVAL_PAUSE_SECONDS", "3")))  # spread out requests, avoid tripping rate limits

    df = pd.DataFrame(records)

    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_emb = LangchainEmbeddingsWrapper(get_embeddings())

    summary = {}
    for name in systems:
        sub = df[df.system == name]
        answerable = sub[sub.type != "unanswerable"]
        unanswerable = sub[sub.type == "unanswerable"]
        row = {
            "accuracy": answerable.passed.mean() if len(answerable) else None,
            "abstention_rate": unanswerable.passed.mean() if len(unanswerable) else None,
            "avg_latency_s": sub.latency_s.mean(),
            "avg_retries": sub.retries.mean(),
            "avg_rewrites": sub.rewrites.mean(),
            "crashes": int((sub.error != "").sum()),
        }
        if len(answerable) and not no_ragas:
            print(f"Running RAGAS for {name} ...")
            row.update(ragas_scores(answerable.to_dict("records"), ragas_llm, ragas_emb).mean().to_dict())
        summary[name] = row

    summary_df = pd.DataFrame(summary).T.round(3)

    print("\n=== SUMMARY (baseline vs graph) ===")
    print(summary_df.T.to_string())

    print("\n=== PASS RATE BY QUESTION TYPE ===")
    print(df.groupby(["system", "type"]).passed.mean().unstack().round(2).to_string())

    failed = df[(df.system == "graph") & (~df.passed)]
    if len(failed):
        print("\n=== GRAPH FAILURES (read these - they tell you what to fix) ===")
        for _, r in failed.iterrows():
            print(f"\n[{r['id']}] {r['question']}")
            print(f"  expected: {r['ground_truth']}")
            print(f"  got:      {(r['answer'] or '(empty)')[:300]}")
            if r["error"]:
                print(f"  error:    {r['error']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.drop(columns=["contexts"]).to_csv(RESULTS_DIR / f"per_question_{stamp}.csv", index=False)
    (RESULTS_DIR / f"summary_{stamp}.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved results to {RESULTS_DIR}/ (stamp {stamp})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore/delete results/progress.jsonl and start over instead of resuming",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="skip the RAGAS metrics (the biggest spender of requests); accuracy/abstention still run",
    )
    args = parser.parse_args()
    main(args.limit, fresh=args.fresh, no_ragas=args.no_ragas)
