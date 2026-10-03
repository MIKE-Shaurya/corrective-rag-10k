

import argparse

from dotenv import load_dotenv

load_dotenv()

from nodes import detect_tickers, _per_company_queries
from eval import get_vectorstore


def show(docs):
    if not docs:
        print("  No documents returned (filter excludes everything, or index is missing this company).")
        return
    for i, doc in enumerate(docs, 1):
        print(f"\n  [{i}] metadata: {doc.metadata}")
        preview = doc.page_content[:500].replace("\n", " \\n ")
        print(f"      content preview: {preview}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5, help="chunks to retrieve (per company for comparisons)")
    parser.add_argument("--no-filter", action="store_true", help="skip the ticker filter, search everything")
    parser.add_argument("--no-decompose", action="store_true", help="comparison: don't rewrite per company")
    args = parser.parse_args()

    tickers = detect_tickers(args.question)
    print(f"Question:           {args.question}")
    print(f"Detected ticker(s): {tickers or '(none detected)'}")
    print(f"k:                  {args.k}")
    print("-" * 70)

    vectorstore = get_vectorstore()

    if len(tickers) >= 2 and not args.no_filter:
        queries = [args.question] * len(tickers) if args.no_decompose else _per_company_queries(args.question, tickers)
        for ticker, q in zip(tickers, queries):
            print(f"\n=== {ticker} | query: {q}")
            show(vectorstore.similarity_search(q, k=args.k, filter={"ticker": {"$in": [ticker]}}))
        return

    filter_dict = None if args.no_filter else ({"ticker": {"$in": tickers}} if tickers else None)
    print(f"Filter used: {filter_dict}")
    show(vectorstore.similarity_search(args.question, k=args.k, filter=filter_dict))


if __name__ == "__main__":
    main()
