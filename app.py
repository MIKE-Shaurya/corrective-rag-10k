import os
from dotenv import load_dotenv

load_dotenv()  

from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from graph import run_query

K = 5  # see eval.py's TOP_K comment


def build_vectorstore():
    embedder = CohereEmbeddings(model="embed-english-v3.0")
    return PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embedder,
    )


def main():
    vectorstore = build_vectorstore()
    print("RAG system ready. Ask a question about the ingested 10-K filings (Ctrl+C to quit).\n")

    while True:
        question = input("Q: ").strip()
        if not question:
            continue

        final_state = run_query(question, vectorstore, k=K)

        print(f"\nA: {final_state['generation']}")
        print(f"   (grounded: {final_state['grounded']}, retries used: {final_state['retries']})\n")


if __name__ == "__main__":
    main()
