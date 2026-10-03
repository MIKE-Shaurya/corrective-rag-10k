"""
ingest.py
Downloads 10-K filings from SEC EDGAR, chunks them, embeds them, and
upserts them into a Pinecone index.

Run this once (or whenever you want to refresh/expand the corpus):
    python ingest.py
"""

import os
import re
import time
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---- Config ----
COMPANIES = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]   # tickers to pull 10-Ks for
FILINGS_PER_COMPANY = 1                                   # most recent N annual filings
DOWNLOAD_DIR = "sec_filings"
CHUNK_SIZE = 1500        # characters per chunk (not tokens -- keep it simple)
CHUNK_OVERLAP = 200       # overlap so we don't cut facts in half at chunk boundaries
EMBEDDING_MODEL = "embed-english-v3.0"  # Cohere's free trial embedding model
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
EMBEDDING_DIMENSIONS = 1024  # embed-english-v3.0's output size (different from Gemini's 3072!)


def download_filings():
    """Pull raw 10-K HTML filings from SEC EDGAR for each ticker."""
    dl = Downloader(
        os.environ["SEC_EDGAR_COMPANY_NAME"],
        os.environ["SEC_EDGAR_EMAIL"],
        download_folder=DOWNLOAD_DIR,
    )
    for ticker in COMPANIES:
        print(f"Downloading 10-K for {ticker}...")
        # download_details=True is required -- without it, sec-edgar-downloader only
        # saves the raw full-submission.txt (SGML-wrapped), not a clean .htm file.
        dl.get("10-K", ticker, limit=FILINGS_PER_COMPANY, download_details=True)


def html_to_clean_text(html_path: Path) -> str:
    """Strip HTML tags/boilerplate down to readable plain text.

    Tables get special handling. soup.get_text() flattens a <table> into one
    long space-joined string with no row/column boundaries -- so a row like
    "R&D | 2025 $6,411 | 2024 $4,540 | 2023 $3,969" loses all structure, and
    a downstream LLM (or a human) can no longer tell which number belongs to
    which year. SEC filings are almost entirely financial tables, so this
    was silently corrupting the exact numbers this project exists to answer
    questions about.

    Fix: convert each table to one line of text PER ROW (cells joined by " | ")
    BEFORE flattening, using a sentinel to protect those line breaks from the
    whitespace-collapsing regex below (which would otherwise erase them too).
    """
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Remove scripts/styles -- pure noise, no financial content
    for tag in soup(["script", "style"]):
        tag.decompose()

    ROW_BREAK = "@@ROWBREAK@@"  # unlikely to collide with real filing content

    for table in soup.find_all("table"):
        row_lines = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]  # SEC tables are full of empty spacer <td>s
            if cells:
                row_lines.append(" | ".join(cells))
        # Replace the whole table with its row-preserving text, or drop it if
        # every row was empty spacers (common for layout-only tables).
        table.replace_with(f"{ROW_BREAK}{ROW_BREAK.join(row_lines)}{ROW_BREAK}" if row_lines else "")

    text = soup.get_text(separator=" ")
    # Collapse ALL whitespace (including any stray newlines inside raw cell text) to single
    # spaces first. The sentinel is plain text with no whitespace in it, so it survives this
    # untouched -- we don't need to special-case \n to protect it.
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*" + re.escape(ROW_BREAK) + r"\s*", "\n", text)  # sentinel -> real line break
    text = re.sub(r"\n{2,}", "\n", text).strip()  # drop stray blank lines from empty tables
    return text


def load_and_chunk_documents() -> list[Document]:
    """Walk the downloaded filings, extract text, split into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # tries paragraph breaks before brute-force cuts
    )

    all_chunks: list[Document] = []
    base = Path(DOWNLOAD_DIR) / "sec-edgar-filings"

    for ticker_dir in base.iterdir():
        ticker = ticker_dir.name
        for filing_dir in (ticker_dir / "10-K").iterdir():
            html_files = list(filing_dir.glob("*.htm")) + list(filing_dir.glob("*.html"))

            if html_files:
                text = html_to_clean_text(html_files[0])
            else:
                # Fallback: no clean .htm found (e.g. download_details wasn't set).
                # full-submission.txt is SGML-wrapped but BeautifulSoup can still
                # strip the tags reasonably well as a last resort.
                fallback = filing_dir / "full-submission.txt"
                if not fallback.exists():
                    print(f"  WARNING: no .htm or full-submission.txt found in {filing_dir}, skipping")
                    continue
                print(f"  NOTE: no .htm file for {ticker} {filing_dir.name}, falling back to full-submission.txt")
                text = html_to_clean_text(fallback)

            chunks = splitter.split_text(text)
            for i, chunk_text in enumerate(chunks):
                all_chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={"ticker": ticker, "filing": filing_dir.name, "chunk_index": i},
                    )
                )
            print(f"  {ticker}: {len(chunks)} chunks from {filing_dir.name}")

    return all_chunks


def ensure_index_exists(pc: Pinecone):
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def upsert_to_pinecone(chunks: list[Document]):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    ensure_index_exists(pc)

    # Wipe whatever's already in the index before re-upserting. Without this,
    # re-running ingest.py after a chunking change (e.g. the table-flattening
    # fix) just ADDS the new, better chunks alongside the old, worse ones --
    # the retriever can still surface a stale flattened chunk instead of the
    # fixed one. This makes "refresh the corpus" an actual rebuild, not an append.
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    existing_vectors = stats.get("total_vector_count", 0)
    if existing_vectors:
        print(f"Clearing {existing_vectors} existing vectors from '{PINECONE_INDEX_NAME}' before re-upserting...")
        index.delete(delete_all=True)
        time.sleep(5)  # Pinecone's delete is eventually consistent; give it a moment

    embedder = CohereEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embedder)

    # Cohere's trial key: max 96 texts per embed call, 20 requests/min, 1,000 calls/month
    # total across all Cohere endpoints. Batching at 90 texts/call keeps total calls for
    # this whole corpus in the dozens, not thousands -- comfortably inside all three limits.
    BATCH_SIZE = 90
    PAUSE_SECONDS = 4  # keeps us under 20 requests/minute

    total = len(chunks)
    print(f"Embedding {total} chunks in batches of {BATCH_SIZE} via Cohere...")

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        attempt = 0
        while True:
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                attempt += 1
                if attempt > 5:
                    raise
                wait = 30 * attempt
                print(f"  Rate limited or error ({e}); waiting {wait}s and retrying batch...")
                time.sleep(wait)

        done = min(i + BATCH_SIZE, total)
        print(f"  Upserted {done}/{total} chunks")

        if done < total:
            time.sleep(PAUSE_SECONDS)

    print("Done.")


if __name__ == "__main__":
    download_filings()
    chunks = load_and_chunk_documents()
    print(f"\nTotal chunks to embed: {len(chunks)}")

    if not chunks:
        raise SystemExit(
            "No chunks were produced -- check that sec_filings/sec-edgar-filings/<TICKER>/10-K/ "
            "actually contains .htm files (or full-submission.txt as a fallback). Nothing was "
            "upserted to Pinecone."
        )

    upsert_to_pinecone(chunks)
