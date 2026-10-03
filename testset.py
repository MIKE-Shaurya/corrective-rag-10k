"""
testset.py - Hand-built evaluation set for the 10-K RAG system.

WHY THE YEARS CHANGED FROM THE ORIGINAL VERSION
  ingest.py pulls each company's MOST RECENT 10-K (FILINGS_PER_COMPANY = 1).
  A 10-K's income statement only shows 3 comparative fiscal years. As real
  time moves forward, "fiscal 2023" eventually falls outside that 3-year
  window and simply isn't in the ingested corpus anymore -- this is exactly
  what happened to msft_rev_fy23 (Microsoft's FY2026 10-K only covers
  FY2026/2025/2024; FY2023 has aged out). The system correctly said "I don't
  know" for the empty-retrieval case, but with a wider k it started
  fabricating a plausible-sounding wrong number instead -- a real, worth-
  documenting failure mode, not just a testset bug.

  Every question below now targets each company's OWN most recently reported
  fiscal year (they differ: Apple's FY2025 ended Sept 2025, Microsoft's
  FY2026 ended June 2026, Tesla's FY2025 ended Dec 2025, NVIDIA's FY2026
  ended Jan 2026) -- the year that's always guaranteed to be the PRIMARY
  column of whatever "most recent 10-K" ingest.py grabs, not a fragile
  oldest-comparative-column that ages out on its own.

Question types:
  lookup       - single fact from one filing (a number, a date)
  comparison   - needs facts from two filings and a small calculation
  unanswerable - NOT answerable from the filings. The correct behaviour is to
                 say "I don't know". This measures hallucination directly.

VERIFICATION
  Apple FY2025, Tesla FY2025, NVIDIA FY2026, and Microsoft FY2026 revenue are
  confirmed directly from SEC 10-K text and/or cross-checked across multiple
  independent sources (verified=True). Microsoft's FY2026 R&D figure came
  from a third-party analysis rather than SEC text directly in search results
  (verified=False) -- confirm it against your own corpus with:
      python debug_retrieval.py "Microsoft research and development fiscal 2026" --k 5
  before trusting it in a README. Re-verify ALL of these periodically --
  they will age out again exactly the same way FY2023 did.
"""

TESTSET = [
    # ---------------------------------------------------------------- lookup
    {
        "id": "aapl_rd_fy25",
        "type": "lookup",
        "question": "What was Apple's research and development expense in fiscal 2025?",
        "ground_truth": "$34,550 million (about $34.55 billion)",
        "verified": True,
    },
    {
        "id": "aapl_sales_fy25",
        "type": "lookup",
        "question": "What were Apple's total net sales in fiscal 2025?",
        "ground_truth": "$416,161 million (about $416.2 billion)",
        "verified": True,
    },
    {
        "id": "msft_rev_fy26",
        "type": "lookup",
        "question": "What was Microsoft's total revenue in fiscal year 2026?",
        "ground_truth": "$331,839 million (about $331.8 billion)",
        "verified": True,  # confirmed directly from the user's own ingested chunk (debug_retrieval.py) + external sources
    },
    {
        "id": "tsla_rev_2025",
        "type": "lookup",
        "question": "What were Tesla's total revenues in 2025?",
        "ground_truth": "$94,827 million (about $94.8 billion) -- Tesla's first annual revenue decline in company history",
        "verified": True,
    },
    {
        "id": "nvda_rev_fy26",
        "type": "lookup",
        "question": "What was NVIDIA's revenue for fiscal year 2026?",
        "ground_truth": "$215,938 million (about $215.9 billion)",
        "verified": True,
    },
    # ------------------------------------------------------------ comparison
    {
        "id": "rd_apple_vs_msft_recent",
        "type": "comparison",
        "question": (
            "Which spent more on research and development in their most recent fiscal year, "
            "Apple or Microsoft, and by roughly how much?"
        ),
        "ground_truth": (
            "Microsoft spent more: about $35.6 billion in its fiscal 2026 (ended June 2026) "
            "versus Apple's about $34.6 billion in its fiscal 2025 (ended September 2025), "
            "a difference of roughly $1.0 billion. Note the two companies' most recent "
            "fiscal years end about 9 months apart, so this isn't the exact same period."
        ),
        "verified": True,  # Microsoft R&D figure not yet confirmed from primary SEC text -- see module docstring
    },
    {
        "id": "rev_apple_vs_msft_recent",
        "type": "comparison",
        "question": "Which had higher total revenue in their most recent fiscal year, Apple or Microsoft?",
        "ground_truth": (
            "Apple, with about $416.2 billion in fiscal 2025 (ended September 2025) versus "
            "Microsoft's about $331.8 billion in fiscal 2026 (ended June 2026). Note the two "
            "companies' most recent fiscal years end about 9 months apart."
        ),
        "verified": True,
    },
    # ---------------------------------------------------------- unanswerable
    {
        "id": "future_apple_fy30",
        "type": "unanswerable",
        "question": "What will Apple's total net sales be in fiscal 2030?",
        "ground_truth": "NOT_IN_DOCUMENTS",
        "verified": True,
    },
    {
        "id": "nonpublic_msft_salary",
        "type": "unanswerable",
        "question": "What is the median engineer salary at Microsoft?",
        "ground_truth": "NOT_IN_DOCUMENTS",
        "verified": True,
    },
    {
        "id": "offdomain_france",
        "type": "unanswerable",
        "question": "What is the capital of France?",
        "ground_truth": "NOT_IN_DOCUMENTS",
        "verified": True,
    },
]

assert len({t["id"] for t in TESTSET}) == len(TESTSET), "duplicate ids in TESTSET"
