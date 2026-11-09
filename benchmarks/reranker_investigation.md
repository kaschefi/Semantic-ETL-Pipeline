# 🔬 Technical Investigation: Root Causes of Reranking & Hybrid Search Regressions

## 📋 Executive Overview

In the initial benchmark run, adding **Hybrid Search + Groq Cross-Encoder Reranking** unexpectedly resulted in worse retrieval metrics than plain **Dense Vector-Only Search**:
- **MRR:** $0.2083$ (Dense-Only) $\rightarrow 0.1333$ (Hybrid + Rerank) ($-36.01\%$ Regression)
- **NDCG@5:** $0.2715 \rightarrow 0.1296$ ($-52.27\%$ Regression)
- **Recall@5:** $0.40 \rightarrow 0.30$ ($-25.0\%$ Regression)
- **Latency:** $389\text{ms} \rightarrow 9,160\text{ms}$ ($23.5\times$ slower)

This document details the deep-dive investigation into the 5 core architectural areas, identifies the exact bugs responsible, and outlines the fixes implemented.

---

## 🔍 Investigation Findings across 5 Key Areas

### 1. Candidate Pool Size & Reranker Truncation Asymmetry
- **Finding:** In [`benchmarks/config.py`](file:///c:/Users/mkrad/PycharmProjects/Semantic-ETL-Pipeline/benchmarks/config.py) and [`benchmarks/baselines.py`](file:///c:/Users/mkrad/PycharmProjects/Semantic-ETL-Pipeline/benchmarks/baselines.py), `search_hybrid_rerank` was configured with `top_k=8` candidates and `rerank_top_n=3`.
- **Root Cause:** When `calculate_retrieval_metrics_for_query` computed $\text{Recall}@5$ and $\text{NDCG}@5$:
  - `search_vector_only`, `search_lexical_only`, and `search_hybrid` returned 8 candidates (positions 1 through 8).
  - `search_hybrid_rerank` sliced candidates down to `top_n=3` items. Positions 4 and 5 were **completely omitted**, making it mathematically impossible for `search_hybrid_rerank` to find relevant documents at rank 4 or 5.
  - Furthermore, passing only 8 initial candidates to the reranker gave it too narrow a pool to recover relevant documents that were ranked 9th–16th by dense search.
- **Fix Implemented:**
  1. Expanded initial candidate retrieval to `fetch_k = max(top_k * 2, 16)` in `search_hybrid_rerank`.
  2. Set `retrieval_rerank_top_n: int = 8` (matching `retrieval_top_k`), preserving all top-K ranks for fair evaluation at $k=1, 3, 5, 8$.

---

### 2. Reranker Prompt Formatting & JSON Parsing Failure
- **Finding:** In [`src/agents/reranker.py`](file:///c:/Users/mkrad/PycharmProjects/Semantic-ETL-Pipeline/src/agents/reranker.py), the reranker prompt requested:
  ```text
  "Return a JSON array of candidate indices (1-indexed) ordered from MOST relevant to LEAST relevant."
  "Example output format: {\"ranked_indices\": [3, 1, 2]}"
  ```
- **Root Cause:** When presented with 8 candidates, the LLM (`openai/gpt-oss-120b`) in `json_object` mode frequently output `{"ranked_indices": [12345678]}` without commas.
  - When the Python parser checked:
    ```python
    if isinstance(idx, int) and 1 <= idx <= len(candidates):
    ```
    `12345678` failed `idx <= len(candidates)` (8).
  - As a result, `reranked` was empty, and the code silently fell back to the un-reranked candidate order (`candidates[:top_n]`).
  - In addition, the prompt truncated candidate content to only 300 characters (`text[:300]`), discarding tables, code, and definitions that appeared later in the chunk.
- **Fix Implemented:**
  1. Structured the prompt to request typed candidate objects:
     `{"ranked_candidates": [{"index": 2, "relevance_score": 9.5}, {"index": 1, "relevance_score": 3.0}]}`.
  2. Added a robust digit-splitting fallback: if a single merged integer like `12345678` is returned, it is automatically decomposed into `[1, 2, 3, 4, 5, 6, 7, 8]`.
  3. Increased candidate context preview to 800 characters with chunk summary and category metadata included.

---

### 3. Groq LLM Daily Quota (TPD) Exhaustion & Rate Limits (429)
- **Finding:** Inspecting live runtime logs revealed:
  ```text
  [Reranker Warning] Context reranking fallback: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` ... on tokens per day (TPD): Limit 200000, Used 199807'}}
  ```
- **Root Cause:** The `openai/gpt-oss-120b` model reached its 200,000 Tokens Per Day (TPD) limit on the free tier. Every subsequent reranker call caught a 429 exception and fell back to the raw unranked candidate list, while still incurring latency timeouts.
- **Fix Implemented:**
  - Added automatic multi-model fallback in `ContextReranker` and `ChunkEnricher`: tries `openai/gpt-oss-120b` first, and automatically falls back to `openai/gpt-oss-20b` and `qwen/qwen3.6-27b` upon receiving 429 quota exhaustion.

---

### 4. Lexical Score Weighting & Token Overlap Calibration
- **Finding:** Hybrid search fuses dense embedding similarity and lexical exact-token matching:
  $$\text{Hybrid Score} = 0.70 \cdot \text{Dense Score} + 0.30 \cdot \text{Keyword Score}$$
- **Analysis:**
  - $\text{Dense Score}$ from Pinecone cosine similarity ranges between $0.15$ and $0.85$.
  - $\text{Keyword Score} = \frac{|\text{Query Tokens} \cap \text{Chunk Tokens}|}{|\text{Query Tokens}|}$ ranges between $0.0$ and $1.0$.
  - When query tokens contain generic terms (e.g., `"what"`, `"how"`, `"difference"`), small non-relevant chunks with accidental token overlaps received a $+0.15$ to $+0.30$ boost, occasionally displacing high-relevance semantic matches.
- **Fix Implemented:**
  - Ensured token extraction in `ContextRetriever` and `BaselineRetrievalEvaluator` operates on meaningful domain tokens and preserves dense similarity as the primary anchor (70%).

---

### 5. Ground Truth Alignment & Exact Query Set Size
- **Finding:**
  - The evaluation suite contains **exactly 25 ground-truth queries** (`GROUND_TRUTH_EVAL_QUERIES`) spanning 7 technical domains (Agile, Requirements, UML, Testing, DevOps, Research Methods, Databases).
  - In fast mode (`--fast`), exactly the first 10 queries ($Q01$ to $Q10$) are evaluated.
  - When evaluating on a small 2-document corpus (`01.pdf` [DevOps] and `04_ALDA_Sortieren.pdf` [Algorithms]), queries targeting other domains (e.g., $Q02$ Requirements, $Q03$ UML, $Q04$ Testing) find zero relevant chunks across all strategies, producing baseline scores of $0.0$ for those queries.
- **Takeaway:** For full end-to-end evaluation, the entire document corpus must be indexed so all 25 queries find their respective domain document chunks.

---

## 📊 Summary of Implemented Fixes

| Component | Bug / Limitation | Fix Applied |
|---|---|---|
| **Reranker Schema** | LLM output `[12345678]` without commas failed bounds check | Structured `ranked_candidates` JSON schema + digit fallback parsing |
| **Candidate Slicing** | `rerank_top_n=3` truncated list, breaking Recall@5 / NDCG@5 | `retrieval_rerank_top_n=8` preserving all evaluation ranks |
| **Candidate Pool** | Initial pool of 8 candidates was too narrow for reranking | Increased `fetch_k = max(top_k * 2, 16)` for hybrid search |
| **Model Quotas (429)** | `gpt-oss-120b` hit 200k daily token limit, causing silent fallbacks | Added automatic fallback chain (`openai/gpt-oss-20b`, `qwen/qwen3.6-27b`) |
| **Delta Sign Reporting** | Lift percentages formatted with `+{lift}%` showed regressions as lifts | Created `format_signed_delta()` explicitly labeling regressions vs lifts |
