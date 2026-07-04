# Financial RAG Challenge — OfficeQA / U.S. Treasury Bulletins
RAG system over U.S. Treasury Bulletins (2010–2015, a 6-year window), built with a **Baseline**
(naive chunking, no metadata filtering) and an **Engineered** version
(token-aware chunking + Year/Month metadata filtering), evaluated on both
Retriever metrics (Hit Rate@5, MRR, Recall) and Generator metrics
(Groundedness, Factual Accuracy, Hallucination Rate).

## 0. Prereqs

```bash
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token        # OfficeQA is gated, request access first
export ANTHROPIC_API_KEY=your_anthropic_key   # used for the "Generator" (Student)
```

The dataset is gated on Hugging Face (`databricks/officeqa`). Go to
https://huggingface.co/datasets/databricks/officeqa , request access, then
generate a token at https://huggingface.co/settings/tokens with read scope.

## 1. Pipeline

```bash
# Step 1 — download only the transformed .txt bulletins + the answer key CSV
python src/download_data.py

# Step 2 — filter officeqa_full.csv down to 2022-2025 questions whose source
# files are ALL present in our downloaded subset (avoids "manual errors")
python src/filter_questions.py --years 2022 2023 2024 2025

# Step 3 — build the Baseline index (naive fixed-size chunking, no metadata)
# and run retrieval + generation, saving raw predictions
python src/run_pipeline.py --mode baseline

# Step 4 — build the Engineered index (512-token overlap chunking,
# Year/Month metadata tags + query-time filtering) and run the same eval
python src/run_pipeline.py --mode engineered

# Step 5 — print the scorecard (Part 1 of the discussion board post)
python src/scorecard.py
```

All intermediate artifacts land in `results/`:
- `results/baseline_predictions.json`
- `results/engineered_predictions.json`
- `results/scorecard.json` / printed table

## 2. Architecture

| Component | Baseline | Engineered |
|---|---|---|
| Vector DB | ChromaDB (in-memory, `default` collection) | ChromaDB (persistent, `treasury_2022_2025` collection) |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) | same, for apples-to-apples comparison |
| Chunking | Naive fixed-size: 1000 **characters**, 0 overlap, split on `\n\n` only as a fallback | Token-aware: ~512 **tokens**, 15% (≈75 token) overlap, paragraph-boundary aware via `RecursiveCharacterTextSplitter` equivalent |
| Metadata | None — every chunk is anonymous text | Every chunk tagged with `{year, month, source_file}` parsed from the filename `treasury_bulletin_{YEAR}_{MONTH}.txt` |
| Query-time filtering | Pure semantic search over the whole corpus | If the question mentions a specific year/month (regex + simple date parse), the Chroma `where` filter restricts search to that year (±1 month) before ranking; otherwise falls back to full corpus |
| Retrieval | top-k=5 cosine similarity | top-k=5 cosine similarity, filtered subset first, backfilled from full corpus if filter returns <5 |
| Generation | Single-shot Claude call with raw top-5 chunks concatenated, no citation instructions | Claude call with top-5 chunks labeled by source/year/month, explicit "only use the provided context, cite which chunk each number comes from" instruction |

### Why Year/Month metadata matters here
Treasury Bulletins repeat the same statistics (e.g., "average interest rate
on public debt") every year — the token-similarity signal alone often pulls
the *wrong year's* table because the surrounding prose is nearly identical
across issues. Tagging chunks with `year`/`month` and filtering by the
year mentioned in the question removes those near-duplicate distractors
before ranking, which is where most of the Hit Rate/MRR gains come from
(see `results/scorecard.json` after you run it).

## 3. Metrics — how each one is computed

All metrics use **K=5**.

- **Hit Rate@5** = (# queries where at least one gold `source_file` appears
  in the top-5 retrieved chunks) / (# queries)
- **MRR** = mean over queries of `1 / rank_of_first_correct_chunk` (0 if no
  correct chunk in top-5)
- **Recall@5** = (# gold source files found across top-5) / (# gold source
  files required for that question), averaged over queries — this is why
  multi-document questions cap below 100% if only 1 of 2 needed docs is retrieved
- **Groundedness** = each factual claim extracted from the generated answer
  (split on sentences containing a number or named entity) is checked for
  entailment against the retrieved context using a second Claude call
  ("Is this claim supported by the context? yes/no"); Groundedness =
  supported claims / total claims
- **Factual Accuracy** = generated final numeric/text answer compared to
  `officeqa_full.csv` ground truth using ±1% relative tolerance for numbers,
  exact-ish string match otherwise
- **Hallucination Rate** = 1 − Groundedness (fabricated claims / total claims)

See `src/metrics.py` for the exact implementation.
