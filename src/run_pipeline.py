import os
import re
import json
import argparse
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

from chunking import load_bulletins, naive_chunks, token_overlap_chunks
from generate import answer_question, extract_claims, check_claim_supported
from metrics import (
    hit_rate_at_k, mrr_at_k, recall_at_k, factual_accuracy,
    groundedness_and_hallucination,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EMB_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# crude "what year/month is this question about" extractor for the
# engineered pipeline's metadata pre-filter
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}


def extract_year_month_hint(question):
    year = None
    m = re.search(r"\b(19|20)\d{2}\b", question)
    if m:
        year = int(m.group(0))
    month = None
    for name, num in MONTHS.items():
        if name in question.lower():
            month = num
            break
    return year, month


def build_index(mode):
    bulletins = load_bulletins(os.path.join(DATA_DIR, "bulletins_raw"))
    if not bulletins:
        raise SystemExit("No bulletins found in data/bulletins_raw — run download_data.py first")

    client = chromadb.Client()
    coll_name = f"officeqa_{mode}"
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    collection = client.create_collection(coll_name, embedding_function=EMB_FN)

    ids, docs, metadatas = [], [], []
    for b in bulletins:
        if mode == "baseline":
            chunks = naive_chunks(b["text"], chunk_chars=1000)
        else:
            chunks = token_overlap_chunks(b["text"], chunk_tokens=512, overlap_tokens=75)

        for i, chunk in enumerate(chunks):
            ids.append(f"{b['filename']}_{i}")
            docs.append(chunk)
            meta = {"source_file": b["filename"]}
            if mode == "engineered":
                meta["year"] = b["year"]
                meta["month"] = b["month"]
            metadatas.append(meta)

    BATCH = 512
    for i in range(0, len(ids), BATCH):
        collection.add(
            ids=ids[i:i + BATCH],
            documents=docs[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )

    print(f"[{mode}] indexed {len(ids)} chunks from {len(bulletins)} bulletins")
    return collection


def retrieve(collection, question, mode, k=5):
    where = None
    if mode == "engineered":
        year, month = extract_year_month_hint(question)
        if year:
            where = {"year": year}

    if where:
        res = collection.query(query_texts=[question], n_results=k, where=where)
        if not res["ids"][0]:
            res = collection.query(query_texts=[question], n_results=k)
        elif len(res["ids"][0]) < k:
            extra = collection.query(query_texts=[question], n_results=k)
            seen = set(res["ids"][0])
            for j, doc_id in enumerate(extra["ids"][0]):
                if doc_id not in seen and len(res["ids"][0]) < k:
                    res["ids"][0].append(doc_id)
                    res["documents"][0].append(extra["documents"][0][j])
                    res["metadatas"][0].append(extra["metadatas"][0][j])
    else:
        res = collection.query(query_texts=[question], n_results=k)

    hits = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        hits.append({
            "text": doc,
            "filename": meta.get("source_file"),
            "year": meta.get("year"),
            "month": meta.get("month"),
        })
    return hits


def gold_files(row):
    val = row["_source_files_parsed"] if "_source_files_parsed" in row else row["source_files"]
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("["):
            import ast
            try:
                parsed = ast.literal_eval(val)
                return [p.strip() for p in parsed]
            except Exception:
                pass
        parts = re.split(r"[\r\n,;]+", val)
        return [p.strip() for p in parts if p.strip()]
    return list(val)


def main(mode, years, limit=None):
    q_path = os.path.join(DATA_DIR, f"officeqa_filtered_{min(years)}_{max(years)}.csv")
    df = pd.read_csv(q_path)
    if limit:
        df = df.head(limit)

    collection = build_index(mode)

    gold_lists, retrieved_lists = [], []
    predictions, golds_answers = [], []
    claim_judgments = []
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {mode}"):
        question = row["question"]
        gold = gold_files(row)
        gold_lists.append(gold)

        contexts = retrieve(collection, question, mode, k=5)
        retrieved_lists.append([c["filename"] for c in contexts])

        answer = answer_question(question, contexts, engineered=(mode == "engineered"))
        predictions.append(answer)
        golds_answers.append(row.get("answer", row.get("ground_truth", "")))

        claims = extract_claims(answer)
        judgments = [check_claim_supported(c, contexts) for c in claims]
        claim_judgments.append(judgments)

        records.append({
            "question": question,
            "gold_files": gold,
            "retrieved_files": [c["filename"] for c in contexts],
            "answer": answer,
            "gold_answer": golds_answers[-1],
            "claims": claims,
            "claim_supported": judgments,
        })

    hr = hit_rate_at_k(gold_lists, retrieved_lists, k=5)
    mrr = mrr_at_k(gold_lists, retrieved_lists, k=5)
    rec = recall_at_k(gold_lists, retrieved_lists, k=5)
    acc = factual_accuracy(predictions, golds_answers)
    grounded, halluc = groundedness_and_hallucination(claim_judgments)

    summary = {
        "mode": mode,
        "n_questions": len(df),
        "hit_rate_at_5": hr,
        "mrr": mrr,
        "recall_at_5": rec,
        "factual_accuracy": acc,
        "groundedness": grounded,
        "hallucination_rate": halluc,
    }

    out_path = os.path.join(RESULTS_DIR, f"{mode}_predictions.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, default=str)

    print(json.dumps(summary, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "engineered"], required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    ap.add_argument("--limit", type=int, default=None, help="cap #questions for a quick smoke test")
    args = ap.parse_args()
    main(args.mode, args.years, args.limit)