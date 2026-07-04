import re


def hit_rate_at_k(gold_files_per_q, retrieved_files_per_q, k=5):
    hits = 0
    for gold, retrieved in zip(gold_files_per_q, retrieved_files_per_q):
        top_k = set(retrieved[:k])
        if top_k & set(gold):
            hits += 1
    return hits / len(gold_files_per_q) if gold_files_per_q else 0.0


def mrr_at_k(gold_files_per_q, retrieved_files_per_q, k=5):
    total = 0.0
    for gold, retrieved in zip(gold_files_per_q, retrieved_files_per_q):
        rank = None
        for i, f in enumerate(retrieved[:k], start=1):
            if f in gold:
                rank = i
                break
        total += (1.0 / rank) if rank else 0.0
    return total / len(gold_files_per_q) if gold_files_per_q else 0.0


def recall_at_k(gold_files_per_q, retrieved_files_per_q, k=5):
    total = 0.0
    for gold, retrieved in zip(gold_files_per_q, retrieved_files_per_q):
        if not gold:
            continue
        top_k = set(retrieved[:k])
        found = len(top_k & set(gold))
        total += found / len(set(gold))
    return total / len(gold_files_per_q) if gold_files_per_q else 0.0


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def numeric_match(pred, gold, tolerance=0.01):
    """±1% relative tolerance for numeric answers, else normalized string match."""
    pred_nums = _NUM_RE.findall(str(pred).replace(",", ""))
    gold_nums = _NUM_RE.findall(str(gold).replace(",", ""))
    if gold_nums:
        try:
            gv = float(gold_nums[0])
            if gv == 0:
                return str(gold).strip().lower() in str(pred).strip().lower()
            for pn in pred_nums:
                pv = float(pn)
                if abs(pv - gv) / abs(gv) <= tolerance:
                    return True
            return False
        except ValueError:
            pass
    # fallback: normalized substring match
    return str(gold).strip().lower() in str(pred).strip().lower()


def factual_accuracy(predictions, golds, tolerance=0.01):
    correct = sum(numeric_match(p, g, tolerance) for p, g in zip(predictions, golds))
    return correct / len(golds) if golds else 0.0


def groundedness_and_hallucination(claim_judgments):
    """
    claim_judgments: list of lists of booleans (True = claim supported by
    context), one inner list per question, produced by the claim-checking
    LLM call in generate.py.
    """
    all_claims = [c for q in claim_judgments for c in q]
    if not all_claims:
        return 0.0, 0.0
    supported = sum(all_claims)
    groundedness = supported / len(all_claims)
    hallucination = 1 - groundedness
    return groundedness, hallucination
