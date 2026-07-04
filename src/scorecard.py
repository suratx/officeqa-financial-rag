import os
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load(mode):
    path = os.path.join(RESULTS_DIR, f"{mode}_predictions.json")
    with open(path) as f:
        return json.load(f)["summary"]


def pct(x):
    return f"{x * 100:.1f}%"


def main():
    b = load("baseline")
    e = load("engineered")

    rows = [
        ("Hit Rate (K=5)", pct(b["hit_rate_at_5"]), pct(e["hit_rate_at_5"])),
        ("MRR", f"{b['mrr']:.2f}", f"{e['mrr']:.2f}"),
        ("Recall (K=5)", pct(b["recall_at_5"]), pct(e["recall_at_5"])),
        ("Groundedness", pct(b["groundedness"]), pct(e["groundedness"])),
        ("Factual Accuracy", pct(b["factual_accuracy"]), pct(e["factual_accuracy"])),
        ("Hallucination Rate", pct(b["hallucination_rate"]), pct(e["hallucination_rate"])),
    ]

    print(f"{'Metric':<20}{'Baseline':<15}{'Engineered':<15}")
    print("-" * 50)
    for name, bv, ev in rows:
        print(f"{name:<20}{bv:<15}{ev:<15}")

    with open(os.path.join(RESULTS_DIR, "scorecard.json"), "w") as f:
        json.dump({"baseline": b, "engineered": e}, f, indent=2)


if __name__ == "__main__":
    main()
