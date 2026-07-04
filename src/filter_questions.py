"""
Filters officeqa_full.csv down to questions that are FULLY answerable from
our downloaded subset (i.e. every file listed in `source_files` must be
within the target years). This avoids the "manual filtering error"
trap called out in the assignment: a question needing a 2019 bulletin AND
a 2023 bulletin is dropped entirely rather than silently scored wrong.
"""
import os
import re
import ast
import argparse
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FILENAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})")


def parse_source_files(cell):
    """source_files column may be a Python-list-as-string, or a single string
    with entries separated by CRLF/LF, commas, or semicolons."""
    if isinstance(cell, str) and cell.strip().startswith("["):
        try:
            parsed = ast.literal_eval(cell)
            return [p.strip() for p in parsed]
        except Exception:
            pass
    if isinstance(cell, str):
        parts = re.split(r"[\r\n,;]+", cell)
        return [p.strip() for p in parts if p.strip()]
    return []


def years_of(files):
    yrs = set()
    for f in files:
        m = FILENAME_RE.search(f)
        if m:
            yrs.add(int(m.group(1)))
    return yrs


def main(years):
    csv_path = os.path.join(DATA_DIR, "officeqa_full.csv")
    df = pd.read_csv(csv_path)

    if "source_files" not in df.columns:
        raise SystemExit(f"Expected a 'source_files' column, got: {list(df.columns)}")

    df["_source_files_parsed"] = df["source_files"].apply(parse_source_files)
    df["_years_needed"] = df["_source_files_parsed"].apply(years_of)

    target = set(years)
    mask = df["_years_needed"].apply(lambda ys: len(ys) > 0 and ys.issubset(target))
    filtered = df[mask].drop(columns=["_source_files_parsed", "_years_needed"])

    out_path = os.path.join(DATA_DIR, f"officeqa_filtered_{min(years)}_{max(years)}.csv")
    filtered.to_csv(out_path, index=False)
    print(f"Kept {len(filtered)} / {len(df)} questions "
          f"(years {sorted(target)}) -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    args = ap.parse_args()
    main(args.years)