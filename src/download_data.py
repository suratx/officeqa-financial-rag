"""
Downloads:
  - officeqa_full.csv (the answer key)
  - treasury_bulletins_parsed/transformed/*.txt  (Markdown-ified bulletins)

Only pulls .txt files whose filename year falls in TARGET_YEARS to avoid
downloading the full ~460MB corpus when we only need 4 years.

Filename convention (per the repo README):
  treasury_bulletin_{YEAR}_{MONTH_NUM}.txt
"""
import os
import re
import argparse
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "databricks/officeqa"
REPO_TYPE = "dataset"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main(years):
    os.makedirs(os.path.join(DATA_DIR, "bulletins"), exist_ok=True)

    api = HfApi()
    print("Listing repo files...")
    all_files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)

    # 1. answer key CSV
    csv_candidates = [f for f in all_files if f.endswith("officeqa_full.csv")]
    if not csv_candidates:
        raise SystemExit("officeqa_full.csv not found in repo file listing")
    csv_path = hf_hub_download(REPO_ID, csv_candidates[0], repo_type=REPO_TYPE,
                                local_dir=DATA_DIR)
    print(f"Downloaded answer key -> {csv_path}")

    # 2. transformed txt bulletins, filtered to target years
    pattern = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.txt$")
    txt_files = [f for f in all_files if "transformed" in f and f.endswith(".txt")]

    wanted = []
    for f in txt_files:
        m = pattern.search(f)
        if m and int(m.group(1)) in years:
            wanted.append(f)

    print(f"Found {len(wanted)} bulletins for years {sorted(years)} "
          f"(out of {len(txt_files)} total transformed .txt files)")

    for f in wanted:
        hf_hub_download(REPO_ID, f, repo_type=REPO_TYPE,
                         local_dir=os.path.join(DATA_DIR, "bulletins_raw"))

    print("Done. Files under data/bulletins_raw/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    args = ap.parse_args()
    main(set(args.years))
