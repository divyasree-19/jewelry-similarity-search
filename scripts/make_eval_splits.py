"""Add a fixed tune / holdout `split` column to eval query files (Stage 10).

About 30% of each group goes to holdout, where a group is the first edge-case tag (stratified), shuffled
with a fixed seed. Re-running gives the same split. Files with fewer than --min-queries rows are left
alone (report only).

Usage (from the project root):
    python scripts/make_eval_splits.py                      # eval/queries.csv and eval/text_queries.csv
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILES = {"queries.csv": "edge_case_tags", "text_queries.csv": "tags"}


def assign(rows: list[dict], tag_column: str, holdout_share: float, seed: int) -> dict[str, str]:
    groups = defaultdict(list)
    for row in rows:
        first_tag = (row.get(tag_column) or "").split(";")[0].strip() or "(none)"
        groups[first_tag].append(row["query_id"])
    rng = random.Random(seed)
    split = {}
    for tag in sorted(groups):
        ids = sorted(groups[tag])
        rng.shuffle(ids)
        holdout = round(len(ids) * holdout_share)
        for position, query_id in enumerate(ids):
            split[query_id] = "holdout" if position < holdout else "tune"
    return split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout-share", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-queries", type=int, default=15)
    args = parser.parse_args()

    for name, tag_column in FILES.items():
        path = PROJECT_ROOT / "eval" / name
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows, columns = list(reader), list(reader.fieldnames)
        if len(rows) < args.min_queries:
            print(f"{name}: {len(rows)} queries < {args.min_queries}; left for reporting only")
            continue
        split = assign(rows, tag_column, args.holdout_share, args.seed)
        if "split" not in columns:
            columns.append("split")
        for row in rows:
            row["split"] = split[row["query_id"]]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        holdout = sorted(q for q, s in split.items() if s == "holdout")
        print(f"{name}: {len(rows) - len(holdout)} tune, {len(holdout)} holdout: {', '.join(holdout)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
