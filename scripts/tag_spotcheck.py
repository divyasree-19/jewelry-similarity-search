"""Spot-check VLM tags by hand (Stage 7).

export: writes eval/tag_spotcheck.csv with 50 random tagged images. For each field there is the VLM value,
        a ref_<field> column from data/products.csv to help you judge, and an empty <field>_ok column.
        Mark each <field>_ok as 1 (correct) or 0 (wrong); leave blank to skip. Refuses to overwrite a CSV
        that already has marks unless --force.
score:  prints accuracy per field from your marks, plus how often the VLM agrees with products.csv
        (automatic, no marks needed; products.csv was written by hand from the same images, so it is a
        reference, not ground truth).

Usage (from the project root):
    python scripts/tag_spotcheck.py export
    python scripts/tag_spotcheck.py score
"""

import argparse
import csv
import random
import sys
from pathlib import Path

from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.tagging import TAGGER_VERSION  # noqa: E402
from app.tagging.vocab import LIST_FIELDS, PROVISIONAL_FIELDS, normalize  # noqa: E402

CSV_PATH = PROJECT_ROOT / "eval" / "tag_spotcheck.csv"
FIELDS = ["category", "metal", "gemstones", "stone_colors", "motifs", "style", "description"]
# products.csv column for each tag field (description has no reference)
REFERENCE = {"category": "category", "metal": "metal", "gemstones": "gemstones", "stone_colors": "stone_colours",
             "motifs": "motifs", "style": "style"}
TRUE, FALSE = {"1", "y", "yes", "true", "ok", "correct"}, {"0", "n", "no", "false", "wrong"}


def as_text(value) -> str:
    return ";".join(value) if isinstance(value, list) else (value or "")


def export(sample_size: int, seed: int, force: bool) -> int:
    if CSV_PATH.exists() and not force:
        with CSV_PATH.open(newline="", encoding="utf-8") as f:
            if any(row.get(f"{field}_ok", "").strip() for row in csv.DictReader(f) for field in FIELDS):
                sys.exit(f"{CSV_PATH.relative_to(PROJECT_ROOT)} already has marks; use --force to overwrite")

    settings = get_settings()
    db = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=settings.mongo_timeout_ms)[settings.mongo_db]
    docs = list(db.images.find({"vlm_tags.tagger_version": TAGGER_VERSION}).sort("_id", 1))
    if not docs:
        sys.exit("No tagged images; run scripts/tag_catalogue.py first")
    sample = random.Random(seed).sample(docs, min(sample_size, len(docs)))
    with (PROJECT_ROOT / "data" / "products.csv").open(newline="", encoding="utf-8") as f:
        products = {row["product_id"]: row for row in csv.DictReader(f)}

    columns = ["image_id", "product_id", "image_path"]
    for field in FIELDS:
        columns += [field] + ([f"ref_{field}"] if field in REFERENCE else []) + [f"{field}_ok"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for doc in sorted(sample, key=lambda d: d["_id"]):
            product = products.get(doc["product_id"], {})
            row = {"image_id": doc["_id"], "product_id": doc["product_id"], "image_path": doc["image_path"]}
            for field in FIELDS:
                row[field] = as_text(doc["vlm_tags"].get(field))
                if field in REFERENCE:
                    row[f"ref_{field}"] = product.get(REFERENCE[field], "")
                row[f"{field}_ok"] = ""
            writer.writerow(row)
    print(f"wrote {len(sample)} rows to {CSV_PATH.relative_to(PROJECT_ROOT)} (tagger {TAGGER_VERSION}, seed {seed})")
    print("Open each image_path, then mark <field>_ok as 1 or 0. Then run: python scripts/tag_spotcheck.py score")
    return 0


def reference_values(field: str, text: str) -> set[str]:
    values = {normalize(field, part) for part in text.split(";") if part.strip()}
    return {value for value in values if value}


def score() -> int:
    if not CSV_PATH.exists():
        sys.exit("No spot-check file; run: python scripts/tag_spotcheck.py export")
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Manual spot-check ({len(rows)} images in {CSV_PATH.relative_to(PROJECT_ROOT)})\n")
    print(f"{'field':14} {'marked':>6} {'correct':>7} {'accuracy':>8}")
    unmarked_total = 0
    for field in FIELDS:
        marks = [row.get(f"{field}_ok", "").strip().lower() for row in rows]
        correct, wrong = sum(m in TRUE for m in marks), sum(m in FALSE for m in marks)
        marked = correct + wrong
        unmarked_total += len(rows) - marked
        accuracy = f"{100 * correct / marked:.0f}%" if marked else "–"
        note = "  (provisional vocabulary)" if field in PROVISIONAL_FIELDS else ""
        print(f"{field:14} {marked:6} {correct:7} {accuracy:>8}{note}")
    if unmarked_total:
        print(f"\n{unmarked_total} field(s) not marked yet; accuracy above covers marked fields only.")

    print("\nAgreement with data/products.csv (automatic; both mapped to the vocabulary)\n")
    print(f"{'field':14} {'rows':>5} {'exact':>6} {'overlap':>8}")
    for field, ref_column in REFERENCE.items():
        exact = overlap = counted = 0
        for row in rows:
            ref = reference_values(field, row.get(f"ref_{field}", ""))
            got = reference_values(field, row.get(field, ""))
            if not ref and not got:
                continue
            counted += 1
            exact += ref == got
            overlap += bool(ref & got) or (not ref and not got)
        print(f"{field:14} {counted:5} {100 * exact / counted if counted else 0:5.0f}% {100 * overlap / counted if counted else 0:7.0f}%")
    print("\nexact = same set of values; overlap = at least one value in common.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--n", type=int, default=50)
    exp.add_argument("--seed", type=int, default=7)
    exp.add_argument("--force", action="store_true")
    sub.add_parser("score")
    args = parser.parse_args()
    return export(args.n, args.seed, args.force) if args.command == "export" else score()


if __name__ == "__main__":
    sys.exit(main())
