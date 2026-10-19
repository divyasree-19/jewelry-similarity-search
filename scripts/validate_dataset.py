"""Validate the catalogue and the labelled evaluation set.

Usage (from the project root):
    python scripts/validate_dataset.py              # errors fail; coverage gaps are warnings
    python scripts/validate_dataset.py --strict     # coverage gaps fail too (Stage 2 "done" check)
    python scripts/validate_dataset.py --markdown   # also print the counts table for the README

Layout:
    data/catalogue.csv    image_id,product_id,project_id,category,subcategory,image_path
    data/categories.txt   allowed categories, one per line
    data/catalogue/<category>/<image_id>.jpg
    eval/queries.csv      query_id,query_path,category,expected_product_ids,edge_case_tags
    data/queries/<query_id>.jpg

image_path and query_path are relative to the project root. expected_product_ids and
edge_case_tags are separated by ";".
"""

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATALOGUE_COLUMNS = ["image_id", "product_id", "project_id", "category", "subcategory", "image_path"]
QUERY_COLUMNS = ["query_id", "query_path", "category", "expected_product_ids", "edge_case_tags"]
EDGE_CASE_TAGS = [
    "colour_mismatch", "lighting", "reflection", "angle",
    "rotation", "background", "distance", "near_duplicate",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_QUERIES = 30
MIN_QUERIES_PER_TAG = 3


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    catalogue: list[dict] = field(default_factory=list)
    queries: list[dict] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def split_list(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def load_categories(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def read_csv(path: Path, columns: list[str], report: Report) -> list[dict]:
    if not path.exists():
        report.error(f"{path.name}: file not found at {path}")
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            report.error(f"{path.name}: missing columns {missing}")
            return []
        rows = []
        for line_no, row in enumerate(reader, start=2):
            row = {k: (v or "").strip() for k, v in row.items() if k is not None}
            row["_line"] = line_no
            rows.append(row)
        return rows


def check_image(path: Path, label: str, report: Report) -> str | None:
    """Return the file's sha256 if it is a readable image, else record an error."""
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        report.error(f"{label}: unsupported extension {path.suffix!r} ({path})")
    if not path.is_file():
        report.error(f"{label}: file not found ({path})")
        return None
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
    except Exception as exc:
        report.error(f"{label}: unreadable image ({path}): {exc}")
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_catalogue(root: Path, categories: set[str], report: Report) -> dict[str, str]:
    """Validate data/catalogue.csv. Returns file hash -> image_id for leak checks."""
    rows = read_csv(root / "data" / "catalogue.csv", CATALOGUE_COLUMNS, report)
    report.catalogue = rows
    if not rows and not report.errors:
        report.error("catalogue.csv: no rows")

    seen_ids: dict[str, int] = {}
    seen_paths: dict[str, int] = {}
    hashes: dict[str, str] = {}
    product_category: dict[str, str] = {}

    for row in rows:
        where = f"catalogue.csv line {row['_line']}"
        empty = [c for c in CATALOGUE_COLUMNS if c != "subcategory" and not row.get(c)]
        if empty:
            report.error(f"{where}: empty {empty}")
            continue

        image_id, product_id, category = row["image_id"], row["product_id"], row["category"]
        if image_id in seen_ids:
            report.error(f"{where}: duplicate image_id {image_id!r} (first on line {seen_ids[image_id]})")
        seen_ids.setdefault(image_id, row["_line"])

        if row["image_path"] in seen_paths:
            report.error(f"{where}: image_path reused (first on line {seen_paths[row['image_path']]})")
        seen_paths.setdefault(row["image_path"], row["_line"])

        if category not in categories:
            report.error(f"{where}: unknown category {category!r}")

        if product_id in product_category and product_category[product_id] != category:
            report.error(
                f"{where}: product {product_id!r} is in {category!r} here but "
                f"{product_category[product_id]!r} elsewhere"
            )
        product_category.setdefault(product_id, category)

        rel = Path(row["image_path"])
        if rel.parts[:3] != ("data", "catalogue", category) or rel.stem != image_id:
            report.error(f"{where}: image_path should be data/catalogue/{category}/{image_id}.<ext>, got {rel}")

        digest = check_image(root / rel, where, report)
        if digest:
            if digest in hashes:
                report.warn(f"{where}: same file content as image_id {hashes[digest]!r}")
            hashes.setdefault(digest, image_id)

    return hashes


def validate_queries(root: Path, categories: set[str], catalogue_hashes: dict[str, str], report: Report) -> None:
    rows = read_csv(root / "eval" / "queries.csv", QUERY_COLUMNS, report)
    report.queries = rows
    product_category = {r["product_id"]: r["category"] for r in report.catalogue if r.get("product_id")}

    seen_ids: dict[str, int] = {}
    for row in rows:
        where = f"queries.csv line {row['_line']}"
        query_id, category = row["query_id"], row["category"]
        empty = [c for c in ("query_id", "query_path", "category") if not row.get(c)]
        if empty:
            report.error(f"{where}: empty {empty}")
            continue

        if query_id in seen_ids:
            report.error(f"{where}: duplicate query_id {query_id!r} (first on line {seen_ids[query_id]})")
        seen_ids.setdefault(query_id, row["_line"])

        if category not in categories:
            report.error(f"{where}: unknown category {category!r}")

        expected = split_list(row["expected_product_ids"])
        if not expected:
            report.error(f"{where}: query {query_id!r} has no expected_product_ids")
        for pid in expected:
            if pid not in product_category:
                report.error(f"{where}: expected product {pid!r} is not in the catalogue")
            elif product_category[pid] != category:
                report.error(
                    f"{where}: expected product {pid!r} is in {product_category[pid]!r}, "
                    f"but the query category is {category!r} (category search would never find it)"
                )

        for tag in split_list(row["edge_case_tags"]):
            if tag not in EDGE_CASE_TAGS:
                report.error(f"{where}: unknown edge_case_tag {tag!r} (allowed: {', '.join(EDGE_CASE_TAGS)})")

        rel = Path(row["query_path"])
        if rel.parts[:2] != ("data", "queries") or rel.stem != query_id:
            report.error(f"{where}: query_path should be data/queries/{query_id}.<ext>, got {rel}")

        digest = check_image(root / rel, where, report)
        if digest and digest in catalogue_hashes:
            report.warn(
                f"{where}: query image is byte-identical to catalogue image {catalogue_hashes[digest]!r} "
                "(this measures nothing; use a different photo)"
            )


def check_coverage(report: Report) -> list[str]:
    gaps = []
    if len(report.queries) < MIN_QUERIES:
        gaps.append(f"only {len(report.queries)} queries (aim for at least {MIN_QUERIES})")
    tag_counts = Counter(tag for q in report.queries for tag in split_list(q.get("edge_case_tags", "")))
    for tag in EDGE_CASE_TAGS:
        if tag_counts[tag] < MIN_QUERIES_PER_TAG:
            gaps.append(f"edge_case_tag {tag!r} has {tag_counts[tag]} queries (need {MIN_QUERIES_PER_TAG})")
    return gaps


def counts_table(report: Report, categories: set[str], markdown: bool) -> str:
    images = Counter(r.get("category") for r in report.catalogue)
    products: dict[str, set] = defaultdict(set)
    for r in report.catalogue:
        products[r.get("category")].add(r.get("product_id"))
    queries = Counter(q.get("category") for q in report.queries)

    rows = [(c, len(products[c]), images[c], queries[c]) for c in sorted(categories | set(images) | set(queries)) if c]
    total = ("total", sum(len(products[c]) for c, *_ in rows), sum(r[2] for r in rows), sum(r[3] for r in rows))

    if markdown:
        lines = ["| Category | Products | Images | Queries |", "|---|---:|---:|---:|"]
        lines += [f"| {c} | {p} | {i} | {q} |" for c, p, i, q in rows]
        lines.append(f"| **{total[0]}** | **{total[1]}** | **{total[2]}** | **{total[3]}** |")
    else:
        lines = [f"{'category':<14}{'products':>10}{'images':>10}{'queries':>10}"]
        lines += [f"{c:<14}{p:>10}{i:>10}{q:>10}" for c, p, i, q in rows + [total]]
    return "\n".join(lines)


def tag_table(report: Report) -> str:
    tag_counts = Counter(tag for q in report.queries for tag in split_list(q.get("edge_case_tags", "")))
    return "\n".join(f"{tag:<18}{tag_counts[tag]:>4}" for tag in EDGE_CASE_TAGS)


def validate(root: Path) -> tuple[Report, set[str]]:
    report = Report()
    categories_path = root / "data" / "categories.txt"
    if not categories_path.exists():
        report.error(f"categories.txt: file not found at {categories_path}")
        return report, set()
    categories = load_categories(categories_path)
    hashes = validate_catalogue(root, categories, report)
    validate_queries(root, categories, hashes, report)
    return report, categories


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="project root (default: this repo)")
    parser.add_argument("--strict", action="store_true", help="treat coverage gaps as errors")
    parser.add_argument("--markdown", action="store_true", help="print the counts table as Markdown")
    args = parser.parse_args(argv)

    report, categories = validate(args.root)
    gaps = check_coverage(report)

    print("Counts per category:")
    print(counts_table(report, categories, markdown=False))
    print("\nQueries per edge-case tag:")
    print(tag_table(report))

    for w in report.warnings:
        print(f"WARNING  {w}")
    for gap in gaps:
        print(f"{'ERROR' if args.strict else 'WARNING'}  coverage: {gap}")
    for e in report.errors:
        print(f"ERROR    {e}")

    error_count = len(report.errors) + (len(gaps) if args.strict else 0)
    warning_count = len(report.warnings) + (0 if args.strict else len(gaps))
    print(f"\n{error_count} error(s), {warning_count} warning(s)")

    if args.markdown:
        print("\n" + counts_table(report, categories, markdown=True))
    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())
