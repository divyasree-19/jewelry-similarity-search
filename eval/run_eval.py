"""Run every query in eval/queries.csv through POST /search/image and score the results.

Metrics (a query is a hit at k if any of its expected products is in the top k products):
Recall@1, Recall@5, Recall@10 and MRR@10, overall, per edge-case tag and per category. Also latency per
query (wall clock and per pipeline step) and memory of the embedding container and this process.

Writes eval/results/<run_name>.json, eval/results/<run_name>.md and, for the worst failures, a strip per
query in eval/results/<run_name>_failures/: query | top 5 | an image of the expected product.

Usage (from the project root; the backend's index for that backend must exist, see scripts/ingest.py):
    python eval/run_eval.py --backend siglip2_224
    python eval/run_eval.py --backend siglip2_384 --name siglip2_384_try2
    python eval/run_eval.py --compare siglip2_224 siglip2_384      # Markdown table of saved runs
"""

import argparse
import csv
import json
import logging
import os
import resource
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
CONTAINERS = {"siglip2_224": "siglip2-api", "siglip2_384": "siglip2-api-384"}
CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
KS = (1, 5, 10)
WORST = 10


def split(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def read_queries(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{**row, "expected": split(row["expected_product_ids"]), "tags": split(row["edge_case_tags"])}
                for row in csv.DictReader(f)]


def first_hit_rank(product_ids: list[str], expected: list[str]) -> int | None:
    for rank, product_id in enumerate(product_ids, start=1):
        if product_id in expected:
            return rank
    return None


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    result = {"n": n}
    for k in KS:
        result[f"recall@{k}"] = round(sum(1 for r in ranks if r is not None and r <= k) / n, 4)
    result["mrr@10"] = round(sum(1 / r for r in ranks if r is not None and r <= 10) / n, 4)
    return result


def latency_summary(values: list[float]) -> dict:
    values = sorted(values)
    return {
        "mean": round(statistics.fmean(values), 1),
        "p50": round(values[len(values) // 2], 1),
        "p95": round(values[min(len(values) - 1, int(len(values) * 0.95))], 1),
    }


def container_memory_mb(name: str | None) -> dict | None:
    """Peak (since the container started) and current memory of the embedding container, from its cgroup."""
    if not name:
        return None
    values = {}
    for key in ("peak", "current"):
        try:
            out = subprocess.run(["docker", "exec", name, "cat", f"/sys/fs/cgroup/memory.{key}"],
                                 capture_output=True, text=True, timeout=30, check=True).stdout
            values[key] = round(int(out) / 2**20, 1)
        except (OSError, ValueError, subprocess.SubprocessError):
            values[key] = None
    return values


def run(backend: str, name: str, queries_path: Path) -> dict:
    os.environ["EMBED_BACKEND"] = backend
    from fastapi.testclient import TestClient

    from app import main
    from app.config import get_settings

    get_settings.cache_clear()
    logging.getLogger("search").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    queries = read_queries(queries_path)

    with TestClient(main.app) as client:
        health = client.get("/health").json()
        if health["status"] != "ok":
            sys.exit(f"Backend is not healthy for {backend}: {json.dumps(health['dependencies'])}")
        store = main.app.state.indexes.get()
        if store.total == 0:
            sys.exit(f"No index for {backend}; run: python scripts/ingest.py --backend {backend}")
        settings = main.app.state.settings

        def post(row: dict):
            path = PROJECT_ROOT / row["query_path"]
            files = {"file": (path.name, path.read_bytes(), CONTENT_TYPES[path.suffix.lower()])}
            return client.post("/search/image", files=files, data={"category": row["category"], "top_k": "10"})

        post(queries[0])  # warm-up, not timed
        rows = []
        for row in queries:
            started = time.perf_counter()
            response = post(row)
            wall_ms = (time.perf_counter() - started) * 1000
            if response.status_code != 200:
                sys.exit(f"{row['query_id']}: HTTP {response.status_code} {response.text}")
            body = response.json()
            results = body["results"]
            rank = first_hit_rank([r["product_id"] for r in results], row["expected"])
            rows.append({
                "query_id": row["query_id"], "query_path": row["query_path"], "category": row["category"],
                "tags": row["tags"], "expected": row["expected"], "rank": rank,
                "wall_ms": round(wall_ms, 1), "timings_ms": body["timings_ms"],
                "results": [{key: r[key] for key in ("product_id", "image_id", "image_url", "similarity_score")}
                            for r in results],
            })

    by_tag, by_category = defaultdict(list), defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row["rank"])
        for tag in row["tags"] or ["(none)"]:
            by_tag[tag].append(row["rank"])

    return {
        "run_name": name,
        "backend": backend,
        "model": settings.embed_model_name,
        "embedding_version": settings.embed_version,
        "preprocess_version": main.app.state.preprocess_options.signature,
        "dimension": store.dimension,
        "index_counts": store.counts(),
        "queries_file": str(queries_path.relative_to(PROJECT_ROOT)),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall": metrics([row["rank"] for row in rows]),
        "by_tag": {tag: metrics(ranks) for tag, ranks in sorted(by_tag.items())},
        "by_category": {cat: metrics(ranks) for cat, ranks in sorted(by_category.items())},
        "latency_ms": {
            "wall": latency_summary([row["wall_ms"] for row in rows]),
            "embed": latency_summary([row["timings_ms"]["embed"] for row in rows]),
            "preprocess": latency_summary([row["timings_ms"]["preprocess"] for row in rows]),
            "faiss": latency_summary([row["timings_ms"]["faiss"] for row in rows]),
        },
        "memory_mb": {
            "embedding_container": container_memory_mb(CONTAINERS.get(backend)),
            "eval_process_peak_rss": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        },
        "queries": rows,
    }


def worst_failures(report: dict) -> list[dict]:
    """Misses first (expected product not in top 10), then the lowest-ranked hits; best wrong score breaks ties."""
    failed = [row for row in report["queries"] if row["rank"] != 1]
    top_score = lambda row: row["results"][0]["similarity_score"] if row["results"] else 0  # noqa: E731
    return sorted(failed, key=lambda row: (row["rank"] or 99, top_score(row)), reverse=True)[:WORST]


def expected_image(product_id: str) -> Path | None:
    with (PROJECT_ROOT / "data" / "catalogue.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["product_id"] == product_id:
                return PROJECT_ROOT / row["image_path"]
    return None


def failure_strip(row: dict, catalogue_dir: Path, out: Path) -> None:
    size, label_h = 200, 20
    font = ImageFont.load_default(size=13)
    panels = [(f"{row['query_id']} query", PROJECT_ROOT / row["query_path"])]
    for i, result in enumerate(row["results"][:5], start=1):
        mark = " *" if result["product_id"] in row["expected"] else ""
        path = catalogue_dir / result["image_url"].removeprefix("/catalogue-images/")
        panels.append((f"#{i} {result['product_id']} {result['similarity_score']:.3f}{mark}", path))
    expected_path = expected_image(row["expected"][0])
    rank = f"rank {row['rank']}" if row["rank"] else "not in top 10"
    panels.append((f"expected {row['expected'][0]} ({rank})", expected_path))

    sheet = Image.new("RGB", (size * len(panels), size + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(panels):
        if path and path.exists():
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            sheet.paste(ImageOps.pad(image, (size, size), color=(255, 255, 255)), (i * size, label_h))
        draw.text((i * size + 4, 2), label, fill="black", font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)


def pct(value) -> str:
    return "–" if value is None else f"{value * 100:.0f}%"


def metrics_table(groups: dict[str, dict], label: str) -> list[str]:
    lines = [f"| {label} | n | R@1 | R@5 | R@10 | MRR@10 |", "|---|---:|---:|---:|---:|---:|"]
    for key, m in groups.items():
        lines.append(f"| {key} | {m['n']} | {pct(m['recall@1'])} | {pct(m['recall@5'])} | "
                     f"{pct(m['recall@10'])} | {m['mrr@10']:.3f} |")
    return lines


def write_markdown(report: dict, catalogue_dir: Path) -> Path:
    name = report["run_name"]
    lat, mem = report["latency_ms"], report["memory_mb"]
    lines = [
        f"# Eval run `{name}`",
        "",
        f"Backend `{report['backend']}` · model `{report['model']}` · embedding `{report['embedding_version']}` · "
        f"preprocess `{report['preprocess_version']}` · {report['dimension']}-dim · {report['created_at']}",
        "",
        *metrics_table({"**overall**": report["overall"]}, "Scope"),
        "",
        "## Per edge-case tag", "",
        *metrics_table(report["by_tag"], "Tag"),
        "",
        "## Per category", "",
        *metrics_table(report["by_category"], "Category"),
        "",
        "## Latency and memory", "",
        "| Step | mean ms | p50 ms | p95 ms |", "|---|---:|---:|---:|",
        *[f"| {step} | {v['mean']} | {v['p50']} | {v['p95']} |" for step, v in lat.items()],
        "",
        f"Embedding container memory (cgroup, MiB): {mem['embedding_container'] or 'n/a'} · "
        f"eval process peak RSS (backend + FAISS): {mem['eval_process_peak_rss']} MB",
        "",
        f"## Worst {WORST} failures", "",
        "Strip: query | top 5 (* = expected product) | an image of the expected product.", "",
    ]
    for row in worst_failures(report):
        strip = Path(f"{name}_failures") / f"{row['query_id']}.jpg"
        failure_strip(row, catalogue_dir, RESULTS_DIR / strip)
        top = ", ".join(f"{r['product_id']} {r['similarity_score']:.3f}" for r in row["results"][:5])
        rank = row["rank"] or "miss"
        lines += [f"**{row['query_id']}** · {row['category']} · tags {', '.join(row['tags'])} · "
                  f"expected {', '.join(row['expected'])} · rank {rank}  ", f"top 5: {top}", "",
                  f"![{row['query_id']}]({strip.as_posix()})", ""]
    out = RESULTS_DIR / f"{name}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def compare(names: list[str]) -> str:
    reports = [json.loads((RESULTS_DIR / f"{name}.json").read_text()) for name in names]
    lines = ["| Run | Model | dim | R@1 | R@5 | R@10 | MRR@10 | embed p50 ms | wall p50 ms | container peak MiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        m, lat = r["overall"], r["latency_ms"]
        lines.append(f"| {r['run_name']} | {r['model']} | {r['dimension']} | {pct(m['recall@1'])} | "
                     f"{pct(m['recall@5'])} | {pct(m['recall@10'])} | {m['mrr@10']:.3f} | {lat['embed']['p50']} | "
                     f"{lat['wall']['p50']} | {(r['memory_mb']['embedding_container'] or {}).get('peak', 'n/a')} |")
    tags = sorted({tag for r in reports for tag in r["by_tag"]})
    lines += ["", "| Tag (R@1 / R@5) | " + " | ".join(r["run_name"] for r in reports) + " |",
              "|---|" + "---:|" * len(reports)]
    for tag in tags:
        cells = [f"{pct(r['by_tag'][tag]['recall@1'])} / {pct(r['by_tag'][tag]['recall@5'])}"
                 if tag in r["by_tag"] else "–" for r in reports]
        lines.append(f"| {tag} (n={reports[0]['by_tag'].get(tag, {}).get('n', '?')}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", help="siglip2_224 | siglip2_384 | vertex")
    parser.add_argument("--name", help="run name (default: the backend name)")
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "eval" / "queries.csv")
    parser.add_argument("--compare", nargs="+", metavar="RUN", help="print a comparison of saved runs")
    args = parser.parse_args()

    if args.compare:
        print(compare(args.compare))
        return 0
    if not args.backend:
        parser.error("--backend or --compare is required")

    name = args.name or args.backend
    report = run(args.backend, name, args.queries.resolve())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    from app.config import get_settings
    summary = write_markdown(report, get_settings().catalogue_dir)
    m = report["overall"]
    print(f"{name}: R@1 {pct(m['recall@1'])}  R@5 {pct(m['recall@5'])}  R@10 {pct(m['recall@10'])}  "
          f"MRR@10 {m['mrr@10']:.3f}  ({m['n']} queries)")
    print(f"embed p50 {report['latency_ms']['embed']['p50']} ms, wall p50 {report['latency_ms']['wall']['p50']} ms, "
          f"container memory {report['memory_mb']['embedding_container']} MiB")
    print(f"wrote {summary.relative_to(PROJECT_ROOT)} and {name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
