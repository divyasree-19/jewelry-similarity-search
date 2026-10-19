"""Run every query in eval/queries.csv through POST /search/image and score the results.

Metrics (a query is a hit at k if any of its expected products is in the top k products):
Recall@1, Recall@5, Recall@10 and MRR@10, overall, per edge-case tag and per category. Also latency per
query (wall clock and per pipeline step) and memory of the embedding container and this process.

Writes eval/results/<run_name>.json, eval/results/<run_name>.md and, for the worst failures, a strip per
query in eval/results/<run_name>_failures/: query | top 5 | an image of the expected product.

Modes (Stage 8):
    image       eval/queries.csv: query photo -> POST /search/image
    text        eval/text_queries.csv: query text -> POST /search/text (the category is never sent; it must
                come from the text, and the category column is only used to score detection)
    image_text  eval/image_text_queries.csv: query photo + refinement text -> POST /search/image
--category given sends each query's category (image modes); --category detect leaves it out, so the run also
measures category detection. --set KEY=VALUE overrides a backend setting for the run, e.g. SEARCH_HYBRID=false.

Usage (from the project root; the backend's index for that backend must exist, see scripts/ingest.py):
    python eval/run_eval.py --backend siglip2_224
    python eval/run_eval.py --backend siglip2_224 --name s8_baseline --set SEARCH_HYBRID=false --set SEARCH_BOOSTS=false
    python eval/run_eval.py --backend siglip2_224 --mode text --name s8_text
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


QUERY_FILES = {"image": "queries.csv", "text": "text_queries.csv", "image_text": "image_text_queries.csv"}


def read_queries(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{**row, "expected": split(row["expected_product_ids"]),
                 "tags": split(row.get("edge_case_tags") or row.get("tags") or "")}
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


def run(backend: str, name: str, queries_path: Path, mode: str = "image", category_mode: str = "given",
        overrides: dict | None = None, split_name: str = "all") -> dict:
    os.environ["EMBED_BACKEND"] = backend
    os.environ.update(overrides or {})
    from fastapi.testclient import TestClient

    from app import main
    from app.config import get_settings

    get_settings.cache_clear()
    logging.getLogger("search").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    queries = read_queries(queries_path)
    if split_name != "all":
        if any("split" not in row for row in queries):
            sys.exit(f"{queries_path.name} has no split column; run scripts/make_eval_splits.py")
        queries = [row for row in queries if row["split"] == split_name]

    with TestClient(main.app) as client:
        health = client.get("/health").json()
        if health["status"] != "ok":
            sys.exit(f"Backend is not healthy for {backend}: {json.dumps(health['dependencies'])}")
        store = main.app.state.indexes.get()
        if store.total == 0:
            sys.exit(f"No index for {backend}; run: python scripts/ingest.py --backend {backend}")
        settings = main.app.state.settings

        def post(row: dict):
            if mode == "text":
                return client.post("/search/text", json={"text": row["text"], "top_k": 10})
            path = PROJECT_ROOT / row["query_path"]
            files = {"file": (path.name, path.read_bytes(), CONTENT_TYPES[path.suffix.lower()])}
            data = {"top_k": "10"}
            if category_mode == "given":
                data["category"] = row["category"]
            if mode == "image_text":
                data["text"] = row["text"]
            return client.post("/search/image", files=files, data=data)

        post(queries[0])  # warm-up, not timed
        if hasattr(main.app.state, "judge_cache"):
            main.app.state.judge_cache._items.clear()  # the warm-up must not pre-fill the VLM judge cache
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
                "query_id": row["query_id"], "query_path": row.get("query_path"), "text": row.get("text"),
                "category": row["category"], "detected_category": body.get("category"),
                "category_source": body.get("category_source"), "category_confidence": body.get("category_confidence"),
                "needs_category": body.get("needs_category", False), "parsed_text": body.get("parsed_text"),
                "candidates": body.get("category_candidates") or [],
                "vlm_judge": body.get("vlm_judge"),
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
        "mode": mode,
        "split": split_name,
        "category_mode": "from text" if mode == "text" else category_mode,
        "settings_overrides": overrides or {},
        "flags": {"hybrid": settings.search_hybrid, "boosts": settings.search_boosts,
                  "query_tagging": settings.search_query_tagging, "category_detection": settings.category_detection,
                  "category_confidence_threshold": settings.category_confidence_threshold,
                  "refine_metal_hard_filter": settings.refine_metal_hard_filter},
        "category_detection": detection_summary(rows) if mode == "text" or category_mode == "detect" else None,
        "vlm_judge": {
            "qwen_calls_per_query": round(statistics.fmean(r["vlm_judge"]["calls"] for r in rows), 2),
            "cache_hits_per_query": round(statistics.fmean(r["vlm_judge"]["cache_hits"] for r in rows), 2),
            "queries_failed": sum(1 for r in rows if r["vlm_judge"]["failed"]),
            "queries_reordered": sum(1 for r in rows if r["vlm_judge"]["reordered"]),
        } if rows and all(r.get("vlm_judge") for r in rows) else None,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall": metrics([row["rank"] for row in rows]),
        "by_tag": {tag: metrics(ranks) for tag, ranks in sorted(by_tag.items())},
        "by_category": {cat: metrics(ranks) for cat, ranks in sorted(by_category.items())},
        "latency_ms": {
            "wall": latency_summary([row["wall_ms"] for row in rows]),
            **{step: latency_summary([row["timings_ms"][step] for row in rows if step in row["timings_ms"]])
               for step in sorted({step for row in rows for step in row["timings_ms"]})},
        },
        "memory_mb": {
            "embedding_container": container_memory_mb(CONTAINERS.get(backend)),
            "eval_process_peak_rss": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        },
        "queries": rows,
    }


def detection_summary(rows: list[dict]) -> dict:
    """How often the category the backend chose (detected, or parsed from text) was right; asked = needs_category."""
    asked = [row for row in rows if row["needs_category"]]
    decided = [row for row in rows if not row["needs_category"] and row["detected_category"]]
    correct = sum(row["detected_category"] == row["category"] for row in decided)
    return {
        "n": len(rows), "decided": len(decided), "correct": correct, "asked_user": len(asked),
        "no_category": len(rows) - len(decided) - len(asked),
        "accuracy_when_decided": round(correct / len(decided), 4) if decided else None,
        "accuracy_overall": round(correct / len(rows), 4) if rows else None,
        "asked_top3_contains_true": sum(1 for row in asked if row["category"] in
                                        [c["category"] for c in row.get("candidates") or []]),
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
    panels = [(f"{row['query_id']} {(row.get('text') or 'query')[:22]}",
               PROJECT_ROOT / row["query_path"] if row.get("query_path") else None)]
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
        f"Mode `{report.get('mode', 'image')}` · split {report.get('split', 'all')} · "
        f"category {report.get('category_mode', 'given')} · "
        f"flags {report.get('flags', {})}",
        "",
        *([f"Category detection: {report['category_detection']}", ""] if report.get("category_detection") else []),
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
        text = f" · text \"{row['text']}\"" if row.get("text") else ""
        lines += [f"**{row['query_id']}**{text} · {row['category']} · tags {', '.join(row['tags'])} · "
                  f"expected {', '.join(row['expected'])} · rank {rank}  ", f"top 5: {top}", "",
                  f"![{row['query_id']}]({strip.as_posix()})", ""]
    out = RESULTS_DIR / f"{name}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def compare(names: list[str]) -> str:
    reports = [json.loads((RESULTS_DIR / f"{name}.json").read_text()) for name in names]
    lines = ["| Run | Model | dim | R@1 | R@5 | R@10 | MRR@10 | cat. acc. | embed p50 ms | wall p50 ms | wall p95 ms "
             "| container peak MiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        m, lat = r["overall"], r["latency_ms"]
        lines.append(f"| {r['run_name']} | {r['model']} | {r['dimension']} | {pct(m['recall@1'])} | "
                     f"{pct(m['recall@5'])} | {pct(m['recall@10'])} | {m['mrr@10']:.3f} | "
                     f"{pct((r.get('category_detection') or {}).get('accuracy_overall'))} | "
                     f"{lat.get('embed', {}).get('p50', 'n/a')} | {lat['wall']['p50']} | {lat['wall']['p95']} | "
                     f"{(r['memory_mb']['embedding_container'] or {}).get('peak', 'n/a')} |")
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
    parser.add_argument("--mode", choices=sorted(QUERY_FILES), default="image")
    parser.add_argument("--category", choices=["given", "detect"], default="given",
                        help="image modes: send each query's category, or leave it out to test detection")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="backend setting for this run, e.g. SEARCH_HYBRID=false (repeatable)")
    parser.add_argument("--queries", type=Path, help="query file (default: the mode's file in eval/)")
    parser.add_argument("--split", choices=["all", "tune", "holdout"], default="all",
                        help="only queries of this split (Stage 10; see scripts/make_eval_splits.py)")
    parser.add_argument("--compare", nargs="+", metavar="RUN", help="print a comparison of saved runs")
    args = parser.parse_args()

    if args.compare:
        print(compare(args.compare))
        return 0
    if not args.backend:
        parser.error("--backend or --compare is required")

    name = args.name or args.backend
    overrides = dict(item.split("=", 1) for item in args.set)
    queries = (args.queries or PROJECT_ROOT / "eval" / QUERY_FILES[args.mode]).resolve()
    report = run(args.backend, name, queries, args.mode, args.category, overrides, args.split)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    from app.config import get_settings
    summary = write_markdown(report, get_settings().catalogue_dir)
    m = report["overall"]
    print(f"{name}: R@1 {pct(m['recall@1'])}  R@5 {pct(m['recall@5'])}  R@10 {pct(m['recall@10'])}  "
          f"MRR@10 {m['mrr@10']:.3f}  ({m['n']} queries)")
    if report.get("category_detection"):
        print(f"category: {report['category_detection']}")
    if report.get("vlm_judge"):
        print(f"vlm judge: {report['vlm_judge']}")
    print(f"embed p50 {report['latency_ms'].get('embed', {}).get('p50')} ms, wall p50 {report['latency_ms']['wall']['p50']} ms, "
          f"p95 {report['latency_ms']['wall']['p95']} ms, "
          f"container memory {report['memory_mb']['embedding_container']} MiB")
    print(f"wrote {summary.relative_to(PROJECT_ROOT)} and {name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
