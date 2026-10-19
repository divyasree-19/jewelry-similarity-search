"""Tune the Stage 10 reranker weights on the TUNE split only.

1. Runs every tune query once through the real search pipeline (SEARCH_RERANKER on, base weights) and records
   each candidate's raw signals (image, caption, gray, region cosines and boost).
2. Scores weight combinations offline with the same combine() the backend uses: grid (step --step) or random
   (--samples) over the signals that exist in the data. Weights sum to 1; ranges, boost scale and cap stay as in
   the base file.
3. Objective: Recall@5, then MRR@10; remaining ties go to the combination closest to the base weights.
4. Writes config/reranker_weights_<new version>.json with the tuned weights for this mode and the tune-split numbers.
   Run once per mode; the second run adds its mode to the same file.

Holdout queries are dropped as the file is read and never reach the pipeline.

Usage (from the project root, stack running):
    python eval/tune_weights.py --mode image --new-version v2
    python eval/tune_weights.py --mode text --new-version v2
"""

import argparse
import csv
import functools
import itertools
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

QUERY_FILES = {"image": "queries.csv", "text": "text_queries.csv"}
CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def tune_rows(mode: str) -> list[dict]:
    """The tune split of the mode's query file. Holdout rows are discarded here, before anything else."""
    with (PROJECT_ROOT / "eval" / QUERY_FILES[mode]).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "split" not in rows[0]:
        sys.exit(f"{QUERY_FILES[mode]} has no split column; run scripts/make_eval_splits.py")
    tune = [row for row in rows if row["split"] == "tune"]
    del rows  # holdout rows are never used
    return [{**row, "expected": [p.strip() for p in row["expected_product_ids"].split(";") if p.strip()]}
            for row in tune]


def collect_signals(mode: str, rows: list[dict], base_version: str) -> list[dict]:
    """Per tune query: the expected products and every re-ranked candidate's raw signals."""
    os.environ["SEARCH_RERANKER"] = "true"
    os.environ["RERANKER_MODES"] = "image,text"
    os.environ["RERANKER_WEIGHTS_VERSION"] = base_version
    from fastapi.testclient import TestClient

    from app import main
    from app.config import get_settings
    from app.search.image_search import search_by_image
    from app.search.text_search import search_by_text

    get_settings.cache_clear()
    logging.getLogger("search").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    queries = []
    with TestClient(main.app) as client:
        for row in rows:
            if mode == "text":
                call = functools.partial(search_by_text, main.app, row["text"], category=None, project_id=None,
                                         top_k=10, debug=False, collect_signals=True)
            else:
                data = (PROJECT_ROOT / row["query_path"]).read_bytes()
                call = functools.partial(search_by_image, main.app, data, category=row["category"], text=None,
                                         project_id=None, subcategory=None, top_k=10, debug=False,
                                         collect_signals=True)
            response = client.portal.call(call)
            queries.append({"query_id": row["query_id"], "expected": row["expected"],
                            "candidates": response.get("candidate_signals", [])})
            print(f"  {row['query_id']}: {len(queries[-1]['candidates'])} candidates", flush=True)
    return queries


def evaluate(queries: list[dict], mode_weights, boost_scale: float, boost_cap: float) -> dict:
    from app.search.reranker import combine
    hits = {1: 0, 5: 0, 10: 0}
    reciprocal = 0.0
    for query in queries:
        scored = []
        for candidate in query["candidates"]:
            score, _ = combine(candidate["signals"], candidate["boost"], mode_weights, boost_scale, boost_cap)
            scored.append((score, candidate["signals"].get("image") or -1.0, candidate["product_id"]))
        scored.sort(reverse=True)
        products, seen = [], set()
        for _, _, product_id in scored:
            if product_id not in seen:
                seen.add(product_id)
                products.append(product_id)
        rank = next((i for i, p in enumerate(products[:10], start=1) if p in query["expected"]), None)
        for k in hits:
            hits[k] += rank is not None and rank <= k
        reciprocal += 1 / rank if rank else 0.0
    n = len(queries)
    return {"n": n, **{f"recall@{k}": round(v / n, 4) for k, v in hits.items()}, "mrr@10": round(reciprocal / n, 4)}


def weight_sets(signals: list[str], search: str, step: float, samples: int, seed: int):
    if search == "grid":
        units = round(1 / step)
        for parts in itertools.product(range(units + 1), repeat=len(signals) - 1):
            if sum(parts) <= units:
                yield dict(zip(signals, [p * step for p in parts] + [(units - sum(parts)) * step]))
    else:
        rng = random.Random(seed)
        for _ in range(samples):
            raw = [rng.random() for _ in signals]
            yield {s: v / sum(raw) for s, v in zip(signals, raw)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=sorted(QUERY_FILES), default="image")
    parser.add_argument("--base-version", default="v1")
    parser.add_argument("--new-version", required=True)
    parser.add_argument("--search", choices=["grid", "random"], default="grid")
    parser.add_argument("--step", type=float, default=0.05, help="grid step")
    parser.add_argument("--samples", type=int, default=2000, help="random search samples")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from app.config import Settings
    from app.search.reranker import ModeWeights, load_weights, parse_weights, weights_path

    directory = Settings().reranker_weights_dir
    base = load_weights(directory, args.base_version)
    rows = tune_rows(args.mode)
    print(f"collecting signals for {len(rows)} tune queries ({args.mode})...")
    queries = collect_signals(args.mode, rows, args.base_version)

    base_mode = base.for_mode(args.mode)
    signals = [s for s in ("image", "caption", "gray", "region")
               if s in base_mode.ranges and any(c["signals"].get(s) is not None for q in queries for c in q["candidates"])]
    print(f"signals present: {signals}")

    def distance(weights):
        return sum(abs(weights.get(s, 0) - base_mode.weights.get(s, 0)) for s in signals)

    results = []
    for weights in weight_sets(signals, args.search, args.step, args.samples, args.seed):
        full = {s: round(weights.get(s, 0.0), 4) for s in ("image", "caption", "gray", "region")}
        metrics = evaluate(queries, ModeWeights(full, base_mode.ranges), base.boost_scale, base.boost_cap)
        results.append((metrics["recall@5"], metrics["mrr@10"], -distance(full), full, metrics))
    results.sort(key=lambda r: r[:3], reverse=True)
    base_metrics = evaluate(queries, base_mode, base.boost_scale, base.boost_cap)
    best_weights, best_metrics = results[0][3], results[0][4]

    print(f"\n{len(results)} weight sets evaluated on {len(queries)} tune queries")
    print(f"base {args.base_version}: {base_mode.weights} -> {base_metrics}")
    for _, _, _, weights, metrics in results[:5]:
        print(f"  {weights} -> {metrics}")

    path = weights_path(directory, args.new_version)
    data = json.loads(path.read_text()) if path.exists() else json.loads(weights_path(directory, args.base_version).read_text())
    data["version"] = args.new_version
    data["description"] = (f"Tuned on the tune split from {args.base_version}; ranges, boost scale and cap "
                           f"unchanged from {args.base_version}.")
    data["modes"][args.mode]["weights"] = best_weights
    data.setdefault("tuned", {})[args.mode] = {
        "split": "tune", "queries": len(queries), "query_ids": [q["query_id"] for q in queries],
        "search": f"grid step {args.step}" if args.search == "grid" else f"random {args.samples} seed {args.seed}",
        "weight_sets_evaluated": len(results), "objective": "recall@5, then mrr@10",
        "base_version": args.base_version, "base_tune_metrics": base_metrics, "tune_metrics": best_metrics,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    parse_weights(data)  # validate before writing
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path.relative_to(PROJECT_ROOT)}: {args.mode} weights {best_weights} -> {best_metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
