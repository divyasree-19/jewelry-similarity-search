"""Stage 10 Part B: optional Qwen same-design check on the top results (FEATURE_VLM_JUDGE, default off).

For the top VLM_JUDGE_TOP_N results, Qwen sees the query and the candidate image and answers
{same_design, confidence, reason}. Each judged result's score moves by
+VLM_JUDGE_WEIGHT * confidence (same design) or -VLM_JUDGE_WEIGHT * confidence (not), and only the top N are
re-sorted; results below N never move. If any judgement fails or times out, the Part A order is kept.
Calls run in parallel up to VLM_JUDGE_CONCURRENCY and are cached by (query image hash, candidate image_id,
tagger version + judge prompt version).
"""

import asyncio
import hashlib
import logging
from collections import OrderedDict

from app.config import PROJECT_ROOT
from app.preprocessing import encode, preprocess
from app.tagging import TAGGER_VERSION
from app.tagging.prompts import JUDGE_PROMPT_VERSION

log = logging.getLogger("search")
JUDGE_VERSION = f"{TAGGER_VERSION}/judge-p{JUDGE_PROMPT_VERSION}"


class JudgeCache:
    """Least-recently-used cache of judgements."""

    def __init__(self, size: int):
        self.size = size
        self._items: OrderedDict = OrderedDict()

    def get(self, key):
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        return None

    def put(self, key, value) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.size:
            self._items.popitem(last=False)


def cache_key(query_png: bytes, image_id: str) -> tuple[str, str, str]:
    return hashlib.sha256(query_png).hexdigest(), image_id, JUDGE_VERSION


async def judge_top(app, results: list[dict], query_png: bytes) -> dict:
    """Judge the top N results and re-sort them in place. Returns a summary for the response."""
    settings = app.state.settings
    top = results[:settings.vlm_judge_top_n]
    cache: JudgeCache = app.state.judge_cache
    semaphore = asyncio.Semaphore(settings.vlm_judge_concurrency)
    summary = {"top_n": len(top), "calls": 0, "cache_hits": 0, "failed": 0, "reordered": False}

    async def one(result):
        key = cache_key(query_png, result["image_id"])
        cached = cache.get(key)
        if cached is not None:
            summary["cache_hits"] += 1
            return cached
        async with semaphore:
            summary["calls"] += 1
            candidate_png = encode(await asyncio.to_thread(
                preprocess, (PROJECT_ROOT / result["_image_path"]).read_bytes(), app.state.preprocess_options))
            judgement = await asyncio.wait_for(app.state.vlm_client.judge(query_png, candidate_png),
                                               settings.vlm_judge_timeout_seconds)
        cache.put(key, judgement)
        return judgement

    outcomes = await asyncio.gather(*(one(r) for r in top), return_exceptions=True)
    failures = [o for o in outcomes if isinstance(o, BaseException)]
    if failures:
        summary["failed"] = len(failures)
        log.warning("VLM judge: %d of %d calls failed (%r); keeping the Part A order",
                    len(failures), len(top), failures[0])
        return summary

    before = [r["image_id"] for r in top]
    for result, judgement in zip(top, outcomes):
        sign = 1.0 if judgement.same_design else -1.0
        adjustment = sign * judgement.confidence * settings.vlm_judge_weight
        result["vlm_judge"] = {"same_design": judgement.same_design, "confidence": judgement.confidence,
                               "reason": judgement.reason, "adjustment": round(adjustment, 4)}
        result["score"] = round(result["score"] + adjustment, 6)
    top.sort(key=lambda r: r["score"], reverse=True)
    results[:len(top)] = top
    summary["reordered"] = [r["image_id"] for r in top] != before
    return summary
