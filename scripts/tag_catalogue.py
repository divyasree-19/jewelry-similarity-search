"""Tag every catalogue image in MongoDB with the VLM (Stage 7).

Resumable: an image is tagged only if it has no vlm_tags for the current TAGGER_VERSION and VLM model,
and each result is saved as soon as it arrives, so an interrupted run continues where it stopped.
Saved on the image document:

    vlm_tags: {category, metal, gemstones, stone_colors, motifs, style, description,
               dropped, attempts, tagger_model, quantization, tagger_version, tagged_at}

A failure is logged, saved as vlm_tag_error and skipped. Every run ends by re-validating all saved tags
against the schema.

Usage (from the project root):
    python scripts/tag_catalogue.py                  # tag what is missing
    python scripts/tag_catalogue.py --limit 10       # try a few first
    python scripts/tag_catalogue.py --retag          # tag everything again
    python scripts/tag_catalogue.py --check          # only re-validate saved tags
"""

import argparse
import asyncio
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.clients.vlm_client import VLMClient, VLMError  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.preprocessing import ImagePreprocessError, PreprocessOptions, encode, preprocess  # noqa: E402
from app.tagging import TAGGER_VERSION  # noqa: E402
from app.tagging.schema import Tags  # noqa: E402

log = logging.getLogger("tag_catalogue")
TAG_FIELDS = list(Tags.model_fields)


def pending_filter(model: str, retag: bool) -> dict:
    if retag:
        return {}
    return {"$or": [{"vlm_tags.tagger_version": {"$ne": TAGGER_VERSION}}, {"vlm_tags.tagger_model": {"$ne": model}}]}


async def tag_all(db, settings, docs: list[dict]) -> tuple[int, int]:
    options = PreprocessOptions.from_settings(settings)
    client = VLMClient.from_settings(settings)
    semaphore = asyncio.Semaphore(settings.vlm_concurrency)
    done = failed = 0
    started = time.perf_counter()

    async def one(doc: dict) -> None:
        nonlocal done, failed
        async with semaphore:
            try:
                data = (PROJECT_ROOT / doc["image_path"]).read_bytes()
                png = encode(await asyncio.to_thread(preprocess, data, options))
                result = await client.tag(png)
            except (OSError, ImagePreprocessError, VLMError) as exc:
                failed += 1
                log.warning("skipped %s: %s", doc["_id"], exc)
                db.images.update_one({"_id": doc["_id"]}, {"$set": {"vlm_tag_error": {
                    "message": str(exc)[:500], "tagger_version": TAGGER_VERSION, "at": datetime.now(timezone.utc)}}})
                return
            record = {
                **result.tags.model_dump(), "dropped": result.dropped, "attempts": result.attempts,
                "tagger_model": settings.vlm_model_name, "quantization": settings.vlm_quantization,
                "tagger_version": TAGGER_VERSION, "tagged_at": datetime.now(timezone.utc),
            }
            db.images.update_one({"_id": doc["_id"]}, {"$set": {"vlm_tags": record}, "$unset": {"vlm_tag_error": ""}})
            done += 1
            if result.dropped:
                log.info("%s: dropped values not in the vocabulary %s", doc["_id"], result.dropped)
            if (done + failed) % 10 == 0 or done + failed == len(docs):
                rate = (time.perf_counter() - started) / (done + failed)
                log.info("%d/%d done (%d failed), %.1f s/image", done + failed, len(docs), failed, rate)

    try:
        await asyncio.gather(*(one(doc) for doc in docs))
    finally:
        await client.aclose()
    return done, failed


def check_saved(db) -> bool:
    """Re-validate every saved vlm_tags record against the schema; report coverage per category."""
    docs = list(db.images.find({}, {"category": 1, "vlm_tags": 1, "vlm_tag_error": 1}))
    tagged = [doc for doc in docs if doc.get("vlm_tags", {}).get("tagger_version") == TAGGER_VERSION]
    invalid = []
    for doc in tagged:
        try:
            Tags(**{key: doc["vlm_tags"].get(key) for key in TAG_FIELDS})
        except ValidationError as exc:
            invalid.append((doc["_id"], exc.errors()[0]["msg"]))
    for image_id, message in invalid:
        log.error("invalid saved tags on %s: %s", image_id, message)

    per_category = Counter(doc["category"] for doc in tagged)
    totals = Counter(doc["category"] for doc in docs)
    for category in sorted(totals):
        log.info("%-10s tagged %3d / %3d", category, per_category[category], totals[category])
    category_mismatch = sum(1 for doc in tagged if doc["vlm_tags"].get("category") not in (None, doc["category"]))
    dropped = Counter(f for doc in tagged for f in doc["vlm_tags"].get("dropped", {}))
    errors = sum(1 for doc in docs if "vlm_tag_error" in doc)
    log.info("schema check: %d/%d saved tag records valid (%s)", len(tagged) - len(invalid), len(tagged),
             "100%" if tagged and not invalid else "FAILED" if invalid else "nothing tagged")
    log.info("VLM category differs from catalogue category on %d image(s); values dropped per field: %s; "
             "images with vlm_tag_error: %d", category_mismatch, dict(dropped) or "none", errors)
    return not invalid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="tag at most this many images")
    parser.add_argument("--retag", action="store_true", help="tag every image again")
    parser.add_argument("--check", action="store_true", help="only re-validate saved tags")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    mongo = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=settings.mongo_timeout_ms)
    db = mongo[settings.mongo_db]
    if not args.check:
        query = pending_filter(settings.vlm_model_name, args.retag)
        docs = list(db.images.find(query, {"image_path": 1}).sort("_id", 1))
        if args.limit:
            docs = docs[: args.limit]
        log.info("tagger %s, model %s at %s, %d image(s) to tag, %d at a time",
                 TAGGER_VERSION, settings.vlm_model_name, settings.vlm_service_url, len(docs), settings.vlm_concurrency)
        if docs:
            done, failed = asyncio.run(tag_all(db, settings, docs))
            log.info("tagged %d, failed %d", done, failed)
    ok = check_saved(db)
    mongo.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
