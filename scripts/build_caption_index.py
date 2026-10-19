"""Embed every VLM description with the SigLIP text encoder and rebuild the caption indexes (Stage 7).

Uses images that have vlm_tags for the current TAGGER_VERSION and a vector in the current EMBED_BACKEND's
image index; each caption vector gets the image's vector_id. The run ends by checking that every category
has as many captions as images, with the same vector IDs.

Usage (from the project root, after scripts/ingest.py and scripts/tag_catalogue.py):
    python scripts/build_caption_index.py
    python scripts/build_caption_index.py --check     # only compare the saved caption and image indexes
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.caption_store import load_caption_store, new_caption_store  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.embed_backends import make_embed_client  # noqa: E402
from app.embed_client import EmbeddingServiceError  # noqa: E402
from app.preprocessing import PreprocessOptions  # noqa: E402
from app.tagging import TAGGER_VERSION  # noqa: E402
from app.vector_index import CategoryIndexes  # noqa: E402

log = logging.getLogger("build_caption_index")


async def build(db, settings, concurrency: int) -> CategoryIndexes:
    backend = settings.embed_backend
    docs = list(db.images.find(
        {"vlm_tags.tagger_version": TAGGER_VERSION, f"embeddings.{backend}": {"$exists": True}},
        {"vector_id": 1, "category": 1, "vlm_tags.description": 1},
    ).sort("_id", 1))
    log.info("%d tagged image(s) with a %s vector; tagger %s", len(docs), backend, TAGGER_VERSION)

    client = make_embed_client(settings)
    if not hasattr(client, "embed_text"):
        await client.aclose()
        sys.exit(f"EMBED_BACKEND={backend} has no text encoder; caption indexes need a SigLIP backend")
    store = new_caption_store(settings, TAGGER_VERSION)
    semaphore = asyncio.Semaphore(concurrency)
    failed = 0

    async def one(doc):
        nonlocal failed
        async with semaphore:
            try:
                vector = await client.embed_text(doc["vlm_tags"]["description"])
            except EmbeddingServiceError as exc:
                failed += 1
                log.warning("skipped %s: %s", doc["_id"], exc)
                return
            store.add(doc["category"], [doc["vector_id"]], [vector])

    try:
        await asyncio.gather(*(one(doc) for doc in docs))
    finally:
        await client.aclose()
    log.info("embedded %d caption(s), %d failed; saved to %s", store.total, failed, store.save(
        Path(settings.faiss_index_dir) / "captions"))
    return store


def compare(settings, captions: CategoryIndexes) -> bool:
    """Caption count and vector IDs equal the image index's, per category."""
    images = CategoryIndexes.load_or_create(
        settings.faiss_index_dir, settings.embed_model_name, settings.embed_version,
        PreprocessOptions.from_settings(settings).signature,
    )
    ok = True
    for category in sorted(set(images.counts()) | set(captions.counts())):
        image_ids, caption_ids = set(images.ids(category).tolist()), set(captions.ids(category).tolist())
        match = image_ids == caption_ids
        ok &= match
        detail = "ok" if match else f"MISMATCH: {len(image_ids - caption_ids)} image(s) without caption, " \
                                    f"{len(caption_ids - image_ids)} caption(s) without image"
        log.info("%-10s image index %4d  caption index %4d  %s", category, len(image_ids), len(caption_ids), detail)
    log.info("caption check: %s", "ok" if ok else "FAILED")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only compare saved caption and image indexes")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    if args.check:
        captions = load_caption_store(settings, TAGGER_VERSION)
    else:
        mongo = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=settings.mongo_timeout_ms)
        captions = asyncio.run(build(mongo[settings.mongo_db], settings, args.concurrency))
        mongo.close()
    return 0 if compare(settings, captions) else 1


if __name__ == "__main__":
    sys.exit(main())
