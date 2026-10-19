"""Embed every catalogue image and store it in MongoDB and the per-category FAISS indexes.

Works on one embedding backend at a time (EMBED_BACKEND, or --backend). Each image has one permanent
vector_id, shared by every backend's index; what a backend embedded is recorded on the image document under
embeddings.<backend> (embedding_model, embedding_version, preprocess_version, image_sha256, ingested_at).

By default the run is incremental: only images that are new, whose file changed, whose category changed
or that were embedded with other settings are sent to the embedding service. Vectors of images that left
the catalogue are removed. --rebuild clears this backend's vectors and re-embeds everything.

A bad image is logged, recorded as ingest_errors.<backend>, and skipped; the run continues.

Usage (from the project root, with docker compose up):
    python scripts/ingest.py                          # incremental, EMBED_BACKEND from .env
    python scripts/ingest.py --backend siglip2_384    # another backend, its own index directory
    python scripts/ingest.py --rebuild                # recreate this backend's indexes
    python scripts/ingest.py --verify                 # also search with every catalogue image (re-embeds them all)
"""

import argparse
import asyncio
import hashlib
import logging
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient, UpdateOne

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.embed_backends import BACKENDS, make_embed_client  # noqa: E402
from app.embed_client import EmbeddingServiceError  # noqa: E402
from app.preprocessing import ImagePreprocessError, PreprocessOptions, encode, preprocess  # noqa: E402
from app.vector_index import CategoryIndexes, IndexMismatchError  # noqa: E402
from scripts.seed_mongo import build_documents, read_rows, write_metadata  # noqa: E402

log = logging.getLogger("ingest")

# Stage 4 kept one backend's fields at the top level of the image document; they now live in embeddings.<backend>
LEGACY_FIELDS = ("embedding_model", "embedding_version", "preprocess_version", "image_sha256", "ingested_at", "ingest_error")


@dataclass
class Item:
    image_id: str
    category: str
    path: Path
    sha256: str
    vector_id: int | None = None


@dataclass
class Plan:
    keep_ids: set[int] = field(default_factory=set)  # vector IDs that are already correct in the index
    to_embed: list[Item] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # image_id -> reason


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(images: list[dict], existing: dict[str, dict], indexed: dict[int, str], current: dict, root: Path) -> Plan:
    """Decide which catalogue images need embedding.

    images: image metadata from catalogue.csv; existing: MongoDB image documents by _id;
    indexed: vector_id -> category for every vector currently in the FAISS indexes;
    current: the embedding_model / embedding_version / preprocess_version this run will write.
    """
    plan = Plan()
    for image in images:
        path = root / image["image_path"]
        try:
            digest = sha256_file(path)
        except OSError as exc:
            plan.failed[image["_id"]] = f"cannot read {image['image_path']}: {exc.strerror or exc}"
            continue

        doc = existing.get(image["_id"], {})
        vector_id = doc.get("vector_id")
        up_to_date = (
            vector_id is not None
            and indexed.get(vector_id) == image["category"]
            and doc.get("image_sha256") == digest
            and all(doc.get(key) == value for key, value in current.items())
        )
        if up_to_date:
            plan.keep_ids.add(vector_id)
        else:
            plan.to_embed.append(Item(image["_id"], image["category"], path, digest, vector_id))
    return plan


async def embed_items(items: list[Item], client, options: PreprocessOptions, batch_size: int):
    """Yield (item, vector or None, error or None) per item, batch_size requests in flight at a time."""

    async def one(item: Item):
        try:
            image = await asyncio.to_thread(preprocess, item.path.read_bytes(), options)
            vector = await client.embed(encode(image), f"{item.image_id}.png", "image/png")
            return item, vector, None
        except (ImagePreprocessError, EmbeddingServiceError, OSError) as exc:
            return item, None, str(exc)

    for start in range(0, len(items), batch_size):
        for result in await asyncio.gather(*(one(item) for item in items[start:start + batch_size])):
            yield result


async def ingest(args) -> int:
    settings = get_settings()
    backend = settings.embed_backend
    options = PreprocessOptions.from_settings(settings)
    current = {
        "embedding_model": settings.embed_model_name,
        "embedding_version": settings.embed_version,
        "preprocess_version": options.signature,
    }
    log.info("backend %s (%s, %s)", backend, settings.embed_model_name, settings.embed_version)

    products, images = build_documents(
        read_rows(PROJECT_ROOT / "data" / "products.csv"), read_rows(PROJECT_ROOT / "data" / "catalogue.csv")
    )
    mongo = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=settings.mongo_timeout_ms)
    db = mongo[settings.mongo_db]
    db.images.update_many({}, {"$unset": {key: "" for key in LEGACY_FIELDS}})

    if args.rebuild:
        log.info("--rebuild: clearing every %s vector", backend)
        db.images.update_many({}, {"$unset": {f"embeddings.{backend}": "", f"ingest_errors.{backend}": ""}})
        store = CategoryIndexes(settings.embed_model_name, settings.embed_version, options.signature)
    else:
        store = CategoryIndexes.load_or_create(
            settings.faiss_index_dir, settings.embed_model_name, settings.embed_version, options.signature
        )
    write_metadata(db, products, images)

    existing = {
        doc["_id"]: {"vector_id": doc.get("vector_id"), **doc.get("embeddings", {}).get(backend, {})}
        for doc in db.images.find({}, {"vector_id": 1, f"embeddings.{backend}": 1})
    }
    indexed = store.id_to_category()
    plan = make_plan(images, existing, indexed, current, PROJECT_ROOT)

    # Drop every vector that is not already correct: stale, changed, recategorized or orphaned
    stale_by_category = defaultdict(list)
    for vector_id, category in indexed.items():
        if vector_id not in plan.keep_ids:
            stale_by_category[category].append(vector_id)
    removed = sum(store.remove(category, ids) for category, ids in stale_by_category.items())

    # vector_id is permanent per image and shared by every backend; new images get the next free one
    used_ids = [doc["vector_id"] for doc in existing.values() if doc.get("vector_id") is not None]
    next_id = max([*used_ids, *indexed, 0]) + 1
    for item in plan.to_embed:
        if item.vector_id is None:
            item.vector_id, next_id = next_id, next_id + 1

    log.info(
        "%d catalogue images: %d up to date, %d to embed, %d unreadable, %d stale vectors removed",
        len(images), len(plan.keep_ids), len(plan.to_embed), len(plan.failed), removed,
    )

    failed = dict(plan.failed)
    embedded = 0
    client = make_embed_client(settings)
    try:
        updates = []
        async for item, vector, error in embed_items(plan.to_embed, client, options, args.batch_size):
            if error:
                failed[item.image_id] = error
                continue
            store.add(item.category, [item.vector_id], [vector])
            embedded += 1
            record = {**current, "image_sha256": item.sha256, "ingested_at": datetime.now(timezone.utc)}
            updates.append(UpdateOne({"_id": item.image_id}, {
                "$set": {"vector_id": item.vector_id, f"embeddings.{backend}": record},
                "$unset": {f"ingest_errors.{backend}": ""},
            }))
            if len(updates) >= args.batch_size:
                db.images.bulk_write(updates)
                updates = []
            if embedded % 50 == 0:
                log.info("embedded %d/%d", embedded, len(plan.to_embed))
        if updates:
            db.images.bulk_write(updates)
    finally:
        await client.aclose()

    now = datetime.now(timezone.utc)
    for image_id, reason in failed.items():
        log.warning("skipped %s: %s", image_id, reason)
        db.images.update_one({"_id": image_id}, {
            "$unset": {f"embeddings.{backend}": ""},
            "$set": {f"ingest_errors.{backend}": {"message": reason, "at": now}},
        })

    directory = store.save(settings.faiss_index_dir)
    log.info("embedded %d, skipped %d; indexes saved to %s", embedded, len(failed), directory)

    ok = verify_consistency(db, store, backend, {image["_id"] for image in images})
    if args.verify:
        ok = await verify_self_search(db, store, backend, options, settings, args.batch_size) and ok
    mongo.close()
    return 0 if ok else 1


def indexed_docs(db, backend: str, fields: dict) -> list[dict]:
    return list(db.images.find({f"embeddings.{backend}": {"$exists": True}}, fields))


def verify_consistency(db, store: CategoryIndexes, backend: str, catalogue_ids: set[str]) -> bool:
    """Index count per category equals MongoDB vectors per category, and IDs map one-to-one."""
    ok = True
    docs = indexed_docs(db, backend, {"vector_id": 1, "category": 1})
    mongo_counts = Counter(doc["category"] for doc in docs)
    index_counts = store.counts()
    for category in sorted(set(mongo_counts) | set(index_counts)):
        match = mongo_counts.get(category, 0) == index_counts.get(category, 0)
        ok &= match
        log.info("%-10s index %4d  mongo %4d  %s", category, index_counts.get(category, 0),
                 mongo_counts.get(category, 0), "ok" if match else "MISMATCH")

    owners = defaultdict(list)
    for doc in docs:
        owners[doc["vector_id"]].append(doc)
    for category in store.indexes:
        for vector_id in store.ids(category).tolist():
            matches = owners.get(vector_id, [])
            if len(matches) != 1 or matches[0]["category"] != category:
                ok = False
                log.error("vector_id %d in %s maps to %s", vector_id, category, [d["_id"] for d in matches] or "nothing")
    indexed = store.id_to_category()
    for vector_id, matches in owners.items():
        if vector_id not in indexed:
            ok = False
            log.error("vector_id %d of %s is not in any index", vector_id, matches[0]["_id"])

    missing = catalogue_ids - {doc["_id"] for doc in docs}
    if missing:
        log.warning("%d catalogue image(s) have no vector: %s", len(missing), ", ".join(sorted(missing)))
    log.info("consistency: %s (%d vectors)", "ok" if ok else "FAILED", store.total)
    return ok


async def verify_self_search(db, store: CategoryIndexes, backend: str, options, settings, batch_size: int) -> bool:
    """Re-embed every indexed catalogue image and check it comes back as its own top result."""
    docs = indexed_docs(db, backend, {"vector_id": 1, "category": 1, "image_path": 1})
    items = [Item(d["_id"], d["category"], PROJECT_ROOT / d["image_path"], "", d["vector_id"]) for d in docs]
    misses = 0
    client = make_embed_client(settings)
    try:
        async for item, vector, error in embed_items(items, client, options, batch_size):
            if error:
                misses += 1
                log.error("self-search %s: %s", item.image_id, error)
                continue
            top_id, score = store.search(item.category, vector, k=1)[0]
            if top_id != item.vector_id:
                misses += 1
                log.error("self-search %s: top result is vector_id %d (%.4f), expected %d",
                          item.image_id, top_id, score, item.vector_id)
    finally:
        await client.aclose()
    log.info("self-search: %d/%d images return themselves first", len(items) - misses, len(items))
    return misses == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=sorted(BACKENDS), help="embedding backend (default: EMBED_BACKEND)")
    parser.add_argument("--rebuild", action="store_true", help="clear this backend's vectors and re-embed everything")
    parser.add_argument("--verify", action="store_true", help="search with every catalogue image after ingesting")
    parser.add_argument("--batch-size", type=int, default=8, help="concurrent /embed requests (default 8)")
    args = parser.parse_args()
    if args.backend:
        os.environ["EMBED_BACKEND"] = args.backend
        get_settings.cache_clear()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        return asyncio.run(ingest(args))
    except IndexMismatchError as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
