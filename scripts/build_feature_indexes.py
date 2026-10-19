"""Build Stage 9 feature stores for every image in the current EMBED_BACKEND's image index.

    --feature gray    preprocess -> grayscale -> /embed/batch -> per-category FAISS gray index
    --feature edge    same with an edge map (optional experiment)
    --feature tiles   preprocess -> centre crop + grid x grid tiles -> /embed/batch -> tiles_<g>x<g>.npy store
                      (--grid, default FEATURE_TILE_GRID); not in FAISS, only used for re-scoring

Every vector uses the image's vector_id. Resumable: work is saved after each chunk, and each image records
features.<backend>.<feature> = {signature, image_sha256, at} in MongoDB, so a rerun only embeds images that are
new, changed, or were built with another signature. Ends by comparing counts with the image index.

Usage (from the project root, after scripts/ingest.py):
    python scripts/build_feature_indexes.py --feature gray
    python scripts/build_feature_indexes.py --feature tiles --grid 3
    python scripts/build_feature_indexes.py --feature gray --rebuild
"""

import argparse
import asyncio
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymongo import MongoClient, UpdateOne

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.embed_backends import make_embed_client  # noqa: E402
from app.embed_client import EmbeddingServiceError  # noqa: E402
from app.preprocessing import (ImagePreprocessError, PreprocessOptions, encode, make_tiles, preprocess,  # noqa: E402
                               to_edge_map, to_grayscale, variant_signature)
from app.search.features import new_variant_store, save_tiles, tile_dir, tile_names, variant_root  # noqa: E402
from app.vector_index import CategoryIndexes, IndexMismatchError  # noqa: E402

log = logging.getLogger("build_feature_indexes")
MAX_BATCH = 32


def feature_images(feature: str, data: bytes, options: PreprocessOptions, grid: int) -> list:
    image = preprocess(data, options)
    if feature == "gray":
        return [to_grayscale(image)]
    if feature == "edge":
        return [to_edge_map(image)]
    return make_tiles(image, grid)


async def embed_images(client, docs: list[dict], feature: str, options, grid: int, batch_size: int):
    """{image_id: [vectors]} for the docs that worked, and {image_id: error}."""
    per_image, failed = {}, {}
    for doc in docs:
        try:
            per_image[doc["_id"]] = await asyncio.to_thread(
                feature_images, feature, (PROJECT_ROOT / doc["image_path"]).read_bytes(), options, grid)
        except (OSError, ImagePreprocessError) as exc:
            failed[doc["_id"]] = str(exc)
    flat = [(image_id, picture) for image_id, pictures in per_image.items() for picture in pictures]
    vectors: dict[str, list] = defaultdict(list)
    for start in range(0, len(flat), batch_size):
        chunk = flat[start:start + batch_size]
        try:
            result = await client.embed_batch([(encode(p), f"{i}.png", "image/png") for i, p in chunk])
        except EmbeddingServiceError as exc:
            for image_id, _ in chunk:
                failed[image_id] = str(exc)
            continue
        for (image_id, _), vector in zip(chunk, result):
            vectors[image_id].append(vector)
    done = {i: v for i, v in vectors.items() if i not in failed and len(v) == len(per_image[i])}
    for image_id in per_image:
        if image_id not in done and image_id not in failed:
            failed[image_id] = "incomplete batch"
    return done, failed


def record_updates(docs, backend: str, key: str, signature: str) -> list[UpdateOne]:
    now = datetime.now(timezone.utc)
    return [UpdateOne({"_id": d["_id"]}, {"$set": {f"features.{backend}.{key}": {
        "signature": signature, "image_sha256": d["embeddings"][backend].get("image_sha256"), "at": now}}})
        for d in docs]


def is_current(doc: dict, backend: str, key: str, signature: str) -> bool:
    record = doc.get("features", {}).get(backend, {}).get(key, {})
    return (record.get("signature") == signature
            and record.get("image_sha256") == doc["embeddings"][backend].get("image_sha256"))


async def build(args, settings, db) -> bool:
    backend = settings.embed_backend
    options = PreprocessOptions.from_settings(settings)
    grid = args.grid or settings.feature_tile_grid
    signature = variant_signature(options.signature, args.feature, grid)
    key = f"tiles_{grid}x{grid}" if args.feature == "tiles" else args.feature
    image_store = CategoryIndexes.load_or_create(settings.faiss_index_dir, settings.embed_model_name,
                                                 settings.embed_version, options.signature)
    docs = list(db.images.find({f"embeddings.{backend}": {"$exists": True}},
                               {"vector_id": 1, "category": 1, "image_path": 1, f"embeddings.{backend}": 1,
                                f"features.{backend}.{key}": 1}).sort("_id", 1))
    by_id = {d["vector_id"]: d for d in docs}
    log.info("feature %s (%s), backend %s, %d indexed image(s)", key, signature, backend, len(docs))

    client = make_embed_client(settings)
    if not hasattr(client, "embed_batch"):
        await client.aclose()
        sys.exit(f"EMBED_BACKEND={backend} has no /embed/batch; feature stores need a SigLIP backend")
    batch_size = min(args.batch_size, MAX_BATCH)

    try:
        if args.feature in ("gray", "edge"):
            root = variant_root(settings, args.feature)
            if args.rebuild:
                store = new_variant_store(settings, signature)
            else:
                try:
                    store = CategoryIndexes.load_or_create(root, settings.embed_model_name, settings.embed_version,
                                                           signature)
                except IndexMismatchError as exc:
                    sys.exit(f"{exc}")
            indexed = store.id_to_category()
            keep = {v for v, c in indexed.items()
                    if v in by_id and by_id[v]["category"] == c and is_current(by_id[v], backend, key, signature)}
            stale = defaultdict(list)
            for vector_id, category in indexed.items():
                if vector_id not in keep:
                    stale[category].append(vector_id)
            for category, ids in stale.items():
                store.remove(category, ids)
            todo = [d for d in docs if d["vector_id"] not in keep]
            log.info("%d up to date, %d to embed, %d stale removed", len(keep), len(todo), sum(map(len, stale.values())))
            for start in range(0, len(todo), batch_size * 4):
                chunk = todo[start:start + batch_size * 4]
                done, failed = await embed_images(client, chunk, args.feature, options, grid, batch_size)
                for doc in chunk:
                    if doc["_id"] in done:
                        store.add(doc["category"], [doc["vector_id"]], done[doc["_id"]])
                store.save(root)  # save first, then record: a crash never leaves records without vectors
                ok_docs = [d for d in chunk if d["_id"] in done]
                if ok_docs:
                    db.images.bulk_write(record_updates(ok_docs, backend, key, signature))
                for image_id, error in failed.items():
                    log.warning("skipped %s: %s", image_id, error)
                log.info("%d/%d embedded", min(start + len(chunk), len(todo)), len(todo))
            if not todo:
                store.save(root)
            counts = store.counts()
            ok = all(set(store.ids(c).tolist()) == set(image_store.ids(c).tolist())
                     for c in set(counts) | set(image_store.counts()))
            location = store.directory(root)
        else:
            directory = tile_dir(settings)
            matrix_name, ids_name, _ = tile_names(grid)
            existing: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            if not args.rebuild:
                for category in image_store.counts():
                    folder = directory / category
                    if (folder / matrix_name).exists():
                        existing[category] = (np.load(folder / matrix_name), np.load(folder / ids_name))
            keep_rows: dict[str, list] = defaultdict(list)
            keep_ids = set()
            for category, (matrix, ids) in existing.items():
                for vector_id in np.unique(ids[:, 0]):
                    doc = by_id.get(int(vector_id))
                    if doc and doc["category"] == category and is_current(doc, backend, key, signature):
                        rows = ids[:, 0] == vector_id
                        keep_rows[category].append((matrix[rows], ids[rows]))
                        keep_ids.add(int(vector_id))
            todo = [d for d in docs if d["vector_id"] not in keep_ids]
            log.info("%d up to date, %d to embed", len(keep_ids), len(todo))
            dimension = None

            def write():
                per_category = {}
                for category in image_store.counts():
                    parts = keep_rows.get(category, [])
                    if parts:
                        per_category[category] = (np.vstack([m for m, _ in parts]), np.vstack([i for _, i in parts]))
                    else:
                        per_category[category] = (np.zeros((0, dimension or 0), np.float32), np.zeros((0, 2), np.int64))
                save_tiles(directory, grid, signature, per_category, dimension or 0)

            images_per_chunk = max(1, (batch_size * 4) // (1 + grid * grid))
            for start in range(0, len(todo), images_per_chunk):
                chunk = todo[start:start + images_per_chunk]
                done, failed = await embed_images(client, chunk, "tiles", options, grid, batch_size)
                for doc in chunk:
                    if doc["_id"] in done:
                        vectors = np.asarray(done[doc["_id"]], dtype=np.float32)
                        dimension = vectors.shape[1]
                        ids = np.column_stack([np.full(len(vectors), doc["vector_id"]), np.arange(len(vectors))])
                        keep_rows[doc["category"]].append((vectors, ids))
                if dimension is None and keep_rows:
                    dimension = next(iter(keep_rows.values()))[0][0].shape[1]
                write()
                ok_docs = [d for d in chunk if d["_id"] in done]
                if ok_docs:
                    db.images.bulk_write(record_updates(ok_docs, backend, key, signature))
                for image_id, error in failed.items():
                    log.warning("skipped %s: %s", image_id, error)
                log.info("%d/%d images embedded", min(start + len(chunk), len(todo)), len(todo))
            if dimension is None and keep_rows:
                dimension = next(iter(keep_rows.values()))[0][0].shape[1]
            if not todo:
                write()
            counts, ok = {}, True
            for category in image_store.counts():
                parts = keep_rows.get(category, [])
                ids = np.vstack([i for _, i in parts]) if parts else np.zeros((0, 2), np.int64)
                images = set(np.unique(ids[:, 0]).tolist())
                per_image_ok = all((ids[:, 0] == v).sum() == 1 + grid * grid for v in images)
                ok &= images == set(image_store.ids(category).tolist()) and per_image_ok
                counts[category] = f"{len(images)} images / {len(ids)} tiles"
            location = directory
            size = sum(p.stat().st_size for p in directory.rglob(f"tiles_{grid}x{grid}*") if p.is_file())
            log.info("tile store size on disk: %.2f MB", size / 1e6)
    finally:
        await client.aclose()

    for category in sorted(image_store.counts()):
        log.info("%-10s image index %4d  %s %s", category, image_store.counts()[category], key,
                 counts.get(category, 0))
    log.info("saved to %s; check against image index: %s", location, "ok" if ok else "MISMATCH")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature", choices=["gray", "edge", "tiles"], required=True)
    parser.add_argument("--grid", type=int, help="tiles only: grid size (default FEATURE_TILE_GRID)")
    parser.add_argument("--rebuild", action="store_true", help="ignore existing vectors and embed everything")
    parser.add_argument("--batch-size", type=int, default=16, help="images per /embed/batch request (max 32)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    mongo = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=settings.mongo_timeout_ms)
    ok = asyncio.run(build(args, settings, mongo[settings.mongo_db]))
    mongo.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
