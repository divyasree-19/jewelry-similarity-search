"""POST /search/image and the catalogue image route.

Flow: validate upload -> preprocess -> embed -> search the category's FAISS index for candidates ->
fetch image and product metadata from MongoDB -> apply project_id / subcategory filters ->
keep the best image per product -> return top_k.
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.config import PROJECT_ROOT, Settings
from app.embed_client import EmbeddingServiceError
from app.preprocessing import ImagePreprocessError, PreprocessOptions, encode, preprocess
from app.vector_index import MANIFEST_NAME, CategoryIndexes, IndexMismatchError

log = logging.getLogger("search")
router = APIRouter()

ALLOWED_TYPES = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
IMAGE_ROUTE = "/catalogue-images"
# Product fields that are internal bookkeeping, not attributes for the client
HIDDEN_PRODUCT_FIELDS = {"_id", "image_ids", "source", "updated_at", "category"}


def read_categories(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip().lower() for line in lines if line.strip() and not line.startswith("#")}


class IndexCache:
    """Holds the loaded FAISS indexes and reloads them when ingest rewrites the manifest."""

    def __init__(self, settings: Settings, options: PreprocessOptions):
        self.settings = settings
        self.options = options
        self._store: CategoryIndexes | None = None
        self._mtime: float | None = None

    def get(self) -> CategoryIndexes:
        empty = CategoryIndexes(self.settings.embed_model_name, self.settings.embed_version, self.options.signature)
        manifest = empty.directory(self.settings.faiss_index_dir) / MANIFEST_NAME
        try:
            mtime = manifest.stat().st_mtime
        except FileNotFoundError:
            return empty
        if self._store is None or mtime != self._mtime:
            self._store = CategoryIndexes.load_or_create(
                self.settings.faiss_index_dir, self.settings.embed_model_name,
                self.settings.embed_version, self.options.signature,
            )
            self._mtime = mtime
            log.info("loaded FAISS indexes: %s", self._store.counts())
        return self._store


def image_url(image_path: str, catalogue_dir: Path) -> str | None:
    """Public URL for a catalogue image, or None if it lies outside the catalogue directory."""
    path = Path(image_path)
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    try:
        return f"{IMAGE_ROUTE}/{path.relative_to(catalogue_dir.resolve()).as_posix()}"
    except ValueError:
        return None


def sniff_type(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def read_upload(file: UploadFile, max_bytes: int) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported file type {file.content_type!r}; upload a JPEG, PNG or WebP image")
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"Image is larger than {max_bytes / 1_000_000:g} MB")
    if not data:
        raise HTTPException(400, "Uploaded file is empty")
    if sniff_type(data) is None:
        raise HTTPException(415, "File content is not a JPEG, PNG or WebP image")
    return data


@router.post("/search/image")
async def search_image(
    request: Request,
    file: UploadFile = File(..., description="JPEG, PNG or WebP photo"),
    category: str = Form(...),
    project_id: str | None = Form(None),
    subcategory: str | None = Form(None),
    top_k: int = Form(10),
):
    app = request.app
    settings: Settings = app.state.settings
    query_id = uuid.uuid4().hex
    timings: dict[str, float] = {}
    started = clock = time.perf_counter()

    def lap(step: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[step] = round((now - clock) * 1000, 1)
        clock = now

    category = category.strip().lower()
    if category not in app.state.categories:
        raise HTTPException(400, f"Unknown category {category!r}; expected one of {sorted(app.state.categories)}")
    if not 1 <= top_k <= settings.search_max_top_k:
        raise HTTPException(400, f"top_k must be between 1 and {settings.search_max_top_k}")
    project_id = (project_id or "").strip() or None
    subcategory = (subcategory or "").strip().lower() or None

    try:
        store = app.state.indexes.get()
    except IndexMismatchError as exc:
        log.error("query %s: %s", query_id, exc)
        raise HTTPException(503, "Search index was built with different settings; re-run scripts/ingest.py --rebuild")
    if store.counts().get(category, 0) == 0:
        raise HTTPException(404, f"No catalogue images are indexed for category {category!r}")

    data = await read_upload(file, int(settings.max_upload_mb * 1_000_000))
    if settings.save_query_images:
        settings.query_image_dir.mkdir(parents=True, exist_ok=True)
        (settings.query_image_dir / f"{query_id}.{ALLOWED_TYPES[sniff_type(data)]}").write_bytes(data)
    lap("read")

    try:
        image = await asyncio.to_thread(preprocess, data, app.state.preprocess_options)
    except ImagePreprocessError as exc:
        raise HTTPException(400, str(exc))
    lap("preprocess")

    try:
        vector = await app.state.embed_client.embed(encode(image), f"{query_id}.png", "image/png")
    except EmbeddingServiceError as exc:
        log.error("query %s: %s", query_id, exc)
        raise HTTPException(502, "Embedding service is unavailable; try again shortly")
    lap("embed")

    hits = store.search(category, vector, k=settings.search_candidates)
    lap("faiss")

    db = app.state.mongo[settings.mongo_db]
    scores = dict(hits)
    try:
        images = {
            doc["vector_id"]: doc
            async for doc in db.images.find({"vector_id": {"$in": list(scores)}, "category": category})
        }
        product_ids = list({doc["product_id"] for doc in images.values()})
        products = {doc["_id"]: doc async for doc in db.products.find({"_id": {"$in": product_ids}})}
    except Exception as exc:  # pymongo raises many error types; all mean the same to the caller
        log.error("query %s: MongoDB error %s", query_id, exc)
        raise HTTPException(503, "Catalogue database is unavailable; try again shortly")
    lap("metadata")

    results, seen = [], set()
    for vector_id, score in hits:  # best first, so the first image of a product is its best one
        doc = images.get(vector_id)
        if doc is None:
            log.warning("query %s: vector_id %d has no MongoDB image; re-run ingest", query_id, vector_id)
            continue
        if project_id and doc.get("project_id") != project_id:
            continue
        if subcategory and (doc.get("subcategory") or "").lower() != subcategory:
            continue
        if doc["product_id"] in seen:
            continue
        seen.add(doc["product_id"])
        product = products.get(doc["product_id"], {})
        results.append({
            "product_id": doc["product_id"],
            "image_id": doc["_id"],
            "image_url": image_url(doc["image_path"], settings.catalogue_dir),
            "category": doc["category"],
            "similarity_score": round(score, 4),
            "attributes": {key: value for key, value in product.items() if key not in HIDDEN_PRODUCT_FIELDS},
        })
        if len(results) == top_k:
            break
    lap("rank")
    timings["total"] = round((time.perf_counter() - started) * 1000, 1)

    log.info(
        "query %s category=%s project_id=%s subcategory=%s top_k=%d candidates=%d timings_ms=%s results=%s",
        query_id, category, project_id, subcategory, top_k, len(hits), timings,
        [r["image_id"] for r in results],
    )
    return {
        "query_id": query_id,
        "embed_backend": settings.embed_backend,
        "model": settings.embed_model_name,
        "embedding_version": settings.embed_version,
        "timings_ms": timings,
        "category": category,
        "results": results,
    }


@router.get(IMAGE_ROUTE + "/{image_path:path}", include_in_schema=False)
async def catalogue_image(image_path: str, request: Request):
    root = request.app.state.settings.catalogue_dir.resolve()
    path = (root / image_path).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(404, "Image not found")
    return FileResponse(path)
