"""Helpers shared by image and text search: index caches, upload checks, image URLs, step timing."""

import logging
import time
from pathlib import Path
from typing import Callable

import faiss
import numpy as np
from fastapi import HTTPException, UploadFile

from app.config import PROJECT_ROOT
from app.vector_index import MANIFEST_NAME, CategoryIndexes, IndexMismatchError

log = logging.getLogger("search")

ALLOWED_TYPES = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
IMAGE_ROUTE = "/catalogue-images"
# Product fields that are internal bookkeeping, not attributes for the client
HIDDEN_PRODUCT_FIELDS = {"_id", "image_ids", "source", "updated_at", "category"}


def read_categories(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip().lower() for line in lines if line.strip() and not line.startswith("#")}


class IndexCache:
    """Holds loaded FAISS indexes and reloads them when a script rewrites the manifest.

    `make_empty` returns an empty CategoryIndexes with the expected model/version/source; `root` is the folder
    that holds its <model>_<version>/ directory (FAISS_INDEX_DIR for images, FAISS_INDEX_DIR/captions for captions).
    A category whose index does not match the current settings (or `dimension`) is not loaded; it is logged as
    an error and listed in the store's `rejected`, and the other categories are still served.
    """

    def __init__(self, root: Path, make_empty: Callable[[], CategoryIndexes], label: str, dimension: int | None = None):
        self.root = Path(root)
        self.make_empty = make_empty
        self.label = label
        self.dimension = dimension
        self._store: CategoryIndexes | None = None
        self._mtime: float | None = None
        self._matrices: dict[str, tuple[np.ndarray, dict[int, int]]] = {}
        self._for_store = None

    def get(self) -> CategoryIndexes:
        empty = self.make_empty()
        manifest = empty.directory(self.root) / MANIFEST_NAME
        try:
            mtime = manifest.stat().st_mtime
        except FileNotFoundError:
            return empty
        if self._store is None or mtime != self._mtime:
            self._store = CategoryIndexes.load_or_create(
                self.root, empty.model_name, empty.embedding_version, empty.preprocess_version,
                dimension=self.dimension, strict=False,
            )
            self._mtime = mtime
            log.info("loaded %s indexes: %s", self.label, self._store.counts())
            for category, reason in self._store.rejected.items():
                log.error("%s index for %r NOT SERVED: %s", self.label, category, reason)
        return self._store

    def vectors(self, category: str, vector_ids) -> dict[int, np.ndarray]:
        """Stored (normalized) vectors by vector_id, from a per-category matrix cached until the index reloads."""
        store = self.get_or_none()
        if store is None or category not in store.indexes:
            return {}
        if self._for_store is not store:  # reloaded: rebuild the lookups
            self._matrices, self._for_store = {}, store
        if category not in self._matrices:
            index = store.indexes[category]
            ids = faiss.vector_to_array(index.id_map)
            self._matrices[category] = (index.index.reconstruct_n(0, index.ntotal),
                                        {int(v): row for row, v in enumerate(ids)})
        matrix, rows = self._matrices[category]
        return {int(v): matrix[rows[int(v)]] for v in vector_ids if int(v) in rows}

    def get_or_none(self) -> CategoryIndexes | None:
        """Like get(), but a missing or mismatched index is None (used for optional caption indexes)."""
        try:
            store = self.get()
        except IndexMismatchError as exc:
            log.warning("%s indexes ignored: %s", self.label, exc)
            return None
        return store if store.total else None


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


class StepTimer:
    """Milliseconds per pipeline step, for logs and the response."""

    def __init__(self):
        self.started = self._clock = time.perf_counter()
        self.timings: dict[str, float] = {}

    def lap(self, step: str) -> None:
        now = time.perf_counter()
        self.timings[step] = round((now - self._clock) * 1000, 1)
        self._clock = now

    def record(self, step: str, ms: float) -> None:
        self.timings[step] = round(ms, 1)

    def finish(self) -> dict[str, float]:
        self.timings["total"] = round((time.perf_counter() - self.started) * 1000, 1)
        return self.timings
