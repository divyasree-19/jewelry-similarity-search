"""Per-category FAISS indexes stored under FAISS_INDEX_DIR/<model>_<version>/.

Each category has one IndexIDMap(IndexFlatIP). Vectors are L2-normalized before they are added or
searched, so the inner product is the cosine similarity. The FAISS ID of a vector is the vector_id
stored on its MongoDB image document.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

MANIFEST_NAME = "manifest.json"


class IndexMismatchError(RuntimeError):
    """The stored indexes were built with a different model, embedding version or preprocessing."""


def index_dir_name(model_name: str, embedding_version: str) -> str:
    """"google/siglip2-base-patch16-224" + "siglip2-base-v1" -> "siglip2-base-patch16-224_siglip2-base-v1"."""
    return f"{model_name.rstrip('/').split('/')[-1]}_{embedding_version}"


def normalize(vectors) -> np.ndarray:
    array = np.ascontiguousarray(np.atleast_2d(np.asarray(vectors, dtype=np.float32)))
    faiss.normalize_L2(array)
    return array


@dataclass
class CategoryIndexes:
    model_name: str
    embedding_version: str
    preprocess_version: str
    dimension: int | None = None
    indexes: dict[str, faiss.IndexIDMap] = field(default_factory=dict)
    created_at: str | None = None

    # ---- persistence -------------------------------------------------------------------------

    @classmethod
    def load_or_create(cls, root: Path, model_name: str, embedding_version: str, preprocess_version: str) -> "CategoryIndexes":
        """Load the indexes for this model/version, or start empty if none exist yet.

        Raises IndexMismatchError if the stored manifest records a different preprocessing version.
        """
        store = cls(model_name, embedding_version, preprocess_version)
        directory = store.directory(root)
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.exists():
            return store

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, expected in (
            ("model", model_name), ("embedding_version", embedding_version), ("preprocess_version", preprocess_version)
        ):
            if manifest.get(key) != expected:
                raise IndexMismatchError(
                    f"{manifest_path} has {key}={manifest.get(key)!r}, current settings give {expected!r}; "
                    "run with --rebuild"
                )
        store.dimension = manifest["dimension"]
        store.created_at = manifest.get("created_at")
        for category, info in manifest["categories"].items():
            index = faiss.read_index(str(directory / info["file"]))
            if index.ntotal != info["count"] or index.d != store.dimension:
                raise IndexMismatchError(f"{category}: index file does not match {manifest_path}; run with --rebuild")
            store.indexes[category] = index
        return store

    def directory(self, root: Path) -> Path:
        return Path(root) / index_dir_name(self.model_name, self.embedding_version)

    def save(self, root: Path) -> Path:
        """Write every category index and then the manifest. Each file is replaced atomically."""
        directory = self.directory(root)
        directory.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.created_at = self.created_at or now

        for category, index in self.indexes.items():
            _atomic_write(directory / f"{category}.index", lambda tmp, index=index: faiss.write_index(index, str(tmp)))
        for stale in directory.glob("*.index"):
            if stale.stem not in self.indexes:
                stale.unlink()

        manifest = {
            "model": self.model_name,
            "embedding_version": self.embedding_version,
            "preprocess_version": self.preprocess_version,
            "dimension": self.dimension,
            "metric": "inner_product_on_l2_normalized (cosine)",
            "index_type": "IndexIDMap(IndexFlatIP)",
            "categories": {
                category: {"file": f"{category}.index", "count": int(index.ntotal)}
                for category, index in sorted(self.indexes.items())
            },
            "total": self.total,
            "created_at": self.created_at,
            "updated_at": now,
        }
        _atomic_write(
            directory / MANIFEST_NAME,
            lambda tmp: tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8"),
        )
        return directory

    # ---- contents ----------------------------------------------------------------------------

    @property
    def total(self) -> int:
        return sum(int(index.ntotal) for index in self.indexes.values())

    def counts(self) -> dict[str, int]:
        return {category: int(index.ntotal) for category, index in sorted(self.indexes.items())}

    def ids(self, category: str) -> np.ndarray:
        index = self.indexes.get(category)
        return faiss.vector_to_array(index.id_map).astype(np.int64) if index is not None else np.empty(0, np.int64)

    def id_to_category(self) -> dict[int, str]:
        return {int(vector_id): category for category in self.indexes for vector_id in self.ids(category)}

    def add(self, category: str, vector_ids, vectors) -> None:
        vectors = normalize(vectors)
        if self.dimension is None:
            self.dimension = vectors.shape[1]
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")
        if category not in self.indexes:
            self.indexes[category] = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
        self.indexes[category].add_with_ids(vectors, np.asarray(vector_ids, dtype=np.int64))

    def remove(self, category: str, vector_ids) -> int:
        index = self.indexes.get(category)
        if index is None or len(vector_ids) == 0:
            return 0
        removed = index.remove_ids(np.asarray(list(vector_ids), dtype=np.int64))
        if index.ntotal == 0:
            del self.indexes[category]
        return int(removed)

    def search(self, category: str, vector, k: int = 10) -> list[tuple[int, float]]:
        """Return up to k (vector_id, cosine similarity) pairs, best first."""
        index = self.indexes.get(category)
        if index is None or index.ntotal == 0:
            return []
        scores, ids = index.search(normalize(vector), min(k, index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)
