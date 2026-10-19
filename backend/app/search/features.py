"""Stage 9 feature stores and similarity: colour-reduced (gray / edge) vectors and region (tile) vectors.

Gray and edge: one FAISS IndexIDMap(IndexFlatIP) per category under
FAISS_INDEX_DIR/features/<variant>/<model>_<version>/, same vector_id as the image index. Used as an extra
RRF list (search) and for re-scoring (vectors looked up by vector_id from a cached matrix).

Tiles: FEATURES_DIR/<model>_<version>/<category>/tiles_<g>x<g>.npy (float32, one row per tile) plus
tiles_<g>x<g>_ids.npy (int64 rows of [vector_id, tile index]) and a manifest per grid. Not in FAISS: only used
to re-score the top candidates. Loaded memory-mapped.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

from app.search.common import IndexCache
from app.vector_index import CategoryIndexes, index_dir_name, normalize

log = logging.getLogger("search")


# ---- gray / edge -------------------------------------------------------------------------------

def variant_root(settings, variant: str) -> Path:
    return Path(settings.faiss_index_dir) / "features" / variant


def new_variant_store(settings, signature: str) -> CategoryIndexes:
    return CategoryIndexes(settings.embed_model_name, settings.embed_version, signature)


class VariantIndexCache(IndexCache):
    """IndexCache for a gray/edge store (vector lookup by vector_id comes from IndexCache.vectors)."""

    def __init__(self, settings, variant: str, signature: str):
        super().__init__(variant_root(settings, variant), lambda: new_variant_store(settings, signature), variant)


def feature_similarity(query_vector, candidate_vectors: dict[int, np.ndarray]) -> dict[int, float]:
    """Cosine between the query's feature vector and each candidate's (all L2-normalized)."""
    query = normalize(query_vector)[0]
    return {vector_id: float(vector @ query) for vector_id, vector in candidate_vectors.items()}


def gray_similarity(query_gray_vector, candidate_vector_ids, category: str, cache: VariantIndexCache) -> dict[int, float]:
    """Gray cosine for each candidate that has a gray vector."""
    return feature_similarity(query_gray_vector, cache.vectors(category, candidate_vector_ids))


# ---- tiles -------------------------------------------------------------------------------------

def tile_dir(settings) -> Path:
    return Path(settings.features_dir) / index_dir_name(settings.embed_model_name, settings.embed_version)


def tile_names(grid: int) -> tuple[str, str, str]:
    stem = f"tiles_{grid}x{grid}"
    return f"{stem}.npy", f"{stem}_ids.npy", f"{stem}_manifest.json"


def save_tiles(directory: Path, grid: int, signature: str, per_category: dict[str, tuple[np.ndarray, np.ndarray]],
               dimension: int) -> None:
    """Write each category's tile matrix and ids, then the manifest (each file replaced atomically)."""
    matrix_name, ids_name, manifest_name = tile_names(grid)
    counts = {}
    for category, (matrix, ids) in per_category.items():
        folder = directory / category
        folder.mkdir(parents=True, exist_ok=True)
        for name, array in ((matrix_name, matrix.astype(np.float32)), (ids_name, ids.astype(np.int64))):
            tmp = folder / (name + ".tmp")
            with tmp.open("wb") as f:
                np.save(f, array)
            os.replace(tmp, folder / name)
        counts[category] = {"images": int(len(np.unique(ids[:, 0]))) if len(ids) else 0, "tiles": int(len(ids))}
    manifest = {"grid": grid, "tiles_per_image": 1 + grid * grid, "signature": signature, "dimension": dimension,
                "categories": counts}
    tmp = directory / (manifest_name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, directory / manifest_name)


class TileStore:
    """Memory-mapped tile vectors for one grid size, reloaded when the manifest changes."""

    def __init__(self, directory: Path, grid: int, signature: str):
        self.directory, self.grid, self.signature = Path(directory), grid, signature
        self._mtime = None
        self._data: dict[str, tuple[np.ndarray, dict[int, slice]]] = {}

    @property
    def manifest_path(self) -> Path:
        return self.directory / tile_names(self.grid)[2]

    def load(self) -> bool:
        """(Re)load if the manifest changed. False if there is no usable store."""
        try:
            mtime = self.manifest_path.stat().st_mtime
        except FileNotFoundError:
            return False
        if mtime == self._mtime:
            return bool(self._data)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._data, self._mtime = {}, mtime
        if manifest.get("signature") != self.signature:
            log.warning("tile store %s has %s, expected %s; ignored until rebuilt",
                        self.manifest_path, manifest.get("signature"), self.signature)
            return False
        matrix_name, ids_name, _ = tile_names(self.grid)
        for category in manifest["categories"]:
            matrix = np.load(self.directory / category / matrix_name, mmap_mode="r")
            ids = np.load(self.directory / category / ids_name)
            rows: dict[int, slice] = {}
            if len(ids):  # rows of one image are contiguous
                starts = np.flatnonzero(np.r_[True, ids[1:, 0] != ids[:-1, 0]])
                ends = np.r_[starts[1:], len(ids)]
                rows = {int(ids[s, 0]): slice(int(s), int(e)) for s, e in zip(starts, ends)}
            self._data[category] = (matrix, rows)
        log.info("loaded %dx%d tile store: %s", self.grid, self.grid,
                 {c: len(rows) for c, (_, rows) in self._data.items()})
        return True

    def tiles(self, category: str, vector_id: int) -> np.ndarray | None:
        matrix, rows = self._data.get(category, (None, {}))
        piece = rows.get(int(vector_id))
        return None if piece is None else np.asarray(matrix[piece])

    def ids(self, category: str) -> list[int]:
        return sorted(self._data.get(category, (None, {}))[1])


def region_similarity(query_tiles: np.ndarray, candidate_tiles: np.ndarray) -> float:
    """For each query tile take its best-matching candidate tile (cosine), then average over query tiles."""
    query = normalize(query_tiles)
    candidate = normalize(candidate_tiles)
    return float((query @ candidate.T).max(axis=1).mean())


# ---- query-time re-scoring ---------------------------------------------------------------------

def rerank_with_features(candidates: list, query: dict, settings, caches: dict, tile_store: TileStore | None) -> dict:
    """Add weight * similarity for the enabled features to the top candidates (sorted best first, in place).

    query: {"gray": vector, "edge": vector, "tiles": matrix} for the features computed for this request.
    Returns how many candidates each feature scored (for the debug block).
    """
    scored = {}
    for name in ("gray", "edge"):
        if not getattr(settings, f"feature_{name}_rerank") or query.get(name) is None or caches.get(name) is None:
            continue
        weight, top = getattr(settings, f"feature_{name}_rerank_weight"), candidates[:settings.feature_rerank_top_n]
        by_category: dict[str, list] = {}
        for candidate in top:
            by_category.setdefault(candidate.category, []).append(candidate)
        count = 0
        for category, group in by_category.items():
            vectors = caches[name].vectors(category, [c.vector_id for c in group])
            similarities = feature_similarity(query[name], vectors)
            for candidate in group:
                if candidate.vector_id in similarities:
                    score = similarities[candidate.vector_id]
                    candidate.rerank[name] = {"score": round(score, 4), "added": weight * score}
                    count += 1
        scored[name] = count

    if settings.feature_region_rerank and query.get("tiles") is not None and tile_store is not None and tile_store.load():
        weight, count = settings.feature_region_weight, 0
        for candidate in candidates[:settings.feature_region_top_n]:
            tiles = tile_store.tiles(candidate.category, candidate.vector_id)
            if tiles is not None and tiles.shape[1] != query["tiles"].shape[1]:
                log.error("tile store has %d-dim vectors, query has %d; rebuild it for this model",
                          tiles.shape[1], query["tiles"].shape[1])
                break
            if tiles is not None:
                score = region_similarity(query["tiles"], tiles)
                candidate.rerank["region"] = {"score": round(score, 4), "added": weight * score}
                count += 1
        scored["region"] = count
    return scored
