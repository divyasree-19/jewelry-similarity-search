"""Stage 10 de-duplication of ranked results.

- Same product: only its best image is kept (the first one in score order).
- Different products whose images are near-identical (image cosine >= DEDUP_NEAR_DUPLICATE_THRESHOLD): both are
  kept; the lower-ranked one is marked possible_duplicate_of the higher one. Nothing is hidden: the catalogue may
  really contain two products photographed identically, and only a person can decide.
"""

import numpy as np


def best_image_per_product(ranked: list, product_of) -> list:
    """The first candidate of each product, in the given order."""
    seen, kept = set(), []
    for candidate in ranked:
        product_id = product_of(candidate)
        if product_id not in seen:
            seen.add(product_id)
            kept.append(candidate)
    return kept


def mark_near_duplicates(results: list[dict], vectors: dict[int, np.ndarray], threshold: float) -> None:
    """Set result["possible_duplicate_of"] (image_id, product_id, image cosine) on lower-ranked look-alikes.

    results: in rank order, each with "vector_id", "image_id" and "product_id"; vectors: normalized image vectors.
    """
    earlier: list[tuple[dict, np.ndarray]] = []
    for result in results:
        vector = vectors.get(result["vector_id"])
        if vector is None:
            continue
        for other, other_vector in earlier:
            similarity = float(vector @ other_vector)
            if other["product_id"] != result["product_id"] and similarity >= threshold:
                result["possible_duplicate_of"] = {"image_id": other["image_id"], "product_id": other["product_id"],
                                                   "image_similarity": round(similarity, 4)}
                break
        earlier.append((result, vector))
