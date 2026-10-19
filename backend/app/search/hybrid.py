"""Hybrid retrieval: search the image index and the caption index with one query vector, fuse with RRF.

Both indexes live in the same SigLIP space, so the same vector (an image embedding, or a text embedding for
text queries) can search both. Reciprocal rank fusion: score = sum over lists of 1 / (k + rank), rank from 1.
With only the image list (hybrid off) the order is exactly the image cosine order, as in Stage 5.
"""

from dataclasses import dataclass, field

from app.vector_index import CategoryIndexes


@dataclass
class Candidate:
    vector_id: int
    category: str
    image_rank: int | None = None
    image_score: float | None = None
    caption_rank: int | None = None
    caption_score: float | None = None
    rrf_score: float = 0.0
    boost: float = 0.0
    boosts: dict = field(default_factory=dict)  # field -> {"matched": [...], "boost": x}
    list_ranks: dict = field(default_factory=dict)  # extra RRF lists (Stage 9): name -> rank
    list_scores: dict = field(default_factory=dict)  # name -> cosine
    rerank: dict = field(default_factory=dict)  # Stage 9 re-scoring: name -> {"score": s, "added": weight * s}

    @property
    def rerank_total(self) -> float:
        return sum(item["added"] for item in self.rerank.values())

    @property
    def final_score(self) -> float:
        return self.rrf_score + self.boost + self.rerank_total


Hit = tuple[int, str, float]  # (vector_id, category, cosine)


def ranked_hits(store: CategoryIndexes, categories: list[str], vector, k: int) -> list[Hit]:
    """Top k hits over one or more categories, best first."""
    hits = [(vector_id, category, score)
            for category in categories for vector_id, score in store.search(category, vector, k=k)]
    return sorted(hits, key=lambda hit: hit[2], reverse=True)[:k]


def rrf_merge(image_hits: list[Hit], caption_hits: list[Hit], rrf_k: int = 60,
              extra_lists: dict[str, list[Hit]] | None = None) -> list[Candidate]:
    """Fuse ranked lists by vector_id: image, caption and any extra lists (e.g. "gray").

    Ties keep the higher image score first.
    """
    candidates: dict[int, Candidate] = {}
    for rank, (vector_id, category, score) in enumerate(image_hits, start=1):
        candidate = candidates.setdefault(vector_id, Candidate(vector_id, category))
        candidate.image_rank, candidate.image_score = rank, score
        candidate.rrf_score += 1 / (rrf_k + rank)
    for rank, (vector_id, category, score) in enumerate(caption_hits, start=1):
        candidate = candidates.setdefault(vector_id, Candidate(vector_id, category))
        candidate.caption_rank, candidate.caption_score = rank, score
        candidate.rrf_score += 1 / (rrf_k + rank)
    for name, hits in (extra_lists or {}).items():
        for rank, (vector_id, category, score) in enumerate(hits, start=1):
            candidate = candidates.setdefault(vector_id, Candidate(vector_id, category))
            candidate.list_ranks[name], candidate.list_scores[name] = rank, score
            candidate.rrf_score += 1 / (rrf_k + rank)
    return sorted(candidates.values(), key=lambda c: (c.rrf_score, c.image_score or -1.0), reverse=True)


def hybrid_search(image_store: CategoryIndexes, caption_store: CategoryIndexes | None, categories: list[str],
                  vector, k: int, rrf_k: int, use_captions: bool,
                  extra: dict[str, tuple[CategoryIndexes, list]] | None = None) -> tuple[list[Candidate], dict]:
    """Search the image and caption indexes (and any extra (store, vector) lists) and fuse.

    Returns (candidates best first, info for the debug block).
    """
    image_hits = ranked_hits(image_store, categories, vector, k)
    caption_hits = []
    captions_used = use_captions and caption_store is not None and any(caption_store.counts().get(c) for c in categories)
    if captions_used:
        caption_hits = ranked_hits(caption_store, categories, vector, k)
    extra_hits = {name: ranked_hits(store, categories, extra_vector, k)
                  for name, (store, extra_vector) in (extra or {}).items() if store is not None}
    info = {"image_hits": len(image_hits), "caption_hits": len(caption_hits), "captions_used": captions_used,
            "rrf_k": rrf_k, **{f"{name}_hits": len(hits) for name, hits in extra_hits.items()}}
    return rrf_merge(image_hits, caption_hits, rrf_k, extra_hits), info
