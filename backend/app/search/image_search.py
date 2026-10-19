"""Image search flow (Stage 8), optionally refined by text:

preprocess -> [embed image | tag query image (optional) | parse text (optional)] in parallel -> QueryPlan
(category given / from text / detected) -> hybrid search (image + caption index, RRF) -> MongoDB metadata
-> filters (project_id, subcategory, optional hard metal filter) -> boosts -> one result per product -> top_k.

similarity_score in each result is the image cosine (for display); results are ordered by the fused score.
"""

import asyncio
import logging
import time
import uuid

from fastapi import HTTPException

from app.clients.llm_client import LLMError
from app.clients.vlm_client import VLMError
from app.embed_client import EmbeddingServiceError
import numpy as np

from app.preprocessing import ImagePreprocessError, encode, make_tiles, preprocess, to_edge_map, to_grayscale
from app.search.boosts import apply_boosts
from app.search.dedup import best_image_per_product, mark_near_duplicates
from app.search.features import rerank_with_features
from app.search.vlm_judge import judge_top
from app.search.reranker import combine, labels, load_weights, weights_path
from app.vector_index import normalize
from app.search.common import HIDDEN_PRODUCT_FIELDS, StepTimer, image_url
from app.search.hybrid import Candidate, hybrid_search
from app.search.query_analysis import plan_image_query
from app.vector_index import IndexMismatchError

log = logging.getLogger("search")


def image_store_or_503(app, query_id: str):
    try:
        return app.state.indexes.get()
    except IndexMismatchError as exc:
        log.error("query %s: %s", query_id, exc)
        raise HTTPException(503, "Search index was built with different settings; re-run scripts/ingest.py --rebuild")


def indexed_categories(app, store) -> list[str]:
    return sorted(c for c, n in store.counts().items() if n and c in app.state.categories)


async def timed(coro):
    """(result or exception, milliseconds)."""
    started = time.perf_counter()
    try:
        result = await coro
    except Exception as exc:  # the caller decides which failures are fatal
        result = exc
    return result, (time.perf_counter() - started) * 1000


async def fetch_metadata(app, candidates: list[Candidate], query_id: str):
    """(image docs by vector_id, product docs by product_id) for the candidates."""
    db = app.state.mongo[app.state.settings.mongo_db]
    try:
        images = {doc["vector_id"]: doc
                  async for doc in db.images.find({"vector_id": {"$in": [c.vector_id for c in candidates]}})}
        product_ids = list({doc["product_id"] for doc in images.values()})
        products = {doc["_id"]: doc async for doc in db.products.find({"_id": {"$in": product_ids}})}
    except Exception as exc:  # pymongo raises many error types; all mean the same to the caller
        log.error("query %s: MongoDB error %s", query_id, exc)
        raise HTTPException(503, "Catalogue database is unavailable; try again shortly")
    return images, products


class RerankerCache:
    """The reranker weights file for RERANKER_WEIGHTS_VERSION, reloaded when it changes."""

    def __init__(self, settings):
        self.settings = settings
        self._weights, self._mtime = None, None

    def get(self):
        path = weights_path(self.settings.reranker_weights_dir, self.settings.reranker_weights_version)
        try:
            mtime = path.stat().st_mtime
            if mtime != self._mtime:
                self._weights, self._mtime = load_weights(self.settings.reranker_weights_dir,
                                                          self.settings.reranker_weights_version), mtime
        except (OSError, ValueError, KeyError) as exc:
            log.error("reranker weights %s unusable (%s); using the Stage 9 order", path, exc)
            return None
        return self._weights


def candidate_signals(app, candidates: list[Candidate], query_vectors: dict) -> dict[int, dict]:
    """Raw cosine signals per candidate. A value missing from retrieval is looked up by vector_id."""
    settings = app.state.settings
    sources = {"image": app.state.indexes, "caption": app.state.caption_indexes if settings.search_hybrid else None}
    if query_vectors.get("gray") is not None:
        sources["gray"] = app.state.feature_indexes.get("gray")
    found: dict[str, dict[int, float]] = {}
    for name, cache in sources.items():
        query = query_vectors.get(name)
        values = {}
        for candidate in candidates:
            known = {"image": candidate.image_score, "caption": candidate.caption_score}.get(
                name, candidate.list_scores.get(name, (candidate.rerank.get(name) or {}).get("score")))
            if known is not None:
                values[candidate.vector_id] = float(known)
        missing = [c for c in candidates if c.vector_id not in values]
        if missing and cache is not None and query is not None:
            query_unit = normalize(query)[0]
            by_category: dict[str, list[int]] = {}
            for candidate in missing:
                by_category.setdefault(candidate.category, []).append(candidate.vector_id)
            for category, ids in by_category.items():
                for vector_id, vector in cache.vectors(category, ids).items():
                    values[vector_id] = float(vector @ query_unit)
        found[name] = values
    return {c.vector_id: {"image": found["image"].get(c.vector_id),
                          "caption": found.get("caption", {}).get(c.vector_id),
                          "gray": found.get("gray", {}).get(c.vector_id),
                          "region": (c.rerank.get("region") or {}).get("score")}
            for c in candidates}


def rank_results(app, candidates: list[Candidate], images: dict, products: dict, plan, *, project_id, subcategory,
                 top_k: int, debug: bool, display_score: str, query_id: str, rerank=None,
                 query_vectors: dict | None = None, collect: list | None = None) -> list[dict]:
    """Filter, boost, re-score (Stage 9, if `rerank` is given), re-rank (Stage 10, if on), keep the best image
    per product, mark near-duplicates, build results.

    collect: if a list is given, one row of raw signals per re-ranked candidate is appended (for tuning).
    """
    settings = app.state.settings
    kept = []
    for candidate in candidates:
        doc = images.get(candidate.vector_id)
        if doc is None or doc["category"] != candidate.category:
            log.warning("query %s: vector_id %d has no matching MongoDB image; re-run ingest",
                        query_id, candidate.vector_id)
            continue
        if project_id and doc.get("project_id") != project_id:
            continue
        if subcategory and (doc.get("subcategory") or "").lower() != subcategory:
            continue
        wanted_metal = plan.hard_filters.get("metal")
        if wanted_metal and not set(wanted_metal) & set((doc.get("vlm_tags") or {}).get("metal") or []):
            continue
        kept.append(candidate)

    if settings.search_boosts:
        apply_boosts(kept, {c.vector_id: images[c.vector_id].get("vlm_tags") for c in kept}, plan, settings)
    kept.sort(key=lambda c: (c.final_score, c.image_score or -1.0), reverse=True)
    if rerank is not None:
        rerank(kept)
        kept.sort(key=lambda c: (c.final_score, c.image_score or -1.0), reverse=True)

    weights = app.state.reranker.get() if reranker_on(settings, plan.mode) and query_vectors is not None else None
    scored: dict[int, tuple[float, dict, dict]] = {}
    if weights is not None:
        top, rest = kept[:settings.reranker_top_n], kept[settings.reranker_top_n:]
        signals = candidate_signals(app, top, query_vectors)
        mode = weights.for_mode(plan.mode)
        for position, candidate in enumerate(top):
            score, detail = combine(signals[candidate.vector_id], candidate.boost, mode, weights.boost_scale,
                                    weights.boost_cap)
            scored[candidate.vector_id] = (score, detail, signals[candidate.vector_id])
            if collect is not None:
                collect.append({"vector_id": candidate.vector_id, "product_id": images[candidate.vector_id]["product_id"],
                                "signals": signals[candidate.vector_id], "boost": candidate.boost,
                                "stage9_position": position})
        top.sort(key=lambda c: (scored[c.vector_id][0], scored[c.vector_id][2]["image"] or -1.0), reverse=True)
        kept = top + rest

    results = []
    for candidate in best_image_per_product(kept, lambda c: images[c.vector_id]["product_id"]):
        doc = images[candidate.vector_id]
        product = products.get(doc["product_id"], {})
        if weights is not None:
            score = scored[candidate.vector_id][0] if candidate.vector_id in scored else None
        else:
            score = candidate.caption_score if display_score == "caption" else candidate.image_score
            if score is None:
                score = candidate.image_score if display_score == "caption" else None
        result = {
            "product_id": doc["product_id"],
            "image_id": doc["_id"],
            "image_url": image_url(doc["image_path"], settings.catalogue_dir),
            "category": doc["category"],
            "similarity_score": round(score, 4) if score is not None else None,
            "score": round(scored[candidate.vector_id][0] if candidate.vector_id in scored else candidate.final_score, 6),
            "attributes": {key: value for key, value in product.items() if key not in HIDDEN_PRODUCT_FIELDS},
            "vector_id": candidate.vector_id,  # internal; removed before the response is sent
            "_image_path": doc["image_path"],
        }
        if debug:
            result["debug"] = {
                "image_rank": candidate.image_rank, "image_score": _round(candidate.image_score),
                "caption_rank": candidate.caption_rank, "caption_score": _round(candidate.caption_score),
                "rrf_score": round(candidate.rrf_score, 6), "boosts": candidate.boosts,
                "boost_total": round(candidate.boost, 6),
                **{f"{name}_rank": rank for name, rank in candidate.list_ranks.items()},
                **{f"{name}_score": _round(score) for name, score in candidate.list_scores.items()},
                "rerank": {name: {"score": item["score"], "added": round(item["added"], 6)}
                           for name, item in candidate.rerank.items()},
                "stage9_score": round(candidate.final_score, 6),
                "vlm_tags": {k: v for k, v in (doc.get("vlm_tags") or {}).items()
                             if k in ("metal", "gemstones", "stone_colors", "motifs", "style")},
            }
            if candidate.vector_id in scored:
                _, detail, _ = scored[candidate.vector_id]
                result["debug"]["signals"] = detail
                result["debug"]["labels"] = labels(detail, candidate.boosts)
        results.append(result)
        if len(results) == top_k:
            break

    if weights is not None:  # Stage 10: flag look-alike products (never hidden)
        by_category: dict[str, list[int]] = {}
        for result in results:
            by_category.setdefault(result["category"], []).append(result["vector_id"])
        vectors = {}
        for category, ids in by_category.items():
            vectors.update(app.state.indexes.vectors(category, ids))
        mark_near_duplicates(results, vectors, settings.dedup_near_duplicate_threshold)
        for result in results:
            duplicate = result.pop("possible_duplicate_of", None)
            if duplicate and debug:
                result["debug"]["possible_duplicate_of"] = duplicate
    return results


def public(results: list[dict]) -> list[dict]:
    """Drop the internal fields from results before they are returned."""
    return [{k: v for k, v in r.items() if k not in ("vector_id", "_image_path")} for r in results]


def _round(value):
    return round(value, 4) if value is not None else None


def flags(settings) -> dict:
    return {"hybrid": settings.search_hybrid, "boosts": settings.search_boosts,
            "query_tagging": settings.search_query_tagging, "category_detection": settings.category_detection,
            "refine_metal_hard_filter": settings.refine_metal_hard_filter,
            **{name: getattr(settings, f"feature_{name}") for name in FEATURE_FLAGS}}


FEATURE_FLAGS = ("gray_retrieval", "gray_rerank", "edge_retrieval", "edge_rerank", "region_rerank")


def wanted_features(settings) -> list[str]:
    """Which query-side feature images this request needs (Stage 9 flags)."""
    wanted = []
    for name in ("gray", "edge"):
        if getattr(settings, f"feature_{name}_retrieval") or getattr(settings, f"feature_{name}_rerank"):
            wanted.append(name)
    if settings.feature_region_rerank:
        wanted.append("tiles")
    return wanted


async def embed_with_features(app, image, png: bytes, features: list[str], query_id: str):
    """One /embed/batch call for the colour image plus its gray / edge / tile versions -> (vector, {feature: ...})."""
    pictures = [("colour", png)]
    if "gray" in features:
        pictures.append(("gray", encode(to_grayscale(image))))
    if "edge" in features:
        pictures.append(("edge", encode(to_edge_map(image))))
    if "tiles" in features:
        pictures += [("tiles", encode(tile)) for tile in make_tiles(image, app.state.settings.feature_tile_grid)]
    vectors = await app.state.embed_client.embed_batch(
        [(data, f"{query_id}_{i}.png", "image/png") for i, (_, data) in enumerate(pictures)])
    found: dict = {}
    for (name, _), vector in zip(pictures, vectors):
        found.setdefault(name, []).append(vector)
    query = {name: found[name][0] for name in ("gray", "edge") if name in found}
    if "tiles" in found:
        query["tiles"] = np.asarray(found["tiles"], dtype=np.float32)
    return found["colour"][0], query


def reranker_on(settings, mode: str) -> bool:
    modes = {m.strip() for m in settings.reranker_modes.split(",") if m.strip()}
    return settings.search_reranker and ("text" if mode == "text" else "image") in modes


def score_meaning(app, mode: str) -> dict:
    """What similarity_score means in this response."""
    settings = app.state.settings
    weights = app.state.reranker.get() if reranker_on(settings, mode) else None
    if weights is None:
        return {"similarity_score": "image cosine similarity (caption cosine for text search)"}
    return {"similarity_score": "0-1 similarity score (weighted signals); not a probability",
            "reranker_weights_version": weights.version}


async def search_by_image(app, data: bytes, *, category: str | None, text: str | None, project_id: str | None,
                          subcategory: str | None, top_k: int, debug: bool, collect_signals: bool = False) -> dict:
    settings = app.state.settings
    query_id = uuid.uuid4().hex
    timer = StepTimer()
    warnings: list[str] = []

    store = image_store_or_503(app, query_id)
    indexed = indexed_categories(app, store)
    if category and category not in indexed:
        raise HTTPException(404, f"No catalogue images are indexed for category {category!r}")
    if not indexed:
        raise HTTPException(404, "No catalogue images are indexed; run scripts/ingest.py")

    try:
        image = await asyncio.to_thread(preprocess, data, app.state.preprocess_options)
    except ImagePreprocessError as exc:
        raise HTTPException(400, str(exc))
    png = encode(image)
    timer.lap("preprocess")

    # Model calls in parallel: the embedding is required, tagging and text parsing are optional
    want_tags = settings.search_query_tagging or (settings.category_detection == "qwen" and not category)
    features = wanted_features(settings)
    if features:  # Stage 9: colour + feature images in one batch call
        calls = {"embed": embed_with_features(app, image, png, features, query_id)}
    else:  # flags off: exactly the Stage 8 call
        calls = {"embed": app.state.embed_client.embed(png, f"{query_id}.png", "image/png")}
    if want_tags:
        calls["tag_query"] = asyncio.wait_for(app.state.vlm_client.tag(png), settings.search_query_tagging_timeout_seconds)
    if text:
        calls["parse_text"] = app.state.llm_client.parse_query(text)
    outcomes = dict(zip(calls, await asyncio.gather(*(timed(call) for call in calls.values()))))
    for step, (_, ms) in outcomes.items():
        timer.record(step, ms)
    timer.lap("model_calls")

    vector, query_features = outcomes["embed"][0], {}
    if isinstance(vector, Exception):
        log.error("query %s: %s", query_id, vector)
        raise HTTPException(502, "Embedding service is unavailable; try again shortly")
    if features:
        vector, query_features = vector
    query_tags = None
    if "tag_query" in outcomes:
        result = outcomes["tag_query"][0]
        if isinstance(result, (VLMError, asyncio.TimeoutError)):
            warnings.append("query image tagging failed; searched without tag boosts")
            log.warning("query %s: query tagging failed: %r", query_id, result)
        elif isinstance(result, Exception):
            raise result
        else:
            query_tags = result.tags.model_dump()
    parsed = None
    if "parse_text" in outcomes:
        result = outcomes["parse_text"][0]
        if isinstance(result, LLMError):
            warnings.append("could not understand the text; searched by image only")
            log.warning("query %s: text parsing failed: %s", query_id, result)
        elif isinstance(result, Exception):
            raise result
        else:
            parsed = result.tags

    try:
        plan = await plan_image_query(
            settings=settings, detector=app.state.category_detector, embed_client=app.state.embed_client,
            vector=vector, user_category=category, indexed=indexed, query_tags=query_tags, parsed_text=parsed,
        )
    except EmbeddingServiceError as exc:
        log.error("query %s: category detection failed: %s", query_id, exc)
        raise HTTPException(502, "Embedding service is unavailable; try again shortly")
    timer.lap("analyse")

    response = {
        "query_id": query_id, "mode": plan.mode, "embed_backend": settings.embed_backend,
        "model": settings.embed_model_name, "embedding_version": settings.embed_version,
        "category": plan.category, "category_source": plan.category_source,
        "category_confidence": plan.category_confidence, "needs_category": plan.needs_category,
        "category_candidates": plan.category_candidates, "results": [],
    }
    if text:
        response["text"] = text
        response["parsed_text"] = parsed.model_dump(exclude_defaults=True) if parsed else None
    if plan.needs_category:
        response["message"] = "Not sure which category this is; choose one of category_candidates and search again"
    else:
        extra = {name: (app.state.feature_indexes[name].get_or_none(), query_features[name])
                 for name in ("gray", "edge")
                 if getattr(settings, f"feature_{name}_retrieval") and name in query_features}
        candidates, hybrid_info = hybrid_search(
            store, app.state.caption_indexes.get_or_none() if settings.search_hybrid else None,
            plan.search_categories, vector, settings.search_candidates, settings.search_rrf_k, settings.search_hybrid,
            extra=extra,
        )
        timer.lap("search")
        images, products = await fetch_metadata(app, candidates, query_id)
        timer.lap("metadata")
        rescored = {}

        def rerank(kept):
            rescored.update(rerank_with_features(kept, query_features, settings, app.state.feature_indexes,
                                                 app.state.tile_store))
        collected = [] if collect_signals else None
        results = rank_results(
            app, candidates, images, products, plan, project_id=project_id, subcategory=subcategory,
            top_k=top_k, debug=debug, display_score="image", query_id=query_id,
            rerank=rerank if query_features else None,
            query_vectors={"image": vector, "caption": vector, "gray": query_features.get("gray")},
            collect=collected,
        )
        timer.lap("rank")
        if settings.feature_vlm_judge and results:
            response["vlm_judge"] = await judge_top(app, results, png)
            timer.lap("vlm_judge")
        response["results"] = public(results)
        response.update(score_meaning(app, plan.mode))
        if collected is not None:
            response["candidate_signals"] = collected
        if rescored:
            hybrid_info["rescored"] = rescored
        if debug:
            response["debug"] = {"plan": plan.debug(), "hybrid": hybrid_info, "flags": flags(settings)}
    if warnings:
        response["warnings"] = warnings
    response["timings_ms"] = timer.finish()
    log.info(
        "query %s mode=%s category=%s source=%s confidence=%s project_id=%s subcategory=%s top_k=%d "
        "timings_ms=%s results=%s",
        query_id, plan.mode, plan.category, plan.category_source, plan.category_confidence, project_id, subcategory,
        top_k, response["timings_ms"], [r["image_id"] for r in response["results"]],
    )
    return response
