"""Text search flow (Stage 8):

parse text with the LLM -> QueryPlan (category is a hard filter; none -> all categories or ask) ->
embed semantic_query with the SigLIP text encoder -> search caption index AND image index with that vector ->
RRF -> MongoDB metadata -> boosts on the attributes the user wrote -> one result per product -> top_k.

similarity_score is the caption cosine (text vs. description), falling back to the image cosine.
Text search needs Qwen: if it is unreachable or slower than LLM_QUERY_TIMEOUT_SECONDS the answer is a 503.
If it answers but the reply is unusable, the raw text is embedded and searched without attribute boosts.
"""

import asyncio
import logging
import uuid

from fastapi import HTTPException

from app.clients.llm_client import LLMError, LLMUnavailableError
from app.embed_client import EmbeddingServiceError
from app.search.common import StepTimer
from app.search.hybrid import hybrid_search
from app.search.image_search import (EMBEDDING_DOWN, fetch_metadata, flags, image_store_or_503, public,
                                     rank_results, refused_warning, score_meaning, searchable_categories)
from app.search.query_analysis import plan_text_query

log = logging.getLogger("search")


async def search_by_text(app, text: str, *, category: str | None, project_id: str | None, top_k: int,
                         debug: bool, collect_signals: bool = False) -> dict:
    settings = app.state.settings
    query_id = uuid.uuid4().hex
    timer = StepTimer()
    warnings: list[str] = []

    store = image_store_or_503(app, query_id)
    indexed = searchable_categories(app, store, category)

    parsed = None
    try:
        parsed = (await asyncio.wait_for(app.state.llm_client.parse_query(text),
                                         settings.llm_query_timeout_seconds)).tags
    except (LLMUnavailableError, asyncio.TimeoutError) as exc:
        log.error("query %s: query parser unavailable: %r", query_id, exc)
        raise HTTPException(503, "The query parser (Qwen) is unavailable or too slow, so text search cannot run; "
                                 "try again shortly, or use image search")
    except LLMError as exc:
        warnings.append("could not parse the query; searched by the raw text without attribute boosts")
        log.warning("query %s: text parsing failed: %s", query_id, exc)
    timer.lap("parse_text")

    semantic_query = ((parsed.semantic_query if parsed else None) or text).lower()
    try:
        vector = await app.state.embed_client.embed_text(semantic_query)
        timer.lap("embed")
        plan = await plan_text_query(settings=settings, detector=app.state.category_detector,
                                     embed_client=app.state.embed_client, vector=vector, user_category=category,
                                     indexed=indexed, parsed_text=parsed)
    except EmbeddingServiceError as exc:
        log.error("query %s: %s", query_id, exc)
        raise HTTPException(503, EMBEDDING_DOWN)
    plan.semantic_query = semantic_query
    timer.lap("analyse")

    response = {
        "query_id": query_id, "mode": "text", "text": text, "embed_backend": settings.embed_backend,
        "model": settings.embed_model_name, "embedding_version": settings.embed_version,
        "parsed_text": parsed.model_dump(exclude_defaults=True) if parsed else None,
        "category": plan.category, "category_source": plan.category_source,
        "searched_categories": plan.search_categories, "needs_category": plan.needs_category,
        "category_candidates": plan.category_candidates, "results": [],
    }
    if plan.needs_category:
        response["message"] = "The query names no category; choose one of category_candidates and search again"
    else:
        # Text vectors always search both indexes: captions are the natural match for text
        candidates, hybrid_info = hybrid_search(
            store, app.state.caption_indexes.get_or_none(), plan.search_categories, vector,
            settings.search_candidates, settings.search_rrf_k, use_captions=True,
        )
        timer.lap("search")
        images, products = await fetch_metadata(app, candidates, query_id)
        timer.lap("metadata")
        collected = [] if collect_signals else None
        response["results"] = public(rank_results(
            app, candidates, images, products, plan, project_id=project_id, subcategory=None, top_k=top_k,
            debug=debug, display_score="caption", query_id=query_id,
            query_vectors={"image": vector, "caption": vector}, collect=collected,
        ))
        timer.lap("rank")
        response.update(score_meaning(app, "text"))
        if collected is not None:
            response["candidate_signals"] = collected
        if debug:
            response["debug"] = {"plan": plan.debug(), "hybrid": hybrid_info, "flags": flags(settings)}
    if not category:
        warnings += refused_warning(store)
    if warnings:
        response["warnings"] = warnings
    response["timings_ms"] = timer.finish()
    log.info("query %s mode=text text=%r category=%s searched=%s timings_ms=%s results=%s", query_id, text,
             plan.category, plan.search_categories, response["timings_ms"], [r["image_id"] for r in response["results"]])
    return response
