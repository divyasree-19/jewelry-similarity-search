"""Turn a request into a QueryPlan: which categories to search, where the category came from, and which
tags to boost on or filter by.

Category, in order: the user's category field > a category written in the query text > detection.
Detection (no category given): SigLIP zero-shot by default - the query vector is compared with cached text
embeddings of "a photo of a <category>" (several templates averaged) for every indexed category, and a softmax
gives a confidence. CATEGORY_DETECTION=qwen uses the VLM tag of the query image instead (falls back to SigLIP).
Below CATEGORY_CONFIDENCE_THRESHOLD the plan asks the user to pick from the top 3 instead of searching.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

TEMPLATES = ("a photo of a {}", "a jewellery product photo of a {}", "a close-up photo of a {}")
CATEGORY_NAMES = {
    "rings": "ring", "earrings": "pair of earrings", "necklaces": "necklace", "pendants": "pendant",
    "bracelets": "bracelet", "bangles": "bangle", "chains": "chain", "nose_pins": "nose pin", "anklets": "anklet",
}
OVERRIDE_FIELDS = ("metal", "gemstones", "stone_colors", "motifs", "style")


@dataclass
class QueryPlan:
    mode: str  # "image", "text" or "image_text"
    category: str | None = None
    category_source: str | None = None  # "user", "text" or "detected"
    category_confidence: float | None = None
    category_candidates: list[dict] = field(default_factory=list)  # top 3 [{"category", "confidence"}]
    needs_category: bool = False
    search_categories: list[str] = field(default_factory=list)
    query_tags: dict = field(default_factory=dict)  # from tagging the query image
    overrides: dict = field(default_factory=dict)  # attribute values the user wrote in text
    hard_filters: dict = field(default_factory=dict)  # e.g. {"metal": ["silver"]} with REFINE_METAL_HARD_FILTER
    allow_stone_colors: bool = False  # only when the colours come from the user's text
    semantic_query: str | None = None
    notes: list[str] = field(default_factory=list)

    def debug(self) -> dict:
        return asdict(self)


class CategoryDetector:
    """SigLIP zero-shot category detection with cached category text embeddings."""

    def __init__(self):
        self._vectors: dict[str, np.ndarray] = {}

    async def category_vectors(self, embed_client, categories: list[str]) -> dict[str, np.ndarray]:
        for category in categories:
            if category not in self._vectors:
                name = CATEGORY_NAMES.get(category, category.replace("_", " "))
                vectors = [await embed_client.embed_text(template.format(name)) for template in TEMPLATES]
                mean = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
                self._vectors[category] = mean / np.linalg.norm(mean)
        return {category: self._vectors[category] for category in categories}

    async def rank(self, embed_client, vector, categories: list[str], logit_scale: float) -> list[tuple[str, float]]:
        """[(category, confidence)] best first; confidences sum to 1 over `categories`."""
        if len(categories) == 1:
            return [(categories[0], 1.0)]
        vectors = await self.category_vectors(embed_client, categories)
        query = np.asarray(vector, dtype=np.float32)
        query = query / np.linalg.norm(query)
        logits = np.array([float(query @ vectors[c]) for c in categories]) * logit_scale
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return sorted(zip(categories, probabilities.round(4).tolist()), key=lambda item: item[1], reverse=True)


def text_overrides(parsed_tags) -> dict:
    """Attribute values the user wrote, from a parsed text query (QueryTags)."""
    if parsed_tags is None:
        return {}
    values = parsed_tags.model_dump()
    return {f: values[f] for f in OVERRIDE_FIELDS if values.get(f)}


async def resolve_category(plan: QueryPlan, *, user_category, text_category, vector, indexed: list[str],
                           settings, detector: CategoryDetector, embed_client, vlm_category=None) -> None:
    """Fill the category fields of the plan, or mark it as needing the user's choice."""
    if user_category:
        plan.category, plan.category_source, plan.category_confidence = user_category, "user", 1.0
    elif text_category:
        plan.category, plan.category_source, plan.category_confidence = text_category, "text", 1.0
    elif settings.category_detection == "qwen" and vlm_category in indexed:
        plan.category, plan.category_source, plan.category_confidence = vlm_category, "detected", 1.0
        plan.notes.append("category detected by the VLM")
    else:
        if settings.category_detection == "qwen":
            plan.notes.append("VLM gave no usable category; used SigLIP zero-shot")
        ranking = await detector.rank(embed_client, vector, indexed, settings.category_logit_scale)
        plan.category_candidates = [{"category": c, "confidence": p} for c, p in ranking[:3]]
        best, confidence = ranking[0]
        plan.category_source, plan.category_confidence = "detected", confidence
        if confidence < settings.category_confidence_threshold:
            plan.needs_category = True
            return
        plan.category = best
    plan.search_categories = [plan.category]


def _indexed_text_category(plan: QueryPlan, parsed_text, indexed: list[str]) -> str | None:
    """The category written in the text, if this catalogue has an index for it (a note explains otherwise)."""
    category = parsed_text.category if parsed_text else None
    if category and category not in indexed:
        plan.notes.append(f"the text asks for {category!r}, which has no catalogue images; ignored")
        return None
    return category


async def plan_image_query(*, settings, detector, embed_client, vector, user_category, indexed: list[str],
                           query_tags=None, parsed_text=None) -> QueryPlan:
    """Plan for an image query, optionally refined by parsed text (QueryTags, "similar but in silver")."""
    plan = QueryPlan(mode="image_text" if parsed_text is not None else "image")
    if query_tags is not None:
        plan.query_tags = {f: query_tags.get(f) for f in ("category", "gemstones", "motifs") if query_tags.get(f)}
    if parsed_text is not None:
        plan.overrides = text_overrides(parsed_text)
        plan.allow_stone_colors = bool(plan.overrides.get("stone_colors"))
        if settings.refine_metal_hard_filter and plan.overrides.get("metal"):
            plan.hard_filters["metal"] = plan.overrides["metal"]
    await resolve_category(
        plan, user_category=user_category, text_category=_indexed_text_category(plan, parsed_text, indexed),
        vector=vector, indexed=indexed, settings=settings, detector=detector, embed_client=embed_client,
        vlm_category=(query_tags or {}).get("category"),
    )
    return plan


async def plan_text_query(*, settings, detector, embed_client, vector, user_category, indexed: list[str],
                          parsed_text) -> QueryPlan:
    """Plan for a text-only query (parsed_text is QueryTags, or None if parsing failed).

    The category is a hard filter; without one, search all indexed categories or ask (TEXT_NO_CATEGORY).
    """
    plan = QueryPlan(mode="text")
    plan.overrides = text_overrides(parsed_text)
    plan.allow_stone_colors = bool(plan.overrides.get("stone_colors"))
    plan.semantic_query = parsed_text.semantic_query if parsed_text else None
    category = user_category or _indexed_text_category(plan, parsed_text, indexed)
    if category:
        plan.category, plan.category_confidence = category, 1.0
        plan.category_source = "user" if user_category else "text"
        plan.search_categories = [category]
    elif settings.text_no_category == "all":
        plan.search_categories = list(indexed)
        plan.notes.append("no category in the query; searched all categories")
    else:
        ranking = await detector.rank(embed_client, vector, indexed, settings.category_logit_scale)
        plan.category_candidates = [{"category": c, "confidence": p} for c, p in ranking[:3]]
        plan.needs_category = True
    return plan
