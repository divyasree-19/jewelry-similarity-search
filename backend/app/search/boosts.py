"""Soft boosts: small score bonuses for candidates whose tags match the query. They only reorder results.

Where query values come from:
- image queries: gemstones and motifs from tagging the query image (QueryPlan.query_tags). Never metal or
  stone colour: a photo's colour cast or lighting would otherwise push the right product down
  (colour_mismatch / lighting edge cases).
- text (text-only, or image + text refinement): every field the user wrote (QueryPlan.overrides). A field
  given in text replaces the image's values for that field. Stone colours only come from text.

Boost = weight per matching value (BOOST_<FIELD>), summed and capped at BOOST_CAP.
"""

IMAGE_TAG_FIELDS = ("gemstones", "motifs")
TEXT_FIELDS = ("metal", "gemstones", "stone_colors", "motifs", "style")


def weights(settings) -> dict[str, float]:
    return {field: getattr(settings, f"boost_{field}") for field in TEXT_FIELDS}


def boost_targets(plan) -> dict[str, set[str]]:
    """The values each candidate is compared against, per field."""
    targets = {field: set(plan.query_tags.get(field) or []) for field in IMAGE_TAG_FIELDS}
    for field, values in plan.overrides.items():
        if field in TEXT_FIELDS and values:
            targets[field] = {values} if isinstance(values, str) else set(values)
    if not plan.allow_stone_colors:
        targets.pop("stone_colors", None)
    return {field: values for field, values in targets.items() if values}


def apply_boosts(candidates, tags_by_vector_id: dict[int, dict], plan, settings) -> None:
    """Set boost and boosts (debug detail) on each candidate in place. Never removes a candidate."""
    targets = boost_targets(plan)
    field_weights = weights(settings)
    for candidate in candidates:
        tags = tags_by_vector_id.get(candidate.vector_id) or {}
        detail, total = {}, 0.0
        for field, wanted in targets.items():
            have = tags.get(field) or []
            have = {have} if isinstance(have, str) else set(have)
            matched = sorted(wanted & have)
            if matched:
                amount = field_weights[field] * len(matched)
                detail[field] = {"matched": matched, "boost": round(amount, 5)}
                total += amount
        candidate.boost = min(total, settings.boost_cap)
        candidate.boosts = detail
