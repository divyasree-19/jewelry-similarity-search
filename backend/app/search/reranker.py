"""Stage 10 multi-factor re-ranking: one 0-1 similarity score from several signals.

Signals per candidate (raw cosines; None when not available):
    image    query vector vs the candidate's image-index vector
    caption  query vector vs the candidate's caption vector
    gray     query gray vector vs the candidate's gray vector (Stage 9; only if gray features are on)
    region   tile similarity (Stage 9; only for the top FEATURE_REGION_TOP_N)
Each cosine maps to 0-1 by clipping to a fixed range from the weights file: (raw - lo) / (hi - lo), clipped.
No per-query min-max, so a score means the same thing on every query.

final = sum(w_s * unit_s over available signals) / sum(w_s over available signals) + capped boost, clipped to 0-1.
A missing signal is dropped and the remaining weights are re-normalized for that candidate.
Boost points are the Stage 8 boosts (RRF units) times boost_scale, capped at boost_cap.

Weights live in config/reranker_weights_<version>.json, one block per query mode ("image" also covers
image + text refinement; "text" is text-only search, where vectors are text embeddings).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

SIGNALS = ("image", "caption", "gray", "region")


@dataclass(frozen=True)
class ModeWeights:
    weights: dict[str, float]
    ranges: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class RerankerWeights:
    version: str
    modes: dict[str, ModeWeights]
    boost_scale: float
    boost_cap: float
    meta: dict = field(default_factory=dict)

    def for_mode(self, mode: str) -> ModeWeights:
        return self.modes["text" if mode == "text" else "image"]


def weights_path(directory: Path, version: str) -> Path:
    return Path(directory) / f"reranker_weights_{version}.json"


def parse_weights(data: dict) -> RerankerWeights:
    modes = {}
    for mode in ("image", "text"):
        block = data["modes"][mode]
        weights = {s: float(block["weights"].get(s, 0.0)) for s in SIGNALS}
        if any(w < 0 for w in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"{mode} weights must be non-negative and sum to 1, got {weights}")
        ranges = {s: (float(lo), float(hi)) for s, (lo, hi) in block["ranges"].items()}
        if any(hi <= lo for lo, hi in ranges.values()):
            raise ValueError(f"{mode} ranges need lo < hi, got {ranges}")
        missing = [s for s, w in weights.items() if w > 0 and s not in ranges]
        if missing:
            raise ValueError(f"{mode} has weights but no range for {missing}")
        modes[mode] = ModeWeights(weights, ranges)
    return RerankerWeights(data["version"], modes, float(data["boost_scale"]), float(data["boost_cap"]),
                           {k: v for k, v in data.items() if k not in ("version", "modes", "boost_scale", "boost_cap")})


def load_weights(directory: Path, version: str) -> RerankerWeights:
    path = weights_path(directory, version)
    weights = parse_weights(json.loads(path.read_text(encoding="utf-8")))
    if weights.version != version:
        raise ValueError(f"{path} says version {weights.version!r}")
    return weights


def to_unit(raw: float, lo: float, hi: float) -> float:
    return min(1.0, max(0.0, (raw - lo) / (hi - lo)))


def combine(raw: dict[str, float | None], boost_raw: float, mode: ModeWeights, boost_scale: float,
            boost_cap: float) -> tuple[float, dict]:
    """(final score 0-1, detail per signal) for one candidate.

    If none of the weighted signals is available (e.g. weights that ignore the image signal, with captions and
    region switched off), the image signal alone is used, so the score never collapses to boosts only.
    """
    available = {s: v for s, v in raw.items() if v is not None and mode.weights.get(s, 0) > 0}
    weights = {s: mode.weights[s] for s in available}
    fallback = not available and raw.get("image") is not None and "image" in mode.ranges
    if fallback:
        available, weights = {"image": raw["image"]}, {"image": 1.0}
    total = sum(weights.values())
    detail, weighted = {}, 0.0
    for signal in SIGNALS:
        value = raw.get(signal)
        unit = to_unit(value, *mode.ranges[signal]) if value is not None and signal in mode.ranges else None
        used = signal in available
        weight = weights[signal] / total if used else 0.0
        if used:
            weighted += weight * unit
        detail[signal] = {"raw": _r(value), "value": None if unit is None else round(unit, 4),
                          "weight": round(weight, 4), "base_weight": mode.weights.get(signal, 0.0), "used": used}
    boost_points = min(max(boost_raw, 0.0) * boost_scale, boost_cap)
    final = min(1.0, max(0.0, weighted + boost_points))
    detail["boost"] = {"raw": round(boost_raw, 6), "points": round(boost_points, 4), "cap": boost_cap}
    if fallback:
        detail["note"] = "no weighted signal available; image signal used alone"
    return final, detail


def labels(detail: dict, boosts: dict) -> dict[str, str]:
    """Plain-language summary for the debug block, e.g. {"design": "high", "stones": "match"}."""
    def level(value):
        return None if value is None else "high" if value >= 0.75 else "medium" if value >= 0.4 else "low"

    names = {"image": "design", "caption": "description", "gray": "shape (no colour)", "region": "details"}
    result = {names[s]: level(detail[s]["value"]) for s in SIGNALS if detail[s]["value"] is not None}
    for field_name, label in (("gemstones", "stones"), ("stone_colors", "stone colour"), ("metal", "metal"),
                              ("motifs", "motifs"), ("style", "style")):
        if field_name in boosts:
            result[label] = "match"
    return result


def _r(value):
    return None if value is None else round(float(value), 4)
