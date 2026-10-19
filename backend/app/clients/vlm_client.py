"""Client for the tagging VLM (Qwen3-VL-8B behind an OpenAI-compatible chat API, e.g. vLLM).

tag() sends one preprocessed image, retries once if the reply is not valid JSON, maps synonyms to the
vocabulary, drops values it does not know (reported in TagResult.dropped) and validates with Tags.
"""

import base64
import json
import re
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.tagging.prompts import JUDGE_PROMPT, RETRY_PROMPT, TAG_PROMPT
from app.tagging.schema import Tags
from app.tagging.vocab import LIST_FIELDS, PROVISIONAL_FIELDS, normalize


class VLMError(Exception):
    """The VLM is unreachable, or did not return usable tags after a retry."""


@dataclass
class TagResult:
    tags: Tags
    dropped: dict[str, list[str]] = field(default_factory=dict)  # field -> raw values not in the vocabulary
    attempts: int = 1
    raw: str = ""


class Judgement(BaseModel):
    """Qwen's same-design verdict for a (query, candidate) pair (Stage 10 Part B)."""

    same_design: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=300)


def parse_json_object(text: str) -> dict | None:
    """The JSON object in a reply, tolerating code fences or a sentence around it; None if there is none."""
    candidates = [text.strip()]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_values(raw: dict) -> tuple[dict, dict[str, list[str]]]:
    """Map the vocabulary fields of raw model output to allowed values.

    Returns (values for category, metal, gemstones, stone_colors, motifs, style; dropped raw values per field).
    """
    dropped: dict[str, list[str]] = {}
    values: dict = {}

    def keep(field_name: str, raw_value):
        canonical = normalize(field_name, raw_value)
        if canonical is None and raw_value not in (None, ""):
            dropped.setdefault(field_name, []).append(str(raw_value))
        return canonical

    for field_name in ("category", "style"):
        value = raw.get(field_name)
        if isinstance(value, list):  # "style": ["modern"] -> first allowed value
            value = next((v for v in value if normalize(field_name, v)), value[0] if value else None)
        values[field_name] = keep(field_name, value)

    for field_name in LIST_FIELDS:
        value = raw.get(field_name) or []
        if isinstance(value, str):
            value = [part for part in re.split(r"[;,]", value)]
        kept = []
        for item in value:
            canonical = keep(field_name, item)
            if canonical and canonical not in kept:
                kept.append(canonical)
        values[field_name] = kept
    return values, dropped


def normalize_tags(raw: dict) -> tuple[Tags, dict[str, list[str]]]:
    """Map raw VLM output to the vocabulary. Unknown values are dropped and returned separately."""
    values, dropped = normalize_values(raw)
    description = raw.get("description")
    values["description"] = description.strip() if isinstance(description, str) else ""
    try:
        return Tags(**values), dropped
    except ValidationError as exc:
        raise VLMError(f"Tags failed validation: {exc.errors()[0]['msg']}") from exc


class VLMClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 180.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, headers=headers, transport=transport)

    @classmethod
    def from_settings(cls, settings) -> "VLMClient":
        return cls(settings.vlm_service_url, settings.vlm_model_name, settings.vlm_api_key, settings.vlm_timeout_seconds)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def tag(self, png_bytes: bytes) -> TagResult:
        image_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": TAG_PROMPT},
        ]}]
        reply = await self._chat(messages)
        raw, attempts = parse_json_object(reply), 1
        if raw is None:  # one retry, showing the model its own reply
            messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": RETRY_PROMPT}]
            reply = await self._chat(messages)
            raw, attempts = parse_json_object(reply), 2
        if raw is None:
            raise VLMError(f"VLM did not return a JSON object after a retry: {reply[:200]!r}")
        tags, dropped = normalize_tags(raw)
        return TagResult(tags, dropped, attempts, reply)

    async def judge(self, query_png: bytes, candidate_png: bytes) -> Judgement:
        """Same design or not, ignoring colour, lighting, background and angle. Retries once on bad JSON."""
        def image(png):
            return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}

        messages = [{"role": "user", "content": [image(query_png), image(candidate_png),
                                                 {"type": "text", "text": JUDGE_PROMPT}]}]
        for attempt in (1, 2):
            reply = await self._chat(messages, json_mode=True, max_tokens=120)
            raw = parse_json_object(reply)
            if raw is not None:
                try:
                    return Judgement(**{k: raw.get(k) for k in ("same_design", "confidence", "reason")
                                        if raw.get(k) is not None})
                except ValidationError:
                    pass
            if attempt == 1:
                messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": RETRY_PROMPT}]
        raise VLMError(f"VLM judgement was not valid JSON after a retry: {reply[:200]!r}")

    async def _chat(self, messages: list[dict], json_mode: bool = False, max_tokens: int = 400) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            response = await self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise VLMError(f"VLM timed out at {self.base_url}") from exc
        except httpx.RequestError as exc:
            raise VLMError(f"Cannot reach VLM at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise VLMError(f"VLM returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            return response.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise VLMError(f"VLM returned an unexpected response: {exc}") from exc


__all__ = ["PROVISIONAL_FIELDS", "TagResult", "VLMClient", "VLMError", "normalize_tags", "normalize_values",
           "parse_json_object"]
