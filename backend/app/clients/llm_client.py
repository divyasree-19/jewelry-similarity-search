"""Text-query parsing with an LLM (Stage 8). Reuses the hosted Qwen endpoint unless LLM_* settings say otherwise.

parse_query("silver peacock earrings with green stones") ->
    ParsedQuery(tags=QueryTags(category="earrings", metal=["silver"], stone_colors=["green"], motifs=["peacock"],
                               semantic_query="silver peacock earrings with green stones"), dropped={...})

The reply is mapped to the vocabulary (synonyms, unknown values dropped) and validated with QueryTags.
Stone colours and the category are kept only when the shopper wrote them: "ruby ring" never gets a "red"
boost, and "evil eye with an elephant" never gets a guessed category, which would be a hard filter.
"""

import re
from dataclasses import dataclass, field

import httpx
from pydantic import ValidationError

from app.clients.vlm_client import parse_json_object, normalize_values
from app.tagging.prompts import QUERY_PARSE_PROMPT, RETRY_PROMPT
from app.tagging.schema import QueryTags
from app.tagging.vocab import GEMSTONES, SYNONYMS


class LLMError(Exception):
    """The LLM is unreachable or did not return a usable JSON object after a retry."""


@dataclass
class ParsedQuery:
    tags: QueryTags
    dropped: dict[str, list[str]] = field(default_factory=dict)
    attempts: int = 1


# Words that name a metal or a gemstone, not a stone colour, even though they are colour synonyms
_NOT_COLOUR_WORDS = {"gold", "golden", "silver"} | set(GEMSTONES) | set(SYNONYMS["gemstones"])
_CATEGORY_WORDS = {"earrings": {"earring", "studs", "stud earrings", "jhumka", "jhumkas"},
                   "nose_pins": {"nose pin", "nose pins", "nose ring", "nose stud"}}


def _mentions(text: str, words) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in words)


def colours_written(text: str, colours: list[str]) -> list[str]:
    """The colours from `colours` that the text names, directly or by a synonym ("gray" -> "grey")."""
    kept = []
    for colour in colours:
        words = {colour} | {raw for raw, target in SYNONYMS["stone_colors"].items() if target == colour}
        if _mentions(text, words - _NOT_COLOUR_WORDS):
            kept.append(colour)
    return kept


def category_written(text: str, category: str | None) -> bool:
    """Whether the text names the category: "rings", "ring", or a synonym such as "studs" for earrings."""
    if not category:
        return False
    words = {category, category.replace("_", " "), category.rstrip("s").replace("_", " ")}
    words |= {raw for raw, target in SYNONYMS["category"].items() if target == category}
    words |= _CATEGORY_WORDS.get(category, set())
    return _mentions(text, words)


def validate_query(raw: dict, text: str) -> ParsedQuery:
    values, dropped = normalize_values(raw)
    unwritten = [c for c in values["stone_colors"] if c not in colours_written(text, values["stone_colors"])]
    if unwritten:
        dropped.setdefault("stone_colors", []).extend(unwritten)
        values["stone_colors"] = [c for c in values["stone_colors"] if c not in unwritten]
    if values["category"] and not category_written(text, values["category"]):
        dropped.setdefault("category", []).append(values["category"])
        values["category"] = None
    semantic = raw.get("semantic_query")
    values["semantic_query"] = semantic if isinstance(semantic, str) and semantic.strip() else text
    try:
        return ParsedQuery(QueryTags(**values), dropped)
    except ValidationError as exc:
        raise LLMError(f"Parsed query failed validation: {exc.errors()[0]['msg']}") from exc


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, headers=headers, transport=transport)
        self._json_mode = True  # switched off if the server rejects response_format

    @classmethod
    def from_settings(cls, settings) -> "LLMClient":
        return cls(
            settings.llm_service_url or settings.vlm_service_url,
            settings.llm_model_name or settings.vlm_model_name,
            settings.llm_api_key or settings.vlm_api_key,
            settings.vlm_timeout_seconds,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def parse_query(self, text: str) -> ParsedQuery:
        text = " ".join(text.split())
        if not text:
            raise LLMError("Query text is empty")
        messages = [{"role": "system", "content": QUERY_PARSE_PROMPT}, {"role": "user", "content": text}]
        reply = await self._chat(messages)
        raw, attempts = parse_json_object(reply), 1
        if raw is None:
            messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": RETRY_PROMPT}]
            reply = await self._chat(messages)
            raw, attempts = parse_json_object(reply), 2
        if raw is None:
            raise LLMError(f"LLM did not return a JSON object after a retry: {reply[:200]!r}")
        parsed = validate_query(raw, text)
        parsed.attempts = attempts
        return parsed

    async def _chat(self, messages: list[dict]) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 300}
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            response = await self._client.post("/chat/completions", json=body)
            if response.status_code == 400 and self._json_mode and "response_format" in response.text:
                self._json_mode = False  # server without JSON mode: plain request, the parser copes with prose
                body.pop("response_format")
                response = await self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise LLMError(f"LLM timed out at {self.base_url}") from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Cannot reach LLM at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise LLMError(f"LLM returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            return response.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM returned an unexpected response: {exc}") from exc
