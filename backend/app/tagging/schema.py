"""Pydantic models for VLM tags and parsed text queries. Only vocabulary values are accepted; normalization
(synonyms, dropping unknown values) happens before these models."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.tagging.vocab import ALLOWED, LIST_FIELDS


class _VocabFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str | None = None
    metal: list[str] = Field(default_factory=list)
    gemstones: list[str] = Field(default_factory=list)
    stone_colors: list[str] = Field(default_factory=list)
    motifs: list[str] = Field(default_factory=list)
    style: str | None = None

    @field_validator("category", "style")
    @classmethod
    def _single_in_vocab(cls, value, info):
        if value is not None and value not in ALLOWED[info.field_name]:
            raise ValueError(f"{value!r} is not an allowed {info.field_name}")
        return value

    @field_validator(*LIST_FIELDS)
    @classmethod
    def _list_in_vocab(cls, values, info):
        unknown = [value for value in values if value not in ALLOWED[info.field_name]]
        if unknown:
            raise ValueError(f"{unknown} not allowed for {info.field_name}")
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate values in {info.field_name}")
        return values


class Tags(_VocabFields):
    """Tags of one catalogue image (Stage 7)."""

    description: str = Field(min_length=1, max_length=500)

    @field_validator("description")
    @classmethod
    def _clean_description(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("description is empty")
        return value


class QueryTags(_VocabFields):
    """A parsed text query (Stage 8): attribute values plus the text to embed for semantic search."""

    semantic_query: str | None = Field(default=None, max_length=300)

    @field_validator("semantic_query")
    @classmethod
    def _clean_query(cls, value: str | None) -> str | None:
        value = " ".join((value or "").split())
        return value or None
