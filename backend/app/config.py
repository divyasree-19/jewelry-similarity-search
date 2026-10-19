from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.embed_backends import BACKENDS, EmbedBackend, get_backend

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables or the project .env file."""

    model_config = SettingsConfigDict(
        # .env.backend wins over .env when both exist
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.backend"), env_file_encoding="utf-8", extra="ignore"
    )

    # Which embedding model to use; each has its own index directory (see app/embed_backends.py).
    # siglip2_224 was chosen in Stage 6: same accuracy as siglip2_384 at less than half the latency (eval/DECISION.md)
    embed_backend: str = "siglip2_224"
    embed_service_url: str = "http://localhost:8000"  # siglip2_224 container
    siglip2_384_service_url: str = "http://localhost:8003"
    vertex_project: str | None = None
    vertex_location: str = "us-central1"
    embed_timeout_seconds: float = 30.0

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "jewellery_search"
    mongo_timeout_ms: int = 2000

    faiss_index_dir: Path = PROJECT_ROOT / "data" / "faiss_index"
    catalogue_dir: Path = PROJECT_ROOT / "data" / "catalogue"

    # Stage 3 preprocessing; changing any of these means re-ingesting the catalogue
    preprocess_size: int = 512
    preprocess_min_side: int = 128
    preprocess_remove_background: bool = False
    preprocess_rembg_model: str = "isnet-general-use"
    preprocess_normalize_lighting: bool = False

    # Stage 5 search API
    categories_file: Path = PROJECT_ROOT / "data" / "categories.txt"
    search_candidates: int = 100
    search_max_top_k: int = 50
    max_upload_mb: float = 10.0
    cors_origins: list[str] = ["http://localhost:5173"]
    save_query_images: bool = False
    query_image_dir: Path = PROJECT_ROOT / "data" / "query_uploads"

    vlm_service_url: str = "http://localhost:8002"
    vlm_model_name: str = "Qwen/Qwen3-VL-8B-Instruct"

    @field_validator("embed_backend")
    @classmethod
    def _known_backend(cls, value: str) -> str:
        if value not in BACKENDS:
            raise ValueError(f"expected one of {sorted(BACKENDS)}")
        return value

    @property
    def backend(self) -> EmbedBackend:
        return get_backend(self.embed_backend)

    @property
    def embed_model_name(self) -> str:
        return self.backend.model_name

    @property
    def embed_version(self) -> str:
        return self.backend.version


@lru_cache
def get_settings() -> Settings:
    return Settings()
