from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.embed_backends import BACKENDS, EmbedBackend, get_backend

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables or the project .env file."""

    model_config = SettingsConfigDict(
        # .env.backend wins over .env when both exist
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.backend"), env_file_encoding="utf-8", extra="ignore",
        env_ignore_empty=True,  # "KEY=" in .env means "use the default"
    )

    # Which embedding model to use; each has its own index directory (see app/embed_backends.py).
    # siglip2_224 was chosen in Stage 6: same accuracy as siglip2_384 at less than half the latency (eval/DECISION.md)
    embed_backend: str = "siglip2_224"
    embed_service_url: str = "http://localhost:8000"  # siglip2_224 container
    siglip2_384_service_url: str = "http://localhost:8003"
    vertex_project: str | None = None
    vertex_location: str = "us-central1"
    embed_timeout_seconds: float = 30.0
    embed_dimension: int | None = None  # override the backend's vector length (tests with fake embeddings only)

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

    # Stage 8 search flags; each step can be switched off to compare with the Stage 6 baseline
    search_hybrid: bool = True  # also search the caption index and fuse with RRF
    search_boosts: bool = True  # soft re-ranking on matching tags; never removes results
    search_query_tagging: bool = False  # tag the query image with Qwen for boosts (adds a VLM call per search)
    search_query_tagging_timeout_seconds: float = 20.0
    search_rrf_k: int = 60
    # Category detection when the user gives none: "siglip" (zero-shot, default) or "qwen" (VLM tag of the query)
    category_detection: str = "siglip"
    category_confidence_threshold: float = 0.5  # below this, return the top 3 categories and ask the user
    category_logit_scale: float = 100.0  # softmax temperature over SigLIP cosine similarities
    # Boost per matching value, added to the RRF score (~0.00027 separates rank 1 and 2 of one list); capped in total
    boost_gemstones: float = 0.002
    boost_motifs: float = 0.001
    boost_metal: float = 0.006  # only from text ("in silver"); image queries never boost on metal
    boost_stone_colors: float = 0.003  # only from text; image queries never use stone colour
    boost_style: float = 0.001  # only from text
    boost_cap: float = 0.01
    refine_metal_hard_filter: bool = False  # image + text: drop results without the requested metal
    text_no_category: str = "all"  # text query with no category: "all" categories, or "ask" the user

    # Stage 9 colour-reduced and region features (each needs scripts/build_feature_indexes.py first).
    # Decided in eval/results/stage9_decision.md: gray and edge did not help (off); 2x2 region rerank did (on).
    feature_gray_retrieval: bool = False  # gray index as a third RRF list
    feature_gray_rerank: bool = False  # add weight * gray cosine to the top candidates
    feature_gray_rerank_weight: float = 0.02
    feature_edge_retrieval: bool = False  # optional experiment: edge-map index as an RRF list
    feature_edge_rerank: bool = False
    feature_edge_rerank_weight: float = 0.02
    feature_rerank_top_n: int = 100  # gray/edge rerank applies to this many candidates
    feature_region_rerank: bool = True  # tile (region) similarity added to the top candidates; ~+1.2 s on CPU
    feature_region_weight: float = 0.02
    feature_region_top_n: int = 50
    feature_tile_grid: int = 2  # 2 -> centre + 2x2 tiles (5), 3 -> centre + 3x3 (10)
    features_dir: Path = PROJECT_ROOT / "data" / "features"

    # Stage 10 re-ranking: one 0-1 score from weighted signals (app/search/reranker.py)
    # Holdout (eval/results/stage10_final.md): v2 helped text search and hurt image search, so text only
    search_reranker: bool = True
    reranker_modes: str = "text"  # comma-separated: "image" (also image + text), "text"
    reranker_weights_version: str = "v2"
    reranker_weights_dir: Path = PROJECT_ROOT / "config"
    reranker_top_n: int = 100  # candidates scored by the reranker; the rest keep the retrieval order below them
    dedup_near_duplicate_threshold: float = 0.98  # image cosine above which two products are marked look-alikes

    # Stage 10 Part B: Qwen same-design check on the top results (vlm_judge.py)
    feature_vlm_judge: bool = False
    vlm_judge_top_n: int = 5
    vlm_judge_weight: float = 0.1  # final += weight * confidence (same design) or - weight * confidence (not)
    vlm_judge_timeout_seconds: float = 20.0
    vlm_judge_concurrency: int = 5
    vlm_judge_cache_size: int = 5000

    # Stage 8 query parsing LLM; empty means reuse the tagging VLM endpoint, model and key (text-only request)
    llm_service_url: str | None = None
    llm_model_name: str | None = None
    llm_api_key: str | None = None
    # A search waits at most this long for query parsing; slower or unreachable means Qwen is unavailable
    llm_query_timeout_seconds: float = 20.0

    # Stage 7 tagging VLM: any OpenAI-compatible chat API (base URL ending in /v1)
    vlm_service_url: str = "http://localhost:8002/v1"
    vlm_model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    vlm_api_key: str | None = Field(default=None, validation_alias=AliasChoices("VLM_API_KEY", "QWEN_API_KEY"))
    vlm_quantization: str = "none"  # recorded with every tag; the hosted vLLM server runs the full model
    vlm_timeout_seconds: float = 180.0
    vlm_concurrency: int = 4

    @field_validator("embed_backend")
    @classmethod
    def _known_backend(cls, value: str) -> str:
        if value not in BACKENDS:
            raise ValueError(f"expected one of {sorted(BACKENDS)}")
        return value

    @field_validator("category_detection")
    @classmethod
    def _known_detection(cls, value: str) -> str:
        if value not in ("siglip", "qwen"):
            raise ValueError("expected siglip or qwen")
        return value

    @field_validator("text_no_category")
    @classmethod
    def _known_text_fallback(cls, value: str) -> str:
        if value not in ("all", "ask"):
            raise ValueError("expected all or ask")
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

    @property
    def expected_dimension(self) -> int:
        return self.embed_dimension or self.backend.dimension


@lru_cache
def get_settings() -> Settings:
    return Settings()
