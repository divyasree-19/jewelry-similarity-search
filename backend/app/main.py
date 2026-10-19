from contextlib import asynccontextmanager

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import AsyncMongoClient

from app.caption_store import caption_root, new_caption_store
from app.clients.llm_client import LLMClient
from app.clients.vlm_client import VLMClient
from app.config import get_settings
from app.embed_backends import make_embed_client
from app.embed_client import EmbeddingServiceError
from app.preprocessing import PreprocessOptions
from app.routes import catalogue, search_image, search_text
from app.search.common import IndexCache, read_categories
from app.preprocessing import variant_signature
from app.search.features import TileStore, VariantIndexCache, tile_dir
from app.search.image_search import RerankerCache
from app.search.vlm_judge import JudgeCache
from app.search.query_analysis import CategoryDetector
from app.tagging import TAGGER_VERSION
from app.vector_index import CategoryIndexes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.embed_client = make_embed_client(settings)
    app.state.mongo = AsyncMongoClient(settings.mongo_uri, serverSelectionTimeoutMS=settings.mongo_timeout_ms)
    app.state.categories = read_categories(settings.categories_file)
    app.state.preprocess_options = PreprocessOptions.from_settings(settings)
    app.state.indexes = IndexCache(
        settings.faiss_index_dir, lambda: CategoryIndexes(
            settings.embed_model_name, settings.embed_version, app.state.preprocess_options.signature), "image",
    )
    app.state.caption_indexes = IndexCache(
        caption_root(settings), lambda: new_caption_store(settings, TAGGER_VERSION), "caption")
    app.state.category_detector = CategoryDetector()
    app.state.reranker = RerankerCache(settings)
    app.state.judge_cache = JudgeCache(settings.vlm_judge_cache_size)
    signature = app.state.preprocess_options.signature
    app.state.feature_indexes = {name: VariantIndexCache(settings, name, variant_signature(signature, name))
                                 for name in ("gray", "edge")}
    grid = settings.feature_tile_grid
    app.state.tile_store = TileStore(tile_dir(settings), grid, variant_signature(signature, "tiles", grid))
    if settings.feature_region_rerank and not app.state.tile_store.load():  # memory-map once at startup
        logging.getLogger("search").warning(
            "FEATURE_REGION_RERANK is on but no %dx%d tile store exists; run scripts/build_feature_indexes.py "
            "--feature tiles --grid %d. Searching without region rerank.", grid, grid, grid)
    app.state.vlm_client = VLMClient.from_settings(settings)
    app.state.llm_client = LLMClient.from_settings(settings)
    yield
    await app.state.embed_client.aclose()
    await app.state.vlm_client.aclose()
    await app.state.llm_client.aclose()
    await app.state.mongo.close()


app = FastAPI(title="Jewellery Search Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=get_settings().cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
)
for module in (search_image, search_text, catalogue):
    app.include_router(module.router)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    """Missing or malformed form fields are a 400 with a readable message, e.g. "category: Field required"."""
    messages = [f"{'.'.join(str(p) for p in err['loc'] if p not in ('body', 'query'))}: {err['msg']}" for err in exc.errors()]
    return JSONResponse({"detail": "; ".join(messages)}, status_code=400)


async def check_mongo(app: FastAPI) -> dict:
    try:
        await app.state.mongo.admin.command("ping")
        return {"status": "ok", "db": app.state.settings.mongo_db}
    except Exception as exc:
        return {"status": "error", "error": str(exc).splitlines()[0]}


async def check_embedding_service(app: FastAPI) -> dict:
    try:
        body = await app.state.embed_client.health()
        return {"status": "ok", "backend": app.state.settings.embed_backend, "model": body.get("model")}
    except EmbeddingServiceError as exc:
        return {"status": "error", "backend": app.state.settings.embed_backend, "error": str(exc)}


@app.get("/health")
async def health():
    settings = app.state.settings
    dependencies = {
        "mongodb": await check_mongo(app),
        "embedding_service": await check_embedding_service(app),
    }
    healthy = all(dep["status"] == "ok" for dep in dependencies.values())
    body = {
        "status": "ok" if healthy else "degraded",
        "embed_backend": settings.embed_backend,
        "embed_model": settings.embed_model_name,
        "embed_version": settings.embed_version,
        "dependencies": dependencies,
    }
    return JSONResponse(body, status_code=200 if healthy else 503)
