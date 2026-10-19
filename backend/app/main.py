from contextlib import asynccontextmanager

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import AsyncMongoClient

from app.config import get_settings
from app.embed_backends import make_embed_client
from app.embed_client import EmbeddingServiceError
from app.preprocessing import PreprocessOptions
from app.search import IndexCache, read_categories, router as search_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.embed_client = make_embed_client(settings)
    app.state.mongo = AsyncMongoClient(settings.mongo_uri, serverSelectionTimeoutMS=settings.mongo_timeout_ms)
    app.state.categories = read_categories(settings.categories_file)
    app.state.preprocess_options = PreprocessOptions.from_settings(settings)
    app.state.indexes = IndexCache(settings, app.state.preprocess_options)
    yield
    await app.state.embed_client.aclose()
    await app.state.mongo.close()


app = FastAPI(title="Jewellery Search Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=get_settings().cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
)
app.include_router(search_router)


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
