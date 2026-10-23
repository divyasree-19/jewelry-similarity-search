"""Embedding backends selectable with EMBED_BACKEND.

Each backend has its own model name and embedding version, and therefore its own FAISS index directory
(FAISS_INDEX_DIR/<model>_<version>/) and its own `embeddings.<backend>` record on MongoDB images.
Bump a backend's version whenever its vectors change, then re-ingest with --rebuild.
"""

import base64
import os
from dataclasses import dataclass

import httpx

from app.embed_client import EmbedClient, EmbeddingServiceError


@dataclass(frozen=True)
class EmbedBackend:
    name: str
    model_name: str
    version: str
    description: str
    dimension: int  # vector length; index manifests must record the same


BACKENDS = {
    "siglip2_224": EmbedBackend(
        "siglip2_224", "google/siglip2-base-patch16-224", "siglip2-base-v1", "SigLIP 2 Base, 224 px input", 768
    ),
    "siglip2_384": EmbedBackend(
        "siglip2_384", "google/siglip2-base-patch16-384", "siglip2-base-384-v1", "SigLIP 2 Base, 384 px input", 768
    ),
    "vertex": EmbedBackend(
        "vertex", "multimodalembedding@001", "vertex-mm-1408-v1", "Vertex AI multimodal embeddings, 1408-dim", 1408
    ),
}


def get_backend(name: str) -> EmbedBackend:
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown EMBED_BACKEND {name!r}; expected one of {sorted(BACKENDS)}") from None


def make_embed_client(settings):
    """The client for settings.embed_backend. Every client has embed(), health() and aclose()."""
    backend = settings.backend
    if backend.name == "vertex":
        return VertexEmbedClient(settings.vertex_project, settings.vertex_location, timeout=settings.embed_timeout_seconds)
    url = settings.siglip2_384_service_url if backend.name == "siglip2_384" else settings.embed_service_url
    return EmbedClient(url, timeout=settings.embed_timeout_seconds)


class VertexEmbedClient:
    """Vertex AI multimodalembedding@001 image embeddings (1408-dim).

    Auth: VERTEX_ACCESS_TOKEN (e.g. from `gcloud auth print-access-token`), or Application Default
    Credentials when the google-auth package is installed.
    """

    def __init__(self, project: str | None, location: str, timeout: float = 30.0, transport=None):
        self.project = project
        self.base_url = f"https://{location}-aiplatform.googleapis.com"
        self.path = (f"/v1/projects/{project}/locations/{location}/publishers/google/models/"
                     "multimodalembedding@001:predict")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        self._token()
        return {"model": "Vertex AI multimodalembedding@001"}

    async def embed(self, image_bytes: bytes, filename: str, content_type: str) -> list[float]:
        body = {"instances": [{"image": {"bytesBase64Encoded": base64.b64encode(image_bytes).decode()}}],
                "parameters": {"dimension": 1408}}
        try:
            response = await self._client.post(self.path, json=body, headers={"Authorization": f"Bearer {self._token()}"})
        except httpx.HTTPError as exc:
            raise EmbeddingServiceError(f"Cannot reach Vertex AI at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise EmbeddingServiceError(f"Vertex AI returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            return response.json()["predictions"][0]["imageEmbedding"]
        except (ValueError, KeyError, IndexError) as exc:
            raise EmbeddingServiceError(f"Vertex AI returned an unexpected response: {exc}") from exc

    def _token(self) -> str:
        if not self.project:
            raise EmbeddingServiceError("EMBED_BACKEND=vertex needs VERTEX_PROJECT")
        token = os.environ.get("VERTEX_ACCESS_TOKEN")
        if token:
            return token
        try:
            import google.auth
            import google.auth.transport.requests
        except ImportError:
            raise EmbeddingServiceError("No Vertex credentials: set VERTEX_ACCESS_TOKEN or install google-auth") from None
        try:
            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            credentials.refresh(google.auth.transport.requests.Request())
        except Exception as exc:
            raise EmbeddingServiceError(f"No Vertex credentials: {exc}") from exc
        return credentials.token
