import base64

import httpx
import pytest

from app.config import Settings
from app.embed_backends import VertexEmbedClient, make_embed_client
from app.embed_client import EmbedClient, EmbeddingServiceError


def test_each_backend_has_its_own_model_version_and_client():
    s224, s384 = Settings(embed_backend="siglip2_224"), Settings(embed_backend="siglip2_384")
    assert (s224.embed_model_name, s224.embed_version) == ("google/siglip2-base-patch16-224", "siglip2-base-v1")
    assert (s384.embed_model_name, s384.embed_version) == ("google/siglip2-base-patch16-384", "siglip2-base-384-v1")
    assert make_embed_client(s384).base_url == s384.siglip2_384_service_url
    assert isinstance(make_embed_client(s224), EmbedClient)
    assert isinstance(make_embed_client(Settings(embed_backend="vertex")), VertexEmbedClient)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="expected one of"):
        Settings(embed_backend="clip")


async def test_vertex_sends_base64_image_and_returns_embedding(monkeypatch):
    monkeypatch.setenv("VERTEX_ACCESS_TOKEN", "tok")

    def handler(request):
        assert request.url.path.endswith("/projects/p1/locations/europe-west4/publishers/google/models/multimodalembedding@001:predict")
        assert request.headers["Authorization"] == "Bearer tok"
        body = request.read().decode()
        assert base64.b64encode(b"img").decode() in body
        return httpx.Response(200, json={"predictions": [{"imageEmbedding": [0.1, 0.2]}]})

    client = VertexEmbedClient("p1", "europe-west4", transport=httpx.MockTransport(handler))
    assert await client.embed(b"img", "q.png", "image/png") == [0.1, 0.2]
    await client.aclose()


async def test_vertex_errors_are_clear(monkeypatch):
    monkeypatch.delenv("VERTEX_ACCESS_TOKEN", raising=False)
    no_project = VertexEmbedClient(None, "us-central1")
    with pytest.raises(EmbeddingServiceError, match="needs VERTEX_PROJECT"):
        await no_project.embed(b"img", "q.png", "image/png")
    await no_project.aclose()

    monkeypatch.setenv("VERTEX_ACCESS_TOKEN", "tok")
    denied = VertexEmbedClient("p1", "us-central1", transport=httpx.MockTransport(lambda r: httpx.Response(403, text="denied")))
    with pytest.raises(EmbeddingServiceError, match="HTTP 403"):
        await denied.embed(b"img", "q.png", "image/png")
    await denied.aclose()
