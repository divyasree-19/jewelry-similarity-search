import httpx
import pytest

from app.embed_client import EmbedClient, EmbeddingServiceError


def make_client(handler) -> EmbedClient:
    return EmbedClient("http://embed.test", timeout=1.0, transport=httpx.MockTransport(handler))


async def test_embed_returns_vector():
    def handler(request):
        assert request.url.path == "/embed"
        return httpx.Response(200, json={"model": "SigLIP 2 Base", "embedding_dimension": 3, "embedding": [0.6, 0.8, 0.0]})

    client = make_client(handler)
    assert await client.embed(b"img", "ring.jpg", "image/jpeg") == [0.6, 0.8, 0.0]
    await client.aclose()


async def test_embed_http_error_includes_detail():
    client = make_client(lambda request: httpx.Response(400, json={"detail": "Invalid image file"}))
    with pytest.raises(EmbeddingServiceError, match="HTTP 400.*Invalid image file"):
        await client.embed(b"bad", "bad.jpg", "image/jpeg")
    await client.aclose()


async def test_embed_timeout_is_clear():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    client = make_client(handler)
    with pytest.raises(EmbeddingServiceError, match="timed out"):
        await client.embed(b"img", "ring.jpg", "image/jpeg")
    await client.aclose()


async def test_embed_unreachable_is_clear():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = make_client(handler)
    with pytest.raises(EmbeddingServiceError, match="Cannot reach"):
        await client.health()
    await client.aclose()


async def test_embed_dimension_mismatch():
    client = make_client(lambda request: httpx.Response(200, json={"embedding_dimension": 768, "embedding": [1.0]}))
    with pytest.raises(EmbeddingServiceError, match="does not match"):
        await client.embed(b"img", "ring.jpg", "image/jpeg")
    await client.aclose()
