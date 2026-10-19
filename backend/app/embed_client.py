"""HTTP client for the SigLIP embedding service (ai-service)."""

import httpx


class EmbeddingServiceError(Exception):
    """Raised when the embedding service is unreachable or returns an unusable response."""


class EmbedClient:
    def __init__(self, base_url: str, timeout: float = 30.0, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        response = await self._request("GET", "/health")
        return response.json()

    async def embed(self, image_bytes: bytes, filename: str, content_type: str) -> list[float]:
        """Return the normalized embedding for one image."""
        files = {"file": (filename, image_bytes, content_type)}
        response = await self._request("POST", "/embed", files=files)
        try:
            body = response.json()
            embedding = body["embedding"]
            dimension = body["embedding_dimension"]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingServiceError(f"Embedding service returned an unexpected response: {exc}") from exc
        if len(embedding) != dimension:
            raise EmbeddingServiceError(
                f"Embedding length {len(embedding)} does not match embedding_dimension {dimension}"
            )
        return embedding

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        url = f"{self.base_url}{path}"
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise EmbeddingServiceError(f"Embedding service timed out calling {url}") from exc
        except httpx.RequestError as exc:
            raise EmbeddingServiceError(f"Cannot reach embedding service at {url}: {exc}") from exc

        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise EmbeddingServiceError(f"Embedding service returned HTTP {response.status_code} for {url}: {detail}")
        return response
