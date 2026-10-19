# SigLIP 2 embedding API

FastAPI service that returns L2-normalized SigLIP 2 Base image embeddings
(`google/siglip2-base-patch16-224`, 768 dimensions).

## Run

In this project it is started by `docker compose up -d` from the project root. Standalone:

```bash
docker build -t siglip2-api .
docker run -d --name siglip2-api -p 8000:8000 \
  -v ~/siglip2-base/model:/models:ro siglip2-api
docker logs -f siglip2-api   # wait for "Application startup complete"
```

The model weights are not baked into the image; they must be mounted at `/models`.

## Endpoints

- `GET /health` → `{"status": "ok", "model": "SigLIP 2 Base"}`
- `POST /embed` (multipart `file`, JPEG/PNG/WebP) →
  `{"model": ..., "embedding_dimension": 768, "embedding": [...]}`
  Wrong content type or an unreadable image returns 400.

## Stage 0 verification (2026-10-18)

- `/health` OK; `/embed` returns 768-dim vectors for JPEG, PNG and WebP, L2 norm 1.0
- Same image twice gives identical vectors; different images differ
- Invalid files return 400, not 500
- CPU latency: ~350 ms per request (warm), ~600 ms first request

Fixes made during Stage 0:
- Container was bound to host port 8001 (already in use) and had no model mount
- `transformers` 5.x returns an output object from `get_image_features`; `app.py` now uses `pooler_output`
- `embedding_dimension` no longer calls `.item()` on an int
- `requirements.txt` pinned to the versions verified here
