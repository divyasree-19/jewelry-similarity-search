# Integration notes for the client system

For the client's engineers integrating the visual search into their existing system. It describes the API as
built in this POC (Stages 0–12), what their catalogue must provide, and what is still needed before
production. The machine-readable contract is [openapi.json](openapi.json) (regenerate with
`python scripts/export_openapi.py`). Accuracy and latency figures are in
[final_eval_report.md](final_eval_report.md).

> **Needs client input.** Two items below rely on documents not in this repository: the client's ID formats
> (section 3) and their existing 7-step pipeline (section 4). Those sections give our side and the questions
> to settle. They do not guess the client's side.

## 1. Endpoints

Base URL: `http://<host>:8001` (`BACKEND_PORT`). Requests and responses are JSON, except image upload
(multipart). There is no authentication yet (section 6).

### POST /search/image: find catalogue products that look like a photo

Multipart form fields:

| Field | Required | Meaning |
|---|---|---|
| `file` | yes | JPEG, PNG or WebP, ≤ `MAX_UPLOAD_MB` (10 MB), shorter side ≥ 128 px. Type is checked from the declared type and the file bytes. |
| `category` | no | One of `data/categories.txt` (rings, earrings, necklaces, pendants, bangles, bracelets, chains, nose_pins, anklets). Omit to detect it. |
| `text` | no | Refinement in plain language, e.g. `similar but in silver` (≤ 500 characters). Needs Qwen. |
| `project_id` | no | Only return products of this project (tenant / collection). |
| `subcategory` | no | Only return products with this subcategory. |
| `top_k` | no | Number of products, 1–50, default 10. |
| `debug` | no | `true` adds scoring details per result (for testing, not for end users). |

### POST /search/text: find products from a description

JSON body: `{"text": "red ruby ring", "category": null, "project_id": null, "top_k": 10, "debug": false}`.
Only `text` is required. A category is used when given, or when the text names one ("ring"); otherwise every
category is searched (`TEXT_NO_CATEGORY=all`). Needs Qwen (query parsing): without it the answer is 503.

### Response (both searches)

Example: `curl -F file=@data/queries/q001.jpg -F top_k=2 localhost:8001/search/image` (no category given):

```json
{
  "query_id": "2636f2ddb96f46799a7f13ef004d66ca",
  "mode": "image",
  "embed_backend": "siglip2_224",
  "model": "google/siglip2-base-patch16-224",
  "embedding_version": "siglip2-base-v1",
  "category": "rings",
  "category_source": "detected",
  "category_confidence": 0.9532,
  "needs_category": false,
  "category_candidates": [
    {
      "category": "rings",
      "confidence": 0.9532
    },
    {
      "category": "bracelets",
      "confidence": 0.0275
    },
    {
      "category": "pendants",
      "confidence": 0.0122
    }
  ],
  "results": [
    {
      "product_id": "RG-001",
      "image_id": "RG-001_01",
      "image_url": "/catalogue-images/rings/RG-001_01.jpg",
      "category": "rings",
      "similarity_score": 0.9444,
      "score": 0.051009,
      "attributes": {
        "project_id": "demo",
        "subcategory": "band",
        "name": "Men's two-tone band with four diamond bars",
        "metal": [
          "yellow gold",
          "black rhodium"
        ],
        "gemstones": [
          "diamond"
        ],
        "stone_colours": [
          "white"
        ],
        "motifs": [
          "bars",
          "stripes"
        ],
        "style": [
          "men's",
          "modern"
        ]
      }
    },
    {
      "product_id": "RG-010",
      "image_id": "RG-010_01",
      "image_url": "/catalogue-images/rings/RG-010_01.jpg",
      "category": "rings",
      "similarity_score": 0.8638,
      "score": 0.048565,
      "attributes": {
        "project_id": "demo",
        "subcategory": "band",
        "name": "Black rhodium pave dome band",
        "metal": [
          "black rhodium"
        ],
        "gemstones": [
          "diamond"
        ],
        "stone_colours": [
          "white"
        ],
        "motifs": [
          "pave",
          "dome"
        ],
        "style": [
          "modern"
        ]
      }
    }
  ],
  "similarity_score": "image cosine similarity (caption cosine for text search)",
  "reduced_mode": false,
  "timings_ms": {
    "preprocess": 49.5,
    "embed": 1052.6,
    "model_calls": 1052.8,
    "analyse": 1360.8,
    "search": 1.0,
    "metadata": 5.1,
    "rank": 4.1,
    "total": 2473.3
  }
}
```

| Field | Meaning |
|---|---|
| `query_id` | ID of this search, also in the backend log line for it. Quote it when reporting a problem. |
| `embed_backend`, `model`, `embedding_version` | Embedding model that produced the vectors (changes only with a rebuild). |
| `mode` | `image`, `image_text` or `text` |
| `category`, `category_source`, `category_confidence` | Category searched and where it came from: `user`, `text` or `detected` (with confidence 0–1). |
| `needs_category`, `category_candidates`, `message` | See "Low confidence" below. |
| `results[]` | Best first, **one per product** (its best-matching image). |
| `results[].product_id`, `image_id` | Catalogue IDs (section 3). |
| `results[].image_url` | Path of the matching catalogue image on this server (`/catalogue-images/<category>/<file>`); prefix the base URL. File-system paths are never returned. |
| `results[].similarity_score` | For display and ordering within one response only (see below). |
| `results[].score` | Internal ranking score that decided the order. Not comparable across queries; for debugging. |
| `results[].attributes` | Product fields from the catalogue: `name`, `project_id`, `subcategory`, `metal`, `gemstones`, `stone_colours`, `motifs`, `style`. |
| `similarity_score` (top level) | Text saying what the scores in this response mean. |
| `reranker_weights_version` | Weights file used for the scores (text search). |
| `parsed_text` | What Qwen understood from `text` (`image_text` and `text` modes). |
| `reduced_mode` | Image search only. `true` if Qwen was unavailable or slow and the search ran without the refinement text, query tags or VLM judge. Show the result, but note that the refinement was ignored. |
| `warnings[]` | Readable notes, e.g. `Reduced mode: Qwen is unavailable or slow; searched without text refinement`. |
| `timings_ms` | Milliseconds per pipeline step and `total`. The first search without a category after a backend restart spends ~1.4 s in `analyse`, computing the category-name embeddings once. |

**About `similarity_score`.**
- For image search it is the cosine similarity of the two images (about 0.5–1.0 in practice). For text search
  it is a 0–1 score from weighted signals.
- Neither is a probability or a "% match", and the two are not comparable with each other.
- Do not apply a fixed cut-off without calibrating it on the client's catalogue. Use the rank.

**Low confidence.** When no category is given and the detected one has confidence below
`CATEGORY_CONFIDENCE_THRESHOLD` (0.5), the response has `needs_category: true`, empty `results`, and the top 3
`category_candidates` (`[{"category": "pendants", "confidence": 0.468}, ...]`). Ask the user to choose, then
send the same request with `category` set. This happened for 4 of 50 test photos, and the right category was
always among the 3.

```json
{
  "query_id": "fd97cc519f224219bc5462aa55aa9786",
  "mode": "image",
  "embed_backend": "siglip2_224",
  "model": "google/siglip2-base-patch16-224",
  "embedding_version": "siglip2-base-v1",
  "category": null,
  "category_source": "detected",
  "category_confidence": 0.4679,
  "needs_category": true,
  "category_candidates": [
    {
      "category": "pendants",
      "confidence": 0.4679
    },
    {
      "category": "rings",
      "confidence": 0.2253
    },
    {
      "category": "necklaces",
      "confidence": 0.1703
    }
  ],
  "results": [],
  "message": "Not sure which category this is; choose one of category_candidates and search again",
  "reduced_mode": false,
  "timings_ms": {
    "preprocess": 50.7,
    "embed": 1183.8,
    "model_calls": 1184.7,
    "analyse": 0.2,
    "total": 1235.6
  }
}
```

### GET /health

200 with `"status": "ok"`, or 503 with the same body and `"status": "degraded"`. The body lists `dependencies`
(MongoDB, embedding service) and `indexes.served` / `indexes.refused` per category. Use it for load-balancer
checks. Qwen is not included because image search works without it.

### GET /catalogue-images/{category}/{file}

Serves catalogue images by the `image_url` in results. In production the client will probably serve images from
their own CDN instead (section 3).

### Errors

Every error is JSON `{"detail": "<readable message>"}`:

| Status | When | Example `detail` |
|---|---|---|
| 400 | Missing/malformed field, unknown category, `top_k` out of range, empty, corrupt or too-small image | `Image is 20x20; the shorter side must be at least 128 px` |
| 404 | Nothing indexed for the requested category | `No catalogue images are indexed for category 'bangles'` |
| 413 | Image larger than `MAX_UPLOAD_MB` | `Image is larger than 10 MB` |
| 415 | Not JPEG/PNG/WebP | `Unsupported file type 'image/gif'; upload a JPEG, PNG or WebP image` |
| 503 | Embedding service (SigLIP) down or overloaded | `Embedding service unavailable; try again shortly` |
| 503 | MongoDB down | `Catalogue database is unavailable; try again shortly` |
| 503 | Text search with Qwen down or slower than `LLM_QUERY_TIMEOUT_SECONDS` (20 s) | `The query parser (Qwen) is unavailable or too slow, so text search cannot run; ...` |
| 503 | Category index built with other settings (operations problem) | `The 'rings' index was built with a different model or preprocessing settings and is not served; ...` |

400, 404, 413 and 415 are the caller's to fix. A 503 is temporary: retry with backoff, or show "try again".
Under overload the embedding service queues requests, and after 30 s (`EMBED_TIMEOUT_SECONDS`) they become 503s
(see the load test). No error returns HTML or a stack trace. Failure tests are in
`backend/tests/test_failures.py`.

### Latency to plan for

Measured on the development laptop, CPU only:
- **Image search:** ~1.2 s at one request at a time. With region rerank off, ~0.33 s.
- **Image + text:** ~1.8 s.
- **Text search:** ~2.1 s, mostly the Qwen call.

The laptop sustains ~0.8 image searches per second with region rerank on, and ~3.5 with it off. Details are in
`eval/results/load_test.md` and the final report. Set client timeouts at 30 s or more for image search.

## 2. How a search works

| Step | What happens | Component |
|---|---|---|
| 1 | Check the upload (type, size, readable, ≥128 px); fix orientation and colour profile; fit onto a 512 px white square | backend |
| 2 | Embed the photo (and, with region rerank, its centre crop and 2×2 tiles) | SigLIP 2 Base 224 |
| 3 | Category: given, named in the text, or detected (SigLIP zero-shot over the category names); ask if unsure | backend + SigLIP |
| 4 | Understand the refinement or text query (metal, stones, colours, motifs, style) | Qwen3-VL-8B (remote) |
| 5 | Search the category's image index and caption index (top 100 each), fuse with RRF | FAISS |
| 6 | Load metadata; apply `project_id` / `subcategory` filters | MongoDB |
| 7 | Small boosts for matching tags (never removes results); region rerank of the top 50 | backend |
| 8 | Text search: one 0–1 score from weighted signals (weights v2) | backend |
| 9 | One result per product, mark near-duplicates, top_k | backend |

Offline (per catalogue image, see [runbook_add_catalogue.md](runbook_add_catalogue.md)):
1. Embed the image.
2. Tag it with Qwen (category, metal, gemstones, stone colours, motifs, style, description).
3. Embed the description (caption index).
4. Embed region tiles.

## 3. Catalogue and IDs

What the system needs per catalogue image (today from `data/catalogue.csv` and `data/products.csv`, see
[runbook_add_catalogue.md](runbook_add_catalogue.md) for the columns):

| Our field | Meaning | Rules | Client field (to confirm) |
|---|---|---|---|
| `product_id` | The sellable design. Results are one per product. | Stable; unique across the catalogue | Their product / SKU / design number? |
| `image_id` | One photo of a product | Stable; unique across the whole catalogue; never reused for another photo | Their image or asset ID, or `<product_id>_<n>` if they have none |
| `project_id` | Tenant or collection; `project_id` filter in search | Any string; the sample uses `demo` | Their brand / store / collection? |
| `category` | One of `data/categories.txt` | Lowercase; must match their taxonomy mapping | Their category codes → our 9 categories |
| `subcategory`, `name`, `metal`, `gemstones`, `stone_colours`, `motifs`, `style` | Shown as attributes | Optional | Their product attributes |
| `vector_id` | Internal FAISS ID, assigned by ingest | Never exposed in the API | none |

Points to settle with the client:
1. **IDs:** does one design have several photos and several SKUs (sizes, metals)? Results should be grouped at
   the level their users buy or browse.
2. **Category mapping:** their categories → ours. A category missing from our list needs a line in
   `data/categories.txt` and enough images to be useful.
3. **How the catalogue reaches us:**
   - Today: CSV files plus image files, loaded with the scripts.
   - For 955k images and daily changes: an export from their system in the same columns, then the incremental
     ingest. Removed products are dropped from the index on the next run.
4. **Image URLs:** we return our own image path; they will probably map `image_id` to their CDN URL instead.

## 4. Their existing pipeline and the new one

The client's current 7-step pipeline is described in the EJIL proposal, which is not in this repository, so
it is not mapped here step by step. What we know of their system: they store **Vertex AI multimodal
embeddings** for their own products, and have a catalogue of about 955k images.

To fill in with the client, for each of their 7 steps:

| Their step | Corresponding step in section 2 | Reused as is / replaced / new |
|---|---|---|
| 1. … | | |
| … | | |
| 7. … | | |

From what this POC shows, the likely outcome:
- **Probably reused:** their catalogue database and image storage (as the source of the CSV export), their
  product pages and UI (the API returns IDs; the client's UI is out of scope), and their category taxonomy.
- **New:**
  - Upload checks and preprocessing.
  - SigLIP embeddings, per-category FAISS indexes and the caption index.
  - Qwen tagging and query parsing.
  - Hybrid retrieval, boosts, region rerank and the reranker.
  - Category detection with "ask the user".
  - One result per product with near-duplicate marking.
- **Replaced, if they adopt SigLIP:** their stored Vertex vectors (section 5).

## 5. Vertex AI vs SigLIP (Stage 6)

**Decision: SigLIP 2 Base 224, self-hosted.**
- On our 50 test photos it reached R@1 86%, R@5 94% and MRR 0.888 at about 0.46 s per embedding on CPU
  (`eval/DECISION.md`).
- SigLIP 2 at 384 px was no more accurate and 2.3× slower.
- **Vertex AI was not tested**: there were no Google Cloud credentials. The client's stored Vertex vectors cover
  their products, not our sample catalogue, so they could not be compared on our test photos.

If the client keeps Vertex AI (`EMBED_BACKEND=vertex`):
- **What works today:** the adapter embeds images with `multimodalembedding@001` (1408 dimensions) into its
  own index folder, and image search with a given category works. It has not been run against the real API.
- **What does not work yet:**
  - The adapter has no text embedding and no batch call. So category detection, text search, the caption index
    (hybrid search) and region rerank do not work with it. They would fail with a server error instead of a
    readable 503.
  - Vertex has a text embedding in the same space (also 1408-d), so this is adapter work. It also needs a new
    run of the evaluation, because hybrid weights, the region weight and the reranker ranges were tuned on
    SigLIP scores.
- **Their stored vectors:** they can only be reused if they were made from the same image preprocessing (we
  correct orientation and colour profile, fit the photo onto a 512 px white square, and send it as PNG). Otherwise every image is embedded again. Either way,
  ingest still assigns our `vector_id`s.
- **Cost and latency:** a paid API call per search and per catalogue image, with network latency, instead of
  self-hosted CPU/GPU. No GPU to run, but the data leaves their infrastructure.
- **Recommendation:** run `scripts/ingest.py --backend vertex` and `eval/run_eval.py --backend vertex` on a
  sample of their real catalogue and real query photos, next to SigLIP, before deciding. Our test set is
  synthetic edge cases of public images.

## 6. Before production

| Area | Today (POC) | Needed |
|---|---|---|
| Authentication | None; CORS limits browsers to `CORS_ORIGINS` only | API keys or OAuth2 client credentials per calling system, checked in the backend or a gateway; per-key `project_id` scoping |
| Rate limits | None. Overload queues in SigLIP and turns into 503 after 30 s | Per-key limits at a gateway (nginx, API gateway). A cap on queued embedding work, answering 429 at once. Retry-After on 503 |
| HTTPS | Plain HTTP on the host port | TLS termination at a reverse proxy or load balancer. Do not expose MongoDB (27017) or SigLIP (8000) ports beyond the host; today compose publishes them for development |
| Capacity | One laptop: ~0.8 image searches/s with region rerank | SigLIP on GPU or several replicas (the only bottleneck); load test on the target hardware; uvicorn workers or replicas for the backend |
| Monitoring | `/health`, container logs with per-step timings | Metrics (request rate, p95 per endpoint, 4xx/5xx, `reduced_mode` rate, SigLIP queue time), alerts on `/health` degraded and on `indexes.refused` not empty, central logs |
| Qwen | Shared dev server; tagging ~30 days for 955k images at the measured rate | Dedicated GPU capacity or agreed concurrency; a decision on where Qwen runs |
| Data | MongoDB without auth or replication; indexes on local disk | MongoDB auth and backups (see [runbook_operations.md](runbook_operations.md)), replica set; index folders on backed-up storage |
| Privacy | Uploads not stored (`SAVE_QUERY_IMAGES=false`) | Confirm retention rules for user photos and logs |
| Scale of FAISS | Exact search (`IndexFlatIP`) per category; under 1 ms for 15 vectors | Exact search scans every vector of the category, and memory is ~2.9 GB for 955k × 768 floats (plus the caption index and tiles). Benchmark at full size, and move to an approximate index (IVF or HNSW) if search time matters |
