# Jewellery visual search

Upload a jewellery photo (optionally with a refinement like "similar but in silver"), or type a description, and
get the catalogue products with the same design. Proof of concept built in stages 0–12; the implementation doc
has the full plan. **Results and limits: [docs/final_eval_report.md](docs/final_eval_report.md).**

## Architecture

```
 browser ──► frontend (nginx, React test harness)            :8090
    │
    └─────► backend (FastAPI)                                 :8001
              ├─ SigLIP 2 Base 224 embedding service (CPU)    :8000   image + text vectors
              ├─ MongoDB                                      :27017  catalogue metadata, Qwen tags, vector IDs
              ├─ FAISS indexes on disk (data/faiss_index)             per category: image vectors, caption vectors
              ├─ region tile store (data/features)                    2×2 tiles for re-scoring the top 50
              └─ Qwen3-VL-8B, remote GPU server (VLM_SERVICE_URL)     query parsing; catalogue tagging (offline)
```

- **Image search:**
  1. Check and preprocess the photo.
  2. Embed it with SigLIP (plus 2×2 region tiles).
  3. Use the given category, or detect it (and ask the user when unsure).
  4. Search that category's image index and caption index, and fuse them (RRF).
  5. Boost matching tags and re-score the top 50 by region.
  6. Return one result per product.
- **Text search:** Qwen parses the text, SigLIP embeds it, and the same retrieval runs, scored by the tuned
  reranker (weights v2).
- **Offline, per catalogue image** (`scripts/`):
  1. Embed it (`ingest.py`).
  2. Tag it with Qwen (`tag_catalogue.py`).
  3. Embed the tag description (`build_caption_index.py`).
  4. Embed the region tiles (`build_feature_indexes.py`).
- **Degradation:** without Qwen, image search still works ("reduced mode"). Without SigLIP or MongoDB, the API
  answers a readable 503, never a 500.

## Documentation

| Document | For |
|---|---|
| [docs/final_eval_report.md](docs/final_eval_report.md) | Client review: accuracy vs the Stage 6 baseline, latency, features in use, limits, open decisions |
| [docs/integration_notes.md](docs/integration_notes.md) + [docs/openapi.json](docs/openapi.json) | Integrating the API: contracts, error codes, IDs, Vertex AI vs SigLIP, production checklist |
| [docs/runbook_add_catalogue.md](docs/runbook_add_catalogue.md) | Adding and removing catalogue images and products |
| [docs/runbook_rebuild_index.md](docs/runbook_rebuild_index.md) | Full rebuild after a model, preprocessing or tagger change |
| [docs/runbook_operations.md](docs/runbook_operations.md) | Health, logs, restarts, reranker weights, backup and restore |
| `eval/results/load_test.md` | Load test (git-ignored like all of `eval/results/`; regenerate with `scripts/load_test.py`) |
| `eval/DECISION.md` | Stage 6 model decision (SigLIP 224 vs 384 vs Vertex) |

## Layout

```
ai-service/   SigLIP 2 embedding API (FastAPI, Docker)
backend/      Search backend (FastAPI); its Docker image also runs the scripts and tests ("tools" service)
frontend/     React + Vite test harness (Stage 11), served by nginx in Docker
data/         Catalogue images, FAISS indexes and feature stores (git-ignored, built by the scripts)
config/       Re-ranker weights (reranker_weights_<version>.json)
scripts/      Dataset, ingestion, tagging and index scripts; run_tests.sh
eval/         Evaluation sets, run_eval.py and results
docs/         Final report, integration notes, OpenAPI file, runbooks
```

## Run with Docker (fresh machine)

Needs Docker with Compose v2, about 12 GB of disk and internet access (model weights, sample images, Qwen).
Qwen3-VL runs on a remote GPU server (`VLM_SERVICE_URL`); it is not part of the stack. Without a Qwen key, image
search works, but catalogue tagging (step 7) and text search do not.

1. **Get the code.**
   ```bash
   git clone <repo-url> jewellery-search && cd jewellery-search
   ```
2. **Download the SigLIP 2 weights** (1.5 GB, once; only Docker needed). Use any folder outside the repo:
   ```bash
   mkdir -p ~/siglip2-base/model
   docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v ~/siglip2-base/model:/model python:3.12-slim \
     sh -c 'pip install -q --user huggingface_hub && python -c "from huggingface_hub import snapshot_download; snapshot_download(\"google/siglip2-base-patch16-224\", local_dir=\"/model\")"'
   ```
3. **Configure.** `cp .env.example .env`, then edit `.env`:
   - `SIGLIP_MODEL_DIR`: the absolute path from step 2, e.g. `/home/you/siglip2-base/model` (`~` does not work here).
   - `QWEN_API_KEY`: the key for `VLM_SERVICE_URL`.
   - `HOST_UID` / `HOST_GID`: the output of `id -u` / `id -g`, if not 1000.
   - If port 8001 is taken: set `BACKEND_PORT` and `VITE_API_URL` to another port (e.g. 8080 and
     `http://localhost:8080`). If you change `FRONTEND_PORT`, add the new origin to `CORS_ORIGINS`.
4. **Create the data folders** (otherwise Docker creates them owned by root and the scripts cannot write):
   ```bash
   mkdir -p data/catalogue data/faiss_index data/features
   ```
5. **Start the stack** (first build ~5 min):
   ```bash
   docker compose up -d --build
   docker compose ps       # mongodb, siglip2-api, backend: (healthy); frontend: Up
   ```
   The backend log says `serving ... with 0 vectors: nothing indexed yet` until step 6.
6. **Build the sample catalogue and the image index** (~6 min; the scripts run in the `tools` container):
   ```bash
   docker compose run --rm tools scripts/build_sample_dataset.py   # 71 catalogue images + 50 queries (downloads)
   docker compose run --rm tools scripts/validate_dataset.py       # 0 error(s)
   docker compose run --rm tools scripts/ingest.py --verify        # seeds MongoDB, embeds, "71/71 return themselves"
   ```
   On a brand-new machine the first `--verify` can report SigLIP timeouts while the weights are read from disk
   for the first time. Run the same command again: it re-embeds nothing and should print 71/71.
7. **Tag the catalogue with Qwen and build the caption index** (~2 min; needs `QWEN_API_KEY`):
   ```bash
   docker compose run --rm tools scripts/tag_catalogue.py          # "71/71 saved tag records valid"
   docker compose run --rm tools scripts/build_caption_index.py    # "caption check: ok"
   ```
8. **Build the region tiles and restart the backend** (tiles are loaded at startup; FAISS indexes reload by themselves):
   ```bash
   docker compose run --rm tools scripts/build_feature_indexes.py --feature tiles --grid 2
   docker compose restart backend
   ```
9. **Check.**
   ```bash
   curl localhost:8001/health      # "status": "ok", indexes.served has 5 categories, indexes.refused is []
   ```
   Open http://localhost:8090 and search with a photo from `data/queries/`.
10. **Test and evaluate.**
    ```bash
    scripts/run_tests.sh                                            # every suite, in Docker
    docker compose run --rm tools eval/run_eval.py --backend siglip2_224 --name docker_check   # R@1 90%, R@5 96%
    ```

Everyday commands: `docker compose logs -f backend`, `docker compose restart backend`, `docker compose down`
(keeps the MongoDB volume; `down -v` deletes it). Any script runs as `docker compose run --rm tools <script> [args]`.

## Run locally (development)

```bash
cp .env.example .env                  # as above; .env.backend, if present, overrides it for local runs
docker compose up -d mongodb siglip2-api
cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080

# second terminal, from the project root
cd frontend && cp .env.example .env && npm install && npm run dev   # http://localhost:5173
```

Scripts run from the project root with the venv active, e.g. `python scripts/ingest.py`.

## Tests

```bash
scripts/run_tests.sh             # in Docker (stack running): backend, scripts + eval, frontend
scripts/run_tests.sh --local     # same suites with backend/.venv and npm (MongoDB on localhost:27017)
```

## Evaluate

The test set is 50 photo queries (`eval/queries.csv`, synthetic edge cases of public images), 20 text queries
and 8 photo + text queries, with a fixed tune / holdout split (`scripts/make_eval_splits.py`).

```bash
E="docker compose run --rm tools eval/run_eval.py --backend siglip2_224"
$E --name my_run                                   # 50 photo queries, category given -> eval/results/my_run.md
$E --split holdout --name my_run_ho                # holdout only (14 photo queries)
$E --category detect --name my_run_detect          # without the category: measures detection too
$E --mode text --name my_text                      # text queries (needs Qwen)
$E --mode image_text --name my_refine              # photo + refinement text
$E --set FEATURE_REGION_RERANK=false --name no_region   # any setting can be overridden for a run
$E --compare final_all_default my_run              # Markdown comparison of saved runs
python scripts/load_test.py --url http://localhost:8001 --machine "laptop"   # eval/results/load_test.md
```

Each run writes `eval/results/<name>.json`, `<name>.md` and, for failures, `<name>_failures/` with a strip of
the query, the top 5 and the expected product.

## Ports

| Service | Host port (`.env`) |
|---|---|
| siglip2-api | 8000 (`EMBED_PORT`) |
| backend in Docker | 8001 (`BACKEND_PORT`) |
| backend run locally (uvicorn) | 8080 |
| frontend in Docker (nginx) | 8090 (`FRONTEND_PORT`) |
| frontend dev server (Vite) | 5173 |
| MongoDB | 27017 (`MONGO_PORT`) |
| Qwen3-VL | remote (`VLM_SERVICE_URL`) |

On the development laptop, port 8001 is used by another project (paddleocr), so `.env` there sets
`BACKEND_PORT=8080`.

## Stage notes

### Stage 0 — SigLIP container verified
768-dim, L2-normalized embeddings for JPEG/PNG/WebP; invalid files return 400; ~350 ms/request on CPU.
Details in `ai-service/README.md`.

### Stage 1 — Project structure and configuration
Added: the folder layout above, `backend/app/config.py` (env-driven settings),
`backend/app/embed_client.py` (async `/embed` client with timeout and `EmbeddingServiceError`),
backend `/health`, `docker-compose.yml`, `.env.example`, `.gitignore`.

`MODEL_PATH` in `ai-service/app.py` is now read from the environment (default `/models`);
the `/embed` response shape is unchanged.

MongoDB uses `mongo:8.2` because 8.0 refuses to start on Linux kernel 6.19+ (SERVER-121912).

How to test:

```bash
cd backend && pytest -q                        # client + health unit tests
curl localhost:8080/health                     # 200 {"status":"ok", ...} when both deps are up
docker compose stop siglip2-api && curl -i localhost:8080/health   # 503, embedding_service: error
docker compose start siglip2-api
git status --ignored                           # .env, data/, .venv, weights are ignored
```

### Stage 2 — Catalogue dataset and labelled evaluation set
Added: the data layout, `data/catalogue.csv` and `eval/queries.csv` (headers only),
`data/categories.txt` (allowed categories, edit to match the catalogue),
`scripts/validate_dataset.py` and its tests. Status: **tooling done, real data not yet added.**

Layout (images are git-ignored; the CSVs and `categories.txt` are tracked):

```
data/catalogue/<category>/<image_id>.jpg     one row per image in data/catalogue.csv
data/queries/<query_id>.jpg                  one row per query in eval/queries.csv
```

- `data/catalogue.csv`: `image_id,product_id,project_id,category,subcategory,image_path`.
  A product can have several images (one row each); all its images must share one category.
- `eval/queries.csv`: `query_id,query_path,category,expected_product_ids,edge_case_tags`.
  Lists are `;`-separated, e.g. `P102;P118` and `lighting;angle`.
- Edge-case tags: `colour_mismatch`, `lighting`, `reflection`, `angle`, `rotation`,
  `background`, `distance`, `near_duplicate`.
- Paths are relative to the project root, e.g. `data/catalogue/rings/img_0001.jpg`.
- Query photos must be different photos of the product, not copies of catalogue images.
  The validator warns when a query is byte-identical to a catalogue image.

The validator reports these as errors: missing columns or files, unreadable images, duplicate IDs,
reused paths, unknown categories or tags, paths that break the layout, queries with no expected
product, expected products that are missing from the catalogue or are in a different category.
Coverage (at least 30 queries, at least 3 per tag) is a warning by default and an error with `--strict`.

How to test:

```bash
pytest -q scripts/tests                              # validator unit tests
python scripts/validate_dataset.py --strict          # Stage 2 done check: must end "0 error(s)"
python scripts/validate_dataset.py --markdown        # counts table to paste below
```

#### Sample dataset (stand-in until the client catalogue is available)

Client product data is in Postgres (`dev1`, ~955k products), but product photos live in IBM Cloud
Object Storage (`<product_id>-<S|T|L>.jpg`) and need COS credentials. Until then, the sample
dataset is built from a public CC BY 4.0 set (see `data/ATTRIBUTION.md`):

- 65 products picked by hand, with metadata written from the images: `data/products.csv`
  (`name, metal, gemstones, stone_colours, motifs, style`, `;`-separated lists).
- 50 queries in `eval/query_specs.csv`. Each starts from a different file than the catalogue image
  (another augmented copy, or a separate real photo for `q006` and `q030`) and gets one seeded
  edge-case transform. Look-alike pairs tagged `near_duplicate`: RG-002/RG-012, RG-003/RG-011,
  NK-004/NK-010, PD-004/PD-009, BR-002/BR-005.
- Limits: the edge cases are synthetic except `q006`/`q030`; the source images were already
  auto-contrasted and brightness-augmented by Roboflow, so some queries are over- or under-exposed.

```bash
pip install -r backend/requirements.txt
python scripts/build_sample_dataset.py     # downloads ~25 MB (cached in data/raw/), writes images + CSVs
python scripts/validate_dataset.py --strict
python scripts/seed_mongo.py               # products + images collections in MONGO_DB
```

Counts per category (`python scripts/validate_dataset.py --markdown`):

| Category | Products | Images | Queries |
|---|---:|---:|---:|
| bracelets | 13 | 15 | 10 |
| earrings | 14 | 15 | 11 |
| necklaces | 12 | 13 | 6 |
| pendants | 12 | 13 | 9 |
| rings | 14 | 15 | 14 |
| **total** | **65** | **71** | **50** |

Queries per edge-case tag: colour_mismatch 7, lighting 8, reflection 7, angle 7, rotation 8,
background 7, distance 8, near_duplicate 10. Categories in `categories.txt` with no data
(anklets, bangles, chains, nose_pins) report 0.

### Stage 3 — Shared preprocessing module
Added: `backend/app/preprocessing.py`, its tests, `scripts/preprocess_samples.py`, and
`backend/requirements-bg.txt` (optional rembg).

Ingestion and search both call `preprocess(image_bytes) -> PIL.Image`, so catalogue and query images are
always prepared the same way. Steps:

1. Decode; empty, corrupt or truncated files raise `ImagePreprocessError`.
2. Apply EXIF orientation.
3. Reject images whose shorter side is below `PREPROCESS_MIN_SIDE` (128 px; the smallest dataset image is 224).
4. Convert to sRGB/RGB: uses the embedded ICC profile if there is one (CMYK JPEGs), flattens transparency
   onto white, and scales 16-bit greyscale down instead of clipping it.
5. Optional, off by default: background removal (`PREPROCESS_REMOVE_BACKGROUND`, rembg) and lighting
   normalization (`PREPROCESS_NORMALIZE_LIGHTING`, luminance autocontrast).
6. Scale the long side to `PREPROCESS_SIZE` (512) and pad to a white square. The aspect ratio is kept.

`encode(image)` returns lossless PNG bytes for `/embed`. `PreprocessOptions.signature`, e.g. `pp-v1/512`,
identifies the exact pipeline; store it with every vector. Bump `PREPROCESS_VERSION` whenever the output
pixels change, then re-ingest the catalogue.

**Visual check: background removal clips jewellery, so it stays off.** With `isnet-general-use` on the
sample set:

- RG-014: the whole shank is removed; only the cluster head is left.
- BR-003: parts of the thin chain are removed, and the grey area inside the loop plus a shadow are kept.
- BR-012: the white area inside the bracelet is kept as product; shadows are kept.

Lighting normalization also shifts gold towards dark orange on white-background shots (e.g. NK-005,
ER-007), so it stays off too. Both are kept behind flags so they can be retried later with other models.

rembg 2.0.85 declares `pillow<13`, so installing `requirements-bg.txt` downgrades Pillow to 12.x.
Run `pip install -r requirements.txt` afterwards to go back to 13.0.0. rembg still works on 13.0.0
(checked with the sample script); pip only reports a version conflict.

How to test:

```bash
cd backend && pytest -q                         # includes tests/test_preprocessing.py
pip install -r backend/requirements-bg.txt      # only for the background panels
python scripts/preprocess_samples.py            # strips in eval/preprocess_samples/ (git-ignored), ~10 s/image on CPU
python scripts/preprocess_samples.py --ids RG-014_01 BR-003_01 --no-background
```

Each strip shows: original | default | background removed | removed area in red | background removed + lighting.
Chains, prongs and settings must never be red.

### Stage 4 — Catalogue ingestion: embeddings, MongoDB, FAISS
Added: `backend/app/vector_index.py` (per-category FAISS indexes and manifest), `scripts/ingest.py`, and their
tests. `faiss-cpu` is now in `backend/requirements.txt`. `scripts/seed_mongo.py` now updates images field by
field, so re-seeding keeps the embedding fields.

- **MongoDB:** `products` holds product attributes. `images` holds `image_id` (`_id`), `product_id`,
  `category`, `subcategory`, `image_path`, plus `vector_id`, `embedding_model`, `embedding_version`,
  `preprocess_version` (the preprocessing signature, e.g. `pp-v1/512`), `image_sha256` and `ingested_at`.
  `vector_id` has a unique index. An image that failed has `ingest_error` and no vector fields.
- **FAISS:** one `IndexIDMap(IndexFlatIP)` per category. Vectors are L2-normalized, so the inner product is the
  cosine similarity. The FAISS ID is the image's `vector_id`.
- **Files:** `FAISS_INDEX_DIR/<model>_<version>/<category>.index` and `manifest.json` (model, embedding
  and preprocess versions, dimension, count per category). For example:
  `data/faiss_index/siglip2-base-patch16-224_siglip2-base-v1/`.
- **Incremental by default:** only images that are new, whose file hash or category changed, or that were
  embedded with other settings are sent to `/embed`. Vectors of images removed from `catalogue.csv` are
  deleted. If the preprocessing settings no longer match the manifest, the script stops and asks for `--rebuild`.
- **`--rebuild`** clears every vector and re-embeds everything, assigning vector IDs from 1.
- **Bad images** (unreadable, too small, embedding error) are logged, recorded as `ingest_error`, and skipped.

Results on the sample catalogue (`python scripts/ingest.py --rebuild --verify`, ~70 s on CPU):

| Category | Index | MongoDB |
|---|---:|---:|
| bracelets | 15 | 15 |
| earrings | 15 | 15 |
| necklaces | 13 | 13 |
| pendants | 13 | 13 |
| rings | 15 | 15 |

Every `vector_id` maps to exactly one image. 71/71 catalogue images return themselves as the top result.
I also tested an incremental run with one new image and one corrupt file: the new image was added
(72 vectors), the corrupt file was skipped with `ingest_error`, and removing both rows removed the vector again.

How to test:

```bash
cd backend && pytest -q && cd ..                 # vector index unit tests
python -m pytest -q scripts/tests               # ingest planning and bad-image handling
python scripts/ingest.py --rebuild --verify     # exits 0 only if counts, ID mapping and self-search all pass
python scripts/ingest.py                        # second run: "71 up to date, 0 to embed"
```

### Stage 5 — Baseline image search API
Added: `backend/app/search.py` (`POST /search/image` and the catalogue image route), wiring in `main.py`
(CORS, readable 400s for form errors), new settings, `tests/test_search.py` and `tests/test_search_live.py`.

```bash
curl -F "file=@data/queries/q014.jpg;type=image/jpeg" -F category=earrings -F top_k=5 \
     localhost:8080/search/image
```

Form fields: `file` (required), `category` (required), `project_id`, `subcategory`, `top_k` (default 10, max
`SEARCH_MAX_TOP_K`=50). Flow: preprocess → embed → search that category's index for `SEARCH_CANDIDATES` (100) →
MongoDB metadata → `project_id`/`subcategory` filters → best image per product → `top_k`.

The response has `query_id`, `model`, `embedding_version`, `category` and `results[]`. Each result has
`product_id`, `image_id`, `image_url`, `category`, `similarity_score` (cosine) and `attributes`.
`image_url` is a `/catalogue-images/<category>/<file>` route under `CATALOGUE_DIR`; file-system paths are never
returned, and `..` escapes return 404.

| Case | Status |
|---|---|
| Missing `category`/`file`, bad `top_k`, empty, corrupt or too-small image | 400 |
| Unknown category (not in `data/categories.txt`) | 400 |
| Known category with nothing indexed, e.g. `bangles` | 404 |
| Larger than `MAX_UPLOAD_MB` (10) | 413 |
| Not JPEG/PNG/WebP (both the declared type and the file's magic bytes are checked) | 415 |
| Embedding service down | 502 |
| MongoDB down, or indexes built with other preprocessing settings | 503 |

- **Logs:** one line per query with `query_id`, category, filters, per-step latency (read, preprocess, embed,
  faiss, metadata, rank, total) and the result image IDs.
- **Uploads:** not stored unless `SAVE_QUERY_IMAGES=true`.
- **CORS:** only `CORS_ORIGINS` (default `http://localhost:5173`).
- **Index reload:** indexes are loaded on first use and reloaded when `manifest.json` changes, so re-running
  `scripts/ingest.py` needs no restart.

Tests:
- `tests/test_search.py` runs the full request against a 5-image fixture catalogue: a throwaway MongoDB
  database, real FAISS indexes and a deterministic fake embedder. It covers own-product-first,
  one-result-per-product, filters, `top_k`, image serving and traversal, every error status, PNG/WebP and CORS.
  It is skipped if MongoDB is down.
- `tests/test_search_live.py` sends every catalogue image through the real stack: 71/71 return their own
  product first. It is skipped unless `/health` is ok and the indexes exist.

```bash
cd backend && pytest -q        # 42 tests; the live test takes ~45 s
```

### Stage 6 — Baseline evaluation and model comparison
Added: `eval/run_eval.py`, `backend/app/embed_backends.py` (`EMBED_BACKEND=siglip2_224 | siglip2_384 | vertex`),
a `siglip2-api-384` compose service (profile `siglip384`, port 8003), and **`eval/DECISION.md`, the decision note**.

**Decision: SigLIP 2 Base 224.** Baseline to beat: **R@1 86%, R@5 94%, MRR@10 0.888** on the 50 queries.
The 384 model scored the same (R@1 84%, R@10 96%) at 2.3× the latency. Vertex was not run (no credentials).
The weakest cases are background, colour_mismatch and lighting (one complete miss each), distance and angle
(right product at rank 2–4), and necklaces (R@1 50%). Details are in `eval/DECISION.md`.

Changes to earlier stages:
- `EMBED_MODEL_NAME` and `EMBED_VERSION` are gone; the model and version come from the backend registry.
  Old lines in `.env`/`.env.backend` are ignored.
- MongoDB images keep one permanent `vector_id` shared by all backends. The Stage 4 top-level fields moved
  to `embeddings.<backend>` (`embedding_model`, `embedding_version`, `preprocess_version`, `image_sha256`,
  `ingested_at`), and errors moved to `ingest_errors.<backend>`. Ingest removes the old top-level fields.
  `--rebuild` now clears only that backend and keeps vector IDs.
- `POST /search/image` responses now include `embed_backend` and `timings_ms`.

How to run:

```bash
SIGLIP_384_MODEL_DIR=~/siglip2-base-384/model docker compose --profile siglip384 up -d   # 384 only
python scripts/ingest.py --backend siglip2_384 --verify
python eval/run_eval.py --backend siglip2_224         # eval/results/siglip2_224.{json,md} + _failures/
python eval/run_eval.py --compare siglip2_224 siglip2_384
python -m pytest -q scripts/tests eval/tests
```

The 384 weights were downloaded with
`hf download google/siglip2-base-patch16-384 --local-dir ~/siglip2-base-384/model` (1.3 GB, outside the repo).

### Stage 7 — VLM attribute tagging and caption embeddings

**Part A, feasibility** (`scripts/qwen_feasibility.py`, results in `eval/results/qwen_feasibility.json`):
- **Hosted:** Qwen3-VL-8B on the hosted vLLM server (`QWEN_API_KEY_API_BASE`, full precision). 20/20 replies
  were valid JSON, at 12.1 s/image one at a time and **2.76 s/image with 4 in parallel**.
- **Local:** not run. This laptop has 16 GB RAM (about 5 GB free) and no NVIDIA GPU, so 4-bit Qwen would run in swap.
- **Verdict:** use the hosted server. That's about 3 min for the 71 sample images, but about 30 days for the
  955k client catalogue at that rate; the full catalogue needs a dedicated GPU or higher agreed concurrency.

**Part B, tagging.** Search ranking is unchanged (that is Stage 8).

| File | What it does |
|---|---|
| `backend/app/tagging/vocab.py` | Allowed values per field, synonyms (`yellow gold` → `gold`), `normalize()`. Motifs are marked provisional. |
| `backend/app/tagging/schema.py` | Pydantic `Tags`: only vocabulary values, no duplicates, no extra keys, non-empty description. |
| `backend/app/tagging/prompts.py` | Tagging prompt (lists the allowed values) and retry prompt. |
| `backend/app/clients/vlm_client.py` | Calls `VLM_SERVICE_URL` (OpenAI-compatible), retries once on invalid JSON, maps synonyms, drops and reports unknown values, validates. |
| `scripts/tag_catalogue.py` | Resumable batch job: saves `vlm_tags` on each image with `tagger_model`, `quantization`, `tagger_version`. |
| `ai-service/app.py` | New `POST /embed/text`: SigLIP 2 text vector (padded to 64 tokens, normalized). `embed_client.embed_text()` calls it. |
| `backend/app/caption_store.py` | Per-category caption FAISS indexes under `FAISS_INDEX_DIR/captions/<model>_<version>/`, same `vector_id`s as the image index. |
| `scripts/build_caption_index.py` | Embeds each description and rebuilds the caption indexes, then compares them with the image index. |
| `scripts/tag_spotcheck.py` | `export` writes `eval/tag_spotcheck.csv` (50 random images) to mark by hand; `score` prints accuracy per field. |

`vlm-service/` was not built: the GPU server already exposes an OpenAI-compatible API. `TAGGER_VERSION`
(`tags-p1-v1`) changes whenever the prompt or vocabulary changes, and `tag_catalogue.py` re-tags images with
another version. Settings: `VLM_SERVICE_URL`, `VLM_MODEL_NAME`, key from `VLM_API_KEY` or `QWEN_API_KEY`,
`VLM_QUANTIZATION`, `VLM_CONCURRENCY` (4).

Results on the sample catalogue:
- **Tagging:** 71/71 images tagged in 3.5 min, 0 failures, 7 needed the JSON retry, 0 values dropped.
  71/71 saved tag records pass the schema.
- **Captions:** every category has as many captions as images, with the same vector IDs.
- **VLM category vs catalogue category:** they differ on 13 images. Most are pendants on a chain tagged
  `necklaces`, or plain necklaces tagged `chains`. Search uses the catalogue category, so this only matters
  if VLM categories are used later.
- **Agreement with `products.csv`** (automatic; the manual spot-check is still to be marked):

  | Field | exact | overlap |
  |---|---:|---:|
  | category | 78% | 78% |
  | metal | 59% | 76% |
  | gemstones | 31% | 41% |
  | stone_colors | 40% | 57% |
  | motifs | 24% | 70% |
  | style | 40% | 40% |

How to run each check:

```bash
python scripts/tag_catalogue.py --check              # schema: "71/71 saved tag records valid (100%)"
python scripts/build_caption_index.py --check        # caption count = image count per category
python scripts/tag_spotcheck.py score                # per-field accuracy (after marking eval/tag_spotcheck.csv)
cd backend && pytest -q tests/test_tagging.py        # schema, synonyms, invalid JSON, unknown values
```

Rebuild the SigLIP image after changing `ai-service/app.py`:
`docker compose build siglip2-api && docker compose up -d siglip2-api`.

### Stage 8 — Query analysis, hybrid search, text queries and refinement
`app/search.py` is split into a package:

```
app/search/   common.py · query_analysis.py · hybrid.py · boosts.py · image_search.py · text_search.py
app/routes/   search_image.py (POST /search/image) · search_text.py (POST /search/text) · catalogue.py
app/clients/  llm_client.py (text query parsing; reuses the hosted Qwen endpoint)
```

- **`POST /search/image`:**
  - `category` is optional. If it's missing, SigLIP zero-shot picks it. Below `CATEGORY_CONFIDENCE_THRESHOLD`
    the response has `needs_category: true` and the top 3 `category_candidates` instead of results.
  - The optional `text` refines the search ("similar but in silver"), and `debug=true` adds image/caption
    ranks, RRF score and boosts per result.
- **Search:** the query vector searches the image index and the caption index (top 100 each), and the two
  lists are fused with RRF (k=60). Boosts then add small bonuses for matching tags; they never remove results.
- **Image-only queries** never use stone colour or metal. Gemstones and motifs come from Qwen tagging of the
  query image, which is off by default because it adds a Qwen call per search.
- **`POST /search/text`** `{"text", "category"?, "project_id"?, "top_k"?, "debug"?}`:
  - Qwen parses the text into vocabulary values plus a `semantic_query`, which SigLIP embeds as text (lowercased).
  - The category is a hard filter, kept only if the text names it. With none, all categories are searched
    (`TEXT_NO_CATEGORY`). Stone colours are kept only if the user wrote them.
  - If the LLM fails, the raw text is searched and the response carries a warning.
- **Flags and weights** are in `.env.example`; `similarity_score` stays the image cosine (the caption cosine for text).

Eval (`eval/results/s8_compare.md`; 50 image queries, 20 text queries, 8 image + text queries):

| Run | R@1 | R@5 | R@10 | MRR@10 | category acc. | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (all flags off) = Stage 6 | 86% | 94% | 94% | 0.888 | 84%* | 360 | 394 |
| + hybrid (**default**) | 86% | 94% | **100%** | 0.901 | 84%* | 358 | 406 |
| + hybrid + boosts (query tagging on) | 84% | **98%** | 100% | **0.903** | – | 3911 | 6685 |
| hybrid, category detected (SigLIP) | 74% | 80% | 84% | 0.772 | **84%** | 359 | 390 |
| hybrid, category detected (Qwen) | 70% | 84% | 86% | 0.763 | 86% | 3674 | 6400 |
| text queries | 85% | 95% | 100% | 0.883 | 80%† | 2142 | 5225 |
| image only (refinement set) | 12% | 88% | 88% | 0.417 | – | 366 | 423 |
| image + text refinement | **62%** | 88% | 100% | **0.771** | – | 1853 | 2112 |

\* SigLIP detection gets 42/50 right (91% of the 46 it decided on). It asked on 4, and the true category was in
the top 3 each time. Errors are necklace/pendant/bracelet confusions. † Text: the category was named in 16 of 20
queries and was always right; the other 4 searched every category.

Decisions and known limits:
- Hybrid is on: R@10 goes 94% → 100%, and the three complete misses (q008, q014, q036) now reach the top 10.
  Trade-off: a catalogue image can lose first place to a look-alike when its own caption ranks low
  (BR-007_02; self-search 70/71 with hybrid, 71/71 without).
- Query tagging stays off: +4 R@5 points (one query) for about 10× latency. Boosts stay on for text and refinements.
- `colour_mismatch` queries are never pushed down: image queries never boost on colour or metal (all 7 keep their rank).
- "Similar but in silver" moves silver items to the top and keeps the rest; `REFINE_METAL_HARD_FILTER=true`
  shows only silver. Boosts rely on Stage 7 tags, e.g. NK-004 is tagged `black metal` only, so it doesn't get the silver boost.
- Refinements with words outside the vocabulary (e.g. "with grooves") don't change the ranking.

How to run:

```bash
cd backend && pytest -q tests/test_hybrid.py tests/test_query_parsing.py tests/test_search.py
python eval/run_eval.py --backend siglip2_224 --name s8_baseline --set SEARCH_HYBRID=false --set SEARCH_BOOSTS=false
python eval/run_eval.py --backend siglip2_224 --name s8_hybrid
python eval/run_eval.py --backend siglip2_224 --name s8_hybrid_boosts --set SEARCH_QUERY_TAGGING=true
python eval/run_eval.py --backend siglip2_224 --name s8_hybrid_detect --category detect
python eval/run_eval.py --backend siglip2_224 --mode text --name s8_text
python eval/run_eval.py --backend siglip2_224 --mode image_text --name s8_it_image_text
python eval/run_eval.py --compare s8_baseline s8_hybrid s8_hybrid_boosts s8_hybrid_detect s8_text s8_it_image_text
curl -s localhost:8080/search/text -H 'content-type: application/json' -d '{"text":"red ruby ring","debug":true}'
```

### Stage 9 — Colour-reduced and region features
Added:
- `POST /embed/batch` in `ai-service/app.py` (several images per request; same vectors as `/embed`).
- `embed_client.embed_batch()`.
- `to_grayscale`, `to_edge_map` and `make_tiles` in `preprocessing.py`, with their own versions; the main
  `PREPROCESS_VERSION` is unchanged.
- `scripts/build_feature_indexes.py --feature gray|edge|tiles [--grid N]` (resumable).
- `backend/app/search/features.py`, the `FEATURE_*` flags, and `tests/test_features.py`.

Stores (same `vector_id` as the image index):
- gray / edge FAISS indexes: `data/faiss_index/features/<gray|edge>/<model>_<version>/`
- tiles: `data/features/<model>_<version>/<category>/tiles_<g>x<g>.npy` + `_ids.npy` + a manifest per grid.
  The grid is in the file name so 2×2 and 3×3 can sit side by side. Tiles are memory-mapped at startup and
  only used to re-score the top 50.

When any feature flag is on, the query's colour, gray and tile images go to SigLIP in one `/embed/batch`
call. With every flag off the request is exactly the Stage 8 one (`/embed`), and a test checks that.
`debug=true` shows `gray_rank`/`gray_score` and `rerank.{gray,edge,region}.score`.

**Decision (`eval/results/stage9_decision.md`): only the 2×2 region rerank stays on.**

| Config | R@1 | R@5 | MRR@10 | p95 ms |
|---|---:|---:|---:|---:|
| Stage 8 default | 86% | 94% | 0.901 | 607 |
| + gray retrieval / rerank / both | 86 / 84 / 80% | 94% | 0.900 / 0.884 / 0.860 | ~900 |
| **+ region 2×2 (now default)** | **90%** | **96%** | **0.931** | 1873 |
| + region 3×3 | 84% | 96% | 0.895 | 3421 |

- **Gray and edge are off.** colour_mismatch didn't improve (q014 got worse), each costs about 300 ms, and
  edge maps hurt reflection queries.
- **Region 2×2 is on** (weight 0.02, top 50). q008 goes from rank 8 to 1. It adds about 1.2 s per image search
  on CPU; turn it off with `FEATURE_REGION_RERANK=false`.
- **3×3 is off:** worse and slower.
- **Patch tokens were skipped:** more than 1 s or about 575 GB of storage. Details are in the decision note.

How to run:

```bash
python scripts/build_feature_indexes.py --feature gray
python scripts/build_feature_indexes.py --feature tiles --grid 2
cd backend && pytest -q tests/test_features.py && cd ..
python eval/run_eval.py --backend siglip2_224 --name s9_base --set FEATURE_REGION_RERANK=false
python eval/run_eval.py --backend siglip2_224 --name s9_gray_retrieval --set FEATURE_REGION_RERANK=false --set FEATURE_GRAY_RETRIEVAL=true
python eval/run_eval.py --backend siglip2_224 --name s9_region_2x2
python eval/run_eval.py --compare s9_base s9_gray_retrieval s9_region_2x2
```

### Stage 10 — Re-ranking, dedup, tuned weights, VLM judge
Added: `backend/app/search/reranker.py`, `dedup.py`, `vlm_judge.py`, `config/reranker_weights_v1.json` and
`_v2.json`, `scripts/make_eval_splits.py`, `eval/tune_weights.py`, a `split` column in `eval/queries.csv` and
`eval/text_queries.csv`, `run_eval.py --split`, and `tests/test_reranker.py` / `tests/test_vlm_judge.py`.
**Full results: `eval/results/stage10_final.md`.**

- **Reranker:**
  - Signals per candidate: image, caption, gray and region cosines, each mapped to 0–1 by clipping to a fixed
    range (from catalogue statistics, never per query).
  - final = weighted sum (missing signals dropped and weights re-normalized) + capped boosts, clipped to 0–1.
  - Weights come from `config/reranker_weights_<version>.json`, and responses return
    `reranker_weights_version`. `debug=true` shows each signal's raw value, 0–1 value and weight, plus labels
    such as `{"design": "high", "stones": "match"}`.
  - When the reranker is on, `similarity_score` is this 0–1 score; the response labels it as "not a probability".
- **Dedup:** one result per product (its best image). Different products with near-identical images (image cosine
  ≥ 0.98) are both kept, and the lower one gets `possible_duplicate_of` in the debug block.
- **Tuning:** the 36 tune queries only (split seed 42, stratified). The holdout (14 image + 6 text) was run once.
- **Outcome:**
  - The tuned image weights (caption 0.8, region 0.2, image 0) overfit: holdout R@1 86% vs 93% for Stage 9.
    So image search keeps the Stage 9 order (`RERANKER_MODES=text`).
  - For text, holdout R@1 went 83% → 100%.
  - The Stage 6 baseline is already at R@5 100% on this holdout, so it can't be beaten there. On all 50 queries
    the default beats it: R@5 96% vs 94%.
- **VLM judge (`FEATURE_VLM_JUDGE`, off):**
  - Qwen checks the top 5 for "same design", with results cached.
  - Holdout R@1 93% → 100%; all 50: 90% → 94%. No query got worse.
  - Cost: 5 Qwen calls and about +4.5 s p95 per search, so it stays off.

```bash
python scripts/make_eval_splits.py
python eval/tune_weights.py --mode image --new-version v2      # tune split only
python eval/run_eval.py --backend siglip2_224 --split holdout --name s10_holdout
python eval/run_eval.py --backend siglip2_224 --split holdout --name s10_judge --set FEATURE_VLM_JUDGE=true
cd backend && pytest -q tests/test_reranker.py tests/test_vlm_judge.py
```

### Stage 11 — Test frontend (internal POC harness)
A React + Vite page for trying the search API by hand. It is an internal test tool, not the client's UI.

```
frontend/src/api/search.js       searchImage(), searchText(), health(): timeout + readable errors
frontend/src/hooks/useSearch.js  one search at a time: loading / error / result, aborts on Clear
frontend/src/components/         Header (health), UploadBox, CategorySelect, ImageSearchTab, TextSearchTab,
                                 Results, ResultCard, DebugPanel
```

How to run (backend on :8080 and the containers up, see Setup):

```bash
cd frontend
cp .env.example .env
npm install
npm run dev          # http://localhost:5173
npm test             # API client unit tests (vitest)
```

`frontend/.env`:

| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8080` | Backend base URL, no trailing slash |
| `VITE_MAX_UPLOAD_MB` | `10` | Client-side size check; keep equal to the backend's `MAX_UPLOAD_MB` |
| `VITE_SEARCH_TIMEOUT_MS` | `60000` | Search request timeout |
| `VITE_HEALTH_TIMEOUT_MS` | `5000` | `/health` timeout (polled every 15 s) |

**Pointing it at another backend:** set `VITE_API_URL` and restart `npm run dev`, because Vite reads `.env` at
startup. That backend's `CORS_ORIGINS` must include `http://localhost:5173`. The dev server always uses port
5173 (`strictPort`) for this reason. Open the page as `localhost`, not `127.0.0.1`, which is a different origin.

What it does:
- **Image search:** drag-and-drop or pick a JPEG/PNG/WebP (type and size checked before upload), category
  "Auto-detect" or a fixed one, optional refinement text. Shows the detected category and its confidence. On a
  low-confidence answer the top 3 categories become buttons, and clicking one re-runs the search with it.
- **Text search:** text, optional category.
- **Results:** image, product ID and name, category, similarity bar, attribute badges. Attributes that earned a
  boost appear as green ✓ badges, and a "possible duplicate" badge appears for look-alikes; both come from the
  debug block, so they need Debug on.
- **Debug toggle** (sends `debug=true`): a table per result with the image, caption, gray and region signals,
  boosts, the final score and labels, plus the weights version, backend and round-trip latency and the step
  timings. With the default `RERANKER_MODES=text`, image searches are not re-ranked. Their cells then show
  raw cosines with no weights, and the panel says so.
- **Errors:** 400/404/413/415/502/503, timeouts and an unreachable backend each show a readable message. The
  header dot turns red when `/health` is degraded or unreachable.

Checked against the local stack in a headless browser at 1366 px and 390 px (no horizontal scroll):
auto-detect (q001 → rings, 95%), low-confidence choice (q005 → pendants 47% / rings 23% / necklaces 17%),
image + "similar but in silver", text "red ruby ring" with debug (weights v2), the client-side type and size
checks, a 20×20 image (400), Clear, and SigLIP stopped (502 message, header shows embedding_service down).

### Stage 12 Part A — Docker stack and failure handling
Added: `backend/Dockerfile`, `frontend/Dockerfile` + `nginx.conf`, the full `docker-compose.yml` (mongodb,
siglip2-api, backend, frontend, and a `tools` profile for scripts and tests), `.dockerignore` files,
`backend/app/db.py`, `backend/tests/test_failures.py` and `scripts/run_tests.sh`. Every setting is in `.env.example`.

- **Containers:**
  - The backend runs as a non-root user with a `HEALTHCHECK` on `/health`, and waits for healthy MongoDB and SigLIP.
  - Catalogue images, FAISS indexes, feature stores, `categories.txt` and `config/` are mounted read-only into
    the backend. The `tools` service mounts `data/`, `eval/` and `config/` read-write and runs as `HOST_UID`.
  - SigLIP has `SIGLIP_MEM_LIMIT` (3 GB; ~1 GB used) and the backend `BACKEND_MEM_LIMIT` (2 GB).
  - `VITE_API_URL` is a build argument of the frontend image.
- **MongoDB indexes at startup:** `images.product_id`, `images.category`, `images.vector_id` (unique where set)
  and `products.category` (`app/db.py`, shared with `seed_mongo.py`). `image_id` is the image document's `_id`,
  so it is unique already. If MongoDB is down at startup, the backend logs it and still starts.
- **Index check at startup:** each manifest's model, embedding version, dimension (768 for SigLIP) and
  preprocessing version are compared with the settings, and each index file with its manifest entry.
  - A mismatch is logged as `image index for 'X' NOT SERVED: ...`, and that category answers 503 with
    "rebuild it with scripts/ingest.py --rebuild". The other categories are still served.
  - `/health` lists `indexes.served` and `indexes.refused`.
  - The scripts still stop and ask for `--rebuild`.
- **Degradation (never a 500):**

  | Failure | Image search | Text search |
  |---|---|---|
  | SigLIP down | 503 "Embedding service unavailable" (was 502) | 503 |
  | MongoDB down | 503 "Catalogue database is unavailable" | 503 |
  | Qwen down or slower than `LLM_QUERY_TIMEOUT_SECONDS` (20 s) | 200, `reduced_mode: true` and a "Reduced mode: ..." warning; searched without the refinement text, query tags and VLM judge | 503 "The query parser (Qwen) is unavailable or too slow" (was: raw-text fallback) |
  | Qwen answers but the reply is unusable | 200, "could not understand the text" warning | 200, raw text searched (unchanged) |
  | `vector_id` with no MongoDB record | candidate skipped and logged | same |

  Exception handlers turn any uncaught SigLIP, MongoDB or Qwen-unavailable error into the same 503s.
- **Settings:** `KEY=` with an empty value now means "use the default" (`env_ignore_empty`). New settings:
  `LLM_QUERY_TIMEOUT_SECONDS`, `EMBED_DIMENSION` (tests only), and the compose ports and memory limits.
  `SIGLIP_384_MODEL_DIR` is optional; leaving it unset no longer breaks every `docker compose` command.
- **Tests:** `test_failures.py` has 20 tests:
  - SigLIP down: image search, text search and category detection.
  - MongoDB down, including at startup.
  - Qwen down or slow: image search in reduced mode (refinement, query tagging, VLM judge); text search 503.
  - A vector with no MongoDB record.
  - Corrupt, oversized and wrong-type uploads; an empty category index.
  - Mismatched manifests (per file, and dimension / preprocessing / version / model).
  - Running ingest twice: same vector IDs, no duplicates.

  `scripts/run_tests.sh`: backend 190 passed locally; in Docker, 188 passed and 2 skipped (gray index and 3×3
  tiles, which are not built by default). Also scripts + eval 21 and frontend 11.

Fresh-machine test (2026-10-20, this laptop):
- **Setup:** a copy of exactly the files git ships, the weights downloaded with the step 2 command (byte-identical
  to the ones in use), a new MongoDB volume, and `BACKEND_PORT=8080` (8001 is taken here). README steps 1–10
  followed as written.
- **Stack:** up in 74 s, all containers healthy.
- **Catalogue build:** 71 images and 50 queries, 0 validation errors. The first `ingest --verify` timed out on
  7 self-searches (cold weights) and the rerun gave 71/71, as step 6 now says. Tags were 71/71 valid,
  the caption check ok, and 2×2 tiles were built.
- **Checks:**
  - Browser test through nginx: the Stage 11 checklist, all passing.
  - `run_tests.sh` in Docker: all suites passed.
  - `run_eval.py` in Docker: R@1 90%, R@5 96%, R@10 100%, MRR@10 0.934 (Stage 9/10 recorded 90% / 96% / 0.931).
- **Failures on the live stack:**
  - SigLIP stopped: 503 in the UI, and the header shows `embedding_service down`.
  - MongoDB stopped: 503 in 3–4 s, and search recovers when it is back.
  - Qwen URL refused: image + text search returns 200 in reduced mode (2.6 s), text search 503.
  - Qwen URL hanging: the same answers, capped at 20 s.


### Stage 12 Part B — Load test, runbooks, integration notes, final report
Added: `scripts/load_test.py`, `scripts/export_openapi.py`, and `docs/` (final report, integration notes, OpenAPI
file, three runbooks). Error responses are now declared in the OpenAPI schema (400/404/413/415/503; the unused 422
is dropped). Compose rotates container logs (5 × 10 MB).

- **Load test (laptop):**
  - Photo search: p50 1.2 s and ~0.8 searches/s. It saturates at concurrency 5. At 10 and 20, requests queue
    in SigLIP past `EMBED_TIMEOUT_SECONDS` and get readable 503s (10% and 72%).
  - With `FEATURE_REGION_RERANK=false`: 0.33 s and ~3.5 searches/s, with no errors at 20.
  - Text search: 2.1 s, scaling to 2.6 req/s at 10.
  - The slowest step is always `embed` (SigLIP). Details: `eval/results/load_test.md`.
- **Final evaluation (`docs/final_eval_report.md`):**
  - Holdout R@1 86% → 93%, MRR 0.917 → 0.964.
  - All 50: R@1 86% → 90%, R@5 94% → 96%, R@10 94% → 100%.
  - Text: holdout R@1 100%, all 95%.
  - New finding: region rerank lowers photo + text refinements (R@1 62% → 38% on 8 queries). This is an open
    decision in the report.
- **Integration notes:** they flag that the Vertex adapter supports only photo search with a given category
  (no text or batch embedding). The client's ID formats and 7-step pipeline still need their input.
