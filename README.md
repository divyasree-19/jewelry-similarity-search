# Jewellery visual search

Upload a jewellery photo, get catalogue items with the same design. Built in stages (0–12);
see the implementation doc for the full plan.

## Layout

```
ai-service/   SigLIP 2 embedding API (FastAPI, Docker) — copied from ~/siglip2-base
backend/      Search backend (FastAPI, run locally in a venv)
frontend/     React + Vite test harness (Stage 11, empty for now)
data/         Catalogue images and FAISS indexes (git-ignored)
scripts/      Ingestion / maintenance scripts
eval/         Evaluation sets and results
```

## Setup

```bash
cp .env.example .env          # set SIGLIP_MODEL_DIR to your local model folder (.env.backend also works for the backend)
docker compose up -d          # MongoDB 8.2 on 27017, siglip2-api on 8000
docker compose ps             # both should show (healthy); SigLIP takes ~15 s to load

cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

All URLs, ports and paths come from `.env` (see `.env.example`). Model weights are mounted
into the container read-only and are never copied into the repo or image.

## Ports

| Service | Port |
|---|---|
| siglip2-api | 8000 |
| (in use by another project: paddleocr) | 8001 |
| VLM service, Stage 7 | 8002 |
| backend | 8080 |
| MongoDB | 27017 |

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
