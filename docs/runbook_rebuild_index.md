# Runbook: full index rebuild

Use this when something that every vector depends on changes. For adding or removing catalogue images, use
[runbook_add_catalogue.md](runbook_add_catalogue.md) instead; that path is incremental.

All commands run from the project root with the stack up (`docker compose up -d`). Each script runs in the
`tools` container, which reads `.env`. If you run the backend locally instead, use `python <script>` with
`backend/.venv` active; the arguments are the same.

## 1. Decide what needs rebuilding

| What changed | Image index (`ingest`) | Tags (`tag_catalogue`) | Caption index | Region tiles / gray | Backend restart |
|---|---|---|---|---|---|
| Embedding model: `EMBED_BACKEND`, or a model/version in `backend/app/embed_backends.py` | yes (new folder) | no | yes | yes | yes, recreate (`up -d`) |
| Preprocessing: any `PREPROCESS_*` setting, or the code in `backend/app/preprocessing.py` | yes, `--rebuild` | no | no | yes, `--rebuild` | yes, recreate (`up -d`) |
| Tagger: `TAGGER_VERSION`, the prompts or the vocabulary in `backend/app/tagging/` | no | yes, `--retag` | yes | no | yes (`up -d --build`, code changed) |
| VLM model: `VLM_MODEL_NAME` | no | yes (tags images not yet tagged by that model) | yes | no | no |
| Tile grid: `FEATURE_TILE_GRID` | no | no | no | tiles for the new grid | yes, recreate (`up -d`) |
| Only `CATEGORIES` in `data/categories.txt` | no | no | no | no | yes (`restart`) |

How to tell that a rebuild is needed: the backend log at startup says
`image index for 'rings' NOT SERVED: ... has preprocess_version='...', current settings give '...'; run with --rebuild`,
and `curl localhost:8001/health` lists those categories under `indexes.refused`. Searches in a refused category
answer 503 "rebuild it with scripts/ingest.py --rebuild". The scripts stop with the same message instead of
writing over the old index.

## 2. Before you start

1. **Back up** MongoDB and the index folders ([runbook_operations.md](runbook_operations.md#backup)). A rebuild
   overwrites `data/faiss_index/` and `data/features/`, and `--retag` overwrites every `vlm_tags`.
2. **Write down the current evaluation numbers** to compare with afterwards. The last run on the sample catalogue
   (`eval/results/final_all_default.md`) was R@1 90%, R@5 96%, R@10 100%, MRR@10 0.93 (image, category given).
   If you have no saved run, take one now:
   ```bash
   docker compose run --rm tools eval/run_eval.py --backend siglip2_224 --name before_rebuild
   ```
3. **Check the time budget.** On the development laptop (CPU), the 71-image sample takes about 3 min to ingest
   (with `--verify`), 2 min to tag (hosted Qwen, 4 in parallel) and under 1 min each for captions and tiles. For
   the client catalogue (~955k images) the same steps take days on CPU; run them on a GPU embedding service.
   Tagging alone was estimated at ~30 days on the shared Qwen server (Stage 7).
4. **Choose a quiet window** for preprocessing changes. From the moment step 3.2 saves the new index until step
   3.6, the running backend still has the old settings, so it refuses the rebuilt categories (503). A model change
   has no such gap, because the new model gets its own index folder. During a tagger change, searches keep working
   without the caption list between 3.4 and 3.6.

## 3. Rebuild, in this order

### 3.1 Change the setting

Edit `.env` (for example `PREPROCESS_SIZE=640`, or `EMBED_BACKEND=siglip2_384`). For a code change (tagger
prompts, vocabulary, `embed_backends.py`), rebuild the image so the tools container gets the new code:
```bash
docker compose build backend
```
The running backend keeps its old settings and code until step 3.6.

A new model also needs its embedding service. For `siglip2_384`, set `SIGLIP_384_MODEL_DIR` and start it with
`docker compose --profile siglip384 up -d siglip2-api-384`.

### 3.2 Image index

```bash
docker compose run --rm tools scripts/ingest.py --rebuild --verify
```
- `--rebuild` clears this backend's vectors and re-embeds every image. Vector IDs stay the same: they are
  permanent per image and shared by all backends.
- For a new model, `--rebuild` is not needed: its folder is new, so everything is embedded anyway.
- The run must end with `self-search: N/N images return themselves first` and exit code 0. If a few
  self-searches time out on a cold SigLIP container, run the same command without `--rebuild`: it re-embeds
  nothing and repeats the check.
- Images that fail (unreadable, too small) are logged as `skipped <image_id>: <reason>`, saved as
  `ingest_errors.<backend>` in MongoDB, and left out. Fix or remove them before continuing.

### 3.3 Tags (only for tagger or VLM changes)

```bash
docker compose run --rm tools scripts/tag_catalogue.py --retag     # tagger change: tag every image again
docker compose run --rm tools scripts/tag_catalogue.py             # VLM model change: tag what is missing
```
- The run ends with `schema check: N/N saved tag records valid (100%)` and `images with vlm_tag_error: 0`.
- It is resumable: if it stops, run the same command without `--retag` and it continues where it stopped.
- Optional manual check of tag quality: `scripts/tag_spotcheck.py export`, mark the CSV, then
  `scripts/tag_spotcheck.py score`.

### 3.4 Caption index

```bash
docker compose run --rm tools scripts/build_caption_index.py
```
It re-embeds every description and must end with `caption check: ok`, with the same count per category as the
image index. Only images tagged with the current `TAGGER_VERSION` get a caption.

### 3.5 Feature stores

```bash
docker compose run --rm tools scripts/build_feature_indexes.py --feature tiles --grid 2 --rebuild
```
- It must end with `check against image index: ok`.
- Use the grid in `FEATURE_TILE_GRID`.
- Build `--feature gray` (or `edge`) only if a `FEATURE_GRAY_*` / `FEATURE_EDGE_*` flag is on; they are off by
  default.
- For a new model, `--rebuild` is not needed.

### 3.6 Switch the backend over

```bash
docker compose up -d backend           # recreates it with the new .env; add --build if code changed
docker compose logs backend | grep -E "startup|NOT SERVED|tile store"
curl -s localhost:8001/health          # "status": "ok"; indexes.refused must be []
```
The log must show `serving <backend> ... with N vectors` with every category, and no `NOT SERVED` lines.
FAISS and caption indexes reload by themselves when their manifest changes. The tile store is only read at
startup, which is why this step is required after 3.5.

### 3.7 Confirm with the evaluation

```bash
docker compose run --rm tools eval/run_eval.py --backend siglip2_224 --name after_rebuild
docker compose run --rm tools eval/run_eval.py --backend siglip2_224 --mode text --name after_rebuild_text
docker compose run --rm tools eval/run_eval.py --compare before_rebuild after_rebuild
```
Use the new backend name in `--backend` after a model change.

- **Same settings, only re-run** (e.g. after restoring data): expect the same numbers, within one query
  (2 points on 50 queries).
- **A setting changed on purpose:** compare overall and per edge case. Keep the change only if it is not worse
  overall. On 50 queries a one-query difference is noise.

## 4. Roll back

- **Model change:** set `EMBED_BACKEND` back and run `docker compose up -d backend`. The old index folder was
  never touched.
- **Preprocessing or tagger change:** set the old values back in `.env` (and `git checkout` the code), restore
  the backup from step 2.1 ([runbook_operations.md](runbook_operations.md#restore)), then run
  `docker compose up -d --build backend`.
