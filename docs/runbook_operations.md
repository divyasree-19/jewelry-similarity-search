# Runbook: day-to-day operations

Commands run from the project root. Container names: `jewellery-backend`, `siglip2-api`, `jewellery-mongodb`,
`jewellery-frontend`. The backend's host port is `BACKEND_PORT` in `.env` (8001 by default, used below).

## Check the system

```bash
docker compose ps                      # mongodb, siglip2-api, backend: (healthy); frontend: Up (healthy)
curl -s localhost:8001/health          # HTTP 200 and "status": "ok"; HTTP 503 and "degraded" names the failing dependency
```

`/health` reports:
- **`dependencies.mongodb`** and **`dependencies.embedding_service`** (SigLIP), each `ok` or `error` with the
  reason.
- **`indexes.served`**: vectors per category.
- **`indexes.refused`**: categories whose index does not match the settings. It must be empty; see
  [runbook_rebuild_index.md](runbook_rebuild_index.md).

Qwen is not part of `/health`: it is optional for image search. To test it, run a text search:
```bash
curl -s localhost:8001/search/text -H 'content-type: application/json' -d '{"text":"red ruby ring","top_k":3}'
```
An HTTP 503 "The query parser (Qwen) is unavailable or too slow" means Qwen is down.

What users see when a dependency fails:

| Failing | Image search | Text search | What to do |
|---|---|---|---|
| SigLIP | 503 "Embedding service unavailable" | 503 | `docker compose logs siglip2-api`; `docker compose restart siglip2-api` (ready in ~15–60 s) |
| MongoDB | 503 "Catalogue database is unavailable" | 503 | `docker compose logs mongodb`; `docker compose restart mongodb`. Search recovers by itself. |
| Qwen (remote) | works, `reduced_mode: true` (no refinement text, query tags or VLM judge) | 503 | Check `VLM_SERVICE_URL` / the key, and the GPU server. Nothing to restart here. |
| One category's index | 503 for that category, others work | that category is skipped, with a warning | [runbook_rebuild_index.md](runbook_rebuild_index.md) |

## Logs

```bash
docker compose logs -f backend                          # follow
docker compose logs --since 1h backend | grep ERROR     # errors in the last hour
docker compose logs backend | grep <query_id>           # one search: query_id is in every response
```

Useful lines:

| Line | Meaning |
|---|---|
| `query <id> mode=image category=rings ... timings_ms={...} results=[...]` | One search: per-step milliseconds and the image IDs returned |
| `startup: serving siglip2_224 ... with 71 vectors: {...}` | Indexes loaded at startup |
| `image index for 'X' NOT SERVED: ...` / `categories NOT served` | Index built with other settings; that category answers 503 |
| `text parsing failed` / `query tagging failed` / `VLM judge: n of 5 calls failed` | Qwen unavailable or slow; the search ran in reduced mode |
| `embedding service error` / `MongoDB error` / `query parser unavailable` | A dependency failed; the client got a 503 |
| `vector_id N has no matching MongoDB image; re-run ingest` | Index and database out of step; run `scripts/ingest.py` |
| `reranker weights ... unusable (...); using the Stage 9 order` | Bad weights file; see "Reranker weights" below |

Logs rotate at 5 files of 10 MB per container (`x-logging` in `docker-compose.yml`), so older lines are dropped.
Uploaded images are not stored unless `SAVE_QUERY_IMAGES=true`.

## Restart services

| What changed | Command |
|---|---|
| Nothing; a service is stuck | `docker compose restart backend` (or `siglip2-api`, `mongodb`) |
| A value in `.env` | `docker compose up -d`. A plain `restart` does **not** re-read `.env`. |
| Backend code, `scripts/`, `eval/` | `docker compose up -d --build backend` |
| `VITE_API_URL` or frontend code | `docker compose up -d --build frontend` |
| Region tiles rebuilt, or `data/categories.txt` changed | `docker compose restart backend` |
| FAISS or caption index rebuilt | nothing: reloaded on the next search when its `manifest.json` changes |
| Weights file added (`config/`) | nothing for the file itself; switching version is a `.env` change (above) |

Everything: `docker compose down` then `docker compose up -d`. The MongoDB volume is kept. **`docker compose
down -v` deletes the database.**

## Reranker weights

The reranker scores text searches with `config/reranker_weights_<version>.json`; the version is
`RERANKER_WEIGHTS_VERSION` in `.env` (now `v2`). Image search does not use it (`RERANKER_MODES=text`, see
[final_eval_report.md](final_eval_report.md)). Text search responses include `reranker_weights_version`.

**Never edit the file of the version in use.** The backend reloads a changed file at once, so a mistake goes
live immediately. Always make a new version:

1. **Create `v3`.**
   - Either tune it on the tune split:
     ```bash
     docker compose run --rm tools eval/tune_weights.py --mode text --base-version v2 --new-version v3
     ```
   - Or copy and edit: `cp config/reranker_weights_v2.json config/reranker_weights_v3.json`, then set
     `"version": "v3"` inside the file and change `modes.<mode>.weights`.
   - Weights per mode must be non-negative and sum to 1, and every weighted signal needs a `ranges` entry.
2. **Compare on holdout before switching** (the backend is not touched yet):
   ```bash
   docker compose run --rm tools eval/run_eval.py --backend siglip2_224 --mode text --split holdout --name text_v2
   docker compose run --rm tools eval/run_eval.py --backend siglip2_224 --mode text --split holdout --name text_v3 --set RERANKER_WEIGHTS_VERSION=v3
   docker compose run --rm tools eval/run_eval.py --compare text_v2 text_v3
   ```
   Switch only if v3 is not worse on R@5 and MRR@10. Each holdout query is 17 points on 6 text queries, so
   check the full set (no `--split`) as well. Using the holdout to decide makes it slightly optimistic next time.
3. **Switch:** set `RERANKER_WEIGHTS_VERSION=v3` in `.env`, then `docker compose up -d backend`.
4. **Verify:** run the text search curl above. The response must have `"reranker_weights_version": "v3"`.
5. **Roll back:** set `v2` again and run `docker compose up -d backend`. Keep old weight files; they are the
   record of what ran.

If the file is invalid (weights do not sum to 1, or the version inside does not match the name), the backend
logs `reranker weights ... unusable` and falls back to the Stage 9 order. Searches keep working.

## Backup

Backs up MongoDB (catalogue metadata, Qwen tags, vector IDs), the indexes and feature stores, the weights, and
the catalogue. For the sample catalogue it takes ~2 s and ~20 MB (most of it the catalogue images). Keep backups
outside the project folder.

```bash
B=~/jewellery-backups/$(date +%F_%H%M) && mkdir -p "$B"
docker exec jewellery-mongodb mongodump --db jewellery_search --archive --gzip > "$B/mongo.archive.gz"
tar czf "$B/indexes.tar.gz" data/faiss_index data/features config
tar czf "$B/catalogue.tar.gz" data/catalogue data/catalogue.csv data/products.csv data/categories.txt
ls -lh "$B"
```

Back up MongoDB and the indexes together: vector IDs in the indexes must match the image documents. Do not run
a script that writes (ingest, tagging, index builds) during the backup. The catalogue archive is only needed if
the images are not kept elsewhere.

## Restore

```bash
B=~/jewellery-backups/2026-10-20_1200        # the backup folder to restore
docker compose stop backend
docker exec -i jewellery-mongodb mongorestore --archive --gzip --drop --nsInclude 'jewellery_search.*' < "$B/mongo.archive.gz"
rm -rf data/faiss_index data/features && tar xzf "$B/indexes.tar.gz"      # also restores config/
tar xzf "$B/catalogue.tar.gz"                                             # only if the images were lost
docker compose start backend
curl -s localhost:8001/health                                             # "ok"; same indexes.served counts as before
docker compose run --rm tools scripts/ingest.py --verify                  # "0 to embed" and N/N self-search
```

`--drop` replaces the collections, so anything written after the backup is lost. To look inside a backup without
touching the live database, restore it under another name:
```bash
docker exec -i jewellery-mongodb mongorestore --archive --gzip --nsFrom 'jewellery_search.*' --nsTo 'restore_check.*' < "$B/mongo.archive.gz"
docker exec jewellery-mongodb mongosh --quiet restore_check --eval 'db.images.countDocuments()'
docker exec jewellery-mongodb mongosh --quiet restore_check --eval 'db.dropDatabase()'
```
