# Runbook: add or remove catalogue images and products

Every step is incremental: only new or changed images are embedded and tagged, and vectors of removed images are
deleted. The backend keeps serving throughout. For a change of model, preprocessing or tagger, use
[runbook_rebuild_index.md](runbook_rebuild_index.md) instead.

All commands run from the project root with the stack up (`docker compose up -d`). The scripts run in the
`tools` container. Locally, use `python <script>` with `backend/.venv` active instead.

The catalogue is described by two CSV files. They are the source of truth: MongoDB and the indexes are rebuilt
from them.

| File | One row per | Columns |
|---|---|---|
| `data/catalogue.csv` | image | `image_id,product_id,project_id,category,subcategory,image_path` |
| `data/products.csv` | product | `product_id,project_id,category,subcategory,name,metal,gemstones,stone_colours,motifs,style,catalogue_sources` |

Rules:
- **`image_id`** is unique across the whole catalogue and never reused. It becomes the MongoDB `_id` and keeps
  the same `vector_id` forever.
- **`image_path`** must be `data/catalogue/<category>/<image_id>.<jpg|png|webp>`.
- **`category`** must be a line of `data/categories.txt`. A new category needs a new line there and
  `docker compose restart backend` after step 4.
- **`subcategory`** may be empty.
- **List columns** (`metal`, `gemstones`, `stone_colours`, `motifs`, `style`, `catalogue_sources`) are
  `;`-separated and may be empty. They are shown as product attributes; search uses the Qwen tags, not these.
- **Every `product_id`** in `catalogue.csv` needs a row in `products.csv`. Otherwise its results have no name or
  attributes.
- **Images:** JPEG, PNG or WebP, at least 128 px on the shorter side, one product per photo.

## Add images or products

1. **Copy the image files** into `data/catalogue/<category>/`, named `<image_id>.<ext>`.
2. **Add the rows**: one line per image in `data/catalogue.csv`, and one line per new product in
   `data/products.csv`. A new photo of an existing product only needs the `catalogue.csv` line.
3. **Validate:**
   ```bash
   docker compose run --rm tools scripts/validate_dataset.py
   ```
   Fix every `ERROR` (duplicate `image_id`, wrong path, unknown category, unreadable or too-small image), then
   run it again until it prints `0 error(s)`. A warning `same file content as image_id ...` means the same photo
   was added twice; remove one unless that is intended.

   Then check that every product has a row:
   ```bash
   docker compose run --rm tools -c "import csv; p={r['product_id'] for r in csv.DictReader(open('data/products.csv'))}; m=sorted({r['product_id'] for r in csv.DictReader(open('data/catalogue.csv'))}-p); print('missing products:', m or 'none')"
   ```
4. **Ingest** (seeds MongoDB from both CSVs, embeds the new images, adds them to the index):
   ```bash
   docker compose run --rm tools scripts/ingest.py
   ```
   Expect `N catalogue images: <old> up to date, <new> to embed, 0 unreadable` and exit code 0. The backend
   picks up the new vectors on its next search, with no restart. A changed file under an existing `image_id` is
   noticed by its hash and re-embedded.
5. **Tag the new images with Qwen** (only images without tags for the current tagger version are sent):
   ```bash
   docker compose run --rm tools scripts/tag_catalogue.py
   ```
   Expect `schema check: N/N saved tag records valid` and `images with vlm_tag_error: 0`.

   If you replaced the file of an existing image, its old tags are kept. Clear them first, then run the command
   above:
   ```bash
   docker exec jewellery-mongodb mongosh --quiet jewellery_search --eval 'db.images.updateOne({_id: "RG-001_01"}, {$unset: {vlm_tags: ""}})'
   ```
6. **Rebuild the caption index** (re-embeds every description, so it takes longer as the catalogue grows):
   ```bash
   docker compose run --rm tools scripts/build_caption_index.py
   ```
   Expect `caption check: ok`. New images without tags (step 5 failed or skipped) are searchable by image only.
7. **Add the region tiles** for the new images:
   ```bash
   docker compose run --rm tools scripts/build_feature_indexes.py --feature tiles --grid 2
   docker compose restart backend          # the tile store is read at startup only
   ```
   Expect `N up to date, <new> to embed` and `check against image index: ok`. Without the restart, new images
   are found but get no region score until the next restart.
8. **Check:**
   ```bash
   curl -s localhost:8001/health          # indexes.served counts include the new images; refused is []
   curl -s -F "file=@data/catalogue/rings/RG-001_01.jpg;type=image/jpeg" -F category=rings -F top_k=3 \
        localhost:8001/search/image       # a new catalogue image should return its own product first
   ```

## Remove images or products

1. **Delete the rows**: remove the images from `data/catalogue.csv`, and the product from `data/products.csv`
   if it has no images left. You can delete the files from `data/catalogue/` too; they are not used once the
   rows are gone.
2. **Validate:** `docker compose run --rm tools scripts/validate_dataset.py` (0 errors).
3. **Ingest:**
   ```bash
   docker compose run --rm tools scripts/ingest.py
   ```
   It deletes the MongoDB documents of removed images and products, and logs `<k> stale vectors removed`.
   Removed products stop appearing in results at once.
4. **Captions and tiles:**
   ```bash
   docker compose run --rm tools scripts/build_caption_index.py
   docker compose run --rm tools scripts/build_feature_indexes.py --feature tiles --grid 2
   docker compose restart backend
   ```
5. **Removing a whole project:** if no row of a `project_id` is left in the CSVs, ingest does not know that
   project any more, so its MongoDB documents stay behind. They are never returned (they have no vectors), but
   delete them to keep the database clean:
   ```bash
   docker exec jewellery-mongodb mongosh --quiet jewellery_search --eval 'const p = "old_project"; printjson([db.images.deleteMany({project_id: p}).deletedCount, db.products.deleteMany({project_id: p}).deletedCount])'
   ```

## If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `skipped <image_id>: ...` in the ingest log | Unreadable or too-small image. Fix the file and run `ingest.py` again; only that image is retried. |
| Ingest stops with `... run with --rebuild` | The index was built with other settings. See [runbook_rebuild_index.md](runbook_rebuild_index.md). |
| Tagging reports `vlm_tag_error` | Qwen timed out or returned invalid JSON twice. Run `tag_catalogue.py` again; only failed images are retried. |
| `caption check: FAILED` | Some images have no tags yet. Finish step 5, then run step 6 again. |
| A new image never ranks first for its own photo | Check the photo itself (crop, lighting). Run `ingest.py --verify` to test every catalogue image. |
