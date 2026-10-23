# Jewellery visual search: final evaluation report (POC)

Status of the proof of concept at the end of Stage 12 (2026-10-20): how accurate and how fast it is, what is
switched on and why, and what is not yet known. Every number below comes from a run that can be repeated with
the commands in the appendix. Where the test set is too small to support a conclusion, the report says so.

## 1. Summary

- **On our test photos, the final system finds the right product first more often than the Stage 6 baseline.**
  - Over all 50 photo queries: **R@1 86% → 90%**, **R@5 94% → 96%**, **R@10 94% → 100%** and
    **MRR@10 0.888 → 0.931** (category given).
  - On the 14-query holdout, which was never used for tuning: **R@1 86% → 93%**, MRR 0.917 → 0.964. R@5 is
    100% for both, so it cannot show a difference there.
- **Text search** ("red ruby ring") gets the right product first for 19 of 20 queries (R@1 95%, MRR 0.956),
  and for 6 of 6 on its holdout.
- **Speed on one laptop (CPU):** about 1.2 s per photo search and 2.1 s per text search. The laptop sustains
  ~0.8 photo searches per second; the embedding model is the only bottleneck (section 4).
- **The test set is small and partly synthetic:** 71 catalogue images of 65 products from a public dataset, and
  50 queries made by transforming those photos. One query is 2 points (photos) or 7–17 points (holdout).
  **These numbers show the approach works on this data. They do not predict accuracy on the client's 955k-image
  catalogue.** That needs real query photos and the real catalogue (section 6).

## 2. Test setup

| | |
|---|---|
| Catalogue | 71 images, 65 products, 5 categories (bracelets 15, earrings 15, necklaces 13, pendants 13, rings 15); public "AI Tool Pool Jewelry Vision" dataset (CC BY 4.0) |
| Photo queries | 50, each a different photo of a catalogue product with one edge case applied: angle, background, colour mismatch, distance, lighting, near-duplicate, reflection, rotation (7–10 each; 2 are real separate photos) |
| Text queries | 20 descriptions (colour, gemstone, metal, motif, style; 4 name no category) |
| Photo + text | 8 photos with a refinement such as "similar but in silver" |
| Split | Fixed tune / holdout split (seed 42, stratified by edge case): photos 36 / 14, text 14 / 6. Weights were tuned on the tune split only. |
| Correct match | A query is a hit at k if the expected product is among the top k **products** (one result per product). Near-duplicate look-alikes do **not** count. |
| Metrics | Recall@1/5/10 (share of queries with a hit at k), MRR@10 (mean of 1/rank of the expected product, 0 if not in the top 10) |
| Hardware | Development laptop: Intel i5-1334U (12 threads), 16 GB RAM, no GPU. Everything in Docker; Qwen3-VL-8B on the remote shared GPU server. |
| Runs | 2026-10-20, final Docker stack; `eval/results/final_*.md` |

"Stage 6 baseline" means SigLIP 2 image similarity only: hybrid search, boosts, region rerank and the reranker
all off. Re-run today, it gives exactly the Stage 6 numbers (R@1 86%, MRR 0.888).

## 3. Accuracy

### 3.1 Photo search, category given

| Run | n | R@1 | R@5 | R@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Stage 6 baseline, holdout | 14 | 86% | 100% | 100% | 0.917 |
| **Final, holdout** | 14 | **93%** | 100% | 100% | **0.964** |
| Stage 6 baseline, all queries | 50 | 86% | 94% | 94% | 0.888 |
| **Final, all queries** | 50 | **90%** | **96%** | **100%** | **0.931** |

The "all queries" rows include the 36 tune queries, which were used to choose features and weights, so they are
somewhat optimistic. The holdout is clean, but with 14 queries one query moves R@1 by 7 points.

Per edge case, R@1 / R@5:

| Edge case | Baseline, holdout | Final, holdout | Baseline, all | Final, all |
|---|---:|---:|---:|---:|
| angle | 50% / 100% (2) | 50% / 100% (2) | 71% / 100% (7) | 71% / 100% (7) |
| background | 100% / 100% (2) | 100% / 100% (2) | 86% / 86% (7) | 86% / 86% (7) |
| colour mismatch | 100% / 100% (2) | 100% / 100% (2) | 86% / 86% (7) | 86% / 86% (7) |
| distance | 50% / 100% (2) | **100% / 100%** (2) | 75% / 100% (8) | **88%** / 100% (8) |
| lighting | 100% / 100% (2) | 100% / 100% (2) | 88% / 88% (8) | **100% / 100%** (8) |
| near-duplicate | 100% / 100% (1) | 100% / 100% (1) | 80% / 100% (10) | 80% / 100% (10) |
| reflection | 100% / 100% (2) | 100% / 100% (2) | 100% / 100% (7) | 100% / 100% (7) |
| rotation | 100% / 100% (2) | 100% / 100% (2) | 100% / 100% (8) | 100% / 100% (8) |

Per category, R@1 / R@5:

| Category | Baseline, holdout | Final, holdout | Baseline, all | Final, all |
|---|---:|---:|---:|---:|
| bracelets | 100% / 100% (4) | 100% / 100% (4) | 90% / 90% (10) | 90% / 90% (10) |
| earrings | 100% / 100% (4) | 100% / 100% (4) | 82% / 82% (11) | **91% / 91%** (11) |
| necklaces | 33% / 100% (3) | **67%** / 100% (3) | 50% / 100% (6) | **67%** / 100% (6) |
| pendants | 100% / 100% (1) | 100% / 100% (1) | 100% / 100% (9) | 100% / 100% (9) |
| rings | 100% / 100% (2) | 100% / 100% (2) | 93% / 100% (14) | 93% / 100% (14) |

Where the gains come from:
- **Distance and lighting:** the region rerank compares the centre and 2×2 tiles of the photo, so a small or
  badly lit product is matched on its details.
- **R@10 94% → 100%:** hybrid search with the Qwen captions brings the three complete misses of the baseline
  (q008, q014, q036) into the top 10.
- **No change on angle, background, colour mismatch or near-duplicates.**
- **Necklaces stay the weakest category.**

### 3.2 Photo search without a category

The user does not pick a category; SigLIP detects it, and below 50% confidence the user is asked to choose
from the top 3.

| Run | n | R@1 | R@5 | R@10 | MRR@10 | Category correct |
|---|---:|---:|---:|---:|---:|---:|
| Final, category detected | 50 | 80% | 82% | 84% | 0.809 | 84% (42/50) |

- **4 queries got "please choose a category"** (q005, q006, q036, q046), and the right one was among the 3
  offered every time. They count as misses here. In the app the user picks and the search runs with that
  category.
- **4 were detected wrongly and then missed:** q009 and q029 (necklace → bracelets), q018 (pendant → necklaces),
  q047 (necklace → pendants).
- **Necklaces are hit hardest:** R@1 17% with detection, against 67% with the category given.
- For production, **asking for the category, or taking it from the client's UI context, is clearly better**
  than relying on detection for necklaces and pendants.

### 3.3 Text search

| Run | n | R@1 | R@5 | R@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Stage 8 text (no reranker), holdout | 6 | 83% | 100% | 100% | 0.889 |
| **Final (reranker v2), holdout** | 6 | **100%** | 100% | 100% | **1.000** |
| Stage 8 text, all | 20 | 85% | 95% | 100% | 0.883 |
| **Final, all** | 20 | **95%** | 95% | 100% | **0.956** |

- The only miss is t18, "evil eye with an elephant" (expected BR-005, rank 8). It names no category, so all
  categories were searched.
- The category came from the text in 16 of 20 queries (always correct); the other 4 search everything. Those
  4 had R@1 50%.
- Text search was added in Stage 8, so there is no Stage 6 baseline for it.

### 3.4 Photo + refinement text

| Run | n | R@1 | R@5 | R@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Final (region rerank on) | 8 | 38% | **100%** | 100% | 0.656 |
| Same, region rerank off (= Stage 8) | 8 | **62%** | 88% | 100% | **0.771** |

- **The region rerank, added in Stage 9 and tested then only on photo queries, works against refinements.** It
  re-scores by visual similarity to the photo, while the refinement asks for something that looks different
  ("in silver", "pink swirl glass instead"). Two queries drop from rank 1 to 2 (it03, it04) and one improves
  (it06, 6 → 4).
- With 8 queries this is a 2-query difference, too few for a firm conclusion. But the direction makes sense.
  It is listed as an open decision (section 7). "Similar but in silver" works either way (it07 and it08 are
  rank 1).

### 3.5 Tag quality (Stage 7)

- Qwen tags feed the caption search and the boosts.
- Agreement with the hand-written product sheet (automatic, not ground truth):
  - category 78%;
  - metal 59% exact, 76% overlap;
  - gemstones 31% / 41%;
  - stone colours 40% / 57%;
  - motifs 24% / 70%;
  - style 40%.
- The manual spot-check of 50 images (`eval/tag_spotcheck.csv`) **has not been marked yet**, so tag accuracy
  is not measured.

## 4. Latency, throughput and Qwen calls

Load test on the laptop, with the Docker stack, real query photos and the category given
(`eval/results/load_test.md`):

| Search | Concurrency | p50 | p95 | Errors | Throughput |
|---|---:|---:|---:|---:|---:|
| Photo | 1 | 1.22 s | 1.30 s | 0% | 0.81 /s |
| Photo | 5 | 5.0 s | 8.6 s | 0% | 0.82 /s |
| Photo | 10 | 12.5 s | 30.5 s | 10% (503) | 0.51 /s |
| Photo | 20 | 30.2 s | 31.9 s | 72% (503) | 0.18 /s |
| Photo, region rerank off | 1 / 10 / 20 | 0.33 / 1.8 / 5.9 s | 0.37 / 6.2 / 7.3 s | 0% | 3.0–3.6 /s |
| Text | 1 / 5 / 10 | 2.1 / 2.5 / 2.9 s | 5.0 / 2.8 / 5.8 s | 0% | 0.35 / 1.7 / 2.6 /s |
| Photo + text (from the evaluation, one at a time) | 1 | 1.9 s | 2.2 s | 0% | |

- **Where the time goes:**
  - Photo search spends ~1.15 s of its 1.2 s in the embedding service. It embeds 6 images per search: the
    photo, its centre crop and 4 tiles.
  - Text search spends ~2 s waiting for Qwen.
  - FAISS search, database and ranking take ~20 ms together.
- **Why it saturates:** the embedding service handles one request at a time on all CPU cores. Above ~0.8
  photo searches/s, requests queue. After 30 s they are answered with a readable 503 ("Embedding service
  unavailable"), never a server error.
- **Production needs a GPU (or several replicas) for the embedding service.** These are laptop numbers; repeat
  the load test on the target server.

Qwen calls per query (remote server):

| Query | Qwen calls | Note |
|---|---:|---|
| Photo search | **0** | Query tagging and the VLM judge are off |
| Photo + text | 1 | Parse the refinement; a second call only if the first reply is not valid JSON |
| Text search | 1 | Same |
| Photo, with `FEATURE_VLM_JUDGE` on | +5 | Same-design check of the top 5; +3.5 s p50 |
| Catalogue image, offline | 1 (+1 retry) | Tagging; 71 images in ~2 min with 4 in parallel |

## 5. What is switched on, and why

| Feature | State | Why |
|---|---|---|
| SigLIP 2 Base 224 embeddings | on | Stage 6: same accuracy as SigLIP 2 at 384 px, at less than half the latency. Vertex AI was not tested (no credentials). |
| Hybrid search: image + Qwen caption indexes, fused with RRF | on | Stage 8: R@10 94% → 100% on all 50 |
| Tag boosts (small bonus for matching tags, never removes results) | on | Needed for text and refinements ("in silver"). Photo queries never boost on colour or metal, so colour-shifted photos are not pushed down. |
| Region rerank (centre + 2×2 tiles, top 50) | on | Stage 9: R@1 86% → 90% on all 50 (distance, lighting). Costs ~0.9 s per photo search and lowers refinements (3.4). |
| Reranker, weights **v2**, text search only | on for text | Stage 10: text holdout R@1 83% → 100%. The tuned photo weights did worse on holdout (R@1 93% → 86%), so photo search keeps the Stage 9 order. |
| Category detection with "ask the user" below 50% | on | 84% correct; when it asks, the right category was always among the 3 |
| Near-duplicate marking (image similarity ≥ 0.98) | on with the reranker, so text search only | Look-alike products are both shown, and the lower one is flagged (`debug` block) |
| Qwen tagging of the query photo | off | +1 query at R@5 for ~10× the latency (Stage 8) |
| Qwen same-design judge (top 5) | off | Holdout R@1 93% → 100% and +2 of 50, but +3.5 s and 5 Qwen calls per search (Stage 10) |
| Gray / edge features | off | No gain on colour mismatch; ~0.3 s each (Stage 9) |

Settings are in `.env.example`. The reranker weights are in `config/reranker_weights_v2.json`, and responses
report the version used.

## 6. Known limitations and failed cases

**Data.**
- 71 catalogue images from a public dataset, with automatic contrast and brightness changes and some rotated
  images with black corners.
- The queries are mostly synthetic transforms of similar photos.
- A real shop catalogue has many more near-identical designs per category, so ranking will be harder and R@1
  lower. This has to be measured on the client's data before any accuracy target is agreed.

**Photo queries that still fail** (final system, category given; strips in `eval/results/final_all_default_failures/`):

| Query | Edge case | Expected | Rank | What goes wrong |
|---|---|---|---:|---|
| q036 | background | BR-013 | 10 | Dark textured background and overexposure; the top results are other dark-background bracelets |
| q014 | colour mismatch | ER-005 | 10 | Emerald shifted to pink; colour outweighs shape, and no pink-tag boost applies to photos |
| q038 | distance, near-duplicate | RG-002 | 3 | Small ring in the frame; the look-alike RG-012 ranks higher |
| q029 | angle | NK-002 | 2 | Perspective-warped necklace; a similar layered chain ranks first |
| q047 | angle, near-duplicate | NK-010 | 2 | Tilted necklace; the look-alike NK-004 ranks first |

**Other limits:**
- **Near-duplicates:** look-alike products (RG-002/RG-012, NK-004/NK-010) are the main cause of rank-2 results.
  Even the Qwen judge called RG-012 the "same design" as RG-002. The business rule for these is open (section 7).
- **Necklaces:** weakest category (R@1 67% with the category given, 17% with detection).
- **Refinement words outside the tag vocabulary** ("with grooves", "on a black cord") cannot boost anything;
  they only change the text embedding.
- **Tags:** motif tagging agrees with the product sheet only 24% exactly. The motif list is provisional, and the
  tag spot-check is not marked.
- **Holdout used for decisions:** choosing "reranker for text only" used the holdout once, so the final holdout
  numbers are slightly optimistic. A fresh query set is needed for a clean final number.
- **Vertex AI:** not evaluated. The adapter only supports photo search with a given category (see
  [integration_notes.md](integration_notes.md#5-vertex-ai-vs-siglip-stage-6)).
- **Scale:** tested with 71 images. Exact FAISS search, memory and index build times at 955k images are
  estimated, not measured. Tagging 955k images on the shared Qwen server would take ~30 days at the measured
  rate.
- **Not production-ready:** no authentication, rate limits or HTTPS
  ([integration_notes.md](integration_notes.md#6-before-production)).

## 7. Open decisions for the client

1. **Business rule for a correct match.** Today only the exact product counts. Should a near-identical design,
   or the same design in another metal or size (another SKU), count as correct? This changes the target metric
   and how results are grouped (one per product vs one per design).
2. **Motif vocabulary sign-off.** The motif list (`backend/app/tagging/vocab.py`) was built from the 65 sample
   products and is marked provisional. The client should review and extend it for their catalogue. Any change
   means re-tagging (`tag_catalogue.py --retag`) and rebuilding captions.
3. **Where Qwen runs.**
   - Today it is the shared dev GPU server, used for text and refinement parsing (~2 s per query) and offline
     tagging.
   - Options: a dedicated GPU server (needed to tag 955k images in reasonable time), a hosted API, or limiting
     Qwen to offline tagging and dropping text search.
   - The VLM judge (+7 points R@1 on holdout) is only affordable with dedicated capacity.
4. **Region rerank for refinements.** Keep it on for photo-only queries, and switch it off when refinement text
   is given (Stage 8 behaviour: R@1 62% vs 38% on 8 queries)? This is a small code change, to be confirmed on
   more refinement queries.
5. **Embedding model.** Keep SigLIP (self-hosted, needs a GPU for throughput) or Vertex AI (their existing
   vectors; adapter work and an evaluation needed first).
6. **Category from the UI.** Detection is 84% correct and weakest on necklaces and pendants. Can the client's
   UI pass the category, or ask the user, before searching?
7. **Evaluation on real data.** We need about 200 real query photos with known correct products, plus the
   matching part of their catalogue, before any accuracy commitment.

## Appendix: reproduce

With the stack up (README "Run with Docker"), from the project root:

```bash
E="docker compose run --rm tools eval/run_eval.py --backend siglip2_224"
BASE="--set SEARCH_HYBRID=false --set SEARCH_BOOSTS=false --set FEATURE_REGION_RERANK=false --set SEARCH_RERANKER=false"
$E --split holdout --name final_ho_baseline $BASE
$E --split holdout --name final_ho_default
$E --name final_all_baseline $BASE
$E --name final_all_default
$E --name final_all_detect --category detect
$E --mode text --split holdout --name final_ho_text_s8 --set SEARCH_RERANKER=false
$E --mode text --split holdout --name final_ho_text
$E --mode text --name final_all_text_s8 --set SEARCH_RERANKER=false
$E --mode text --name final_all_text
$E --mode image_text --name final_image_text
$E --mode image_text --name final_image_text_noregion --set FEATURE_REGION_RERANK=false
$E --compare final_ho_baseline final_ho_default
python scripts/load_test.py --url http://localhost:8001 --machine "laptop"
```
