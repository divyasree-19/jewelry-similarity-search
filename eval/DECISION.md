# Stage 6 decision: embedding model

**Decision: build on SigLIP 2 Base 224 (`EMBED_BACKEND=siglip2_224`).** On our 50-query set the 384 model
gives the same accuracy at about 2.3× the embed latency. Vertex AI was not tested because there are no
credentials on this machine. Stages 7–10 must beat **R@1 86% / MRR@10 0.888** to be worth keeping.

## Results (same 50 queries, 71 catalogue images, 65 products)

| Config | dim | R@1 | R@5 | R@10 | MRR@10 | embed p50 | request p50 | container peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **SigLIP 2 Base 224** | 768 | **86%** | 94% | 94% | **0.888** | **463 ms** | **521 ms** | 1.41 GiB |
| SigLIP 2 Base 384 | 768 | 84% | 94% | 96% | 0.880 | 1092 ms | 1164 ms | 1.43 GiB |
| Vertex AI multimodalembedding@001 | 1408 | not run | | | | | | |

Latency is on CPU (12 cores), one request at a time, after a warm-up query. Memory is the container's
cgroup peak since a fresh start, and is mostly model weights. The FAISS search itself takes under 1 ms.

**Why not 384:** 384 was expected to help with fine stone detail. It did better on `angle` and `distance`
(+1 query each) but worse on `rotation` (−2) and `reflection` (−1). Overall it got one query fewer right at
rank 1 and one more right in the top 10. On 50 queries one query is 2 points, so the two models are tied,
and 224 is clearly cheaper. Re-test 384 when the real client catalogue is available: its photos are larger
and more alike than this sample set, which is where finer input resolution should matter.

**Vertex:** the adapter is built (`EMBED_BACKEND=vertex`, own index directory) but untested against the
real API. To run it: set `VERTEX_PROJECT` and `VERTEX_ACCESS_TOKEN`, then run
`python scripts/ingest.py --backend vertex` and `python eval/run_eval.py --backend vertex`. The client's
stored Vertex vectors cover their own products, not this sample catalogue, so they cannot be compared on
this query set.

## Failing edge cases (SigLIP 224): these drive Stages 7–10

| Tag | R@1 | R@5 | Failures | What goes wrong |
|---|---:|---:|---|---|
| angle | 71% | 100% | q029 (rank 2), q047 (rank 3) | Perspective-warped necklace inside a black frame: a similar layered silver chain (q029) or a plain chain on a textured background (q047) ranks higher. |
| distance | 75% | 100% | q038 (rank 4), q040 (rank 3) | Product is small in the frame; the background dominates the embedding. |
| near_duplicate | 80% | 100% | q038, q047 | Look-alikes (RG-002/RG-012, NK-004/NK-010) win when the query is also small or tilted. |
| background | 86% | 86% | **q036 (miss)** | Dark textured background plus overexposure: top results are all dark-background bracelets, not the bead chain. |
| colour_mismatch | 86% | 86% | **q014 (miss)** | Emerald shifted to pink: nothing with the same shape in the top 10. Colour outweighs shape. |
| lighting | 88% | 88% | **q008 (miss)** | Warm-lit query vs. a badly underexposed catalogue image of ER-014 (a dataset artifact on the catalogue side). |
| reflection, rotation | 100% | 100% | none | |

By category, **necklaces are weakest (R@1 50%, 3 of 6 failing)**, then earrings (82%).

What this suggests for the next stages:

1. **Crop to the product before embedding** (detection or a tight crop): this addresses distance and
   angle, and part of background. Most of the "rank 2–4" failures are this.
2. **Re-rank the top candidates on shape/design** (e.g. a VLM or a second-stage comparison): the right
   product is in the top 5 for 47/50 queries, so re-ranking alone could take R@1 from 86% towards 94%.
3. **Colour-robust matching** for colour_mismatch: compare a greyscale or edge view as well as the colour
   embedding, or let attributes (metal, stone colour) be a soft filter rather than part of the embedding.
4. **Catalogue image quality**: the source images were auto-contrasted and brightness-augmented, and some
   are rotated with black corners or underexposed (ER-014). Clean catalogue photos would remove failures
   like q008 without any model change.

Stage 3 background removal is not an option for (1) or for backgrounds: it clipped ring shanks and thin chains.

## Limits

- 50 queries, 7–10 per tag: a difference of one query moves a tag by 10–14 points. Treat tag-level
  differences of one query as noise.
- The edge cases are synthetic transforms of catalogue-like photos, except q006 and q030.
- The category is given with each query, so these numbers do not include category errors.

Reproduce:

```bash
docker compose --profile siglip384 up -d        # SIGLIP_384_MODEL_DIR must point at the 384 weights
python scripts/ingest.py --backend siglip2_224 && python scripts/ingest.py --backend siglip2_384
python eval/run_eval.py --backend siglip2_224 && python eval/run_eval.py --backend siglip2_384
python eval/run_eval.py --compare siglip2_224 siglip2_384
```

Full per-query results and failure strips: `eval/results/<run>.json`, `<run>.md`, `<run>_failures/` (git-ignored).
