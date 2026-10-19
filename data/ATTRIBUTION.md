# Sample dataset attribution

The sample catalogue and query images are built from **AI Tool Pool Jewelry Vision**
(Roboflow Universe, https://universe.roboflow.com/work-ypnn9/ai-tool-pool-jewelry-vision),
licensed under **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/).

Downloaded from the Hugging Face mirror `Tarunhugging/ai-tool-pool-jewelry-vision`,
revision `71e72b548a2a2b44d73b40e4b0999b3e74f1b3d4`.

Changes: products were selected and given new IDs; product metadata (metal, gemstones, colours,
motifs, style) was written by hand from the images; query images were generated with rotation,
lighting, colour, reflection, perspective, background and scale transforms
(`scripts/build_sample_dataset.py`).

This is stand-in data for development. Replace it with the client catalogue for real evaluation.
