"""Per-category FAISS indexes of VLM caption (description) embeddings.

Same layout and vector_id mapping as the image indexes (app/vector_index.py), so a caption hit and an image
hit with the same vector_id are the same catalogue image. Stored under
FAISS_INDEX_DIR/captions/<model>_<version>/, where <model> is the SigLIP backend whose text encoder embedded
the captions. The manifest's preprocess_version records the tagger version, "captions:<TAGGER_VERSION>",
so captions from another prompt or vocabulary are refused until the index is rebuilt.
"""

from pathlib import Path

from app.vector_index import CategoryIndexes, index_dir_name


def caption_root(settings) -> Path:
    return Path(settings.faiss_index_dir) / "captions"


def caption_source(tagger_version: str) -> str:
    return f"captions:{tagger_version}"


def new_caption_store(settings, tagger_version: str) -> CategoryIndexes:
    return CategoryIndexes(settings.embed_model_name, settings.embed_version, caption_source(tagger_version))


def load_caption_store(settings, tagger_version: str) -> CategoryIndexes:
    """Load the caption indexes, or an empty store if none exist. Raises IndexMismatchError on another version."""
    return CategoryIndexes.load_or_create(
        caption_root(settings), settings.embed_model_name, settings.embed_version, caption_source(tagger_version)
    )


def caption_directory(settings) -> Path:
    return caption_root(settings) / index_dir_name(settings.embed_model_name, settings.embed_version)
