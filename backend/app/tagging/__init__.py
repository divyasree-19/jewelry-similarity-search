"""VLM attribute tagging: vocabulary, schema and prompt.

TAGGER_VERSION identifies the prompt and vocabulary that produced a set of tags; it is saved with every
tag record, and scripts/tag_catalogue.py re-tags images whose saved version differs.
"""

from app.tagging.prompts import PROMPT_VERSION
from app.tagging.vocab import VOCAB_VERSION

TAGGER_VERSION = f"tags-p{PROMPT_VERSION}-v{VOCAB_VERSION}"
