"""POST /search/text: search the catalogue with words, e.g. "red ruby ring" or "silver peacock earrings"."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.routes.search_image import check_top_k, clean_category
from app.search.text_search import search_by_text

router = APIRouter()


class TextSearchRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    category: str | None = None
    project_id: str | None = None
    top_k: int = 10
    debug: bool = False


@router.post("/search/text")
async def search_text(body: TextSearchRequest, request: Request):
    app = request.app
    text = " ".join(body.text.split())
    if not text:
        raise HTTPException(400, "text is empty")
    check_top_k(app, body.top_k)
    return await search_by_text(
        app, text, category=clean_category(app, body.category), project_id=(body.project_id or "").strip() or None,
        top_k=body.top_k, debug=body.debug,
    )
