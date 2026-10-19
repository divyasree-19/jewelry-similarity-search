"""POST /search/image: image search, category optional, optional text refinement ("similar but in silver")."""

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.search.common import ALLOWED_TYPES, read_upload, sniff_type
from app.search.image_search import search_by_image

router = APIRouter()


def clean_category(app, category: str | None) -> str | None:
    category = (category or "").strip().lower() or None
    if category and category not in app.state.categories:
        raise HTTPException(400, f"Unknown category {category!r}; expected one of {sorted(app.state.categories)}")
    return category


def check_top_k(app, top_k: int) -> None:
    limit = app.state.settings.search_max_top_k
    if not 1 <= top_k <= limit:
        raise HTTPException(400, f"top_k must be between 1 and {limit}")


@router.post("/search/image")
async def search_image(
    request: Request,
    file: UploadFile = File(..., description="JPEG, PNG or WebP photo"),
    category: str | None = Form(None, description="Omit to detect it"),
    text: str | None = Form(None, description='Optional refinement, e.g. "similar but in silver"'),
    project_id: str | None = Form(None),
    subcategory: str | None = Form(None),
    top_k: int = Form(10),
    debug: bool = Form(False),
):
    app = request.app
    settings = app.state.settings
    category = clean_category(app, category)
    check_top_k(app, top_k)
    text = " ".join((text or "").split())[:500] or None

    data = await read_upload(file, int(settings.max_upload_mb * 1_000_000))
    response = await search_by_image(
        app, data, category=category, text=text, project_id=(project_id or "").strip() or None,
        subcategory=(subcategory or "").strip().lower() or None, top_k=top_k, debug=debug,
    )
    if settings.save_query_images:
        settings.query_image_dir.mkdir(parents=True, exist_ok=True)
        (settings.query_image_dir / f"{response['query_id']}.{ALLOWED_TYPES[sniff_type(data)]}").write_bytes(data)
    return response
