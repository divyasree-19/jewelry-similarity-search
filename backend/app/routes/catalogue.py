"""GET /catalogue-images/<category>/<file>: serves catalogue images without exposing file-system paths."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.search.common import IMAGE_ROUTE

router = APIRouter()


@router.get(IMAGE_ROUTE + "/{image_path:path}", include_in_schema=False)
async def catalogue_image(image_path: str, request: Request):
    root = request.app.state.settings.catalogue_dir.resolve()
    path = (root / image_path).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(404, "Image not found")
    return FileResponse(path)
