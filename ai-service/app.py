import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoProcessor
from io import BytesIO

MODEL_PATH = os.getenv("MODEL_PATH", "/models")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading SigLIP 2 Base...")
    app.state.processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    app.state.model = AutoModel.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    app.state.model.eval()
    print("SigLIP 2 is ready!")
    yield

app = FastAPI(
    title="SigLIP 2 Image Embedding API",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "ok", "model": "SigLIP 2 Base"}

@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    if file.content_type not in {
        "image/jpeg", "image/png", "image/webp"
    }:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG, PNG, or WebP image"
        )

    try:
        image = Image.open(
            BytesIO(await file.read())
        ).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    inputs = app.state.processor(
        images=image,
        return_tensors="pt"
    )

    with torch.inference_mode():
        features = app.state.model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )
        # transformers 5.x returns an output object; the projected embedding is pooler_output
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = torch.nn.functional.normalize(
            features.float(), p=2, dim=-1
        )

    return {
        "model": "SigLIP 2 Base",
        "embedding_dimension": features.shape[-1],
        "embedding": features[0].cpu().tolist()
    }




MAX_BATCH = 32


@app.post("/embed/batch")
async def embed_batch(files: list[UploadFile] = File(...)):
    """Several images in one forward pass; returns one normalized vector per image, in upload order."""
    if not 1 <= len(files) <= MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"Send between 1 and {MAX_BATCH} images")
    images = []
    for position, file in enumerate(files):
        if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=400, detail=f"Image {position}: upload a JPEG, PNG, or WebP image")
        try:
            images.append(Image.open(BytesIO(await file.read())).convert("RGB"))
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail=f"Image {position}: invalid image file")

    inputs = app.state.processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        features = app.state.model.get_image_features(pixel_values=inputs["pixel_values"])
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)

    return {
        "model": "SigLIP 2 Base",
        "embedding_dimension": features.shape[-1],
        "embeddings": features.cpu().tolist()
    }


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.post("/embed/text")
def embed_text(request: TextRequest):
    """SigLIP 2 text embedding in the same space as /embed. Text longer than 64 tokens is truncated."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    # SigLIP 2 was trained with text padded to 64 tokens; other padding gives worse embeddings
    inputs = app.state.processor(
        text=[text], padding="max_length", max_length=64, truncation=True, return_tensors="pt"
    )

    with torch.inference_mode():
        features = app.state.model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)

    return {
        "model": "SigLIP 2 Base",
        "embedding_dimension": features.shape[-1],
        "embedding": features[0].cpu().tolist()
    }
