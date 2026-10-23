"""Load data/products.csv and data/catalogue.csv into MongoDB (products and images collections).

Safe to re-run: documents are replaced by ID, and products/images of the same project that are no
longer in the CSVs are removed. Uses MONGO_URI and MONGO_DB from the backend settings (.env).

Usage (from the project root, with the stack running):
    python scripts/seed_mongo.py
"""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.db import ensure_indexes  # noqa: E402

LIST_FIELDS = ["metal", "gemstones", "stone_colours", "motifs", "style"]


def split_list(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_documents(products: list[dict], catalogue: list[dict]) -> tuple[list[dict], list[dict]]:
    now = datetime.now(timezone.utc)
    image_ids: dict[str, list[str]] = {}
    images = []
    for row in catalogue:
        image_ids.setdefault(row["product_id"], []).append(row["image_id"])
        images.append({
            "_id": row["image_id"],
            "product_id": row["product_id"],
            "project_id": row["project_id"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "image_path": row["image_path"],
            "updated_at": now,
        })

    docs = []
    for row in products:
        doc = {
            "_id": row["product_id"],
            "project_id": row["project_id"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "name": row["name"],
            "image_ids": image_ids.get(row["product_id"], []),
            "source": {"dataset": "ai-tool-pool-jewelry-vision", "license": "CC BY 4.0",
                       "files": split_list(row["catalogue_sources"])},
            "updated_at": now,
        }
        doc.update({field: split_list(row[field]) for field in LIST_FIELDS})
        docs.append(doc)
    return docs, images


def write_metadata(db, products: list[dict], images: list[dict]) -> None:
    """Upsert products and image metadata and remove documents no longer in the CSVs.

    Products are replaced whole. Images are updated field by field, so the embedding fields written by
    scripts/ingest.py (vector_id, embedding_model, ...) survive a re-seed.
    """
    for doc in products:
        db.products.replace_one({"_id": doc["_id"]}, doc, upsert=True)
    for doc in images:
        fields = {key: value for key, value in doc.items() if key != "_id"}
        db.images.update_one({"_id": doc["_id"]}, {"$set": fields}, upsert=True)

    for collection, docs in (("products", products), ("images", images)):
        projects = {d["project_id"] for d in docs}
        removed = db[collection].delete_many(
            {"project_id": {"$in": list(projects)}, "_id": {"$nin": [d["_id"] for d in docs]}}
        ).deleted_count
        print(f"{collection}: {len(docs)} upserted, {removed} stale removed")

    ensure_indexes(db)


def seed(root: Path = PROJECT_ROOT) -> None:
    settings = get_settings()
    products, images = build_documents(
        read_rows(root / "data" / "products.csv"), read_rows(root / "data" / "catalogue.csv")
    )
    client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=settings.mongo_timeout_ms)
    write_metadata(client[settings.mongo_db], products, images)
    print(f"database {settings.mongo_db!r} at {settings.mongo_uri}")
    client.close()


if __name__ == "__main__":
    seed()
