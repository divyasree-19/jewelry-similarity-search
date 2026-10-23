"""MongoDB indexes, created by the backend at startup and by scripts/seed_mongo.py (creating them is idempotent).

image_id is the images collection's _id, so it is unique without an extra index. vector_id is unique among the
images that have one (images that were never ingested have none).
"""

from pymongo import ASCENDING, IndexModel

INDEXES = {
    "images": [
        IndexModel([("product_id", ASCENDING)]),
        IndexModel([("category", ASCENDING)]),
        IndexModel([("vector_id", ASCENDING)], unique=True, partialFilterExpression={"vector_id": {"$exists": True}}),
    ],
    "products": [IndexModel([("category", ASCENDING)])],
}


def ensure_indexes(db) -> None:
    for collection, models in INDEXES.items():
        db[collection].create_indexes(models)


async def ensure_indexes_async(db) -> None:
    for collection, models in INDEXES.items():
        await db[collection].create_indexes(models)
