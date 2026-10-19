from scripts.seed_mongo import build_documents


def test_build_documents_links_images_and_splits_lists():
    products = [{
        "product_id": "RG-001", "project_id": "demo", "category": "rings", "subcategory": "band",
        "name": "Two-tone band", "metal": "yellow gold;black rhodium", "gemstones": "diamond",
        "stone_colours": "white", "motifs": "bars; stripes", "style": "", "catalogue_sources": "a.jpg;b.jpg",
    }]
    catalogue = [
        {"image_id": "RG-001_01", "product_id": "RG-001", "project_id": "demo", "category": "rings",
         "subcategory": "band", "image_path": "data/catalogue/rings/RG-001_01.jpg"},
        {"image_id": "RG-001_02", "product_id": "RG-001", "project_id": "demo", "category": "rings",
         "subcategory": "band", "image_path": "data/catalogue/rings/RG-001_02.jpg"},
    ]
    docs, images = build_documents(products, catalogue)

    assert docs[0]["_id"] == "RG-001"
    assert docs[0]["image_ids"] == ["RG-001_01", "RG-001_02"]
    assert docs[0]["metal"] == ["yellow gold", "black rhodium"]
    assert docs[0]["motifs"] == ["bars", "stripes"]
    assert docs[0]["style"] == []
    assert docs[0]["source"]["files"] == ["a.jpg", "b.jpg"]
    assert [i["_id"] for i in images] == ["RG-001_01", "RG-001_02"]
    assert images[0]["product_id"] == "RG-001"
