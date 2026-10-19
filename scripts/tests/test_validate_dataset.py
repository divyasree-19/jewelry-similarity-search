import csv
from pathlib import Path

import pytest
from PIL import Image

from scripts import validate_dataset as vd


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_image(path: Path, colour: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), colour).save(path)


@pytest.fixture
def dataset(tmp_path: Path):
    """A small valid dataset; tests mutate it to trigger each error."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "categories.txt").write_text("# comment\nrings\nearrings\n")

    catalogue = [
        {"image_id": "img1", "product_id": "P1", "project_id": "ejil", "category": "rings",
         "subcategory": "solitaire", "image_path": "data/catalogue/rings/img1.jpg"},
        {"image_id": "img2", "product_id": "P1", "project_id": "ejil", "category": "rings",
         "subcategory": "solitaire", "image_path": "data/catalogue/rings/img2.jpg"},
        {"image_id": "img3", "product_id": "P2", "project_id": "ejil", "category": "earrings",
         "subcategory": "", "image_path": "data/catalogue/earrings/img3.jpg"},
    ]
    for i, row in enumerate(catalogue):
        save_image(tmp_path / row["image_path"], (i * 60, 10, 10))

    queries = [
        {"query_id": "q1", "query_path": "data/queries/q1.jpg", "category": "rings",
         "expected_product_ids": "P1", "edge_case_tags": "lighting;angle"},
        {"query_id": "q2", "query_path": "data/queries/q2.jpg", "category": "earrings",
         "expected_product_ids": "P2", "edge_case_tags": ""},
    ]
    for i, row in enumerate(queries):
        save_image(tmp_path / row["query_path"], (10, i * 60 + 100, 10))

    def write(cat=catalogue, qs=queries):
        write_csv(tmp_path / "data" / "catalogue.csv", vd.CATALOGUE_COLUMNS, cat)
        write_csv(tmp_path / "eval" / "queries.csv", vd.QUERY_COLUMNS, qs)
        return vd.validate(tmp_path)[0]

    return tmp_path, catalogue, queries, write


def errors_matching(report, text: str) -> list[str]:
    return [e for e in report.errors if text in e]


def test_valid_dataset_has_no_errors(dataset):
    _, _, _, write = dataset
    report = write()
    assert report.errors == []
    assert report.warnings == []


def test_missing_file(dataset):
    root, cat, _, write = dataset
    (root / cat[0]["image_path"]).unlink()
    assert errors_matching(write(), "file not found")


def test_unreadable_image(dataset):
    root, cat, _, write = dataset
    (root / cat[0]["image_path"]).write_bytes(b"not an image")
    assert errors_matching(write(), "unreadable image")


def test_duplicate_image_id(dataset):
    _, cat, _, write = dataset
    cat[1]["image_id"] = "img1"
    assert errors_matching(write(), "duplicate image_id")


def test_duplicate_query_id(dataset):
    _, _, qs, write = dataset
    qs[1]["query_id"] = "q1"
    assert errors_matching(write(), "duplicate query_id")


def test_unknown_category(dataset):
    root, cat, _, write = dataset
    cat[2]["category"] = "tiaras"
    cat[2]["image_path"] = "data/catalogue/tiaras/img3.jpg"
    save_image(root / cat[2]["image_path"], (1, 2, 3))
    assert errors_matching(write(), "unknown category 'tiaras'")


def test_query_with_no_expected_match(dataset):
    _, _, qs, write = dataset
    qs[0]["expected_product_ids"] = ""
    assert errors_matching(write(), "no expected_product_ids")


def test_expected_product_not_in_catalogue(dataset):
    _, _, qs, write = dataset
    qs[0]["expected_product_ids"] = "P1;P99"
    assert errors_matching(write(), "'P99' is not in the catalogue")


def test_expected_product_in_other_category(dataset):
    _, _, qs, write = dataset
    qs[0]["expected_product_ids"] = "P2"
    assert errors_matching(write(), "category search would never find it")


def test_unknown_edge_case_tag(dataset):
    _, _, qs, write = dataset
    qs[0]["edge_case_tags"] = "lighting;blurry"
    assert errors_matching(write(), "unknown edge_case_tag 'blurry'")


def test_path_must_follow_layout(dataset):
    root, cat, _, write = dataset
    cat[0]["image_path"] = "data/catalogue/earrings/img1.jpg"
    save_image(root / cat[0]["image_path"], (5, 5, 5))
    assert errors_matching(write(), "image_path should be data/catalogue/rings/img1")


def test_query_identical_to_catalogue_image_warns(dataset):
    root, cat, qs, write = dataset
    (root / qs[0]["query_path"]).write_bytes((root / cat[0]["image_path"]).read_bytes())
    report = write()
    assert report.errors == []
    assert any("byte-identical" in w for w in report.warnings)


def test_coverage_gaps_fail_only_in_strict_mode(dataset):
    root, _, _, write = dataset
    write()
    assert vd.main(["--root", str(root)]) == 0
    assert vd.main(["--root", str(root), "--strict"]) == 1


def test_counts_table(dataset):
    _, _, _, write = dataset
    report = write()
    table = vd.counts_table(report, {"rings", "earrings"}, markdown=True)
    assert "| rings | 1 | 2 | 1 |" in table
    assert "| earrings | 1 | 1 | 1 |" in table
    assert "| **total** | **2** | **3** | **2** |" in table
