import pytest

from eval.run_eval import first_hit_rank, metrics, split


def test_first_hit_rank_uses_any_expected_product():
    assert first_hit_rank(["A", "B", "C"], ["C", "B"]) == 2
    assert first_hit_rank(["A", "B"], ["Z"]) is None


def test_metrics():
    # ranks: hit at 1, hit at 3, hit at 7, miss
    m = metrics([1, 3, 7, None])
    assert m == {"n": 4, "recall@1": 0.25, "recall@5": 0.5, "recall@10": 0.75,
                 "mrr@10": pytest.approx(round((1 + 1 / 3 + 1 / 7) / 4, 4))}


def test_split_lists():
    assert split("P1; P2;") == ["P1", "P2"]
    assert split("") == []
