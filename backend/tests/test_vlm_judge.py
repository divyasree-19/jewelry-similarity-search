"""Stage 10 Part B: the VLM same-design judge on the fixture catalogue with a fake Qwen."""

import asyncio

import pytest

from app import main
from app.clients.vlm_client import Judgement, VLMError
from tests.test_search import catalogue, client, search  # noqa: F401


class FakeJudge:
    """Says "same design" only for the listed image_ids' files; counts calls."""

    def __init__(self, same: set[str] = frozenset(), fail: bool = False, slow: bool = False):
        self.same, self.fail, self.slow, self.calls = same, fail, slow, 0
        self.paths: dict[bytes, str] = {}

    async def judge(self, query_png, candidate_png):
        self.calls += 1
        if self.fail:
            raise VLMError("Qwen is down")
        if self.slow:
            await asyncio.sleep(5)
        same = self.paths.get(candidate_png) in self.same
        return Judgement(same_design=same, confidence=0.9, reason="fake")

    async def aclose(self):
        pass


@pytest.fixture
def judged(client, catalogue, monkeypatch):
    settings = main.app.state.settings
    monkeypatch.setattr(settings, "feature_vlm_judge", True)
    monkeypatch.setattr(settings, "vlm_judge_top_n", 2)
    main.app.state.judge_cache._items.clear()

    # Map each catalogue image's preprocessed PNG to its image_id so the fake can recognise candidates
    from app.preprocessing import encode, preprocess
    from tests.test_search import CATALOGUE
    files, tmp_path = catalogue

    def install(fake):
        for image_id, *_ in CATALOGUE:
            fake.paths[encode(preprocess(files[image_id], main.app.state.preprocess_options))] = image_id
        main.app.state.vlm_client = fake
        return fake
    return client, install


def ids(response):
    return [r["image_id"] for r in response.json()["results"]]


def test_judge_reorders_only_the_top_n(judged, catalogue, monkeypatch):
    client, install = judged
    files, _ = catalogue
    monkeypatch.setattr(main.app.state.settings, "feature_vlm_judge", False)
    part_a = ids(search(client, files["RG-1_a"]))
    monkeypatch.setattr(main.app.state.settings, "feature_vlm_judge", True)
    fake = install(FakeJudge(same={part_a[1]}))  # Qwen prefers the second result

    response = search(client, files["RG-1_a"])
    after = ids(response)
    assert after[:2] == [part_a[1], part_a[0]]  # swapped within the top 2
    assert after[2:] == part_a[2:]  # nothing below N moves
    body = response.json()
    assert body["vlm_judge"] == {"top_n": 2, "calls": 2, "cache_hits": 0, "failed": 0, "reordered": True}
    assert body["results"][0]["vlm_judge"]["same_design"] is True
    assert "vlm_judge" not in body["results"][2]
    assert fake.calls == 2


def test_judge_failure_keeps_the_part_a_order(judged, catalogue, monkeypatch):
    client, install = judged
    files, _ = catalogue
    monkeypatch.setattr(main.app.state.settings, "feature_vlm_judge", False)
    part_a = ids(search(client, files["RG-1_a"]))
    monkeypatch.setattr(main.app.state.settings, "feature_vlm_judge", True)
    install(FakeJudge(fail=True))
    response = search(client, files["RG-1_a"])
    assert response.status_code == 200 and ids(response) == part_a
    assert response.json()["vlm_judge"]["failed"] == 2 and response.json()["vlm_judge"]["reordered"] is False


def test_judge_timeout_keeps_the_part_a_order(judged, catalogue, monkeypatch):
    client, install = judged
    files, _ = catalogue
    monkeypatch.setattr(main.app.state.settings, "vlm_judge_timeout_seconds", 0.05)
    install(FakeJudge(slow=True))
    body = search(client, files["RG-1_a"]).json()
    assert body["vlm_judge"]["failed"] == 2 and "vlm_judge" not in body["results"][0]


def test_cache_hit_skips_the_call(judged, catalogue):
    client, install = judged
    files, _ = catalogue
    fake = install(FakeJudge())
    first = search(client, files["RG-1_a"]).json()["vlm_judge"]
    second = search(client, files["RG-1_a"]).json()["vlm_judge"]
    assert (first["calls"], first["cache_hits"]) == (2, 0)
    assert (second["calls"], second["cache_hits"]) == (0, 2) and fake.calls == 2


def test_judge_is_off_by_default():
    from app.config import Settings
    assert Settings().feature_vlm_judge is False
