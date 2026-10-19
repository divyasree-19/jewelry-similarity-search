from fastapi.testclient import TestClient

from app import main
from app.config import get_settings


def test_health_reports_errors_when_dependencies_down(monkeypatch):
    monkeypatch.setenv("EMBED_SERVICE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("MONGO_URI", "mongodb://127.0.0.1:9")
    monkeypatch.setenv("MONGO_TIMEOUT_MS", "200")
    monkeypatch.setenv("EMBED_TIMEOUT_SECONDS", "1")
    get_settings.cache_clear()

    with TestClient(main.app) as client:
        response = client.get("/health")

    get_settings.cache_clear()
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["mongodb"]["status"] == "error"
    assert body["dependencies"]["embedding_service"]["status"] == "error"


def test_health_ok_when_dependencies_up(monkeypatch):
    async def ok(app):
        return {"status": "ok"}

    monkeypatch.setattr(main, "check_mongo", ok)
    monkeypatch.setattr(main, "check_embedding_service", ok)

    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
