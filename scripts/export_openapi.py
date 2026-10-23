"""Write the backend's OpenAPI description to docs/openapi.json (Stage 12, for client integration).

Request schemas and error codes come from FastAPI; response bodies are documented with examples in
docs/integration_notes.md. Re-run after any change to the routes.

Usage (from the project root):
    python scripts/export_openapi.py
    docker compose run --rm tools scripts/export_openapi.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.main import app  # noqa: E402

out = PROJECT_ROOT / "docs" / "openapi.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")
print(f"wrote {out.relative_to(PROJECT_ROOT)}: {', '.join(sorted(app.openapi()['paths']))}")
