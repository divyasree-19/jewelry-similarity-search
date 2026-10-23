#!/usr/bin/env bash
# Run every test suite with one command and print a summary; exits non-zero if any suite fails.
#
#   scripts/run_tests.sh           in Docker; needs only `docker compose up -d` (MongoDB and SigLIP running)
#   scripts/run_tests.sh --local   on this machine; needs backend/.venv, npm, and MongoDB on localhost:27017
#
# Suites: backend (pytest: search, failures, reranker, ...), scripts + eval (pytest), frontend (vitest).
# The live test (every catalogue image through the real stack) runs only when the indexes exist.
set -uo pipefail
cd "$(dirname "$0")/.."

failed=()
run() {
    local name=$1; shift
    echo "=== $name"
    if "$@"; then echo "--- $name: passed"; else echo "--- $name: FAILED"; failed+=("$name"); fi
}

if [[ "${1:-}" == "--local" ]]; then
    python="$PWD/backend/.venv/bin/python"
    run backend bash -c "cd backend && '$python' -m pytest -q -p no:cacheprovider"
    run scripts+eval "$python" -m pytest -q -p no:cacheprovider scripts/tests eval/tests
    run frontend bash -c "cd frontend && npm test"
else
    run backend docker compose run --rm -w /app/backend tools -m pytest -q -p no:cacheprovider
    run scripts+eval docker compose run --rm tools -m pytest -q -p no:cacheprovider scripts/tests eval/tests
    run frontend docker build --progress=plain --target test frontend
fi

echo
if (( ${#failed[@]} )); then echo "FAILED: ${failed[*]}"; exit 1; fi
echo "All test suites passed."
