"""Stage 7 Part A: can Qwen3-VL-8B tag catalogue images fast enough, and where should it run?

Sends 20 catalogue images (4 per category) to an OpenAI-compatible chat endpoint with a prompt asking
for JSON tags, and records per image: seconds, peak RAM of the model server (local mode only) and whether
the reply was valid JSON with every expected key. Writes eval/results/qwen_feasibility.json (or --out)
and prints a summary with an estimate for tagging the whole catalogue.

Local mode starts llama.cpp's llama-server with a 4-bit GGUF and measures its memory from /proc:
    python scripts/qwen_feasibility.py local \
        --server ~/llama.cpp-b11412/build/bin/llama-server \
        --model ~/models/qwen3-vl-8b/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
        --mmproj ~/models/qwen3-vl-8b/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf

Remote mode calls an existing server (RAM is not measurable from here). URL, model and key default to
QWEN_API_KEY_API_BASE, QWEN_MODEL_NAME and QWEN_API_KEY from the environment or .env / .env.backend:
    python scripts/qwen_feasibility.py remote
"""

import argparse
import base64
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.preprocessing import PreprocessOptions, encode, preprocess  # noqa: E402

KEYS = ["category", "metal", "gemstones", "stone_colors", "motifs", "style", "description"]
PROMPT = (
    "You are tagging a jewellery catalogue photo. Reply with ONLY a JSON object, no other text, with these keys:\n"
    '"category": one of ring, earring, necklace, pendant, bracelet, bangle, chain, nose_pin, anklet;\n'
    '"metal": list of metals or metal colours visible (e.g. "yellow gold", "silver");\n'
    '"gemstones": list of stones (e.g. "diamond", "ruby"), empty list if none;\n'
    '"stone_colors": list of stone colours, empty list if none;\n'
    '"motifs": list of design motifs (e.g. "flower", "heart", "leaf");\n'
    '"style": one word such as classic, modern, ethnic, minimalist, statement;\n'
    '"description": one sentence describing the piece.'
)
ENV = {**dotenv_values(PROJECT_ROOT / ".env"), **dotenv_values(PROJECT_ROOT / ".env.backend"), **os.environ}
CLIENT_PRODUCTS = 955_000  # products in the client's Postgres catalogue (README, Stage 2)


def pick_images(per_category: int) -> list[dict]:
    with (PROJECT_ROOT / "data" / "catalogue.csv").open(newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda row: row["image_id"])
    by_category = defaultdict(list)
    for row in rows:
        if len(by_category[row["category"]]) < per_category:
            by_category[row["category"]].append(row)
    return [row for category in sorted(by_category) for row in by_category[category]]


def parse_reply(text: str) -> tuple[bool, bool, dict | None]:
    """(strict: the reply is exactly a JSON object, lenient: JSON after removing ``` fences, parsed object)."""
    def complete(obj) -> bool:
        return isinstance(obj, dict) and all(key in obj for key in KEYS)

    try:
        obj = json.loads(text)
        return complete(obj), complete(obj), obj
    except ValueError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return False, complete(obj), obj
        except ValueError:
            pass
    return False, False, None


def proc_memory_mb(pid: int) -> tuple[float | None, float | None]:
    """(peak RSS, current RSS) of a process in MiB, from /proc/<pid>/status."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None, None
    values = dict(re.findall(r"^(VmHWM|VmRSS):\s+(\d+) kB", status, re.MULTILINE))
    to_mb = lambda key: round(int(values[key]) / 1024, 1) if key in values else None  # noqa: E731
    return to_mb("VmHWM"), to_mb("VmRSS")


def start_llama_server(args) -> subprocess.Popen:
    command = [str(Path(args.server).expanduser()), "-m", str(Path(args.model).expanduser()),
               "--mmproj", str(Path(args.mmproj).expanduser()), "--port", str(args.port), "--host", "127.0.0.1",
               "-c", str(args.ctx), "--temp", "0", "--no-webui"]
    if args.threads:
        command += ["-t", str(args.threads)]
    log = open(PROJECT_ROOT / "eval" / "results" / "qwen_llama_server.log", "w")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + args.load_timeout
    while time.time() < deadline:
        if process.poll() is not None:
            sys.exit(f"llama-server exited with code {process.returncode}; see eval/results/qwen_llama_server.log")
        try:
            if httpx.get(f"http://127.0.0.1:{args.port}/health", timeout=2).status_code == 200:
                return process
        except httpx.HTTPError:
            pass
        time.sleep(2)
    process.terminate()
    sys.exit("llama-server did not become ready in time")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    local = sub.add_parser("local", help="start llama-server with a quantized GGUF")
    local.add_argument("--server", required=True)
    local.add_argument("--model", required=True)
    local.add_argument("--mmproj", required=True)
    local.add_argument("--port", type=int, default=8091)
    local.add_argument("--ctx", type=int, default=4096)
    local.add_argument("--threads", type=int)
    local.add_argument("--load-timeout", type=int, default=600)
    remote = sub.add_parser("remote", help="call an existing OpenAI-compatible server")
    remote.add_argument("--base-url", default=ENV.get("QWEN_API_KEY_API_BASE"))
    remote.add_argument("--model", default=ENV.get("QWEN_MODEL_NAME") or ENV.get("LLM_MODEL"))
    remote.add_argument("--api-key-env", default="QWEN_API_KEY", help="name of the variable holding the key")
    for p in (local, remote):
        p.add_argument("--per-category", type=int, default=4)
        p.add_argument("--timeout", type=float, default=900)
        p.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval" / "results" / "qwen_feasibility.json")
    args = parser.parse_args()

    (PROJECT_ROOT / "eval" / "results").mkdir(parents=True, exist_ok=True)
    process = None
    headers = {}
    if args.mode == "local":
        print("starting llama-server (loading the model can take a minute)...", flush=True)
        load_started = time.perf_counter()
        process = start_llama_server(args)
        load_seconds = round(time.perf_counter() - load_started, 1)
        base_url, model_label = f"http://127.0.0.1:{args.port}/v1", Path(args.model).name
        print(f"ready after {load_seconds}s; memory after load {proc_memory_mb(process.pid)} MiB (peak, current)")
    else:
        load_seconds = None
        base_url, model_label = args.base_url.rstrip("/"), args.model
        if not args.base_url or not args.model:
            sys.exit("Set QWEN_API_KEY_API_BASE and QWEN_MODEL_NAME (or pass --base-url and --model)")
        key = ENV.get(args.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"

    options = PreprocessOptions()
    images = pick_images(args.per_category)
    rows = []
    try:
        with httpx.Client(timeout=args.timeout, headers=headers) as client:
            for i, image in enumerate(images, start=1):
                png = encode(preprocess((PROJECT_ROOT / image["image_path"]).read_bytes(), options))
                body = {
                    "model": args.model if args.mode == "remote" else model_label,
                    "temperature": 0,
                    "max_tokens": 400,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                        {"type": "text", "text": PROMPT},
                    ]}],
                }
                started = time.perf_counter()
                error, text, usage = None, "", {}
                try:
                    response = client.post(f"{base_url}/chat/completions", json=body)
                    response.raise_for_status()
                    payload = response.json()
                    text = payload["choices"][0]["message"]["content"] or ""
                    usage = payload.get("usage", {})
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"[:300]
                seconds = round(time.perf_counter() - started, 2)
                strict, lenient, parsed = parse_reply(text)
                peak, current = proc_memory_mb(process.pid) if process else (None, None)
                rows.append({
                    "image_id": image["image_id"], "category": image["category"], "seconds": seconds,
                    "server_peak_rss_mb": peak, "server_rss_mb": current,
                    "valid_json": strict, "valid_json_after_cleanup": lenient,
                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                    "error": error, "reply": text, "parsed": parsed,
                })
                status = error or ("valid" if strict else "valid after cleanup" if lenient else "INVALID")
                print(f"[{i:2}/{len(images)}] {image['image_id']:10} {seconds:7.1f}s  "
                      f"peak {peak} MiB  {status}", flush=True)
    finally:
        if process:
            process.terminate()
            process.wait(timeout=30)

    ok = [row for row in rows if not row["error"]]
    seconds = [row["seconds"] for row in ok]
    peaks = [row["server_peak_rss_mb"] for row in rows if row["server_peak_rss_mb"]]
    avg = statistics.fmean(seconds) if seconds else None
    summary = {
        "images": len(rows),
        "errors": len(rows) - len(ok),
        "avg_seconds": round(avg, 2) if avg else None,
        "median_seconds": round(statistics.median(seconds), 2) if seconds else None,
        "max_seconds": max(seconds) if seconds else None,
        "max_ram_mb": max(peaks) if peaks else None,
        "valid_json_pct": round(100 * sum(row["valid_json"] for row in rows) / len(rows), 1),
        "valid_json_after_cleanup_pct": round(100 * sum(row["valid_json_after_cleanup"] for row in rows) / len(rows), 1),
        "estimate_hours": {
            "sample_catalogue_71_images": round(avg * 71 / 3600, 2) if avg else None,
            f"client_catalogue_{CLIENT_PRODUCTS}_images": round(avg * CLIENT_PRODUCTS / 3600, 1) if avg else None,
        },
    }
    report = {
        "mode": args.mode, "model": model_label, "base_url": base_url if args.mode == "remote" else "local llama-server",
        "model_load_seconds": load_seconds, "prompt": PROMPT, "image_size": options.target_size,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summary, "images": rows,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{args.mode} {model_label}: {summary['images']} images, {summary['errors']} errors")
    print(f"avg {summary['avg_seconds']} s/image (median {summary['median_seconds']}, max {summary['max_seconds']}), "
          f"max RAM {summary['max_ram_mb']} MiB")
    print(f"valid JSON {summary['valid_json_pct']}% (after removing code fences {summary['valid_json_after_cleanup_pct']}%)")
    for label, hours in summary["estimate_hours"].items():
        print(f"estimate {label}: {hours} h" + (f" = {hours / 24:.1f} days" if hours and hours > 48 else ""))
    print(f"wrote {args.out.relative_to(PROJECT_ROOT) if args.out.is_relative_to(PROJECT_ROOT) else args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
