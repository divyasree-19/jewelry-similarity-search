"""Load test of the running backend (Stage 12): image search at several concurrency levels, plus text search.

For each level it sends real query images (eval/queries.csv, category given, as in run_eval.py) or query texts
(eval/text_queries.csv) from N concurrent clients and records:
- p50 / p95 / max latency, error rate (any non-200 or timeout) and throughput (completed requests per second);
- CPU and RAM per container, sampled with `docker stats` while the level runs (mean and peak);
- per-step timings (p50 / p95 ms) from the backend's own log lines ("query ... timings_ms={...}"), which
  identify the slowest step. Without access to the backend container's logs, the response's timings_ms are used
  (the same numbers).
Writes eval/results/load_test.md and load_test.json.

Usage (from the project root, with the stack running: `docker compose up -d`):
    python scripts/load_test.py --url http://localhost:8001 --machine "laptop: i5-1334U, 12 threads, 16 GB"
    python scripts/load_test.py --levels 1 5 --text-levels 1 --requests 20      # quick run
Text search calls Qwen once per request on the shared server, so its levels are lower by default.
"""

import argparse
import ast
import asyncio
import csv
import json
import math
import os
import platform
import re
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTAINERS = ["jewellery-backend", "siglip2-api", "jewellery-mongodb"]
TIMINGS = re.compile(r"query (\w+) mode=(\w+).* timings_ms=(\{[^}]*\})")
UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "kB": 1e3, "MB": 1e6, "GB": 1e9}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)  # nearest rank
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def to_bytes(text: str) -> float:
    number, unit = re.match(r"([\d.]+)\s*([A-Za-z]+)", text.strip()).groups()
    return float(number) * UNITS[unit]


class DockerStats:
    """Samples CPU % and memory of the containers in a background thread until stopped."""

    def __init__(self, containers: list[str]):
        self.containers = containers
        self.samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.available = True

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}", *self.containers],
                                     capture_output=True, text=True, timeout=15, check=True).stdout
            except (OSError, subprocess.SubprocessError):
                self.available = False
                return
            for line in out.splitlines():
                row = json.loads(line)
                self.samples[row["Name"]].append((float(row["CPUPerc"].rstrip("%")),
                                                  to_bytes(row["MemUsage"].split("/")[0])))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        result = {}
        for name, rows in self.samples.items():
            cpu, mem = [r[0] for r in rows], [r[1] for r in rows]
            result[name] = {"cpu_mean": statistics.mean(cpu), "cpu_peak": max(cpu),
                            "mem_peak_mib": max(mem) / 2**20, "samples": len(rows)}
        return result


def backend_timings(container: str, since: str, query_ids: set[str]) -> list[dict] | None:
    """Per-step timings of these queries from the backend's log lines, or None if the logs are not readable."""
    try:
        out = subprocess.run(["docker", "logs", "--since", since, container], capture_output=True, text=True,
                             timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    found = []
    for line in (out.stdout + out.stderr).splitlines():
        match = TIMINGS.search(line)
        if match and match.group(1) in query_ids:
            found.append(ast.literal_eval(match.group(3)))
    return found


async def run_level(client: httpx.AsyncClient, make_request, concurrency: int, total: int) -> dict:
    latencies, statuses, query_ids, response_timings = [], defaultdict(int), set(), []
    counter = iter(range(total))

    async def worker():
        for i in counter:
            started = time.perf_counter()
            try:
                response = await make_request(client, i)
                status = response.status_code
                if status == 200:
                    body = response.json()
                    query_ids.add(body["query_id"])
                    response_timings.append(body.get("timings_ms", {}))
            except httpx.HTTPError as exc:
                status = type(exc).__name__
            latencies.append((time.perf_counter() - started) * 1000)
            statuses[status] += 1

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - started
    ok = statuses.get(200, 0)
    return {
        "concurrency": concurrency, "requests": total, "ok": ok, "errors": total - ok,
        "error_rate": (total - ok) / total, "statuses": {str(k): v for k, v in statuses.items()},
        "wall_s": wall, "throughput_rps": ok / wall,
        "p50_ms": percentile(latencies, 50), "p95_ms": percentile(latencies, 95), "max_ms": max(latencies),
        "query_ids": sorted(query_ids), "response_timings": response_timings,
    }


def step_table(timings: list[dict]) -> dict[str, tuple[float, float]]:
    steps = defaultdict(list)
    for row in timings:
        for step, ms in row.items():
            steps[step].append(ms)
    return {step: (percentile(v, 50), percentile(v, 95)) for step, v in steps.items()}


def machine_info(label: str) -> dict:
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else ""
    mem = next((int(line.split()[1]) / 2**20 for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemTotal")), None) if Path("/proc/meminfo").exists() else None
    return {"label": label, "cpu": cpu, "threads": os.cpu_count(),
            "ram_gib": round(mem, 1) if mem else None, "os": platform.platform()}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


async def main_async(args) -> dict:
    images = [(PROJECT_ROOT / row["query_path"], row["category"]) for row in read_csv(PROJECT_ROOT / "eval/queries.csv")]
    payloads = [(path.read_bytes(), category) for path, category in images]
    texts = [row["text"] for row in read_csv(PROJECT_ROOT / "eval/text_queries.csv")]

    async def image_request(client, i):
        data, category = payloads[i % len(payloads)]
        return await client.post("/search/image", files={"file": ("q.jpg", data, "image/jpeg")},
                                 data={"category": category, "top_k": "10"})

    async def text_request(client, i):
        return await client.post("/search/text", json={"text": texts[i % len(texts)], "top_k": 10})

    report = {"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "url": args.url,
              "machine": machine_info(args.machine), "levels": {"image": [], "text": []}}
    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout) as client:
        health = (await client.get("/health")).json()
        if health.get("status") != "ok":
            raise SystemExit(f"backend is not healthy: {health}")
        probe = (await client.post("/search/image", files={"file": ("q.jpg", payloads[0][0], "image/jpeg")},
                                   data={"category": payloads[0][1], "debug": "true"})).json()
        report["config"] = {"embed_backend": health["embed_backend"], "flags": probe.get("debug", {}).get("flags"),
                            "reranker_weights_version": probe.get("reranker_weights_version"),
                            "indexed": health.get("indexes", {}).get("served")}
        for _ in range(3):  # warm-up
            await image_request(client, 0)

        for mode, levels, make_request in (("image", args.levels, image_request),
                                           ("text", args.text_levels, text_request)):
            for concurrency in levels:
                total = max(args.requests, args.per_client * concurrency)
                since = datetime.now(timezone.utc).isoformat()
                print(f"{mode} search, {concurrency} concurrent, {total} requests ...", flush=True)
                with DockerStats(args.containers) as stats:
                    level = await run_level(client, make_request, concurrency, total)
                logged = backend_timings(args.backend_container, since, set(level["query_ids"]))
                level["timings_source"] = "backend log" if logged else "response"
                level["steps"] = step_table(logged or level.pop("response_timings"))
                level.pop("response_timings", None)
                level["containers"] = stats.summary() if stats.available else {}
                report["levels"][mode].append(level)
                print(f"  p50 {level['p50_ms']:.0f} ms  p95 {level['p95_ms']:.0f} ms  errors {level['errors']}  "
                      f"{level['throughput_rps']:.2f} req/s", flush=True)
    return report


def fmt(value, digits=0):
    return "–" if value is None else f"{value:,.{digits}f}"


def markdown(report: dict) -> str:
    m = report["machine"]
    lines = [
        "# Load test", "",
        f"Run {report['started_at']} against `{report['url']}` (`python scripts/load_test.py`).", "",
        f"**Machine: {m['label']}.** {m['cpu']}, {m['threads']} threads, {m['ram_gib']} GiB RAM, {m['os']}. "
        "Everything (backend, SigLIP, MongoDB) runs in Docker on this one machine, on CPU; Qwen is the remote "
        "shared server.", "",
        f"Config: `{report['config']['embed_backend']}`, flags `{json.dumps(report['config']['flags'])}`, "
        f"reranker weights `{report['config']['reranker_weights_version'] or 'not used for image search'}`.", "",
    ]
    for mode in ("image", "text"):
        levels = report["levels"][mode]
        if not levels:
            continue
        lines += [f"## {mode.capitalize()} search", "",
                  "| Concurrency | Requests | p50 ms | p95 ms | max ms | Errors | Throughput req/s |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for lv in levels:
            errors = f"{lv['errors']} ({lv['error_rate']:.0%})" + (
                f" {json.dumps({k: v for k, v in lv['statuses'].items() if k != '200'})}" if lv["errors"] else "")
            lines.append(f"| {lv['concurrency']} | {lv['requests']} | {fmt(lv['p50_ms'])} | {fmt(lv['p95_ms'])} | "
                         f"{fmt(lv['max_ms'])} | {errors} | {lv['throughput_rps']:.2f} |")
        containers = sorted({name for lv in levels for name in lv["containers"]})
        if containers:
            lines += ["", "CPU (mean / peak %, 100% = one core) and peak RAM per container:", "",
                      "| Concurrency | " + " | ".join(containers) + " |", "|---:|" + "---|" * len(containers)]
            for lv in levels:
                cells = [f"{c['cpu_mean']:.0f} / {c['cpu_peak']:.0f}% · {c['mem_peak_mib']:.0f} MiB"
                         if (c := lv["containers"].get(name)) else "–" for name in containers]
                lines.append(f"| {lv['concurrency']} | " + " | ".join(cells) + " |")
        steps = list(dict.fromkeys(step for lv in levels for step in lv["steps"]))
        lines += ["", f"Per-step backend time, p50 / p95 ms (from the {levels[0]['timings_source']}):", "",
                  "| Step | " + " | ".join(f"c={lv['concurrency']}" for lv in levels) + " |",
                  "|---|" + "---:|" * len(levels)]
        for step in steps:
            cells = [f"{fmt(lv['steps'][step][0])} / {fmt(lv['steps'][step][1])}" if step in lv["steps"] else "–"
                     for lv in levels]
            lines.append(f"| {step} | " + " | ".join(cells) + " |")
        slowest = {lv["concurrency"]: max((s for s in lv["steps"] if s not in ("total", "model_calls")),
                                          key=lambda s: lv["steps"][s][0]) for lv in levels}
        lines += ["", "Slowest step (by p50, excluding total): "
                  + ", ".join(f"c={c}: `{s}`" for c, s in slowest.items()) + ".", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8001", help="backend base URL")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 5, 10, 20], help="image search concurrency")
    parser.add_argument("--text-levels", type=int, nargs="*", default=[1, 5, 10], help="text search concurrency")
    parser.add_argument("--requests", type=int, default=20, help="minimum requests per level")
    parser.add_argument("--per-client", type=int, default=3, help="requests per concurrent client (if more)")
    parser.add_argument("--timeout", type=float, default=180, help="client timeout per request, seconds")
    parser.add_argument("--machine", default="laptop", help="describe the machine, e.g. 'laptop' or 'GPU server'")
    parser.add_argument("--containers", nargs="+", default=CONTAINERS)
    parser.add_argument("--backend-container", default="jewellery-backend")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval" / "results" / "load_test")
    args = parser.parse_args()

    report = asyncio.run(main_async(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(f"wrote {args.out.with_suffix('.md')} and .json")


if __name__ == "__main__":
    main()
