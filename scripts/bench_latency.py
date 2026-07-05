"""bench_latency.py — measure real serving latency + throughput against a running service.

Stdlib only (urllib + threading), so it runs from a fresh clone with no installs,
on the CPU quickstart box and on the GB10 arm64 co-tenant deploy identically:

    # bring the service up first (docker compose up, or uvicorn), then:
    python scripts/bench_latency.py                        # 500 reqs, concurrency 1
    python scripts/bench_latency.py --n 1000 --concurrency 8
    python scripts/bench_latency.py --url http://gb10:8000 --json out.json

Reports client-side wall-clock latency (what a caller sees, queueing included):
p50 / p95 / p99 and achieved throughput, plus the service's own reported
compute-only latency_ms (median) for reference. `--concurrency 1` is the honest
single-request latency; raise it to find the throughput knee under load.

The GB10 runbook (docs/DEPLOY.md) uses this script for both the isolated and the
LLM-co-tenant latency captures — same tool, same map, two hardware conditions.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Reuse the quickstart's synthetic map so the benchmark needs no dataset.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from send_sample import synthetic_map, from_mixed  # noqa: E402


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0,100]); values need not be sorted."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(q / 100.0 * (len(s) - 1))))
    return s[k]


def _post_once(url: str, payload: bytes) -> tuple[float, float]:
    """POST /predict once. Returns (client wall-clock ms, server latency_ms)."""
    req = urllib.request.Request(f"{url}/predict", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read().decode())
    dt_ms = (time.perf_counter() - t0) * 1e3
    return dt_ms, float(body.get("latency_ms", float("nan")))


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark /predict latency + throughput")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=500, help="total requests (after warmup)")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel in-flight requests")
    ap.add_argument("--warmup", type=int, default=20, help="warmup requests (excluded)")
    ap.add_argument("--from-mixed", type=int, default=None,
                    help="use real test-split map at this index (needs sibling checkout)")
    ap.add_argument("--label", default=None,
                    help="free-text label recorded in --json output (e.g. 'gb10-cotenant')")
    ap.add_argument("--json", default=None, help="write the summary to this path")
    args = ap.parse_args()

    wmap = from_mixed(args.from_mixed) if args.from_mixed is not None else synthetic_map()
    payload = json.dumps({"wafer_map": wmap}).encode()

    # Warmup — JIT of the first forward passes, page-ins, connection setup — excluded.
    for _ in range(args.warmup):
        _post_once(args.url, payload)

    client_ms: list[float] = []
    server_ms: list[float] = []
    errors = {"n": 0}
    lock = threading.Lock()
    counter = {"i": 0}

    def worker() -> None:
        while True:
            with lock:
                if counter["i"] >= args.n:
                    return
                counter["i"] += 1
            try:
                c, s = _post_once(args.url, payload)
            except Exception:  # record the failure, keep the worker alive
                with lock:
                    errors["n"] += 1
                continue
            with lock:
                client_ms.append(c)
                server_ms.append(s)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(max(1, args.concurrency))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_s = time.perf_counter() - t0

    n = len(client_ms)
    if n == 0:
        print(f"all {errors['n']} requests failed — is the service up at {args.url}?")
        return 1
    throughput = n / wall_s if wall_s > 0 else float("nan")
    summary = {
        "label": args.label,
        "url": args.url,
        "n": n,
        "errors": errors["n"],
        "concurrency": args.concurrency,
        "map_source": "mixed" if args.from_mixed is not None else "synthetic",
        "client_ms": {
            "p50": round(_percentile(client_ms, 50), 2),
            "p95": round(_percentile(client_ms, 95), 2),
            "p99": round(_percentile(client_ms, 99), 2),
            "mean": round(statistics.fmean(client_ms), 2),
            "min": round(min(client_ms), 2),
            "max": round(max(client_ms), 2),
        },
        "server_latency_ms_p50": round(_percentile(server_ms, 50), 2),
        "throughput_rps": round(throughput, 1),
        "wall_s": round(wall_s, 2),
    }

    c = summary["client_ms"]
    print(f"requests      : {n} ok, {errors['n']} failed  "
          f"(concurrency {args.concurrency}, {summary['map_source']} map)")
    print(f"client p50/p99: {c['p50']:.1f} / {c['p99']:.1f} ms   (p95 {c['p95']:.1f}, mean {c['mean']:.1f})")
    print(f"server compute: {summary['server_latency_ms_p50']:.1f} ms (median, model-only)")
    print(f"throughput    : {throughput:.1f} req/s over {wall_s:.1f} s wall")

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"wrote         : {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
