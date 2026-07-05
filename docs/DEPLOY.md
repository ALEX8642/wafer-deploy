# DEPLOY.md — GB10 co-tenant deploy + real latency capture

The CPU quickstart (`docker compose up`, README) reproduces the whole board on any
x86 box with zero external deps. **This runbook is the other half:** build the
**arm64** image and run the service as a **co-tenant on the GB10 Grace-Blackwell
cluster**, next to the standing LLM stack, and capture *real* latency, throughput
and memory footprint — under LLM co-tenancy **and** isolated.

Hardware policy (from `PLAN-wafer-deploy.md`): the GB10 hosts **only** the standing
FastAPI service + the bounded online drift sidecar — footprint < 1 GB (45 MB
checkpoint, single-map CPU inference), so it co-tenants with the LLM stack **without
a spin-down**. Spin the LLMs down *only* to capture the one clean isolated baseline.
No training and no offline drift science ever run on the cluster.

This is a hands-on-the-cluster procedure (the machine Claude is not on); run it in
your own terminal and paste the four numbers it prints into the README latency table
+ `STATUS.md` resume bullet. Everything is scripted so the capture is a copy-paste.

---

## 0. Prerequisites (on the GB10)

- Docker with buildx (native arm64 — Grace-Blackwell is aarch64, so the image
  builds natively; no cross-build/QEMU needed).
- A `wafer-mixed` checkout **on the GB10** providing the read-only artifacts
  `outputs/{best.pt,thresholds.json,calibration.json}`. Nothing is vendored into
  this image; the 45 MB checkpoint is bind-mounted (repo policy). If you don't want
  the whole sibling repo on the cluster, copy just those three files into a dir and
  point `WAFER_MIXED_ROOT` at its parent.
- This repo checked out (the committed reference snapshot travels with it).

## 1. Build the arm64 image

The default `Dockerfile` is arch-neutral — the PyTorch **CPU** wheel index
(`download.pytorch.org/whl/cpu`) serves linux-aarch64 wheels, so the same
Dockerfile builds a native arm64 image on the GB10:

```bash
docker build -t wafer-deploy:arm64 .
docker image inspect wafer-deploy:arm64 --format '{{.Architecture}} {{.Size}}'
# expect: arm64 <bytes>   (CPU torch → ~1.8 GB image, same as the x86 build)
```

(Cross-building from an x86 box instead: `docker buildx build --platform linux/arm64
-t wafer-deploy:arm64 --load .` — slower via QEMU, only if you can't build on the
cluster.)

## 2. Run as a polite co-tenant

Cap CPU and memory so the service can never starve the LLM stack — the drift state
is bounded by design, so a small cap is safe:

```bash
docker run -d --name wafer-deploy \
  -p 8000:8000 \
  --cpus 4 --memory 1g \
  -v /path/to/wafer-mixed:/wafer-mixed:ro \
  -e WAFER_MIXED_ROOT=/wafer-mixed \
  -e WAFER_DEPLOY_DEVICE=cpu \
  wafer-deploy:arm64

# confirm it loaded the checkpoint (not degraded):
curl -s localhost:8000/healthz | python3 -m json.tool     # "model_loaded": true
```

Footprint check (this is the memory number for the README):

```bash
docker stats --no-stream wafer-deploy       # MEM USAGE column — expect < 1 GB
```

## 3. Capture latency — isolated vs co-tenant

Same tool (`scripts/bench_latency.py`), same synthetic map, two hardware conditions.
`--concurrency 1` is the honest single-request p50/p99; a higher concurrency run
finds the throughput knee.

**(a) Co-tenant (LLM stack up — the realistic production number):**

```bash
python3 scripts/bench_latency.py --concurrency 1 --n 500 \
    --label gb10-cotenant --json docs/bench_gb10_cotenant.json
python3 scripts/bench_latency.py --concurrency 8 --n 2000 \
    --label gb10-cotenant-c8 --json docs/bench_gb10_cotenant_c8.json
```

**(b) Isolated (spin the LLM stack down — the *one* clean baseline):**

```bash
# stop the LLM stack; confirm with glances that the GPUs/CPU are idle, then:
python3 scripts/bench_latency.py --concurrency 1 --n 500 \
    --label gb10-isolated --json docs/bench_gb10_isolated.json
# bring the LLM stack back up afterwards.
```

Each run prints:

```
requests      : 500  (concurrency 1, synthetic map)
client p50/p99: <p50> / <p99> ms   (p95 …, mean …)
server compute: <median> ms (median, model-only)
throughput    : <rps> req/s over … s wall
```

## 4. Record the numbers

Fill the README **Real GB10 latency** table and the `STATUS.md` Phase-4 resume
bullet from the four captures:

| condition | p50 (ms) | p99 (ms) | throughput (req/s) | mem (MB) |
|---|---|---|---|---|
| GB10 arm64, isolated, c=1 | _(3a-isolated)_ | | | _(docker stats)_ |
| GB10 arm64, co-tenant, c=1 | _(3a-cotenant)_ | | | |
| GB10 arm64, co-tenant, c=8 | _(3b)_ | | _(knee)_ | |

The honest story is the **co-tenant vs isolated delta**: how much the standing LLM
load costs the service's tail latency. Report it either way.

## 5. Teardown

```bash
docker rm -f wafer-deploy        # image cached; re-`docker run` to relaunch
```

The committed `docs/bench_gb10_*.json` files are the raw captures behind the table
(small, no IP). The service can be left running as the standing co-tenant deploy;
this runbook is only the measurement pass.
