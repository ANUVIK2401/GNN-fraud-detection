"""Latency / throughput benchmark.

In-process (model + state, no network), per batch size:
    python scripts/benchmark.py --bundle artifacts/bundle --requests artifacts/data/sample_requests.jsonl

Over HTTP against a running service (local Docker or Cloud Run):
    python scripts/benchmark.py --url https://<service>.run.app --requests ... --concurrency 8
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pct(a):
    a = np.asarray(a)
    return {"p50_ms": round(float(np.percentile(a, 50)), 3), "p95_ms": round(float(np.percentile(a, 95)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3), "mean_ms": round(float(a.mean()), 3)}


def in_process(bundle, recs, batch_sizes, device):
    import torch
    from fraudgnn.serve.scorer import OnlineScorer
    rows = []
    for bs in batch_sizes:
        t0 = time.perf_counter()
        s = OnlineScorer(bundle, device=device)
        load_s = time.perf_counter() - t0
        s.score(recs[:bs], ingest=False)                       # warmup
        lat, n, t0 = [], 0, time.perf_counter()
        for i in range(0, len(recs) - bs + 1, bs):
            _, ms = s.score(recs[i:i + bs], ingest=True)
            lat.append(ms)
            n += bs
        wall = time.perf_counter() - t0
        row = {"mode": "in_process", "device": str(s.device), "batch_size": bs,
               "txn_per_sec": round(n / wall, 1), "cold_start_s": round(load_s, 2), **pct(lat)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if s.device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def over_http(url, recs, concurrency, batch_size):
    import httpx
    url = url.rstrip("/") + "/score"
    batches = [recs[i:i + batch_size] for i in range(0, len(recs), batch_size)]
    client = httpx.Client(timeout=30)
    client.post(url, json={"transactions": batches[0], "ingest": False})   # warmup

    def call(b):
        t = time.perf_counter()
        r = client.post(url, json={"transactions": b, "ingest": True})
        r.raise_for_status()
        return (time.perf_counter() - t) * 1000, r.json()["model_latency_ms"]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as ex:
        res = list(ex.map(call, batches))
    wall = time.perf_counter() - t0
    e2e, model = zip(*res)
    row = {"mode": "http", "url": url, "concurrency": concurrency, "batch_size": batch_size,
           "txn_per_sec": round(len(recs) / wall, 1),
           "end_to_end": pct(e2e), "model_only": pct(model)}
    print(json.dumps(row, indent=2))
    return [row]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="artifacts/bundle")
    ap.add_argument("--requests", default="artifacts/data/sample_requests.jsonl")
    ap.add_argument("--batch-sizes", default="1,8,32,128")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    ap.add_argument("--url")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="results/benchmark.json")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.requests)]
    if a.url:
        rows = over_http(a.url, recs, a.concurrency, int(a.batch_sizes.split(",")[0]))
    else:
        rows = in_process(a.bundle, recs, [int(x) for x in a.batch_sizes.split(",")], a.device)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    old = json.load(open(a.out)) if os.path.exists(a.out) else []
    json.dump(old + rows, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
