"""FastAPI service.

    BUNDLE_DIR=artifacts/bundle uvicorn fraudgnn.serve.app:app --port 8080
"""
import os
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .scorer import OnlineScorer

_scorer = None
_lat = deque(maxlen=10000)


def scorer():
    global _scorer
    if _scorer is None:
        _scorer = OnlineScorer(os.environ.get("BUNDLE_DIR", "artifacts/bundle"))
    return _scorer


class ScoreRequest(BaseModel):
    transactions: list[dict[str, Any]] = Field(..., min_length=1, max_length=1024)
    ingest: bool = True   # add these txns to the graph after scoring


@asynccontextmanager
async def lifespan(_app):
    scorer()          # load model + warm graph state before taking traffic
    yield


app = FastAPI(title="Temporal GNN fraud scoring", version="2.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    s = scorer()
    return {"status": "ok", "device": str(s.device), "cached_vectors": s.n_slots,
            "relations": len(s.relations), "threshold": s.threshold}


@app.post("/score")
def score(req: ScoreRequest):
    for t in req.transactions:
        if "TransactionDT" not in t:
            raise HTTPException(422, "every transaction needs TransactionDT")
    results, ms = scorer().score(req.transactions, ingest=req.ingest)
    _lat.append(ms)
    return {"results": results, "model_latency_ms": round(ms, 3)}


@app.get("/stats")
def stats():
    if not _lat:
        return {"requests": 0}
    a = np.array(_lat)
    return {"requests": len(a), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99))}
