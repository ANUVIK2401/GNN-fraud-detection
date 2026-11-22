"""Raw Kaggle CSVs -> model-ready arrays.

    python -m fraudgnn.preprocess --raw-dir ieee-data --out-dir artifacts/data
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .config import Config
from .data import EntityIndex, FeatureEncoder, load_raw, save_json, time_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="ieee-data")
    ap.add_argument("--out-dir", default="artifacts/data")
    ap.add_argument("--n-sample-requests", type=int, default=2000)
    args = ap.parse_args()
    cfg = Config()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    df = load_raw(args.raw_dir)
    split = time_split(len(df), cfg.train_frac, cfg.val_frac)
    train_df = df[split == 0]
    print(f"loaded {len(df):,} rows in {time.time() - t0:.0f}s")

    enc = FeatureEncoder().fit(train_df, cfg.relations)
    ents = EntityIndex(cfg.relations).fit(train_df)

    X = enc.transform(df)
    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "E_emb.npy"), ents.transform(df))
    E_graph, graph_vocab = ents.raw_keys(df, return_vocab=True)
    np.save(os.path.join(args.out_dir, "E_graph.npy"), E_graph)
    save_json(graph_vocab, os.path.join(args.out_dir, "graph_vocab.json"))
    np.save(os.path.join(args.out_dir, "t.npy"), df["TransactionDT"].values.astype(np.int64))
    np.save(os.path.join(args.out_dir, "y.npy"), df["isFraud"].values.astype(np.int8))
    np.save(os.path.join(args.out_dir, "split.npy"), split)
    np.save(os.path.join(args.out_dir, "txn_id.npy"), df["TransactionID"].values.astype(np.int64))
    save_json(enc.to_dict(), os.path.join(args.out_dir, "encoder.json"))
    save_json(ents.to_dict(), os.path.join(args.out_dir, "entities.json"))
    cfg.save(os.path.join(args.out_dir, "config.json"))

    y = df["isFraud"].values
    meta = {
        "n_rows": int(len(df)), "n_features": int(X.shape[1]),
        "n_relations": len(cfg.relations),
        "entity_vocab_sizes": dict(zip(cfg.relations, ents.sizes())),
        "splits": {name: {"rows": int((split == s).sum()),
                          "fraud_rate": float(y[split == s].mean()),
                          "dt_min": int(df["TransactionDT"].values[split == s].min()),
                          "dt_max": int(df["TransactionDT"].values[split == s].max())}
                   for s, name in enumerate(["train", "val", "test"])},
    }
    save_json(meta, os.path.join(args.out_dir, "meta.json"))

    # Raw test rows, exactly as a client would send them, for the HTTP benchmark.
    sample = df[split == 2].head(args.n_sample_requests).drop(columns=["isFraud"])
    with open(os.path.join(args.out_dir, "sample_requests.jsonl"), "w") as f:
        for rec in sample.to_dict(orient="records"):
            rec = {k: (None if isinstance(v, float) and np.isnan(v) else
                       (v.item() if hasattr(v, "item") else v)) for k, v in rec.items()}
            f.write(json.dumps(rec) + "\n")

    print(json.dumps(meta["splits"], indent=2))
    print(f"features: {X.shape[1]}  done in {time.time() - t0:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
