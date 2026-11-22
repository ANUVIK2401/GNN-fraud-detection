"""Raw IEEE-CIS loading, time-based splitting, and feature/entity encoding.

Everything that is "fit" (feature means/stds, one-hot categories, entity
vocabularies) is fit on the TRAIN split only, so val/test never leak into
preprocessing. The same encoders are reused by the online scorer.
"""
import json
import math
import os

import numpy as np
import pandas as pd

from .config import NON_FEATURE_COLS, ONEHOT_COLS


def load_raw(raw_dir):
    """Merge train_transaction + train_identity and sort by event time."""
    txn = pd.read_csv(os.path.join(raw_dir, "train_transaction.csv"))
    ident_path = os.path.join(raw_dir, "train_identity.csv")
    if os.path.exists(ident_path):
        txn = txn.merge(pd.read_csv(ident_path), on="TransactionID", how="left")
    # Kaggle uses "id-01" in test files and "id_01" in train; normalize.
    txn.columns = [c.replace("-", "_") for c in txn.columns]
    return txn.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)


def time_split(n, train_frac, val_frac):
    """Rows are already sorted by time: first chunk trains, later chunks evaluate."""
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    split = np.zeros(n, dtype=np.int8)       # 0 train, 1 val, 2 test
    split[n_train:n_train + n_val] = 1
    split[n_train + n_val:] = 2
    return split


def _entity_key(v):
    """Stable string key so 13926, 13926.0 and '13926' map to the same node."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if v.is_integer():
            return str(int(v))
    if isinstance(v, (np.integer,)):
        return str(int(v))
    s = str(v)
    return None if s in ("", "nan", "None") else s


class FeatureEncoder:
    """Numeric standardization + one-hot of M1..M9. Fit on train only."""

    def __init__(self, numeric_cols=None, onehot=None, mean=None, std=None):
        self.numeric_cols = numeric_cols or []
        self.onehot = onehot or {}          # col -> list of categories
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    @property
    def dim(self):
        return len(self.numeric_cols) + sum(len(v) for v in self.onehot.values())

    def fit(self, df, relations):
        skip = set(NON_FEATURE_COLS) | set(relations) | set(ONEHOT_COLS)
        self.numeric_cols = [c for c in df.columns
                             if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
        self.onehot = {c: sorted(df[c].dropna().astype(str).unique().tolist())
                       for c in ONEHOT_COLS if c in df.columns}
        num = np.empty((len(df), len(self.numeric_cols)), dtype=np.float32)
        for j, c in enumerate(self.numeric_cols):
            num[:, j] = pd.to_numeric(df[c], errors="coerce")
        if "TransactionAmt" in self.numeric_cols:
            j = self.numeric_cols.index("TransactionAmt")
            num[:, j] = np.log1p(np.clip(num[:, j], 0, None))
        self.mean = np.nanmean(num, axis=0).astype(np.float32)
        self.mean = np.nan_to_num(self.mean)
        self.std = np.nanstd(num, axis=0).astype(np.float32)
        self.std[~np.isfinite(self.std) | (self.std < 1e-6)] = 1.0
        return self

    def _finish(self, num, onehot_vals):
        if "TransactionAmt" in self.numeric_cols:
            j = self.numeric_cols.index("TransactionAmt")
            num[:, j] = np.log1p(np.clip(num[:, j], 0, None))
        num = (num - self.mean) / self.std
        num = np.clip(np.nan_to_num(num, nan=0.0), -10, 10)   # missing -> train mean
        parts = [num]
        for c, cats in self.onehot.items():
            lookup = {k: i for i, k in enumerate(cats)}
            oh = np.zeros((num.shape[0], len(cats)), dtype=np.float32)
            for r, v in enumerate(onehot_vals[c]):
                i = lookup.get(v)
                if i is not None:
                    oh[r, i] = 1.0
            parts.append(oh)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def transform(self, df):
        num = np.empty((len(df), len(self.numeric_cols)), dtype=np.float32)
        for j, c in enumerate(self.numeric_cols):
            num[:, j] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
        vals = {c: [None if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v)
                    for v in df[c].values] if c in df.columns else [None] * len(df)
                for c in self.onehot}
        return self._finish(num, vals)

    def transform_records(self, records):
        """Online path: plain-Python parsing of raw dicts, same math as transform()."""
        recs = [{k.replace("-", "_"): v for k, v in r.items()} for r in records]
        num = np.full((len(recs), len(self.numeric_cols)), np.nan, dtype=np.float32)
        for i, r in enumerate(recs):
            row = num[i]
            for j, c in enumerate(self.numeric_cols):
                v = r.get(c)
                if v is None or isinstance(v, bool):
                    continue
                try:
                    row[j] = float(v)
                except (TypeError, ValueError):
                    pass
        vals = {c: [None if r.get(c) is None or (isinstance(r.get(c), float) and math.isnan(r.get(c)))
                    else str(r.get(c)) for r in recs] for c in self.onehot}
        return self._finish(num, vals)

    def to_dict(self):
        return {"numeric_cols": self.numeric_cols, "onehot": self.onehot,
                "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["numeric_cols"], d["onehot"], d["mean"], d["std"])


class EntityIndex:
    """Per-relation vocab: raw value -> int id. 0 = missing or unseen in train."""

    def __init__(self, relations, vocab=None):
        self.relations = list(relations)
        self.vocab = vocab or {r: {} for r in self.relations}

    def fit(self, df):
        for r in self.relations:
            if r not in df.columns:
                continue
            keys = {k for k in (_entity_key(v) for v in df[r].values) if k is not None}
            self.vocab[r] = {k: i + 1 for i, k in enumerate(sorted(keys))}
        return self

    def sizes(self):
        return [len(self.vocab[r]) + 1 for r in self.relations]

    def transform(self, df):
        out = np.zeros((len(df), len(self.relations)), dtype=np.int64)
        for j, r in enumerate(self.relations):
            if r not in df.columns:
                continue
            voc = self.vocab[r]
            out[:, j] = [voc.get(_entity_key(v), 0) for v in df[r].values]
        return out

    def transform_records(self, records):
        out = np.zeros((len(records), len(self.relations)), dtype=np.int64)
        for i, rec in enumerate(records):
            for j, r in enumerate(self.relations):
                out[i, j] = self.vocab[r].get(_entity_key(rec.get(r)), 0)
        return out

    def raw_keys(self, df, return_vocab=False):
        """Entity keys for EVERY value (including unseen) for the history index.

        Unseen-in-train entities still connect transactions to each other at
        val/test time; they just have no learned embedding. We give each its
        own id above the vocab range so the neighbor index can use them.
        """
        out = np.zeros((len(df), len(self.relations)), dtype=np.int64)
        graph_vocab = {}
        for j, r in enumerate(self.relations):
            voc = dict(self.vocab[r])
            graph_vocab[r] = voc
            if r not in df.columns:
                continue
            nxt = len(voc) + 1
            col = []
            for v in df[r].values:
                k = _entity_key(v)
                if k is None:
                    col.append(0)
                    continue
                if k not in voc:
                    voc[k] = nxt
                    nxt += 1
                col.append(voc[k])
            out[:, j] = col
        return (out, graph_vocab) if return_vocab else out

    def to_dict(self):
        return {"relations": self.relations, "vocab": self.vocab}

    @classmethod
    def from_dict(cls, d):
        return cls(d["relations"], d["vocab"])


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)
