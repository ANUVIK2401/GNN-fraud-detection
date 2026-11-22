"""Online scorer: the same model and neighbor logic as offline, on live traffic.

State kept in memory:
  * H: encoded vector (hidden dim) of every transaction still referenced.
    Encoding happens once, at ingest, so a request never re-encodes neighbors.
  * per relation, per entity: deque of the K most recent (time, slot) pairs
    plus the total transaction count on that entity.

score() reads state (strictly earlier transactions only), then ingests the
new transactions so later requests see them as neighbors.
"""
import json
import os
import threading
import time
from collections import deque

import numpy as np
import torch

from ..data import EntityIndex, FeatureEncoder, _entity_key, load_json
from ..model import TemporalHeteroGNN


class OnlineScorer:
    def __init__(self, bundle_dir, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(os.path.join(bundle_dir, "model.pt"), map_location=self.device)
        self.model = TemporalHeteroGNN(**ck["init"]).to(self.device).eval()
        self.model.load_state_dict(ck["state_dict"])
        self.threshold, self.k = float(ck["threshold"]), int(ck["k"])
        self.enc = FeatureEncoder.from_dict(load_json(os.path.join(bundle_dir, "encoder.json")))
        self.ents = EntityIndex.from_dict(load_json(os.path.join(bundle_dir, "entities.json")))
        self.relations = self.ents.relations
        self.graph_vocab = load_json(os.path.join(bundle_dir, "graph_vocab.json"))
        self.next_gid = {r: len(v) + 1 for r, v in self.graph_vocab.items()}
        self.lock = threading.Lock()
        self._load_state(os.path.join(bundle_dir, "state.npz"))

    # ---------- state ----------
    def _load_state(self, path):
        st = np.load(path)
        H = torch.from_numpy(st["H"].astype(np.float32))
        cap = max(1024, int(len(H) * 1.25))
        self.H = torch.zeros(cap, self.model.hidden, device=self.device)
        self.H[:len(H)] = H.to(self.device)
        self.n_slots = len(H)
        self.recent = [dict() for _ in self.relations]
        self.count = [dict() for _ in self.relations]
        for j in range(len(self.relations)):
            gid, ptr = st[f"gid_{j}"], st[f"ptr_{j}"]
            slot, tm, cnt = st[f"slot_{j}"], st[f"time_{j}"], st[f"count_{j}"]
            for i, g in enumerate(gid.tolist()):
                a, b = ptr[i], ptr[i + 1]
                self.recent[j][g] = deque(zip(tm[a:b].tolist(), slot[a:b].tolist()), maxlen=self.k)
                self.count[j][g] = int(cnt[i])

    def _gid(self, j, value, create):
        r = self.relations[j]
        key = _entity_key(value)
        if key is None:
            return 0
        g = self.graph_vocab[r].get(key)
        if g is None and create:
            g = self.next_gid[r]
            self.graph_vocab[r][key] = g
            self.next_gid[r] += 1
        return g or 0

    def _append_h(self, h):
        n = len(h)
        if self.n_slots + n > len(self.H):
            grow = torch.zeros(max(len(self.H), n), self.model.hidden, device=self.device)
            self.H = torch.cat([self.H, grow])
        self.H[self.n_slots:self.n_slots + n] = h
        slots = list(range(self.n_slots, self.n_slots + n))
        self.n_slots += n
        return slots

    # ---------- scoring ----------
    @torch.no_grad()
    def score(self, records, ingest=True):
        t0 = time.perf_counter()
        with self.lock:
            b, r, k = len(records), len(self.relations), self.k
            x = torch.from_numpy(self.enc.transform_records(records)).to(self.device)
            ent = torch.from_numpy(self.ents.transform_records(records)).to(self.device)
            times = [int(rec["TransactionDT"]) for rec in records]
            gids = [[self._gid(j, rec.get(self.relations[j]), create=ingest) for j in range(r)]
                    for rec in records]

            nbr = np.zeros((b, r, k), dtype=np.int64)
            mask = np.zeros((b, r, k), dtype=bool)
            dt = np.zeros((b, r, k), dtype=np.float32)
            cnt = np.zeros((b, r), dtype=np.int64)
            for i in range(b):
                t = times[i]
                for j in range(r):
                    g = gids[i][j]
                    if g == 0 or g not in self.recent[j]:
                        continue
                    d = self.recent[j][g]
                    prior = [(tm, s) for tm, s in d if tm < t]
                    cnt[i, j] = self.count[j][g] - (len(d) - len(prior))   # strictly-earlier count
                    for m, (tm, s) in enumerate(prior):
                        pos = k - len(prior) + m          # right-aligned like the offline index
                        nbr[i, j, pos], mask[i, j, pos], dt[i, j, pos] = s, True, t - tm

            rel_mask = torch.tensor([[g > 0 for g in row] for row in gids], device=self.device)
            nbr_t = torch.from_numpy(nbr).to(self.device)
            logit, alpha, _ = self.model(
                x, self.H[nbr_t], torch.from_numpy(mask).to(self.device),
                torch.from_numpy(dt).to(self.device), ent, rel_mask,
                torch.from_numpy(cnt).to(self.device))
            prob = torch.sigmoid(logit).cpu().numpy()
            alpha = alpha.cpu().numpy()

            if ingest:
                slots = self._append_h(self.model.encode(x))
                for i in range(b):
                    for j in range(r):
                        g = gids[i][j]
                        if g == 0:
                            continue
                        self.recent[j].setdefault(g, deque(maxlen=k)).append((times[i], slots[i]))
                        self.count[j][g] = self.count[j].get(g, 0) + 1

        ms = (time.perf_counter() - t0) * 1000
        out = []
        for i, rec in enumerate(records):
            top = np.argsort(-alpha[i])[:3]
            out.append({
                "TransactionID": rec.get("TransactionID"),
                "fraud_probability": float(prob[i]),
                "flagged": bool(prob[i] >= self.threshold),
                "top_relations": [{"relation": self.relations[j], "weight": round(float(alpha[i, j]), 4)}
                                  for j in top if alpha[i, j] > 0],
            })
        return out, ms


def export_bundle(data_dir, gnn_dir, out_dir, history_splits=(0, 1)):
    """Package model + encoders + warm neighbor state from history rows.

    History = train + val by default, so the service starts in the state a
    live system would be in right before the test period begins.
    """
    os.makedirs(out_dir, exist_ok=True)
    ld = lambda n: np.load(os.path.join(data_dir, n + ".npy"))
    X, E, t, split = ld("X"), ld("E_graph"), ld("t"), ld("split")
    hist = np.where(np.isin(split, history_splits))[0]
    ck = torch.load(os.path.join(gnn_dir, "model.pt"), map_location="cpu")
    model = TemporalHeteroGNN(**ck["init"]).eval()
    model.load_state_dict(ck["state_dict"])
    k = int(ck["k"])

    arrays, needed = {}, []
    Eh, th = E[hist], t[hist]
    for j in range(E.shape[1]):
        order = np.lexsort((th, Eh[:, j]))                 # by entity, then time
        g_sorted = Eh[order, j]
        keep = g_sorted > 0
        order, g_sorted = order[keep], g_sorted[keep]
        uniq, start, counts = np.unique(g_sorted, return_index=True, return_counts=True)
        take_from = start + np.maximum(counts - k, 0)
        lens = np.minimum(counts, k)
        idx = np.concatenate([np.arange(a, a + n) for a, n in zip(take_from, lens)]) if len(uniq) else np.array([], int)
        rows = order[idx]                                  # positions within hist
        arrays[f"gid_{j}"] = uniq
        arrays[f"ptr_{j}"] = np.concatenate([[0], np.cumsum(lens)])
        arrays[f"time_{j}"] = th[rows]
        arrays[f"count_{j}"] = counts
        arrays[f"slot_{j}"] = rows
        needed.append(rows)

    needed = np.unique(np.concatenate(needed)) if needed else np.array([], int)
    remap = -np.ones(len(hist), dtype=np.int64)
    remap[needed] = np.arange(len(needed))
    for j in range(E.shape[1]):
        arrays[f"slot_{j}"] = remap[arrays[f"slot_{j}"]]
    with torch.no_grad():
        Xn = torch.from_numpy(X[hist[needed]])
        H = torch.cat([model.encode(Xn[i:i + 65536]) for i in range(0, len(Xn), 65536)]) \
            if len(Xn) else torch.zeros(0, model.hidden)
    np.savez_compressed(os.path.join(out_dir, "state.npz"), H=H.numpy().astype(np.float16), **arrays)

    import shutil
    for f in ("encoder.json", "entities.json", "graph_vocab.json"):
        shutil.copy(os.path.join(data_dir, f), out_dir)
    shutil.copy(os.path.join(gnn_dir, "model.pt"), out_dir)
    with open(os.path.join(out_dir, "bundle.json"), "w") as f:
        json.dump({"history_rows": int(len(hist)), "cached_vectors": int(len(needed)),
                   "history_max_time": int(th.max()) if len(th) else 0}, f)
    print(f"bundle -> {out_dir}: {len(hist):,} history rows, {len(needed):,} cached vectors")
