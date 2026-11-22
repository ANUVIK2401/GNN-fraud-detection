"""Train the temporal GNN with early stopping on validation PR-AUC.

    python -m fraudgnn.train --data-dir artifacts/data --out-dir artifacts/gnn
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .graph import TemporalNeighborIndex
from .metrics import best_f1_threshold, evaluate
from .model import TemporalHeteroGNN


class GraphData:
    def __init__(self, data_dir, k):
        ld = lambda n: np.load(os.path.join(data_dir, n + ".npy"))
        self.X, self.E_emb, self.E_graph = ld("X"), ld("E_emb"), ld("E_graph")
        self.t, self.y, self.split = ld("t"), ld("y"), ld("split")
        self.k = k
        self.index = TemporalNeighborIndex(self.E_graph, self.t)

    def neighbors(self, idx):
        nbr, mask, cnt = self.index.query(self.E_graph[idx], self.t[idx], self.k)
        dt = self.t[idx][:, None, None] - self.t[np.clip(nbr, 0, None)]
        dt = np.where(mask, dt, 0)
        return {
            "nbr": nbr, "mask": mask, "dt": dt, "cnt": cnt,
            "ent": self.E_emb[idx], "rel_mask": self.E_graph[idx] > 0,
        }


def to_t(b, device):
    return {
        "nbr": torch.from_numpy(np.clip(b["nbr"], 0, None)).to(device),
        "mask": torch.from_numpy(b["mask"]).to(device),
        "dt": torch.from_numpy(b["dt"]).float().to(device),
        "cnt": torch.from_numpy(b["cnt"]).to(device),
        "ent": torch.from_numpy(b["ent"]).to(device),
        "rel_mask": torch.from_numpy(b["rel_mask"]).to(device),
    }


@torch.no_grad()
def predict(model, gd, X, idx, device, batch_size=4096):
    """Inference uses cached encodings for every row (same trick as serving)."""
    model.eval()
    H = torch.cat([model.encode(X[i:i + 65536]) for i in range(0, len(X), 65536)])
    probs, embs, atts = [], [], []
    for s in range(0, len(idx), batch_size):
        bi = idx[s:s + batch_size]
        b = to_t(gd.neighbors(bi), device)
        logit, alpha, emb = model(X[bi], H[b["nbr"]], b["mask"], b["dt"],
                                  b["ent"], b["rel_mask"], b["cnt"])
        probs.append(torch.sigmoid(logit).cpu())
        embs.append(emb.cpu())
        atts.append(alpha.cpu())
    return torch.cat(probs).numpy(), torch.cat(embs).numpy(), torch.cat(atts).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="artifacts/data")
    ap.add_argument("--out-dir", default="artifacts/gnn")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--k", type=int)
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--max-train", type=int, default=0, help="subsample train rows (smoke tests)")
    args = ap.parse_args()

    cfg = Config.load(os.path.join(args.data_dir, "config.json"))
    for name in ("epochs", "k", "hidden"):
        if getattr(args, name):
            setattr(cfg, "k_neighbors" if name == "k" else name, getattr(args, name))
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gd = GraphData(args.data_dir, cfg.k_neighbors)
    X = torch.from_numpy(gd.X).to(device)
    tr = np.where(gd.split == 0)[0]
    va = np.where(gd.split == 1)[0]
    te = np.where(gd.split == 2)[0]
    if args.max_train:
        tr = rng.choice(tr, min(args.max_train, len(tr)), replace=False)

    vocab = [int(gd.E_emb[:, j].max()) + 1 for j in range(gd.E_emb.shape[1])]
    vocab = [max(v, 2) for v in vocab]
    model = TemporalHeteroGNN(gd.X.shape[1], vocab, cfg.hidden, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos = gd.y[tr].mean()
    pos_weight = torch.tensor(((1 - pos) / max(pos, 1e-6)) ** 0.5, device=device)
    y_t = torch.from_numpy(gd.y.astype(np.float32)).to(device)
    print(f"device={device} rows={len(gd.y):,} train={len(tr):,} val={len(va):,} "
          f"test={len(te):,} pos_weight={pos_weight.item():.2f} "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        t0, total, nb = time.time(), 0.0, 0
        perm = rng.permutation(tr)
        for s in range(0, len(perm), cfg.batch_size):
            bi = perm[s:s + cfg.batch_size]
            b = to_t(gd.neighbors(bi), device)
            h_nbr = model.encode(X[b["nbr"]])
            logit, _, _ = model(X[bi], h_nbr, b["mask"], b["dt"], b["ent"], b["rel_mask"], b["cnt"])
            loss = F.binary_cross_entropy_with_logits(logit, y_t[bi], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
            nb += 1
        p_val, _, _ = predict(model, gd, X, va, device)
        m = evaluate(gd.y[va], p_val, best_f1_threshold(gd.y[va], p_val))
        history.append({"epoch": epoch, "loss": total / nb, "val_pr_auc": m["pr_auc"],
                        "val_roc_auc": m["roc_auc"], "sec": time.time() - t0})
        print(f"epoch {epoch:02d} loss {total / nb:.4f} val PR-AUC {m['pr_auc']:.4f} "
              f"ROC-AUC {m['roc_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if m["pr_auc"] > best:
            best, bad = m["pr_auc"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    p_val, emb_val, _ = predict(model, gd, X, va, device)
    thr = best_f1_threshold(gd.y[va], p_val)
    p_te, emb_te, att_te = predict(model, gd, X, te, device)
    p_tr, emb_tr, _ = predict(model, gd, X, tr, device)

    results = {"model": "temporal_hetero_gnn", "val": evaluate(gd.y[va], p_val, thr),
               "test": evaluate(gd.y[te], p_te, thr), "history": history,
               "relation_attention_test_mean": dict(zip(cfg.relations, att_te.mean(0).round(4).tolist())),
               "config": cfg.__dict__}
    torch.save({"state_dict": model.state_dict(),
                "init": {"in_dim": gd.X.shape[1], "vocab_sizes": vocab,
                         "hidden": cfg.hidden, "dropout": cfg.dropout},
                "threshold": thr, "k": cfg.k_neighbors}, os.path.join(args.out_dir, "model.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.save(os.path.join(args.out_dir, "p_test.npy"), p_te)
    # Embeddings feed the GNN + GBDT hybrid. Train rows only if they were all used.
    np.save(os.path.join(args.out_dir, "emb_val.npy"), emb_val.astype(np.float16))
    np.save(os.path.join(args.out_dir, "emb_test.npy"), emb_te.astype(np.float16))
    np.save(os.path.join(args.out_dir, "emb_train.npy"), emb_tr.astype(np.float16))
    np.save(os.path.join(args.out_dir, "train_idx.npy"), tr)
    print(json.dumps({k: results["test"][k] for k in
                      ("roc_auc", "pr_auc", "recall_at_1pct_fpr", "precision", "recall", "f1")}, indent=2))


if __name__ == "__main__":
    main()
