"""LightGBM baseline and GNN-embedding + LightGBM hybrid, on the same split.

    python -m fraudgnn.baseline --data-dir artifacts/data --gnn-dir artifacts/gnn --out-dir artifacts

Writes artifacts/lgbm/metrics.json and artifacts/hybrid/metrics.json.
Both use the validation split for early stopping and for the decision
threshold; test is touched exactly once.
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np

from .metrics import best_f1_threshold, evaluate

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=100,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_cat_to_onehot=8, cat_smooth=20, verbose=-1, seed=42, metric="average_precision")


def fit_eval(name, Xtr, ytr, Xva, yva, Xte, yte, cat_idx, out_dir):
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat_idx, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, categorical_feature=cat_idx)
    booster = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    p_va = booster.predict(Xva, num_iteration=booster.best_iteration)
    p_te = booster.predict(Xte, num_iteration=booster.best_iteration)
    thr = best_f1_threshold(yva, p_va)
    res = {"model": name, "val": evaluate(yva, p_va, thr), "test": evaluate(yte, p_te, thr),
           "best_iteration": int(booster.best_iteration)}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.save(os.path.join(out_dir, "p_test.npy"), p_te)
    booster.save_model(os.path.join(out_dir, "model.txt"))
    print(f"[{name}] test PR-AUC {res['test']['pr_auc']:.4f} ROC-AUC {res['test']['roc_auc']:.4f} "
          f"recall@1%FPR {res['test']['recall_at_1pct_fpr']:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="artifacts/data")
    ap.add_argument("--gnn-dir", default="artifacts/gnn")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--skip-hybrid", action="store_true")
    args = ap.parse_args()
    ld = lambda n: np.load(os.path.join(args.data_dir, n + ".npy"))
    X, E, y, split = ld("X"), ld("E_emb"), ld("y"), ld("split")

    # Entity ids as LightGBM categoricals (0 = missing/unseen). Same info the GNN gets
    # about which card/device/email a txn used, minus the neighbor structure.
    F = np.hstack([X, E.astype(np.float32)])
    cat_idx = list(range(X.shape[1], F.shape[1]))
    tr, va, te = (np.where(split == s)[0] for s in (0, 1, 2))
    fit_eval("lightgbm", F[tr], y[tr], F[va], y[va], F[te], y[te], cat_idx,
             os.path.join(args.out_dir, "lgbm"))

    if args.skip_hybrid or not os.path.exists(os.path.join(args.gnn_dir, "emb_test.npy")):
        return
    g = lambda n: np.load(os.path.join(args.gnn_dir, n + ".npy")).astype(np.float32)
    tr_used = np.load(os.path.join(args.gnn_dir, "train_idx.npy"))
    order = np.argsort(tr_used)
    tr_h, emb_tr = tr_used[order], g("emb_train")[order]
    fit_eval("gnn_lightgbm_hybrid",
             np.hstack([F[tr_h], emb_tr]), y[tr_h],
             np.hstack([F[va], g("emb_val")]), y[va],
             np.hstack([F[te], g("emb_test")]), y[te],
             cat_idx, os.path.join(args.out_dir, "hybrid"))


if __name__ == "__main__":
    main()
