"""Collect metrics from all runs into results/RESULTS.md and plots.

    python scripts/report.py --artifacts artifacts
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

RUNS = [("lgbm", "LightGBM (features + entity ids)"),
        ("gnn", "Temporal hetero GNN"),
        ("hybrid", "GNN embeddings + LightGBM")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    y = np.load(os.path.join(a.artifacts, "data", "y.npy"))
    split = np.load(os.path.join(a.artifacts, "data", "split.npy"))
    y_te = y[split == 2]
    meta = json.load(open(os.path.join(a.artifacts, "data", "meta.json")))

    lines = ["# Results", "",
             f"Time-based split on TransactionDT. Test = last {meta['splits']['test']['rows']:,} "
             f"transactions ({meta['splits']['test']['fraud_rate']:.2%} fraud). "
             "Threshold chosen on validation, applied once to test.", "",
             "| Model | PR-AUC | ROC-AUC | Recall @1% FPR | Precision | Recall | F1 |",
             "|---|---|---|---|---|---|---|"]
    plt.figure(figsize=(6, 5))
    summary = {}
    for d, name in RUNS:
        p = os.path.join(a.artifacts, d, "metrics.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))["test"]
        summary[d] = m
        lines.append(f"| {name} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | {m['recall_at_1pct_fpr']:.4f} "
                     f"| {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
        pt = os.path.join(a.artifacts, d, "p_test.npy")
        if os.path.exists(pt):
            pr, rc, _ = precision_recall_curve(y_te, np.load(pt))
            plt.plot(rc, pr, label=f"{name} (AP {m['pr_auc']:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Test precision-recall")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(a.out, "pr_curve.png"), dpi=150)

    gm = os.path.join(a.artifacts, "gnn", "metrics.json")
    if os.path.exists(gm):
        g = json.load(open(gm))
        att = sorted(g["relation_attention_test_mean"].items(), key=lambda kv: -kv[1])
        lines += ["", "## Which relations the GNN attends to (mean test attention)", "",
                  "| Relation | Weight |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in att]
        h = g["history"]
        plt.figure(figsize=(6, 3.5))
        plt.plot([e["epoch"] for e in h], [e["val_pr_auc"] for e in h], marker="o")
        plt.xlabel("Epoch"); plt.ylabel("Val PR-AUC"); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(a.out, "val_pr_auc.png"), dpi=150)

    bp = os.path.join(a.out, "benchmark.json")
    if os.path.exists(bp):
        rows = [r for r in json.load(open(bp)) if r["mode"] == "in_process"]
        if rows:
            lines += ["", "## Serving latency (in-process, model + neighbor lookup + state update)", "",
                      "| Device | Batch | p50 ms | p99 ms | txn/s |", "|---|---|---|---|---|"]
            lines += [f"| {r['device']} | {r['batch_size']} | {r['p50_ms']} | {r['p99_ms']} | {r['txn_per_sec']} |"
                      for r in rows]
        for r in json.load(open(bp)):
            if r["mode"] == "http":
                lines += ["", f"HTTP ({r['url']}, concurrency {r['concurrency']}, batch {r['batch_size']}): "
                              f"end-to-end p50 {r['end_to_end']['p50_ms']} ms, p99 {r['end_to_end']['p99_ms']} ms, "
                              f"{r['txn_per_sec']} txn/s"]
    open(os.path.join(a.out, "RESULTS.md"), "w").write("\n".join(lines) + "\n")

    # Paste the tables into README between the markers, so numbers never get hand-typed.
    readme = "README.md"
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if os.path.exists(readme):
        txt = open(readme).read()
        if start in txt and end in txt:
            body = "\n".join(lines[2:])          # drop the "# Results" heading
            txt = txt.split(start)[0] + start + "\n" + body + "\n" + end + txt.split(end)[1]
            open(readme, "w").write(txt)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
