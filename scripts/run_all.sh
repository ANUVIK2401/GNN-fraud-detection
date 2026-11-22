#!/usr/bin/env bash
# Full pipeline: preprocess -> GNN -> LightGBM + hybrid -> serving bundle -> benchmark -> report.
# Usage: bash scripts/run_all.sh [raw_dir]      (default: ieee-data)
set -euo pipefail
RAW=${1:-ieee-data}
mkdir -p results
python -m fraudgnn.preprocess --raw-dir "$RAW" --out-dir artifacts/data
python -m fraudgnn.train      --data-dir artifacts/data --out-dir artifacts/gnn 2>&1 | tee results/train_log.txt
python -m fraudgnn.baseline   --data-dir artifacts/data --gnn-dir artifacts/gnn --out-dir artifacts 2>&1 | tee results/baseline_log.txt
python -m fraudgnn.export     --data-dir artifacts/data --gnn-dir artifacts/gnn --out-dir artifacts/bundle
rm -f results/benchmark.json
python scripts/benchmark.py --bundle artifacts/bundle --requests artifacts/data/sample_requests.jsonl --device cpu
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  python scripts/benchmark.py --bundle artifacts/bundle --requests artifacts/data/sample_requests.jsonl --device cuda
fi
cp artifacts/gnn/metrics.json results/gnn_metrics.json
cp artifacts/lgbm/metrics.json results/lgbm_metrics.json
[ -f artifacts/hybrid/metrics.json ] && cp artifacts/hybrid/metrics.json results/hybrid_metrics.json
python scripts/report.py --artifacts artifacts --out results
