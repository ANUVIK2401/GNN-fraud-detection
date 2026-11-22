#!/usr/bin/env bash
# Run ON the VM (inside tmux so an SSH drop does not kill it):
#   tmux new -s run
#   curl -sL https://raw.githubusercontent.com/ANUVIK2401/GNN-fraud-detection/main/scripts/vm_run.sh | bash
set -euo pipefail
sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip python3-venv git unzip
mkdir -p ~/.kaggle && [ -f ~/kaggle.json ] && mv ~/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json || true
[ -d GNN-fraud-detection ] || git clone https://github.com/ANUVIK2401/GNN-fraud-detection
cd GNN-fraud-detection
python3 -m venv .venv && . .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements.txt
pytest -q
[ -d ieee-data ] || bash scripts/download_data.sh
time bash scripts/run_all.sh
echo "DONE. Results in results/ and README.md. Bundle in artifacts/bundle."
