#!/usr/bin/env bash
# Needs a Kaggle API token at ~/.kaggle/kaggle.json and the competition
# rules accepted once at https://www.kaggle.com/c/ieee-fraud-detection/rules
set -euo pipefail
pip install -q kaggle
mkdir -p ieee-data
kaggle competitions download -c ieee-fraud-detection -p ieee-data
unzip -o -q ieee-data/ieee-fraud-detection.zip -d ieee-data
rm ieee-data/ieee-fraud-detection.zip
ls -lh ieee-data
