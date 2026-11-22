"""Build the serving bundle.

    python -m fraudgnn.export --data-dir artifacts/data --gnn-dir artifacts/gnn --out-dir artifacts/bundle
"""
import argparse

from .serve.scorer import export_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="artifacts/data")
    ap.add_argument("--gnn-dir", default="artifacts/gnn")
    ap.add_argument("--out-dir", default="artifacts/bundle")
    a = ap.parse_args()
    export_bundle(a.data_dir, a.gnn_dir, a.out_dir)


if __name__ == "__main__":
    main()
