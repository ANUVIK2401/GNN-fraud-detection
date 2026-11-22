"""Generate a small IEEE-CIS-shaped dataset for tests, CI and smoke runs.

Planted signal: "fraud rings" reuse the same card / device / email over a
short time window, so graph structure matters. Feature-only signal is weak.
Numbers produced on this data are NOT results; they only prove the code runs.

    python scripts/make_synthetic.py --out ieee-data-synth --n 20000
"""
import argparse
import os

import numpy as np
import pandas as pd


def make(n=20000, seed=0, n_v=10):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(86400, 86400 * 180, n))
    card1 = rng.zipf(1.3, n) % 5000 + 1000
    device = rng.choice([f"dev_{i}" for i in range(800)] + [None] * 400, n)
    email = rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", None], n)
    y = (rng.random(n) < 0.01).astype(int)

    # Fraud rings: 60 rings, each hits 20-60 txns within ~3 days on shared ids.
    for r in range(60):
        start = rng.integers(0, n - 200)
        idx = start + rng.choice(np.arange(200), rng.integers(20, 60), replace=False)
        card1[idx] = 90000 + r
        device[idx] = f"ring_dev_{r}"
        email[idx] = f"ring{r}.com"
        y[idx] = (rng.random(len(idx)) < 0.85).astype(int)

    amt = np.round(np.exp(rng.normal(3.8, 1.0, n)) * (1 + 0.3 * y), 2)
    df = pd.DataFrame({
        "TransactionID": np.arange(3000000, 3000000 + n),
        "isFraud": y, "TransactionDT": t, "TransactionAmt": amt,
        "ProductCD": rng.choice(list("WHCSR"), n),
        "card1": card1, "card2": rng.choice([np.nan] + list(range(100, 600)), n),
        "card3": 150.0, "card4": rng.choice(["visa", "mastercard", "discover"], n),
        "card5": rng.choice([226.0, 224.0, 166.0, np.nan], n),
        "card6": rng.choice(["debit", "credit"], n),
        "addr1": rng.choice([np.nan] + list(range(100, 500)), n), "addr2": 87.0,
        "dist1": rng.exponential(50, n),
        "P_emaildomain": email, "R_emaildomain": rng.choice([None, "gmail.com"], n),
    })
    for i in range(1, 5):
        df[f"C{i}"] = rng.poisson(2, n) + y * rng.poisson(1, n)
    for i in range(1, 4):
        df[f"D{i}"] = np.where(rng.random(n) < 0.3, np.nan, rng.integers(0, 600, n))
    for i in range(1, 5):
        df[f"M{i}"] = rng.choice(["T", "F", None], n)
    V = rng.normal(0, 1, (n, n_v)) + 0.2 * y[:, None]
    df = pd.concat([df, pd.DataFrame(V, columns=[f"V{i}" for i in range(1, n_v + 1)])], axis=1)
    ident = pd.DataFrame({
        "TransactionID": df["TransactionID"], "DeviceType": rng.choice(["desktop", "mobile"], n),
        "DeviceInfo": device, "id_30": rng.choice(["Windows 10", "iOS 13", "Android 9"], n),
        "id_31": rng.choice(["chrome 79.0", "safari 13.0", "edge 18.0"], n),
        "id_33": rng.choice(["1920x1080", "1366x768", "2436x1125"], n),
        "id_01": rng.normal(-5, 5, n),
    })
    ident = ident[rng.random(n) < 0.4]     # identity info exists for ~40% of txns
    return df, ident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ieee-data-synth")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--n-v", type=int, default=10, help="339 matches the real feature width")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    txn, ident = make(args.n, n_v=args.n_v)
    txn.to_csv(os.path.join(args.out, "train_transaction.csv"), index=False)
    ident.to_csv(os.path.join(args.out, "train_identity.csv"), index=False)
    print(f"wrote {len(txn):,} txns ({txn.isFraud.mean():.2%} fraud) to {args.out}")


if __name__ == "__main__":
    main()
