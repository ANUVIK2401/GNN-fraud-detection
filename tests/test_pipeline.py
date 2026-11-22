import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fraudgnn.data import EntityIndex, FeatureEncoder, time_split  # noqa: E402
from fraudgnn.graph import TemporalNeighborIndex  # noqa: E402


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True)


def test_time_split_is_ordered_and_disjoint():
    s = time_split(100, 0.6, 0.2)
    assert (s[:60] == 0).all() and (s[60:80] == 1).all() and (s[80:] == 2).all()


def test_neighbors_are_strictly_past_and_most_recent():
    rng = np.random.default_rng(0)
    n, k = 3000, 5
    t = np.sort(rng.integers(0, 500, n))           # many ties on purpose
    E = rng.integers(0, 40, (n, 3))
    idx = TemporalNeighborIndex(E, t)
    q = rng.choice(n, 300, replace=False)
    nbr, mask, cnt = idx.query(E[q], t[q], k)
    for bi, row in enumerate(q):
        for j in range(3):
            e = E[row, j]
            earlier = np.where((E[:, j] == e) & (t < t[row]))[0] if e > 0 else np.array([], int)
            got = nbr[bi, j][mask[bi, j]]
            assert cnt[bi, j] == len(earlier)
            assert (t[got] < t[row]).all()                     # no same-time or future rows
            assert row not in got
            assert len(got) == min(k, len(earlier))
            if len(got):                                       # they are the latest k
                assert t[got].min() >= np.sort(t[earlier])[-len(got)]


def test_encoders_fit_on_train_and_online_path_matches():
    df = pd.DataFrame({"TransactionAmt": [10.0, 20.0, np.nan, 5.0], "C1": [1, 2, 3, 4],
                       "M1": ["T", "F", None, "T"], "card1": [1, 2, 3, 99],
                       "TransactionDT": [1, 2, 3, 4], "TransactionID": [1, 2, 3, 4], "isFraud": [0, 1, 0, 0]})
    enc = FeatureEncoder().fit(df.iloc[:3], ["card1"])
    ents = EntityIndex(["card1"]).fit(df.iloc[:3])
    assert ents.transform(df)[3, 0] == 0                       # unseen in train -> 0
    off = enc.transform(df)
    on = enc.transform_records(df.to_dict(orient="records"))
    np.testing.assert_allclose(off, on)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    run("scripts/make_synthetic.py", "--out", str(d / "raw"), "--n", "4000")
    run("-m", "fraudgnn.preprocess", "--raw-dir", str(d / "raw"), "--out-dir", str(d / "data"),
        "--n-sample-requests", "150")
    run("-m", "fraudgnn.train", "--data-dir", str(d / "data"), "--out-dir", str(d / "gnn"), "--epochs", "2")
    run("-m", "fraudgnn.export", "--data-dir", str(d / "data"), "--gnn-dir", str(d / "gnn"),
        "--out-dir", str(d / "bundle"))
    return d


def test_online_scores_match_offline(trained):
    from fraudgnn.serve.scorer import OnlineScorer
    s = OnlineScorer(str(trained / "bundle"), device="cpu")
    recs = [json.loads(l) for l in open(trained / "data" / "sample_requests.jsonl")]
    online = np.array([s.score([r])[0][0]["fraud_probability"] for r in recs])
    offline = np.load(trained / "gnn" / "p_test.npy")[:len(recs)]
    assert np.abs(online - offline).max() < 1e-3               # fp16 state cache tolerance


def test_api(trained, monkeypatch):
    monkeypatch.setenv("BUNDLE_DIR", str(trained / "bundle"))
    from fastapi.testclient import TestClient
    import fraudgnn.serve.app as app_mod
    app_mod._scorer = None
    rec = json.loads(open(trained / "data" / "sample_requests.jsonl").readline())
    with TestClient(app_mod.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/score", json={"transactions": [rec]}).json()
        assert 0.0 <= r["results"][0]["fraud_probability"] <= 1.0
        assert c.post("/score", json={"transactions": [{"x": 1}]}).status_code == 422
        assert c.get("/stats").json()["requests"] == 1
