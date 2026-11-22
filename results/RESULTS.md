# Results

Time-based split on TransactionDT. Test = last 118,108 transactions (3.44% fraud). Threshold chosen on validation, applied once to test.

| Model | PR-AUC | ROC-AUC | Recall @1% FPR | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| LightGBM (features + entity ids) | 0.5009 | 0.8764 | 0.4368 | 0.6097 | 0.4355 | 0.5081 |
| Temporal hetero GNN | 0.3994 | 0.8432 | 0.3189 | 0.4951 | 0.3474 | 0.4083 |
| GNN embeddings + LightGBM | 0.4780 | 0.8569 | 0.4134 | 0.5396 | 0.4446 | 0.4875 |

## Which relations the GNN attends to (mean test attention)

| Relation | Weight |
|---|---|
| card1 | 0.373 |
| card2 | 0.233 |
| addr1 | 0.078 |
| card6 | 0.077 |
| R_emaildomain | 0.044 |
| P_emaildomain | 0.036 |
| card3 | 0.036 |
| ProductCD | 0.026 |
| card5 | 0.024 |
| DeviceInfo | 0.015 |
| card4 | 0.014 |
| addr2 | 0.012 |
| id_33 | 0.009 |
| id_30 | 0.009 |
| id_31 | 0.008 |
| DeviceType | 0.007 |

## Serving latency (in-process, model + neighbor lookup + state update)

| Device | Batch | p50 ms | p99 ms | txn/s |
|---|---|---|---|---|
| cpu | 1 | 1.175 | 1.364 | 808.2 |
| cpu | 8 | 3.005 | 3.435 | 2541.7 |
| cpu | 32 | 8.306 | 9.015 | 3693.3 |
| cpu | 128 | 28.723 | 31.843 | 4205.3 |

HTTP on Cloud Run (2 vCPU, concurrency 4, batch 1, from a GCE client in us-central1): end-to-end p50 21.1 ms, p95 36.1 ms, p99 50.3 ms, 174.7 txn/s. Model time inside the service: p50 6.8 ms, p99 21.0 ms.
