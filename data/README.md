# Local datasets

Do not commit downloaded data, conversations, response tables, embeddings, or checkpoints. The source pins are in [configs/data_sources.json](../configs/data_sources.json), with the audit in [docs/data-gate.md](../docs/data-gate.md).

From the repository root, install the locked Python environment and fetch the three public files (about 455 MB total):

```sh
uv sync --locked
uv run --locked python scripts/fetch_data.py --data-dir ./data
uv run --locked python scripts/audit_data.py --data-dir ./data
uv run --locked python -m pytest -q
```

To keep data outside the checkout on this Windows machine:

```powershell
uv run --locked python scripts/fetch_data.py --data-dir 'D:\DOWNLOAD GG\PILOT_DATA'
uv run --locked python scripts/audit_data.py --data-dir 'D:\DOWNLOAD GG\PILOT_DATA'
```

`fetch_data.py` downloads public data from immutable revisions and checks both file size and upstream SHA-256 before accepting it. Rerunning verifies cached files. Partial files are never deserialized. A checksum failure stops the process; inspect it instead of changing the expected hash to match an unexplained file.

`audit_data.py` is offline and verifies hashes again. It audits both RouterBench variants and Arena 55K, prepares only the provisional 0-shot routing data, writes aggregate JSON to `reports/data-gate.json`, and exports local Parquet tables plus deterministic split IDs under `processed/routerbench_0shot/`.

The 1.20 GB raw RouterBench file is not needed for this initial audit and was not downloaded. The gated 33K Arena dataset was not downloaded. There are no embedding/API calls in this workflow.

The preparation is provisional: model aliases and exact paper dataset choices remain unresolved. `estimated_cost` is deliberately null until a tuning-only cost estimator exists. See the audit before using these files for experiments.
