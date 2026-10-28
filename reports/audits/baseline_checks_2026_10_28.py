"""B01–B03 verification only: read local data, use synthetic contexts, save a small run.

Run from the repository root. No download, embedding or research policy training.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.bandit_contracts import ContextTable
from scripts.evaluation import run_learning, write_run
from scripts.fetch_data import sha256, verify
from scripts.splits import load_split_manifest


class FixedArmProbe:
    """Verification fixture only, not the B04 baseline implementation."""

    def __init__(self, arm_id):
        self.arm_id = arm_id

    def reset(self, seed):
        self.updates = 0

    def select(self, context):
        assert context.shape == (2,) and np.isfinite(context).all()
        return self.arm_id

    def observe(self, arm_id, context, reward):
        assert arm_id == self.arm_id and 0 <= reward <= 1
        self.updates += 1

    def predict_rewards(self, context):
        raise AssertionError("Prediction access is unnecessary for this probe")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    report = json.loads((root / "reports/data-gate.json").read_text(encoding="utf-8"))
    sources = json.loads((root / "configs/data_sources.json").read_text(encoding="utf-8"))
    prepared = args.data_dir / "processed/routerbench_0shot"
    pinned_paths = {}
    for key, source in sources.items():
        path = args.data_dir / key / Path(source["file"]).name
        verify(path, source)
        pinned_paths[str(path)] = source["sha256"]
    for name, digest in report["prepared"]["file_sha256"].items():
        path = prepared / name
        assert sha256(path) == digest
        pinned_paths[str(path)] = digest
    for name, digest in report["config_sha256"].items():
        assert sha256(root / "configs" / name) == digest
    queries = pd.read_parquet(prepared / "queries.parquet")
    outcomes = pd.read_parquet(prepared / "outcomes.parquet")
    splits = load_split_manifest(prepared / "splits.csv", queries,
        expected_sha256=report["prepared"]["file_sha256"]["splits.csv"], seed=42)
    assert splits.counts == {"tuning": 1000, "learning": 32270, "deployment": 3227}
    selection = splits.select("learning", limit=10, subset_seed=7)
    contexts = ContextTable({qid: [1.0, i / 10] for i, qid in enumerate(selection.query_ids)},
                            label="synthetic-constant-plus-stream-index; contract verification only")
    arm = sorted(outcomes.arm_id.unique())[0]
    first_policy, second_policy = FixedArmProbe(arm), FixedArmProbe(arm)
    first = run_learning(first_policy, queries, outcomes, contexts, selection, seed=42)
    second = run_learning(second_policy, queries, outcomes, contexts, selection, seed=42)
    assert first == second and first_policy.updates == second_policy.updates == 10
    for path, digest in pinned_paths.items():
        assert sha256(Path(path)) == digest, path
    evidence = {"date": "2026-10-28", "scope": "B01-B03 contract verification, not research results",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "rows": {"queries": len(queries), "outcomes": len(outcomes)},
        "split_counts": splits.counts, "same_seed_trace_identical": True,
        "source_and_prepared_hashes_unchanged": pinned_paths,
        "implementation_sha256": {name: sha256(root / name) for name in (
            "scripts/data_gate.py", "scripts/bandit_contracts.py", "scripts/splits.py", "scripts/evaluation.py")},
        "rounds": 10, "data_files_written": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3)}
    write_run(first, args.output_dir)
    (args.output_dir / "verification.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"verification": evidence, "metrics": first.summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
