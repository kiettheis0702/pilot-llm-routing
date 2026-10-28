"""Read-only G0 evidence probe; exercises existing code without rebuilding data.

Run from the repository root with:
    python -B -m reports.audits.g0_checks_2026_10_28 --data-dir PATH

This is an audit probe, not the future B01 contract or a benchmark runner.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

from scripts.data_gate import (
    OfflineBandit, RandomSmokeLearner, make_splits, query_hash, run_bandit_round,
)
from scripts.fetch_data import sha256, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    report = json.loads((root / "reports/data-gate.json").read_text(encoding="utf-8"))
    sources = json.loads((root / "configs/data_sources.json").read_text(encoding="utf-8"))
    verified = {}
    for key, source in sources.items():
        path = args.data_dir / key / Path(source["file"]).name
        verify(path, source)
        verified[key] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    for name, expected in report["config_sha256"].items():
        assert sha256(root / "configs" / name) == expected, name
    prepared = args.data_dir / "processed/routerbench_0shot"
    for name, expected in report["prepared"]["file_sha256"].items():
        assert sha256(prepared / name) == expected, name
    queries = pd.read_parquet(prepared / "queries.parquet")
    outcomes = pd.read_parquet(prepared / "outcomes.parquet")
    splits = pd.read_csv(prepared / "splits.csv", dtype={"query_id": str, "query_hash": str})
    assert len(queries) == 36497 and len(outcomes) == 401467
    assert queries.query_id.is_unique and splits.query_id.is_unique
    assert not outcomes.duplicated(["query_id", "arm_id"]).any()
    assert set(outcomes.query_id) == set(queries.query_id) == set(splits.query_id)
    assert outcomes.arm_id.nunique() == 11
    assert (outcomes.groupby("query_id").arm_id.nunique() == 11).all()
    assert np.isfinite(outcomes.reward).all() and outcomes.reward.between(0, 1).all()
    assert np.isfinite(outcomes.actual_cost).all() and (outcomes.actual_cost >= 0).all()
    assert outcomes.estimated_cost.isna().all()
    assert queries["query"].map(query_hash).equals(queries.query_hash)
    assert splits.groupby("query_hash").split.nunique().max() == 1
    pd.testing.assert_frame_equal(splits, make_splits(queries, seed=42))
    for _, group in splits.groupby("split"):
        assert group.stream_position.tolist() == list(range(len(group)))
    learning_ids = splits.loc[splits.split == "learning", "query_id"].iloc[:10].tolist()
    small_queries = queries[queries.query_id.isin(learning_ids)]
    small_outcomes = outcomes[outcomes.query_id.isin(learning_ids)]

    def replay():
        environment = OfflineBandit(small_queries, small_outcomes)
        learner = RandomSmokeLearner(42)
        trace = [asdict(run_bandit_round(environment, qid, learner)) for qid in learning_ids]
        assert len(learner.updates) == 10
        return trace

    first, second = replay(), replay()
    assert first == second
    assert first[0] == report["smoke"]["feedback"]

    # These observations reveal missing future contract validation. They are not
    # tests that require incorrect behavior to remain unchanged.
    one_query = small_queries.iloc[:1]
    query_id = one_query.query_id.iloc[0]
    one_outcome = small_outcomes[small_outcomes.query_id == query_id].iloc[:1].copy()
    arm_id = one_outcome.arm_id.iloc[0]
    one_outcome.loc[:, "reward"] = np.nan
    try:
        feedback = OfflineBandit(one_query, one_outcome).step(query_id, arm_id)
        accepts_nan = bool(np.isnan(feedback.reward))
    except ValueError:
        accepts_nan = False
    # Current pool is derived per query. A one-arm pool is valid in isolation;
    # there is no configured catalog against which consistency can be enforced.
    variable_outcomes = small_outcomes[
        (small_outcomes.query_id != query_id) | (small_outcomes.arm_id == arm_id)
    ]
    try:
        variable_env = OfflineBandit(small_queries, variable_outcomes)
        derived_pool_sizes = sorted({len(variable_env.observe(qid).arm_ids) for qid in learning_ids})
    except ValueError:
        derived_pool_sizes = []
    evidence = {
        "audit_date": "2026-10-28",
        "code_commit_audited": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "implementation_sha256": {name: sha256(root / name) for name in (
            "scripts/data_gate.py", "scripts/audit_data.py", "scripts/data_io.py",
            "scripts/fetch_data.py", "tests/test_data_gate.py",
        )},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "pyarrow": pyarrow.__version__},
        "source_verification": verified,
        "stored_prepared_hashes_match": True,
        "saved_split_matches_fresh_in_memory_split": True,
        "query_rows": len(queries), "outcome_rows": len(outcomes),
        "split_counts": splits.split.value_counts().to_dict(),
        "all_stored_query_ids_have_11_unique_arms": True,
        "replay_rounds": 10, "same_seed_replay_identical": True,
        "first_feedback_matches_saved_smoke": True,
        "selected_feedback_keys": list(first[0]),
        "replay_trace_sha256": hashlib.sha256(json.dumps(first, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "negative_probes": {"constructor_accepts_nan_reward": accepts_nan,
                            "per_query_derived_pool_sizes_without_catalog_validation": derived_pool_sizes},
        "data_files_written": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
