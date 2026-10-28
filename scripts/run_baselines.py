"""Bounded learning-only baseline comparison; synthetic features, never paper results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.bandit_contracts import ContextTable
from scripts.data_gate import make_splits, query_hash
from scripts.evaluation import run_learning, write_run
from scripts.fetch_data import sha256
from scripts.policies import EpochGreedy, ExploreOnly, FixedPolicy, LinearUCB, RandomPolicy
from scripts.splits import FrozenSplits, load_split_manifest


def _toy(rows, dimension, seed):
    rng = np.random.default_rng(seed)
    count = rows + math.ceil(rows / 10) + 12
    ids = [f"toy-{i:05d}" for i in range(count)]
    x = rng.normal(size=(count, dimension))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    theta = rng.normal(size=(3, dimension))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True)
    scores = np.clip((x @ theta.T + 1) / 2, 0, 1)
    queries = pd.DataFrame({"query_id": ids, "query": [f"toy query {i}" for i in range(count)], "task": "synthetic"})
    queries["query_hash"] = queries["query"].map(query_hash)
    outcomes = pd.DataFrame({"query_id": np.repeat(ids, 3), "arm_id": np.tile(["a", "b", "c"], count),
                             "reward": scores.ravel(), "actual_cost": np.tile([.001, .002, .003], count)})
    frame = make_splits(queries, seed=42, tuning_size=10)
    digest = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    splits = FrozenSplits(frame, sha256=digest, seed=42, tuning_size=10, learning_ratio=10)
    selection = splits.select("learning", limit=rows, subset_seed=seed)
    contexts = ContextTable(dict(zip(ids, x)), label="synthetic normalized Gaussian toy features")
    return queries, outcomes, contexts, selection


def _routerbench(data_dir, rows, dimension, seed):
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "reports/data-gate.json").read_text(encoding="utf-8"))
    prepared = Path(data_dir) / "processed/routerbench_0shot"
    for name, expected in report["prepared"]["file_sha256"].items():
        if sha256(prepared / name) != expected:
            raise ValueError(f"Prepared data SHA-256 mismatch: {name}")
    queries = pd.read_parquet(prepared / "queries.parquet")
    outcomes = pd.read_parquet(prepared / "outcomes.parquet")
    splits = load_split_manifest(prepared / "splits.csv", queries,
        expected_sha256=report["prepared"]["file_sha256"]["splits.csv"])
    selection = splits.select("learning", limit=rows, subset_seed=seed)
    text = queries.set_index("query_id")["query"]
    vectors = {}
    for query_id in selection.query_ids:
        raw = hashlib.shake_256(text.loc[query_id].encode("utf-8")).digest(dimension * 4)
        x = np.frombuffer(raw, dtype="<u4").astype(float) / (2**32 - 1) * 2 - 1
        vectors[query_id] = x / np.linalg.norm(x)
    contexts = ContextTable(vectors, label="DEVIATION: synthetic SHAKE256 query features; no semantic embedding")
    return queries, outcomes, contexts, selection


def plot_traces(output_dir: Path, methods: tuple[str, ...]) -> None:
    """Regenerate standard Matplotlib figures solely from saved trace CSV files."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in methods:
        trace = pd.read_csv(output_dir / method / "trace.csv")
        ax.plot(trace["round"], trace.cumulative_empirical_oracle_regret, label=method)
    ax.set(xlabel="Learning round", ylabel="Cumulative empirical oracle regret",
           title="Baseline smoke check — synthetic contexts (not paper results)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "regret.png", dpi=150)
    fig.savefig(output_dir / "regret.svg", metadata={"Date": None})
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["toy", "routerbench"], default="toy")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--ridge", type=float, default=1)
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if not 1 <= args.rows <= 1000 or not 1 <= args.dimension <= 32 or args.seed < 0:
        raise ValueError("Smoke limits: rows 1..1000, dimension 1..32, nonnegative seed")
    if (not np.isfinite(args.alpha) or args.alpha < 0 or not np.isfinite(args.ridge)
            or args.ridge <= 0 or args.window < 1):
        raise ValueError("Invalid alpha, ridge or window")
    if args.source == "routerbench" and args.data_dir is None:
        raise ValueError("RouterBench requires --data-dir")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    inputs = (_toy(args.rows, args.dimension, args.seed) if args.source == "toy" else
              _routerbench(args.data_dir, args.rows, args.dimension, args.seed))
    queries, outcomes, contexts, selection = inputs
    arms = tuple(sorted(outcomes.arm_id.unique()))
    methods = {"random": RandomPolicy(arms, args.dimension),
               "linucb": LinearUCB(arms, args.dimension, alpha=args.alpha, ridge=args.ridge),
               "explore-only": ExploreOnly(arms, args.dimension, ridge=args.ridge),
               "epoch-greedy": EpochGreedy(arms, args.dimension, window=args.window, ridge=args.ridge)}
    methods.update({f"fixed-{i:02d}": FixedPolicy(arms, args.dimension, arm_id=arm) for i, arm in enumerate(arms)})
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.strip())
    code = {"git_commit": commit, "working_tree_dirty": dirty,
            "implementation_sha256": {name: sha256(root / name) for name in (
                "scripts/policies.py", "scripts/run_baselines.py", "scripts/evaluation.py",
                "scripts/bandit_contracts.py", "scripts/splits.py", "scripts/data_gate.py", "uv.lock")}}
    config = {"seed": args.seed, "rows": args.rows, "dimension": args.dimension,
              "alpha": args.alpha, "ridge": args.ridge, "window": args.window}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics = {}
    for name, policy in methods.items():
        result = run_learning(policy, queries, outcomes, contexts, selection, seed=args.seed)
        result.summary.update(configuration={**config, "fixed_arm": getattr(policy, "arm_id", None)},
                              code=code, deviations=["Synthetic contexts; smoke verification only"])
        write_run(result, args.output_dir / name)
        metrics[name] = result.summary["metrics"]
    plot_traces(args.output_dir, tuple(methods))
    summary = {"source": args.source, "configuration": config, "code": code,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
               "deviations": ["Synthetic contexts; not comparable to paper results",
                              "ExploreOnly and EpochGreedy schedule are PAPER_INFERRED"],
               "metrics": metrics, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "methods": len(methods),
                      "rows_per_method": args.rows, "elapsed_seconds": summary["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
