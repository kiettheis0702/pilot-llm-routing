"""CPU synthetic integration only: preference prior -> learning -> held-out deployment."""
from __future__ import annotations

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

from scripts.bandit_contracts import ContextTable
from scripts.cost_policy import TokenPrices, TuningTokenCostEstimator
from scripts.data_gate import make_splits, query_hash
from scripts.deployment import run_deployment
from scripts.evaluation import run_learning, write_run
from scripts.fetch_data import sha256
from scripts.pilot import PILOT
from scripts.pretraining import (EmbeddingCache, fit_preference_prior, load_checkpoint,
                                 project_cache, save_checkpoint)
from scripts.splits import FrozenSplits


def _preference_inputs():
    ids = [f"pref-{i}" for i in range(8)]
    recipe = {"model": "synthetic-coordinate-fixture", "version": "1",
              "source": "explicit toy inputs, no provider", "normalization": "none", "synthetic": True}
    vectors = [[1., 3.], [.9, 2.9], [-1., 3.], [-.9, 3.1],
               [1.1, 3.2], [-1.1, 2.8], [1., 3.05], [-1., 2.95]]
    cache = EmbeddingCache(ids, [query_hash(q) for q in ids], vectors, recipe)
    winners = ["large", "large", "small", "small", "large", "small", "large", "small"]
    comparisons = [{"battle_id": f"pair-{i}", "query_id": ids[i], "winner": winner,
                    "loser": "small" if winner == "large" else "large",
                    "split": "train" if i < 6 else "holdout"} for i, winner in enumerate(winners)]
    return cache, comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.seed < 0:
        raise ValueError("Seed must be nonnegative")
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.strip())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    arms = ("small", "large")
    pref_cache, comparisons = _preference_inputs()
    pref_cache.save(args.output_dir / "preference-cache.json")
    (args.output_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    prior = fit_preference_prior(pref_cache, comparisons, arms,
        {"small": 1., "large": 4.}, "synthetic relative per-token prices, small:large=1:4",
        dimension=2, projection_epochs=40, prior_epochs=100, learning_rate=.03, margin=.2, seed=args.seed)
    save_checkpoint(args.output_dir / "preference-prior.json", prior)
    prior = load_checkpoint(args.output_dir / "preference-prior.json")
    validation = prior["metrics"]["validation"]
    if not validation["all_lambdas_ready"]:
        raise ValueError("Synthetic prior validation has unsupported/zero accuracy; no fallback is allowed")
    ids = [f"route-{i:03d}" for i in range(66)]
    queries = pd.DataFrame({"query_id": ids, "query": [f"routing fixture query {i}" for i in range(66)], "task": "synthetic"})
    queries["query_hash"] = queries["query"].map(query_hash)
    vectors = np.array([[1. if i % 2 == 0 else -1., 3.] for i in range(66)])
    routing_cache = EmbeddingCache(ids, queries.query_hash.tolist(), vectors, pref_cache.provenance)
    routing_cache.save(args.output_dir / "routing-cache.json")
    projected = project_cache(prior, routing_cache)
    contexts = ContextTable(dict(zip(ids, projected)), label="synthetic pretrained toy projection")
    rewards = [[float(i % 2 == 1), float(i % 2 == 0)] for i in range(66)]
    outcomes = pd.DataFrame({"query_id": np.repeat(ids, 2), "arm_id": np.tile(arms, 66),
                             "reward": np.array(rewards).ravel(), "actual_cost": .5})
    frame = make_splits(queries, seed=42, tuning_size=10)
    split_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    (args.output_dir / "routing-splits.csv").write_bytes(split_bytes)
    splits = FrozenSplits(frame, sha256=hashlib.sha256(split_bytes).hexdigest(), seed=42,
                          tuning_size=10, learning_ratio=10)
    learned, deployment, tuning = (splits.select(phase) for phase in ("learning", "deployment", "tuning"))
    policy = PILOT(arms, 2, priors=dict(zip(arms, prior["theta"])),
                   lambdas={a: validation["per_arm"][a]["lambda"] for a in arms},
                   provenance="synthetic fitted preference-prior.json sha256=" + sha256(args.output_dir / "preference-prior.json"))
    learning = run_learning(policy, queries, outcomes, contexts, learned, seed=args.seed)
    write_run(learning, args.output_dir / "learning")
    policy.save_checkpoint(args.output_dir / "learned.json")
    restored = PILOT.load_checkpoint(args.output_dir / "learned.json")
    prices = {"small": TokenPrices(500, 1000), "large": TokenPrices(2000, 4000)}
    estimator = TuningTokenCostEstimator.fit(tuning,
        {q: dict.fromkeys(arms, 20) for q in tuning.query_ids}, prices,
        token_provenance="synthetic tuning counts: 20 output tokens per arm",
        price_provenance="synthetic currency per million tokens; not provider pricing")
    before_A, before_b = restored.A, restored.b
    deployed = run_deployment(restored, queries, outcomes, contexts, learned, deployment, estimator,
        dict.fromkeys(deployment.query_ids, 100), input_token_provenance="synthetic query-only input count=100",
        budget=2., bin_size=2, lower_bound=1., upper_bound=10.)
    frozen = np.array_equal(before_A, restored.A) and np.array_equal(before_b, restored.b)
    if not frozen:
        raise AssertionError("Deployment changed learned router state")
    restored.save_checkpoint(args.output_dir / "deployed.json")
    (args.output_dir / "deployment.json").write_text(json.dumps(asdict(deployed), indent=2, allow_nan=False), encoding="utf-8")
    pd.json_normalize([asdict(row) for row in deployed.trace]).to_csv(args.output_dir / "deployment-trace.csv", index=False)
    report = {"synthetic": True, "scope": "integration verification, not paper reproduction results",
              "seed": args.seed, "split_counts": splits.counts, "prior_validation": validation,
              "training": {"projection_loss_start": prior["metrics"]["projection_losses"][0],
                           "projection_loss_end": prior["metrics"]["projection_losses"][-1],
                           "prior_loss_start": prior["metrics"]["prior_losses"][0],
                           "prior_loss_end": prior["metrics"]["prior_losses"][-1]},
              "learning": learning.summary, "deployment": deployed.summary,
              "frozen_router_verified": frozen, "git_commit": commit, "working_tree_dirty": dirty,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
              "dependency_lock_sha256": sha256(root / "uv.lock"),
              "implementation_sha256": {str(p.relative_to(root)).replace("\\", "/"): sha256(p)
                                          for p in sorted((root / "scripts").glob("*.py"))},
              "elapsed_seconds": round(time.perf_counter() - started, 3)}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "learning_rounds": policy.rounds,
                      "deployment_served": deployed.summary["served_queries"], "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
