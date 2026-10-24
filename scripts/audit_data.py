"""Audit pinned data locally. Reports contain aggregate counts, never conversation text."""
from __future__ import annotations

import argparse
import ast
import json
import platform
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

try:
    from scripts.data_gate import Observation, OfflineBandit, RandomSmokeLearner, arm_ids, make_splits, normalize_routerbench, query_hash, run_bandit_round, text_segments
    from scripts.data_io import load_routerbench
    from scripts.fetch_data import ROOT, sha256, verify
except ModuleNotFoundError:
    from data_gate import Observation, OfflineBandit, RandomSmokeLearner, arm_ids, make_splits, normalize_routerbench, query_hash, run_bandit_round, text_segments
    from data_io import load_routerbench
    from fetch_data import ROOT, sha256, verify


def response_profile(values, *, json_format: bool = False) -> dict:
    counts = {"malformed_or_nonlist": 0, "empty_sequences": 0, "null_segments": 0, "other_nonstring_segments": 0}
    for value in values:
        try:
            parts = json.loads(value) if json_format else ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            counts["malformed_or_nonlist"] += 1
            continue
        if not isinstance(parts, list):
            counts["malformed_or_nonlist"] += 1
            continue
        counts["null_segments"] += sum(x is None for x in parts)
        counts["other_nonstring_segments"] += sum(x is not None and not isinstance(x, str) for x in parts)
        counts["empty_sequences"] += int(not parts or all(isinstance(x, str) and not x.strip() for x in parts))
    return counts


def router_profile(frame: pd.DataFrame) -> dict:
    arms = arm_ids(frame)
    scores = frame[arms].apply(pd.to_numeric, errors="raise")
    costs = frame[[a + "|total_cost" for a in arms]].astype(float)
    valid = np.isfinite(scores)
    return {
        "rows": len(frame), "columns": len(frame.columns), "arms": arms,
        "task_count_raw_eval_name": int(frame.eval_name.nunique()),
        "task_counts": {str(k): int(v) for k, v in frame.eval_name.value_counts().sort_index().items()},
        "duplicate_id_excess_rows": int(frame.sample_id.duplicated().sum()),
        "rows_with_duplicate_ids": int(frame.sample_id.duplicated(False).sum()),
        "duplicate_prompt_excess_rows": int(frame.prompt.duplicated().sum()),
        "missing_by_column": {k: int(v) for k, v in frame.isna().sum().items() if v},
        "rows_with_missing_reward": int(scores.isna().any(axis=1).sum()),
        "nonfinite_rewards": int((~valid).to_numpy().sum()),
        "out_of_range_rewards": int(((scores < 0) | (scores > 1)).to_numpy().sum()),
        "nonbinary_finite_rewards": int((valid & ~scores.isin([0, 1])).to_numpy().sum()),
        "nonfinite_costs": int((~np.isfinite(costs)).to_numpy().sum()),
        "negative_costs": int((costs < 0).to_numpy().sum()),
        "reward_range": [float(scores.min().min()), float(scores.max().max())],
        "cost_usd_range": [float(costs.min().min()), float(costs.max().max())],
        "prompt_segment_count": {str(k): int(v) for k, v in frame.prompt.map(lambda p: len(text_segments(p))).value_counts().sort_index().items()},
        "response_content": {a: response_profile(frame[a + "|model_response"]) for a in arms},
        "per_arm": {a: {
            "missing_reward": int(scores[a].isna().sum()),
            "reward_mean": float(scores[a].mean()),
            "cost_mean_usd": float(costs[a + "|total_cost"].mean()),
        } for a in arms},
        "schema": {c: str(t) for c, t in frame.dtypes.items()},
    }


def preference_profile(frame: pd.DataFrame, mappings: list[dict], queries: pd.DataFrame) -> dict:
    required = {"id", "model_a", "model_b", "prompt", "response_a", "response_b",
                "winner_model_a", "winner_model_b", "winner_tie"}
    if required - set(frame):
        raise ValueError("Incomplete preference schema")
    labels = frame[["winner_model_a", "winner_model_b", "winner_tie"]]
    one_hot = labels.isin([0, 1]).all(axis=1) & (labels.sum(axis=1) == 1)
    parts = []
    for value in frame.prompt:
        try:
            parts.append(text_segments(value, json_format=True))
        except (ValueError, TypeError, SyntaxError):
            parts.append(None)
    valid_prompt = pd.Series([x is not None for x in parts], index=frame.index)
    first_hash = pd.Series([query_hash(x[0]) if x and x[0].strip() else None for x in parts], index=frame.index)
    full_rb = set(queries.query_hash)
    segment_rb = {query_hash(segment) for raw in queries.query_segments_json
                  for segment in json.loads(raw) if segment.strip()}
    masks = {}
    for label, selection in [("exact_ids_only", [x for x in mappings if x["status"] == "exact_id"]),
                             ("all_proposed_mappings", mappings)]:
        names = {x["arena"] for x in selection}
        both = frame.model_a.isin(names) & frame.model_b.isin(names)
        usable = both & one_hot & valid_prompt & first_hash.notna() & (frame.winner_tie == 0) & (frame.model_a != frame.model_b)
        subset = frame.loc[usable]
        turn_counts = pd.Series([len(x) if x else 0 for x in parts], index=frame.index)
        pair_counts = subset.apply(lambda r: " / ".join(sorted([r.model_a, r.model_b])), axis=1).value_counts()
        masks[label] = {
            "both_models_mapped": int(both.sum()),
            "ties_with_both_models_mapped": int((both & (frame.winner_tie == 1)).sum()),
            "decisive_valid_pairs": int(usable.sum()),
            "single_turn_decisive_pairs": int((usable & (turn_counts == 1)).sum()),
            "multi_turn_decisive_pairs": int((usable & (turn_counts > 1)).sum()),
            "duplicate_first_prompt_excess_rows": int(first_hash.loc[usable].duplicated().sum()),
            "first_prompt_matches_full_routerbench_query": int(first_hash.loc[usable].isin(full_rb).sum()),
            "first_prompt_matches_any_routerbench_segment": int(first_hash.loc[usable].isin(segment_rb).sum()),
            "per_arm_decisive_pairs": {x["routerbench"]: int(((subset.model_a == x["arena"]) | (subset.model_b == x["arena"])).sum()) for x in selection},
            "pair_counts": {k: int(v) for k, v in pair_counts.sort_index().items()},
        }
    return {
        "rows": len(frame), "columns": list(frame.columns),
        "model_count": len(set(frame.model_a) | set(frame.model_b)),
        "model_names": sorted(set(frame.model_a) | set(frame.model_b)),
        "duplicate_id_excess_rows": int(frame.id.duplicated().sum()),
        "missing_by_column": {k: int(v) for k, v in frame.isna().sum().items() if v},
        "invalid_winner_rows": int((~one_hot).sum()),
        "invalid_prompt_rows": int((~valid_prompt).sum()),
        "winner_counts": {k: int(v) for k, v in labels.sum().items()},
        "response_content": {c: response_profile(frame[c], json_format=True) for c in ("response_a", "response_b")},
        "mapping_scenarios": masks,
        "training_ready": False,
        "note": "Proposed mappings are sensitivity counts, not approved model equivalence. Multi-turn labels apply to whole conversations; no prior-training dataset is exported.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/data-gate.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sources = json.loads((ROOT / "configs/data_sources.json").read_text(encoding="utf-8"))
    mappings = json.loads((ROOT / "configs/model_mapping.json").read_text(encoding="utf-8"))["mappings"]
    frames = {}
    profiles = {}
    for key in ("routerbench_0shot", "routerbench_5shot"):
        frames[key] = load_routerbench(args.data_dir / key / sources[key]["file"], sources[key])
        profiles[key] = router_profile(frames[key])
    queries, outcomes = normalize_routerbench(frames["routerbench_0shot"])
    if {m["routerbench"] for m in mappings} != set(arm_ids(frames["routerbench_0shot"])):
        raise ValueError("Mapping table does not cover exactly the audited RouterBench arms")
    splits = make_splits(queries, seed=args.seed)
    pref_path = args.data_dir / "arena_55k" / sources["arena_55k"]["file"]
    verify(pref_path, sources["arena_55k"])
    preference = preference_profile(pd.read_csv(pref_path), mappings, queries)
    output = args.data_dir / "processed" / "routerbench_0shot"
    output.mkdir(parents=True, exist_ok=True)
    queries.to_parquet(output / "queries.parquet", index=False)
    outcomes.to_parquet(output / "outcomes.parquet", index=False)
    splits.to_csv(output / "splits.csv", index=False, lineterminator="\n")
    environment = OfflineBandit(queries, outcomes)
    first_id = splits.loc[splits.split == "learning", "query_id"].iloc[0]
    learner = RandomSmokeLearner(args.seed)
    feedback = run_bandit_round(environment, first_id, learner)
    report = {
        "gate_status": "conditional: offline routing data usable; faithful reproduction choices unresolved",
        "source_pins": sources,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__},
        "config_sha256": {name: sha256(ROOT / "configs" / name) for name in ("data_sources.json", "model_mapping.json")},
        "routerbench": profiles,
        "arena_55k": preference,
        "legacy_arena_33k": {
            "repo": "lmsys/chatbot_arena_conversations",
            "revision": "1b6335d42a1d2c7e34870c905d03ab964f7f2bd8",
            "access_observation_2026_10_24": "Anonymous file HEAD returned HTTP 401; gated=auto in metadata. Contents not audited.",
        },
        "prepared": {
            "variant": "routerbench_0shot", "variant_status": "provisional; not confirmed as paper artifact",
            "query_rows": len(queries), "outcome_rows": len(outcomes),
            "whitespace_normalized_duplicate_excess_rows": int(queries.query_hash.duplicated().sum()),
            "split_seed": args.seed,
            "split_method": "SHA-256 ordering of seed and prompt group; exact tuning=1000, deployment=ceil(remaining/11); identical normalized queries stay together",
            "split_counts": {k: int(v) for k, v in splits.split.value_counts().items()},
            "split_query_hash_overlap": int((splits.groupby("query_hash").split.nunique() > 1).sum()),
            "file_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
            "estimated_cost_status": "all null; tuning-only token estimator not implemented",
            "actual_cost_semantics": "historical RouterBench response-dependent cost estimate in USD, not current API price or actual invoice",
        },
        "smoke": {"policy": "seeded random selection for one-step I/O check only",
                  "observation_fields": [f.name for f in fields(Observation)],
                  "feedback": asdict(feedback), "learner_update_count": len(learner.updates)},
        "unresolved": [
            "Paper does not identify 0-shot/5-shot artifact, dataset revision, or exact preprocessing.",
            "Both public variants contain 86 raw eval_name labels; paper states 64 tasks.",
            "5-shot contains duplicated IDs and incomplete reward rows; rejected by strict preparation.",
            "Exact ChatArena release, alias/checkpoint identity, tie treatment, and multi-turn handling remain unconfirmed.",
            "RouterBench HF card has no license field/file; code MIT license is not proof of dataset licensing.",
            "Estimated deployment cost requires a tuning-only token estimator and pinned historical prices/tokenizers.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved audit: {args.report}")
    print(json.dumps({"gate": report["gate_status"], "splits": report["prepared"]["split_counts"],
                      "candidate_pairs": preference["mapping_scenarios"]["all_proposed_mappings"]["decisive_valid_pairs"]}))


if __name__ == "__main__":
    main()
