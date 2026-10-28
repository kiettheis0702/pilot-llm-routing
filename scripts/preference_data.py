"""Strict local preference preparation and conservative readiness evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import stat

import pandas as pd

from scripts.data_gate import query_hash
from scripts.fetch_data import ROOT, sha256, verify


PAIR_COLUMNS = ["battle_id", "prompt", "prompt_hash", "arm_a", "arm_b", "winner_arm", "loser_arm"]
LABELS = ["winner_model_a", "winner_model_b", "winner_tie"]
TEXT_COLUMNS = ["prompt", "response_a", "response_b"]


def parse_preference_text(value) -> tuple[list[str] | None, str | None]:
    """Parse a JSON text list without coercing missing segments or empty text."""
    if value is None or (not isinstance(value, str) and pd.api.types.is_scalar(value) and pd.isna(value)):
        return None, "missing"
    if not isinstance(value, str):
        return None, "malformed_json"
    try:
        parts = json.loads(value)
    except (ValueError, TypeError):
        return None, "malformed_json"
    if not isinstance(parts, list):
        return None, "nonlist"
    if not parts:
        return None, "empty"
    if any(not isinstance(part, str) for part in parts):
        return None, "nonstring_segment"
    if any(not part.strip() for part in parts):
        return None, "blank_segment"
    return parts, None


def exact_mapping(mappings: list[dict]) -> dict[str, str]:
    """A status string cannot approve equivalence between unequal model IDs."""
    arena, arms, exact = set(), set(), {}
    for mapping in mappings:
        if not isinstance(mapping, dict) or not {"arena", "routerbench", "status"} <= set(mapping):
            raise ValueError("Incomplete mapping")
        source, target = mapping["arena"], mapping["routerbench"]
        if (not all(isinstance(x, str) and x.strip() for x in (source, target))
                or source in arena or target in arms):
            raise ValueError("Ambiguous or invalid model mapping")
        arena.add(source)
        arms.add(target)
        if mapping["status"] == "exact_id":
            if source != target:
                raise ValueError("Exact mappings must have literal identical model IDs")
            exact[source] = target
    if not arms:
        raise ValueError("Empty model mapping")
    return exact


def prepare_preferences(frame: pd.DataFrame, mappings: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Filter strict, single-turn comparisons; count every reason and first reason."""
    required = {"id", "model_a", "model_b", *LABELS, *TEXT_COLUMNS}
    if not frame.columns.is_unique or not required <= set(frame):
        raise ValueError("Incomplete or duplicate preference schema")
    ids = frame.id
    if not ids.map(lambda value: isinstance(value, (str, Integral))
                   and not isinstance(value, bool) and bool(str(value).strip())).all():
        raise ValueError("Battle identities must be nonempty strings or integers")
    if ids.astype(str).duplicated().any():
        raise ValueError("Battle identities must be unique")
    exact = exact_mapping(mappings)
    reasons, primary, accepted = Counter(), Counter(), []
    exact_both_rows = 0
    for row in frame.to_dict("records"):
        rejected = []
        a, b = row["model_a"], row["model_b"]
        named = all(isinstance(x, str) and bool(x.strip()) for x in (a, b))
        if not named:
            rejected.append("missing_model")
        if not named or a not in exact or b not in exact:
            rejected.append("unapproved_model_mapping")
        else:
            exact_both_rows += 1
        if named and a == b:
            rejected.append("same_model")
        labels = [row[label] for label in LABELS]
        valid = all(isinstance(x, Real) and not isinstance(x, bool) and x in (0, 1) for x in labels)
        valid = valid and sum(labels) == 1
        if not valid:
            rejected.append("invalid_winner")
        elif labels[2] == 1:
            rejected.append("tie")
        decoded = {}
        for column in TEXT_COLUMNS:
            parts, error = parse_preference_text(row[column])
            if error:
                rejected.append(f"{column}_{error}")
            else:
                decoded[column] = parts
        if any(len(parts) != 1 for parts in decoded.values()):
            rejected.append("multi_turn")
        if rejected:
            reasons.update(rejected)
            primary.update([rejected[0]])
            continue
        prompt = decoded["prompt"][0]
        winner, loser = (a, b) if labels[0] == 1 else (b, a)
        accepted.append({"battle_id": str(row["id"]), "prompt": prompt, "prompt_hash": query_hash(prompt),
                         "arm_a": exact[a], "arm_b": exact[b],
                         "winner_arm": exact[winner], "loser_arm": exact[loser]})
    pairs = pd.DataFrame(accepted, columns=PAIR_COLUMNS).sort_values("battle_id").reset_index(drop=True)
    return pairs, {
        "source_rows": len(frame), "exact_both_models_rows": exact_both_rows,
        "eligible_rows": len(pairs), "excluded_rows": len(frame) - len(pairs),
        "missing_cells_by_column": {k: int(v) for k, v in frame.isna().sum().items() if v},
        "exclusion_reason_counts": dict(sorted(reasons.items())),
        "primary_exclusion_counts": dict(sorted(primary.items())),
        "primary_reason_precedence": ["missing_model", "unapproved_model_mapping", "same_model",
                                      "invalid_winner", "tie", "prompt_content", "response_a_content",
                                      "response_b_content", "multi_turn"],
        "count_note": "Reason incidence can overlap; primary exclusions sum to excluded_rows.",
    }


def split_preferences(pairs: pd.DataFrame, seed: int = 42, holdout_fraction: float = 0.2) -> pd.DataFrame:
    """Hold out complete prompt groups; the fraction targets groups, not rows."""
    if not isinstance(holdout_fraction, Real) or not 0 < holdout_fraction < 1:
        raise ValueError("Holdout fraction must be finite and strictly between zero and one")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError("Split seed must be an integer")
    if not pairs.columns.is_unique or not set(PAIR_COLUMNS) <= set(pairs):
        raise ValueError("Incomplete pair table")
    if pairs.battle_id.isna().any() or pairs.battle_id.duplicated().any():
        raise ValueError("Battle identities must be unique and present")
    if not pairs.prompt.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Invalid pair prompt")
    if not pairs.prompt.map(query_hash).eq(pairs.prompt_hash).all():
        raise ValueError("Prompt hash does not match normalized prompt")
    groups = pairs.prompt_hash.unique().tolist()
    if len(groups) < 2:
        raise ValueError("At least two prompt groups are required for train/holdout")
    order = sorted(groups, key=lambda h: hashlib.sha256(f"{seed}:preference:{h}".encode()).hexdigest())
    count = min(len(groups) - 1, math.ceil(len(groups) * holdout_fraction))
    holdout = set(order[:count])
    result = pairs.copy()
    result["split"] = result.prompt_hash.map(lambda h: "holdout" if h in holdout else "train")
    return result.sort_values("battle_id").reset_index(drop=True)


def table_fingerprint(frame: pd.DataFrame) -> str:
    """Canonical JSON multiset fingerprint, independent of row/column ordering."""
    rows = sorted(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                  for row in frame.to_dict("records"))
    payload = json.dumps({"columns": sorted(frame.columns), "rows": rows}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pair_coverage(pairs: pd.DataFrame, arms: list[str]) -> dict:
    counts = Counter(" / ".join(sorted((a, b))) for a, b in zip(pairs.arm_a, pairs.arm_b))
    return {"rows": len(pairs), "prompt_groups": int(pairs.prompt_hash.nunique()),
            "per_arm": {arm: {"comparisons": int(((pairs.arm_a == arm) | (pairs.arm_b == arm)).sum()),
                              "wins": int((pairs.winner_arm == arm).sum()),
                              "losses": int((pairs.loser_arm == arm).sum())} for arm in sorted(arms)},
            "pair_counts": dict(sorted(counts.items()))}


def preference_readiness(frame: pd.DataFrame, mappings: list[dict], queries: pd.DataFrame,
                         seed: int = 42, holdout_fraction: float = 0.2) -> dict:
    pairs, diagnostic = prepare_preferences(frame, mappings)
    arms = sorted(mapping["routerbench"] for mapping in mappings)
    coverage = pair_coverage(pairs, arms)
    split_info = {"status": "blocked", "seed": seed, "holdout_group_fraction": holdout_fraction,
                  "method": "SHA-256 seed:preference:prompt_hash ordering; ceil(fraction * groups), at least one train group"}
    # Validate options even when insufficient data means no split can be made.
    if not isinstance(holdout_fraction, Real) or not 0 < holdout_fraction < 1:
        raise ValueError("Invalid holdout fraction")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError("Split seed must be an integer")
    if coverage["prompt_groups"] >= 2:
        split = split_preferences(pairs, seed, holdout_fraction)
        split_info.update({"status": "prepared", "query_hash_overlap": int((split.groupby("prompt_hash").split.nunique() > 1).sum()),
                           "battle_id_overlap": int((split.groupby("battle_id").split.nunique() > 1).sum()),
                           "manifest_sha256": table_fingerprint(split[["battle_id", "prompt_hash", "split"]]),
                           "partitions": {name: pair_coverage(split.loc[split.split == name], arms)
                                          for name in ("train", "holdout")}})
    else:
        split_info["reason"] = "Fewer than two eligible prompt groups"
    if not {"query_hash", "query_segments_json"} <= set(queries):
        raise ValueError("Prepared RouterBench queries lack overlap-check fields")
    full_hashes = set(queries.query_hash)
    segment_hashes = set()
    for raw in queries.query_segments_json:
        segments, error = parse_preference_text(raw)
        if error:
            raise ValueError("Invalid prepared RouterBench query segments")
        segment_hashes.update(query_hash(segment) for segment in segments)
    missing = [arm for arm in arms if coverage["per_arm"][arm]["comparisons"] == 0]
    return {"protocol": "provisional-exact-id-single-turn-v1", "mapping_policy": "literal exact_id only; no aliases approved",
            "mapping_status_counts": dict(sorted(Counter(m["status"] for m in mappings).items())),
            "model_pool": arms, "diagnostics": diagnostic, "eligible": coverage,
            "eligible_table_sha256": table_fingerprint(pairs), "split": split_info,
            "routerbench_overlap": {"full_query_rows": int(pairs.prompt_hash.isin(full_hashes).sum()),
                                    "segment_rows": int(pairs.prompt_hash.isin(segment_hashes).sum()),
                                    "scope": "Exact whitespace-normalized text only; not semantic decontamination"},
            "missing_eligible_arms": missing,
            "missing_partition_arms": {name: [a for a in arms if details["per_arm"][a]["comparisons"] == 0]
                                       for name, details in split_info.get("partitions", {}).items()},
            "triplets": {"valid_triplets": None, "status": "blocked",
                         "reason": "No pinned token-cost model-size order for paper Section 2.2.1 hard negatives; no sampler/training executed"},
            "all_arm_prior_ready": False,
            "blockers": (["Exact-ID eligible data does not cover the complete requested arm pool"] if missing else [])
                        + ["No validated embedding model/version/dimension/cache provenance",
                           "No verified triplet size ordering or trained preference checkpoint",
                           "Arena 55K source and local filtering/split are not confirmed paper artifacts"]}


def embedding_inventory(roots: list[Path]) -> dict:
    """Filename inventory only; never deserialize unknown checkpoints or follow links."""
    skipped = {".git", ".venv", ".agents", ".pytest_cache", "__pycache__", "node_modules"}
    extensions = {".npy", ".npz", ".pt", ".pth", ".safetensors", ".bin", ".onnx"}
    candidates, errors, resolved = [], [], []
    for root in sorted({Path(p).resolve() for p in roots}, key=str):
        resolved.append(str(root))
        if not root.is_dir():
            errors.append({"root": str(root), "error": "missing directory"})
            continue
        def onerror(error):
            errors.append({"root": str(root), "error": str(error)})
        for directory, directories, files in os.walk(root, followlinks=False, onerror=onerror):
            def regular_directory(name):
                path = Path(directory) / name
                try:
                    attrs = getattr(path.lstat(), "st_file_attributes", 0)
                    return name not in skipped and not path.is_symlink() and not attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT
                except OSError as exc:
                    onerror(exc)
                    return False
            directories[:] = sorted(name for name in directories if regular_directory(name))
            for name in sorted(files):
                path = Path(directory) / name
                # Prose/code that mentions embeddings is not a vector/model artifact.
                if path.suffix.lower() not in extensions and not (
                        path.suffix.lower() in {".json", ".parquet", ".arrow", ".csv"}
                        and any(word in str(path.relative_to(root)).lower() for word in ("embedding", "cache"))):
                    continue
                try:
                    attrs = path.lstat()
                    if path.is_symlink() or getattr(attrs, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                        continue
                    candidates.append({"root": str(root), "relative_path": path.relative_to(root).as_posix(),
                                       "bytes": attrs.st_size})
                except OSError as exc:
                    onerror(exc)
    return {"roots": resolved, "excluded_directory_names": sorted(skipped), "candidates": candidates,
            "scan_errors": errors, "scan_complete": not errors, "method": "Filename/extension inventory; directory links excluded",
            "embedding_model": None, "embedding_version": None, "dimension": None, "provenance": None,
            "validated_cache_ready": False,
            "note": "No cache validation or model generation; candidate filenames alone cannot establish provenance or dimensions"}


def load_verified_queries(path: Path, expected_sha256: str) -> pd.DataFrame:
    if sha256(path) != expected_sha256:
        raise ValueError("Prepared query fingerprint differs from the Data Gate")
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/preference-readiness.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    args = parser.parse_args()
    # Output cannot overwrite raw data, prepared tables, code or configuration.
    report_path = args.report.resolve()
    if not report_path.is_relative_to((ROOT / "reports").resolve()) or report_path.suffix.lower() != ".json":
        raise ValueError("Write aggregate JSON reports only inside the repository reports directory")
    if report_path.is_relative_to(args.data_dir.resolve()):
        raise ValueError("Report output must be outside the data directory")
    config_paths = [ROOT / "configs" / name for name in ("data_sources.json", "model_mapping.json")]
    sources = json.loads(config_paths[0].read_text(encoding="utf-8"))
    mappings = json.loads(config_paths[1].read_text(encoding="utf-8"))["mappings"]
    gate_path = ROOT / "reports/data-gate.json"
    if report_path == gate_path.resolve():
        raise ValueError("Preserve the existing Data Gate report")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    source = sources["arena_55k"]
    pref_path = args.data_dir / "arena_55k" / source["file"]
    verify(pref_path, source)
    query_path = args.data_dir / "processed/routerbench_0shot/queries.parquet"
    queries = load_verified_queries(query_path, gate["prepared"]["file_sha256"]["queries.parquet"])
    if set(m["routerbench"] for m in mappings) != set(gate["routerbench"]["routerbench_0shot"]["arms"]):
        raise ValueError("Mapping table must cover exactly the Data Gate arm pool")
    frame = pd.read_csv(pref_path, dtype={"id": str})
    report = preference_readiness(frame, mappings, queries, args.seed, args.holdout_fraction)
    # Retain the older, broader counts for an explicit before/after comparison.
    from scripts.audit_data import preference_profile
    report["descriptive_mapping_scenarios"] = preference_profile(frame, mappings, queries)["mapping_scenarios"]
    report["embedding_inventory"] = embedding_inventory([ROOT, args.data_dir])
    report["provenance"] = {"source": source, "source_path": str(pref_path.resolve()),
                            "prepared_query_path": str(query_path.resolve()), "prepared_query_sha256": sha256(query_path),
                            "data_gate_report_sha256": sha256(gate_path),
                            "config_sha256": {p.name: sha256(p) for p in config_paths},
                            "code_sha256": {name: sha256(ROOT / "scripts" / name)
                                            for name in ("preference_data.py", "data_gate.py", "audit_data.py", "fetch_data.py")},
                            "runtime": {"python": platform.python_version(), "pandas": pd.__version__}}
    report["operations"] = {"source_writes": 0, "conversation_exports": 0, "training_runs": 0,
                            "external_api_calls": 0, "downloads": 0}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "eligible_rows": report["eligible"]["rows"],
                      "missing_arms": report["missing_eligible_arms"], "all_arm_prior_ready": False}))


if __name__ == "__main__":
    main()
