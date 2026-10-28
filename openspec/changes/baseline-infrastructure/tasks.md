# B01–B03 Implementation Plan

**Goal:** Validated offline learning contracts, frozen splits and evaluation.
**Architecture:** Keep the existing simulator; add context, split-loader and evaluation modules. See [design](design.md) and the three [capability specs](specs/).
**Tech Stack:** Existing Python 3.11+, NumPy, pandas, PyArrow and pytest.
**Execution:** Apply this OpenSpec change inline with TDD. The user's 2026-10-28 approval covers setup and B01–B03 implementation. These tasks are the single plan; no duplicate Superpowers plan or commit-based ledger is needed.

## Global constraints

Preserve data and frozen split bytes. No dependencies, paid calls, training, baseline algorithms, commits or pushes. Keep valid legacy smoke calls. Numeric contexts are synthetic for current verification.

## Review focus

Input frame mutation after construction; policy mutation of context; IDs such as leading-zero strings; checksum-valid but semantically corrupt splits; held-out selections passed into learning. Each has a negative or behavioral test in its owning task.

## 1. B01 contracts

- [x] 1.1 Add constructor regression tests in tests/test_bandit_contracts.py, observe failure, then validate complete fixed catalogs and numeric data in scripts/data_gate.py; existing smoke and new tests pass.
- [x] 1.2 Add vector validation/copy tests, observe failure, then implement OnlinePolicy.reset/select/observe/predict_rewards and ContextTable.get in scripts/bandit_contracts.py; tests pass. Document compatibility and information boundary in docs/baseline-infrastructure.md.

## 2. B02 frozen split loading

- [x] 2.1 Add tests/test_splits.py for checksum/seed/coverage/metadata/groups/order/leading-zero IDs and deterministic subsets, observe failure, then implement load_split_manifest(path, queries, *, expected_sha256, seed=42, tuning_size=1000, learning_ratio=10), FrozenSplits.select and immutable SplitSelection in scripts/splits.py; tests pass. Document provenance and usage.

## 3. B03 learning evaluation

- [x] 3.1 Add tests/test_evaluation.py with hand-derived reward 1.25, mean 0.625, cost 0.3, regret 0.75; policy spy, mutation, perturbed unchosen rewards and lifecycle failures. Observe failure, then implement run_learning(policy, queries, outcomes, contexts, selection, *, seed) returning RunResult in scripts/evaluation.py; tests pass.
- [x] 3.2 Test log roundtrip and overwrite rejection before implementing write_run(result, output_dir); document runnable usage, metric names, seed/context/data provenance and limitations. Full pytest suite passes.

## 4. Integration verification

- [x] 4.1 Verify the existing 36,497-query manifest and run a labeled ten-query synthetic-context learning check twice using a verification-only policy; compare traces, source/prepared hashes and save evidence. Run full pytest, strict OpenSpec validation and diff checks; review implementation against all scenarios and record evidence in docs/reproduction-log.md.

## Workflow follow-up

- Leave the reviewed change available for user inspection; publication and B04+ need separate authorization.
