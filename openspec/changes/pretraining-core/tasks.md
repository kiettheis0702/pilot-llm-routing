# Tasks

## 1. Input and sampling contracts

- [x] 1.1 Implement supplied embedding/cache provenance and comparison/split validation; verify schema, dimensions, nonfinite values, duplicate IDs and prompt leakage fixtures; document input contracts.
- [x] 1.2 Implement deterministic same-winner/strict-smaller-negative triplets with distinct prompt groups; verify exclusions, missing order rejection and train-only sampling; document inferred settings.

## 2. Numerical phases and diagnostics

- [x] 2.1 Implement cosine triplet and two-logit pair loss gradients; verify finite differences and zero/nonfinite rejection on CPU fixtures.
- [x] 2.2 Implement seeded tiny SGD phases with frozen W,b and held-out-only accuracy/lambda diagnostics; verify reproducibility, loss decrease, no holdout influence and zero/absent accuracy blockers; document limitations.

## 3. Artifacts and integrated evidence

- [x] 3.1 Implement exclusive-create validated JSON cache/checkpoint save/load; verify exact reload output, checksum/schema rejection and catalog/provenance fingerprints; document artifact contract.
- [x] 3.2 Run toy-only evidence, full current pytest suite, strict OpenSpec validation and actual diff review; save scoped commands/results and preserve real-data blocked status.

## Workflow follow-up

- Parent owns independent review and separately authorized Git operations. Real-data fitting requires its unresolved scientific inputs and is outside this change.
