# Proposal

## Why

The separate mathematical components need one executable synthetic flow to verify their contracts together before any real preference training or paper result claims.

## What Changes

- Add an offline toy command for two-phase prior fitting, online PILOT learning, JSON checkpoint readback and frozen held-out budget deployment.
- Save generated inputs/provenance, training diagnostics, learning/deployment traces and a clear status summary.
- Verify fresh-checkout execution and document remaining real-data blockers.

## Capabilities

### New Capabilities
- `toy-reproduction`: Reproducible synthetic integration of preference, online learning and budget deployment.

### Modified Capabilities
None.

## Impact

New scripts/run_toy_pipeline.py and integration tests; reuse existing modules. CPU only, no downloads/APIs/provider costs. Toy checkpoints/inputs are explicitly synthetic, not a replacement for unresolved real artifacts.
