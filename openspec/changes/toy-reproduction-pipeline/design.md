# Design

## Context

PILOT, preference pretraining and standalone deployment have isolated contracts. Real embedding/mapping/token-price prerequisites remain blocked. This integration is synthetic engineering evidence only.

## Goals / Non-Goals

Reproduce a small CPU pipeline from a clean checkout. Do not claim Figure 3, real priors, a production embedding choice or a fair multi-seed research experiment.

## Decisions

Use a fixed two-arm/eight-comparison preference geometry and a separate 66-query routing fixture. Preserve prompt/ID separation, an existing grouped seed-42 10-tuning/50-learning/6-deployment split, and synthetic prices/counts. Run seeds are explicit. Fit via pretraining.py, initialize PILOT only if every held-out arm has supported positive accuracy, then use run_learning and run_deployment. Save both checkpoints and read them back. Cost bounds and budget are explicit synthetic assumptions. Include source-file hashes plus Git commit for dirty-worktree reproducibility.

## Risks / Trade-offs

Success validates code integration, not real-world data adequacy. All figures/metrics remain synthetic. A failed phase may leave an explicitly incomplete output directory; never silently overwrite it.
