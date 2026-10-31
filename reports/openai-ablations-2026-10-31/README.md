# OpenAI controlled ablations

All240 conditions independently replayed in30 groups.18 new full learning streams,6 initial prior-only snapshots,6 reused main snapshots. New observed learning rounds:401751.
Paired with the same seed/domain/budget main64D OpenAI condition. No ablation retuning; no new paid API calls.

| Domain | Variant | Quality delta at50% allocation (pp) | Coverage delta (pp) |
|---|---|---:|---:|
| mmlu | prior-only | -13.878 | +0.000 |
| mmlu | projection-dimension-8 | +1.654 | +0.000 |
| mmlu | single-budget-bin | -0.616 | +0.000 |
| mmlu | unit-initialization-strength | +0.324 | +0.000 |
| mmlu | zero-prior-direction | -0.454 | +0.000 |
| routerbench | prior-only | -6.703 | +0.000 |
| routerbench | projection-dimension-8 | -0.903 | +0.000 |
| routerbench | single-budget-bin | -0.355 | +0.000 |
| routerbench | unit-initialization-strength | +1.550 | +0.000 |
| routerbench | zero-prior-direction | +2.142 | +0.000 |

Zero arm-prior directions retain the preference-learned query projection and calibrated lambdas. Prior-only has zero online learning. Single-bin reuses the exact parent main checkpoint.8D retrains both stages; the base OpenAI vectors are unchanged.
This remains an8-arm local study. Mean/min/max across fixed seeds are descriptive; no matched-paper or significance claim.
