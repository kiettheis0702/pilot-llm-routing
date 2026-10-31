# OpenAI fixed-score budget comparison

240 new policy conditions independently replayed, plus48 reused single-bin ablation controls. Same frozen PILOT checkpoint, estimates, order and all-request GPT-4 denominators. No paid API calls.

| MMLU allocation | Policy | Quality/GPT-4 | Recorded cost/GPT-4 | Coverage |
|---:|---|---:|---:|---:|
| 20% | fixed-per-query | 68.94% | 6.93% | 99.92% |
| 30% | fixed-per-query | 69.71% | 7.57% | 100.00% |
| 20% | guarded-skip | 68.90% | 19.09% | 97.31% |
| 30% | guarded-skip | 75.29% | 28.57% | 99.87% |
| 20% | guarded-stop | 1.72% | 1.10% | 2.27% |
| 30% | guarded-stop | 20.17% | 8.25% | 27.45% |
| 20% | paper-literal-stop | 1.72% | 1.10% | 2.27% |
| 30% | paper-literal-stop | 20.17% | 8.25% | 27.45% |
| 20% | retrospective-lagrange | 53.53% | 19.09% | 60.37% |
| 30% | retrospective-lagrange | 64.04% | 28.67% | 70.20% |
| 20% | single-budget-bin | 15.18% | 13.29% | 15.37% |
| 30% | single-budget-bin | 25.23% | 20.65% | 27.45% |

Guarded-stop exactly matches all48 immutable core PILOT conditions, including chosen actions. Paper-literal-stop removes only the threshold affordability guard and retains termination.
Guarded-skip preserves current-bin threshold/fallback, records a zero-reward unserved request when no arm qualifies and advances query time. Fixed-per-query caps each request atB/Q without transfer. Single-bin is the earlier controlled ablation, not new training.
Retrospective-lagrange uses all future queries' predicted scores and estimated costs to choose a feasible P-lambda*C allocation with abstention. It never uses true rewards for selection. This comparator has more information; no exact optimality/oracle guarantee is claimed.
MRR uses historical reward ranks with minimum rank for ties; unserved requests contribute0. It is a declared local diagnostic, not a verified identical author metric.
All variants remain8-arm local experiments. Higher coverage need not increase quality or save cost. Keep adverse results; no retroactive change to the main protocol.
