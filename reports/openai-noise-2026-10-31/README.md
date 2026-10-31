# Local MMLU feedback-noise sensitivity

All12 conditions (three seeds x0/1/3/5% replacement) independently reconstructed, including every learning selection, update and deployment budget decision.
The0-noise stream matches the immutable main PILOT state/actions exactly for each seed.
Deployment budget is1 historical accounting dollar. It is neither a paid API request nor1xGPT4 normalized budget.
Only the chosen learning reward is replaced with an independent fair binary value; true outcomes are retained for reward/regret evaluation. Replacement probability differs from flip probability.

| Replacement probability | Mean true reward / requested | Mean quality / GPT4 | Mean coverage | Mean true-reward change (pp) |
|---:|---:|---:|---:|---:|
|0%|68.8627%|85.4086%|100.0000%|+0.0000|
|1%|68.7582%|85.2789%|100.0000%|-0.1046|
|3%|68.4706%|84.9222%|100.0000%|-0.3922|
|5%|68.5490%|85.0195%|100.0000%|-0.3137|

A missing/unserved query contributes zero reward and remains in every denominator. Actual replacement and changed-label counts are separately recorded per seed; an unflipped replacement still counts as replaced.
Hyperparameters, prior, original stream order, tuning-only costs and guarded stop controller are unchanged. No noise setting is chosen from deployment quality.
Report individual seeds and coverage alongside means; a curve may reflect an earlier/later budget stop as well as changed ranking.

This is a local adaptation of [PILOT Appendix D.5](https://aclanthology.org/2025.findings-emnlp.1301.pdf), using8 verified arms and local split/order/prices. It does not verify the paper Table5 values or its claimed less-than4% drop.

Commands, sequential with one BLAS thread:
```powershell
.venv/Scripts/python.exe -m tools.openai_study_noise freeze
.venv/Scripts/python.exe -m tools.openai_study_noise run
.venv/Scripts/python.exe -m tools.openai_study_noise_verify
.venv/Scripts/python.exe -m tools.openai_study_noise_report
```

No additional paid API calls. Raw runs and this report are immutable.
