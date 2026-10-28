# Baselines and bounded runs

Implemented in `scripts/policies.py`, using the B01–B03 OnlinePolicy interface.
All policies validate contexts and selected feedback. Reset restores seed and
learned state; deterministic ties follow the caller's catalog order.

| Policy | Learning action | Learned ranking |
| --- | --- | --- |
| RandomPolicy | Uniform local seeded RNG | Uniform scores, not fitted rewards |
| FixedPolicy | Configured single arm | Preferred-arm indicator, not fitted rewards |
| LinearUCB | Linear ridge mean plus alpha times uncertainty | Linear ridge mean |
| ExploreOnly | Maximum ridge uncertainty | Linear ridge mean |
| EpochGreedy | Uniform at rounds 0, window, 2*window; otherwise greedy | Linear ridge mean |

Disjoint ridge uses A=lambda I, b=0, theta=solve(A,b); after feedback, only the
chosen arm receives A+=xx-transpose and b+=reward*x. Linear UCB is
x dot theta + alpha sqrt(x dot solve(A,x)). Vectors are not implicitly normalized.
PILOT's cosine exploitation is a separate future policy, not substituted into LinUCB.

The source [PILOT §3.1.3/3.2](https://aclanthology.org/2025.findings-emnlp.1301.pdf)
names these comparisons and epoch window tuning but does not fully specify the
Explore Only selection rule or Epoch Greedy update schedule. Maximum-uncertainty
exploration and one random step per fixed window, learning from every selected
feedback, are explicitly **PAPER_INFERRED** choices here, not an exact reconstruction
of the authors' code. Standard disjoint LinUCB follows
[Li et al.](https://arxiv.org/abs/1003.0146).

Fixed and random policies' deployment ranking semantics are not empirical reward
estimates; callers must preserve their baseline action rule when adding deployment
comparisons. No budget comparison is claimed by this learning-only stage.

## Run comparisons

```powershell
uv sync --locked
uv run --locked python -B -m scripts.run_baselines --rows 100 --dimension 8 --seed 42 --output-dir outputs/toy-baselines
uv run --locked python -B -m scripts.run_baselines --source routerbench --data-dir 'D:\DOWNLOAD GG\PILOT_DATA' --rows 64 --dimension 8 --seed 42 --output-dir outputs/routerbench-smoke
```

Use a new output directory for each execution. Each method receives identical
learning IDs, order, outcomes and context vectors. The CLI saves per-method
trace.csv/summary.json plus comparison.json and Matplotlib PNG/SVG regret plots.
Figures can be regenerated with `plot_traces(output_path, method_names)` from
saved CSVs. Matplotlib is locked in uv.lock; no LLM, tokenizer or embedding model
is downloaded or called. Run limits are 1–1,000 rows and 1–32 dimensions.

The seed controls the toy generator, subset selection and policy RNG. Cross-seed
research comparisons need a separate locked dataset/subset protocol; these smoke
runs do not claim it. RouterBench features are normalized SHAKE256 query-text
vectors: deterministic, label-free and **synthetic**, with no semantic meaning.
Every manifest records that deviation plus configuration, Git commit, dirty state,
implementation/lockfile fingerprints and data/context fingerprints. A failed run
can leave partial artifacts; use a new destination after diagnosing it.

Measured commands used the same arguments above with destinations
`reports/baselines/toy-2026-10-28` and
`reports/baselines/routerbench-64-2026-10-28`: 7 methods × 100 toy rounds in 2.227 s;
15 methods × 64 RouterBench rounds in 11.877 s. Both exited 0. The independent
review recomputed metric totals/cumulative regret from trace files, verified common
query/context/data fingerprints and checked current code hashes; all matched.
Runner tests: 3 passing. The 15-curve diagnostic plot repeats some colors; CSV
method labels and per-method traces remain unambiguous. Publication plots and
real embedding-based performance claims remain future work.
