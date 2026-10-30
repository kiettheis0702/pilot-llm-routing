# Course research analysis — Team 8

Checked 2026-10-30 against the [final ACL paper](https://aclanthology.org/2025.findings-emnlp.1301.pdf), pp. 23934–23949. The final PDF, not the earlier arXiv numbering, is authoritative here. This is a **core-scope, eight-arm local study**, not a matched eleven-arm reproduction.

## Two core experiments

1. Performance versus budget: all declared budget points, PILOT and the same-evaluator bandit/fixed-arm baselines, on RouterBench and its nested MMLU subset. This addresses the paper's central cost–quality comparison (Section 4/Figure 3).
2. Learning size and cumulative empirical oracle regret: uninterrupted stream prefixes at 10%, 25%, 50%, 100%, then frozen deployment. This examines learning behavior; the four prefix points are not a dense round-by-round plot or independent retraining.

The local inventory is 858 primary plus 240 ablation deployment conditions, three seeds (7/42/2026). MMLU is a subset, not an independent replication. The course permits core scope; it has **not specifically approved our changed arm pool as paper-equivalent**.

## Comparison with the paper

| Item | Published result or protocol | Local measurement or protocol | Interpretation |
|---|---|---|---|
| RouterBench headline | 93% of GPT-4 performance at 25% of GPT-4 cost | At 50% of the local reference budget: PILOT utility 0.6764, LinUCB 0.7056 | Different x/y denominators; no matched relative reproduction error |
| MMLU headline | 86% of GPT-4 performance at 27% of GPT-4 cost | At 50% local reference budget: PILOT 0.6675, ExploreOnly 0.6899 | Local PILOT is not best; does not falsify the paper's different experiment |
| Preference prediction (D.3) | PILOT 65.0%, RouteLLM 63.6%, 500 holdout pairs | 189/313, 188/313, 193/313; mean 60.7029% | Descriptive difference −4.2971 percentage points; different release, pairs and models |
| Pool / representation | 11 models; text-embedding-3-small | 8 admitted models; local BGE-M3, 1,024 → 64 dimensions | Representation and pool effects are confounded |
| Data details | 36,497 queries, 64 tasks | Same query count, 86 raw task labels, pinned 0-shot candidate | Same count alone does not establish release equivalence |
| Budget | Published GPT-4-relative headline and absolute-dollar experiments | Tuning-calibrated reference and historical recorded-cost units | No invented curve digitization or matched $1/$2 comparisons |

Exact means, per-seed values and IDs come from `reports/full-experiments-2026-10-29/aggregate/per-seed.csv`. The table deliberately preserves unfavorable results. Plausible deviation sources are the representation, preference subset, mapping exclusions, cost calibration and split construction. This study does not isolate their causal contributions.

## Three strengths with evidence

1. **Feedback efficiency.** The learner updates only the chosen arm. The local evaluator and independent replay enforce this contract while full outcomes are reserved for scoring oracle regret. This avoids requiring an all-arm label at each learning step.
2. **Reusable preference initialization.** Two-stage projection and preference fitting give a practical cold start. Local held-out preference accuracy is about 60.7%, supporting nonrandom preference signal, but this does not guarantee better downstream routing or transfer.
3. **Modular deployment control.** Separating learning from a frozen budget policy makes the budget grid and common baselines auditable. Main full-learning PILOT has zero recorded overspends in its 48 conditions; this is empirical evidence, not a worst-case guarantee.

## Two principal limitations and tests

**L1 — the direction-only reward model discards arm magnitude.** With a constant unit context (1,0) and positive collinear priors (0.2,0), (0.8,0), both cosine means equal one. A ridge linear mean preserves the reward difference. The predeclared deterministic stress gives PILOT 600 cumulative regret after 2,000 rounds versus 1.2 for a linear control at alpha=1. Both start with identical oracle priors, ridge matrices and selected-only feedback. This control is not the zero-prior LinUCB baseline. The case violates the paper's unit-direction reward assumption: it demonstrates sensitivity to misspecification, **not a counterexample to the conditional OFUL theorem**. Nine horizon/alpha cases are retained, including less favorable control values.

**L2 — estimated-cost feasibility does not imply realized-cost feasibility.** Cost estimation uses known input tokens and per-arm mean output tokens learned on tuning data. A fixed-choice counterfactual multiplying recorded selected costs by 1.10 makes 25/48 main, full-learning PILOT conditions exceed budget, compared with 0/48 at 1.00. Across all methods/variants, 12/1,098 conditions already exceed their budgets at 1.00. These populations must not be confused. The multiplier is an assumption, not an estimated probability or a rerun with a robust controller. No provider requests or actual payments occur.

The paper discusses feedback noise and binned budgeting, but does not establish robustness to these two specific mechanisms. The first challenges the representation of expected reward; the second challenges transferring a feasibility claim from estimates to realized charges.

## Two testable future directions

1. Preserve/calibrate reward magnitude, e.g. a linear reward head or learned arm scale with a matching confidence model. Evaluate magnitude sweeps and held-out LLM routing under identical priors and budgets. Success requires lower regret without losing budgeted service utility; this is proposed, not demonstrated.
2. Reserve budget for cost uncertainty using tuning-only cost quantiles or a validated upper-bound model. Evaluate tail overspend, coverage and utility under out-of-distribution response lengths. A deterministic guarantee would additionally require a valid hard cost bound; a quantile alone is not one.

## Theory audit and proof intuition

Proposition 2.1 / Appendix C.1 concern **preference-initialized OFUL (PI-OFUL)**. The familiar confidence radius replaces the zero-prior distance ||theta*|| by ||theta_pref−theta*||. The improvement requires the prior to be closer and the linear/noise assumptions to hold. It is not an unconditional theorem for the implemented cosine-normalized PILOT selection rule.

Let A = X^T X + lambda I and b = X^T y + lambda theta_pref. Then:

`theta_hat − theta* = A^(-1) X^T epsilon − lambda A^(-1)(theta* − theta_pref)`.

The final printed PDF omits rewards in one displayed definition of b (p. 12) and the inverse in the bias term (p. 13). These are algebraic/typographical discrepancies; correcting them recovers the standard ridge identity. They do not by themselves disprove the conditional regret argument. A separate unresolved question is how preference-domain accuracy controls target-domain parameter distance. The inverse-accuracy lambda heuristic is not a proved transfer certificate.

The local dense implementation solves K ridge systems for prediction: O(K d^3) time and O(K d^2) state. A selected-arm update also validates positive definiteness/solves in O(d^3). Rank-one inverse maintenance could reduce arithmetic to O(K d^2), but that optimization is not implemented. Embedding inference is separate from router-state complexity.

## Source and code status

No official implementation link was located on the ACL entry or authors' project site at the check date. A third-party GitHub reproduction is not author code. The project-site BibTeX has an unrelated title; use ACL metadata.

Predecessors: [Li et al., WWW 2010](https://arxiv.org/abs/1003.0146) provides contextual LinUCB exploration; [Ding et al., ICLR 2024](https://openreview.net/pdf?id=02f3mUtqnM) studies quality-aware small/large model routing with supervised quality-gap information. These supply distinct exploration and routing foundations. They are contextual comparisons, not newly executed local baselines.

Remaining submission gaps: public W&B URLs, public GitHub visibility, and Moodle submission. A fresh end-to-end GPU rebuild on another machine has not been tested; the hosted-data reconstruction and historical full-run replays have been tested separately.
