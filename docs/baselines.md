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
