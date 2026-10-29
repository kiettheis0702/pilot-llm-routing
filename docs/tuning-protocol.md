# Tuning-only selection

This stage implements task 3.2. The paper supplies the alpha grid
1/1.5/2/5/10 and EpochGreedy window grid 10/50/100/500 and selects by tuning reward
([Section 3.2](https://arxiv.org/html/2508.21141v2#S3.SS2)). The repository uses
one prequential pass through the frozen tuning stream: predict/select, reveal only
the chosen arm's reward, update, and score the accumulated reward. This exact
selection schedule, seed 42 and smallest-grid-entry tie rule are local choices.
Every candidate, including unselected candidates, retains its complete trace,
final frozen state and score. Tuning state is never reused as the learning state.

Selection receives only tuning contexts and the tuning outcome grid. Non-tuning
rows are removed before outcome validation, hashing or scoring. Saved fingerprints
cover only selected tuning data. Invalid or mutated deployment labels therefore
cannot change any selected parameter, candidate trace or final tuning state.
Malformed tuning data still fails before any output directory is created.

The real runner verifies the prepared source hashes, full embedding identity and
prior/fit lineage, then constructs only tuning contexts from the real projection.
The `mmlu` domain filters existing tuning IDs by `task.startswith('mmlu-')`, keeps
their original order, and never draws replacements from learning/deployment.
Thus its tuning subset is smaller than 1,000; this differs from an independently
resplit MMLU experiment and is disclosed.

For subsequent main experiments the local pretraining choice is dimension 64,
30 projection epochs and 40 prior epochs, learning rate .03, margin .2, seed 42.
The source does not supply all these settings. They are fixed before routing
tuning; no preference holdout score selects them. The earlier dimension-8, 2/5
epoch checkpoint remains a smoke artifact. Replicate priors/seeds and the complete
experiment inventory will be frozen separately before full deployment runs.

Budget calibration, when requested, uses the selected candidates' pre-feedback
predictions and tuning-only token estimates. Shared LB is half the minimum
positive score/cost ratio; UB is twice its maximum. Zero-cost/nonpositive-score
pairs are counted and excluded from these positive bounds. These are empirical
assumptions, not a guarantee for deployment. The reference cost is the mean
estimated tuning cost of a named arm; actual costs and deployment outcomes never
enter the calculation. Final budget multipliers and bin sizes are declared in
the experiment configuration, not selected by this stage.

Validation: exact candidate inventory and tie selection; selected-only feedback;
non-tuning mutation invariance, including malformed held-out rows; missing or
malformed tuning-grid failures before output; immutable run/checkpoint readback;
independent real trace and score replay; full regression. Final reports do not
use tuning reward as held-out deployment performance.
