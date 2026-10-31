# Binary Hybrid hardware check before training

The separate drift classifier keeps its already frozen microbatch2/accumulation64
recipe through all five epochs. Do not interrupt it or edit its pinned producer.

The binary classifier has not trained yet. Its conservative prepared recipe is
also2/64, whereas the primary Microsoft CLI defaults to8/16 (both effective128).
The real2-example backward probe uses4.59GB allocated GPU memory on a12GB card.
Measured drift throughput makes a full32270-query/five-epoch binary fit lengthy.

Before any binary outcome, measure the original8-example microbatch on the same
public model, FP32 parameters/FP16 autocast, fixed128 tokens, one CPU thread and
one GPU process, using synthetic questions only. No paid API. Wait for the drift
job to end and require>=8GiB available physical RAM; inspect GPU free memory.

Decision rule is hardware-only: use the official8/16 default for binary if the
real forward/backward/Adafactor probe fits and improves per-example throughput.
Otherwise retain the conservative recipe. Do not choose precision, epochs,
thresholds or settings from binary validation/test performance. Keep all five
epochs, seed42, partitions, label rule, cap128, learning rate and checkpoint rule.

If8/16 is feasible, create a separately declared additive producer/protocol with
explicit source reuse/patch hashes. Preserve the original frozen2/64 preparation
as superseded-before-fit, never mutate it. Independently verify the actual final
recipe and document its batch/dropout-order difference from drift. The publication
and report must identify the recipe actually run. This check does not authorize
parallel model jobs, extra workers, stopping active work or paid services.

## Actual pre-fit decision and execution

Hardware profile passed: micro2 median10.9261306s/128 examples, micro8
4.0363829s, peak7,028,448,768 allocated GPU bytes. The final real8-example
probe passed backward/update, exact manual head replay and label negative
control. Full audit retains32270/1000/3227 rows with unchanged token hashes.
The final separate binary protocol was frozen before fitting:
manifest839ba902e548dcb671a20119b7d712df8e809e5af6d8640acaaaed7fcfa47cba.
All five epochs completed using8/16/eval8; no binary validation/test result
selected this change. The preliminary default-batch proof/audit was retained
under hybrid-default-batch-before-metadata-clarification after correcting only
its inherited difference-description string, then repeated before declaration.
The final additive adapter is now frozen; do not edit it or its parents.

Final full fit chose epoch4 by validationCE0.5797713691.1265 attempted steps,
1259 applied and6 normal AMP overflow skips; no retry to erase these events.
All5x1000 validation plus3227 test logits independently replayed with maxdiff0;
7-file immutable report packagea05b054621b07681d64fb20df37cc6cd0ecafb2708cd1fa76c898295b880f72e.
