# PILOT online core (synthetic verification)

`scripts/pilot.py` provides `PILOT`, compatible with the existing OnlinePolicy
interface. This implements the online mathematics; it does not train a preference
prior or produce embeddings. All tests use explicitly synthetic contexts/priors.

The [published paper, section 2.2.2 equations 1/2 and Algorithm 1](https://aclanthology.org/2025.findings-emnlp.1301.pdf)
specifies normalized query contexts, A=lambda I, b=lambda theta_pref, theta=solve(A,b),
cosine exploitation plus alpha sqrt(x dot solve(A,x)), and selected-arm ridge updates.
The implementation uses those equations, with a separate positive lambda for each arm.
It does not infer lambda from accuracy: zero accuracy and the operational accuracy
measurement remain unresolved for real pretraining.

```python
from scripts.pilot import PILOT

model = PILOT(
    ("a", "b"), 2,
    priors={"a": [2, 0], "b": [0, 1]},
    lambdas={"a": 2, "b": 1},
    provenance="synthetic-hand-constructed",
    alpha=0.5,
)
model.reset(7)
arm = model.select([3, 0])
model.observe(arm, [3, 0], 1.0)
model.save_checkpoint("pilot-model.json")  # refuses an existing file
restored = PILOT.load_checkpoint("pilot-model.json")
assert restored.scores([1, 1]) == model.scores([1, 1])
```

Contexts are normalized internally for both scoring and feedback. Norm scaling
handles very small/large finite vectors without squaring their original magnitude.
Zero query contexts are rejected. A zero arm estimate has cosine zero: this is an
explicit local numerical convention, not a claim from the paper. Ties use catalog
order. Predictions are cosine affinities in [-1,1], not calibrated probabilities.
Only selected feedback updates state; costs and other-arm outcomes are absent.

Checkpoint JSON version 1 records the ordered catalog, dimension, alpha, original
priors, per-arm lambdas, prior provenance, reset seed, feedback count and learned
A/b. Canonical payload SHA-256 detects accidental corruption. Loading rejects
duplicate keys, unknown/missing fields, invalid settings, nonfinite/mismatched
arrays and nonsymmetric/nonpositive-definite matrices. This is not authentication
against deliberate re-signing. Prior provenance is caller-supplied, not attested.
No pickle is used. Selection has no random behavior, so no RNG state is required.

Load returns a new policy. Continue it directly with select/observe to resume its
model state. `run_learning` deliberately resets policies; passing a loaded model to
that runner starts again from its original prior. Checkpoints do not store query
cursor, evaluator accumulators or prevent repeated feedback across process runs.
Saving serializes and validates before exclusive file creation; parent directories
must already exist. Storage failure can leave an incomplete new file, which load
rejects. Overwrite refusal preserves existing checkpoints.

Run the synthetic suite with:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_pilot.py -q -p no:cacheprovider
```

The analytic example initializes a at theta=[2,0], yet predicts cosine 1 rather
than linear mean 2. After normalized feedback [3,0], reward 1, a has A=diag(3,2)
and b=[5,0]; b remains unchanged. Integration routes two toy rows to a then b,
obtains total reward 2, saves the learned model and reloads identical scores.
These are mathematical checks, not RouterBench results or trained-PILOT evidence.

Evidence (2026-10-28): 37 targeted tests first failed because the module was absent;
the implemented module then passed all 37. The tests cover seeded reset, scale
invariance, zero norms, malformed priors/configuration, atomic invalid updates,
checksum and re-signed invalid schema/state, overwrite refusal, deterministic
model continuation, and evaluator integration.

Full workspace verification at this checkpoint:
`python -B -m pytest -q -p no:cacheprovider` completed with **242 passed in 6.57s**
(exit 0), including other concurrently developed units. `openspec validate
pilot-core --strict` completed with **Change 'pilot-core' is valid** (exit 0).
