# Independent scoped review

One fresh read-only reviewer inspected the importer, profile alignment,
three-way training, tests and scientific scope. It did not edit, run tests,
build, access an API or spawn workers. AO daemon was checked and was not running.

**P2 fixed: mixed synthetic cost-size provenance.** The new runner initially
used only embedding provenance to authorize/label synthetic execution. Real-format
vectors plus a cost-size file marked synthetic could be reported as real.
Added two regression cases; both failed before the fix and passed after it.
The runner now requires an explicit Boolean cost-size synthetic flag, requires
permission when either source is synthetic, and propagates the combined flag to
both final-test and manifest outputs. The final full suite passed564/1skip.

No other material recovery, exact-input, BGE compatibility, partition leakage or
scientific-claim issue was identified. Provider-response authenticity remains
caller-declared, explicitly documented; local file hashes alone are not provider
attestation. Unresolved eleven-model identities remain fidelity limitations.

Concurrent changes to OpenAI setup dependencies/transport were outside this
review and were preserved. Their24 extra offline setup tests are included in
the current full-suite count; the prior preflight snapshot had517 tests.
