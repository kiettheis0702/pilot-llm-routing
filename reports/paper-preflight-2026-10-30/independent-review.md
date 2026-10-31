# Independent review and resolution

One fresh read-only reviewer examined the new numeric controller option,
offline preparation/audit/comparison tools, tests, protocol and actual output
artifacts. It did not run tests, modify files, spawn workers or call providers.
AO was checked first: its daemon was not running; no AO worker was created.

## P2: new preparation workloads missing from resource discovery — fixed

The Windows resource gate's existing command-line matcher omitted
`tools.paper_preflight.compare`, `.audit` and `.embedding_plan`. Another active
preparation command could therefore be missed on a future rerun. The completed
experiments had no competing jobs and >=8 GiB available physical RAM.

Registered all three command names. Added an actual Windows process probe for
each: a tiny owned process waits for EOF without starting a workload; the
preflight must detect its PID and refuse another heavy job. Before the change:
three failures. After the change: three passes; the full final suite then passed
517 tests, one baseline Windows symlink skip. Every owned probe exited normally;
no user process was killed, suspended or capped.

No additional material scientific, accounting, lineage or paid-transport defect
was reported. The reviewer independently checked 96/48/1098 counts, source/file
hashes, grouped split support and absence of paired action/metric changes.

The later report package is a packaging step; the reviewer was told it was still
being assembled. A separate readback validates its inventory and copied data.
The eleven-model and author-protocol uncertainties remain scientific gates,
not bugs corrected by relabeling data.
