# RouteLLM MF versus PILOT on the same local preference500 split

This is the paper preference-prediction experiment, not a RouteLLM budgeted routing curve.
Train847 / calibration211 / test500, grouped by prompt, using the same real cached text-embedding-3-small vectors.
The three seeds reuse the same500 pairs; they are not1500 independent test examples.

| Seed | PILOT correct /500 | RouteLLM MF correct /500 | MF minus PILOT (pp) | Selected MF epoch |
|---:|---:|---:|---:|---:|
|7|319|338|+3.80|84|
|42|313|340|+5.40|88|
|2026|321|343|+4.40|100|

Mean PILOT63.5333%; RouteLLM MF68.0667%; difference+4.5333percentage points.

The official training class is extracted from pinned RouteLLM commit0b64fdafe049e596a3f5657c219329f24af24198.
Architecture:64 model rows,128 latent dimensions, frozen1536D prompt table, normalized model directions, unnormalized linear text projection, unbiased scalar interaction head.
Official example settings:100epochs, batch64, Adam learning rate0.0003, weight decay0.00001, training noise alpha0.1.
Adaptations: matched three-way split instead of upstream95/5; choose minimum noise-free calibration BCE (earliest exact tie), save that checkpoint, and never select on the final test.
The upstream example evaluator calls its noisy forward when calculating loss. This local adaptation evaluates calibration with test=True, stated explicitly.

Independent NumPy reconstruction checks all1558 predictions per seed, selected calibration BCE, source/cache/weight pins and recorded split separation.
An actual seed42 refit after swapping all500 test winner/loser labels produces elementwise identical selected weights; test accuracy340/500 becomes160/500.
Inference/training network connections are denied; additional inference API calls0.

Limits: these500 rows were seen in historical project fits; this is a fresh local resplit, not author-identical or project-wide unseen. The preference pool contains8 verified models rather than the paper11. Do not report a matched reproduction error against the paper accuracy.

Primary source: [official MF training example](https://github.com/lm-sys/RouteLLM/blob/0b64fdafe049e596a3f5657c219329f24af24198/routellm/routers/matrix_factorization/train_matrix_factorization.py).

Commands (sequential, one BLAS/Torch thread):
```powershell
.venv/Scripts/python.exe -m tools.openai_study_routellm_train freeze
.venv-embedding/Scripts/python.exe -m tools.openai_study_routellm_train run --seed 7
.venv-embedding/Scripts/python.exe -m tools.openai_study_routellm_train run --seed 42
.venv-embedding/Scripts/python.exe -m tools.openai_study_routellm_train run --seed 2026
.venv-embedding/Scripts/python.exe -m tools.openai_study_routellm_train run --seed 42 --mutation-probe
.venv/Scripts/python.exe -m tools.openai_study_routellm_verify
.venv/Scripts/python.exe -m tools.openai_study_routellm_report
```

Freeze refuses an existing output directory; this report and raw runs are immutable.
