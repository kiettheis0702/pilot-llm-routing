import json
import hashlib

import numpy as np
import pandas as pd
import pytest


def test_full_toy_pipeline_is_replayable_and_deployment_frozen(tmp_path):
    from scripts.run_toy_pipeline import main
    from scripts.pilot import PILOT
    for name in ("first", "second"):
        main(["--output-dir", str(tmp_path / name)])
    a = json.loads((tmp_path / "first/summary.json").read_text())
    b = json.loads((tmp_path / "second/summary.json").read_text())
    assert a["synthetic"] is True
    assert type(a["working_tree_dirty"]) is bool
    assert len(a["dependency_lock_sha256"]) == 64
    assert set(a["runtime"]) == {"python", "numpy", "pandas"}
    assert a["prior_validation"]["all_lambdas_ready"] is True
    assert a["deployment"]["served_queries"] == 6
    assert a["deployment"]["budget"]["estimated_budget_violation"] is False
    assert a["deployment"]["budget"]["actual_budget_violation_observed"] is True
    assert a["deployment"]["metrics"] == b["deployment"]["metrics"]
    assert a["frozen_router_verified"] is True
    split_bytes = (tmp_path / "first/routing-splits.csv").read_bytes()
    assert b"\r" not in split_bytes  # Git stores LF; the fingerprint must survive checkout.
    assert hashlib.sha256(split_bytes).hexdigest() == a["learning"]["split"]["manifest_sha256"]
    first = PILOT.load_checkpoint(tmp_path / "first/learned.json")
    after = PILOT.load_checkpoint(tmp_path / "first/deployed.json")
    np.testing.assert_array_equal(first.A, after.A)
    np.testing.assert_array_equal(first.b, after.b)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "first/learning/trace.csv"),
                                  pd.read_csv(tmp_path / "second/learning/trace.csv"))
    with pytest.raises(FileExistsError):
        main(["--output-dir", str(tmp_path / "first")])
