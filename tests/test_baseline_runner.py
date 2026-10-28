import json
from pathlib import Path

import pandas as pd
import pytest


def test_cli_replay_artifacts_and_shared_inputs(tmp_path):
    from scripts.run_baselines import main
    for name in ("first", "second"):
        main(["--rows", "12", "--dimension", "2", "--seed", "7", "--output-dir", str(tmp_path / name)])
    first = json.loads((tmp_path / "first/comparison.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "second/comparison.json").read_text(encoding="utf-8"))
    assert first["metrics"] == second["metrics"] and len(first["metrics"]) == 7
    assert first["deviations"] and first["source"] == "toy"
    assert first["code"]["implementation_sha256"]
    assert (tmp_path / "first/regret.png").stat().st_size > 1000
    assert (tmp_path / "first/regret.svg").stat().st_size > 1000
    ids, context_hashes = [], []
    for method in first["metrics"]:
        a = pd.read_csv(tmp_path / "first" / method / "trace.csv")
        b = pd.read_csv(tmp_path / "second" / method / "trace.csv")
        pd.testing.assert_frame_equal(a, b)
        assert len(a) == 12
        ids.append(tuple(a.query_id))
        summary = json.loads((tmp_path / "first" / method / "summary.json").read_text(encoding="utf-8"))
        context_hashes.append(summary["context"]["sha256"])
        assert summary["configuration"]["seed"] == 7
    assert len(set(ids)) == len(set(context_hashes)) == 1


def test_cli_refuses_overwrite_and_oversized_runs(tmp_path):
    from scripts.run_baselines import main
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "marker.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        main(["--rows", "4", "--output-dir", str(target)])
    assert marker.read_text() == "preserve"
    for bad in (["--rows", "1001"], ["--rows", "0"], ["--dimension", "65"],
                ["--source", "routerbench"], ["--alpha", "nan"]):
        with pytest.raises(ValueError):
            main([*bad, "--output-dir", str(tmp_path / "bad")])
        assert not (tmp_path / "bad").exists()


def test_plot_can_be_regenerated_from_saved_traces(tmp_path):
    from scripts.run_baselines import main, plot_traces
    main(["--rows", "4", "--output-dir", str(tmp_path / "run")])
    manifest = json.loads((tmp_path / "run/comparison.json").read_text(encoding="utf-8"))
    plot_traces(tmp_path / "run", tuple(manifest["metrics"]))
    assert (tmp_path / "run/regret.png").stat().st_size > 1000
