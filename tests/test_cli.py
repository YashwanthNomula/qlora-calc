"""CLI tests for qlora-calc."""

import json
import os
import tempfile

import pytest

from qloracalc.cli import main


def run(argv):
    return main(argv)


def test_estimate_text(capsys):
    assert run(["estimate", "--model", "llama-3-8b", "--gpu", "24"]) == 0
    out = capsys.readouterr().out
    assert "llama-3-8b" in out
    assert "FITS" in out
    assert "total" in out


def test_estimate_json_parses(capsys):
    assert run(["estimate", "--model", "llama-3-8b", "--gpu", "24",
                "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "llama-3-8b"
    assert payload["fits"] is True
    assert payload["total_gb"] > 0
    assert "base weights (4-bit)" in payload["components_gb"]


def test_strict_exits_1_when_over_budget(capsys):
    rc = run(["estimate", "--model", "llama-3-70b", "--gpu", "24",
              "--batch", "1", "--strict"])
    assert rc == 1
    assert "EXCEEDS" in capsys.readouterr().out


def test_strict_exits_0_when_fitting(capsys):
    rc = run(["estimate", "--model", "llama-3-8b", "--gpu", "24", "--strict"])
    assert rc == 0


def test_fit_reports_max_batch(capsys):
    assert run(["fit", "--model", "llama-3-8b", "--gpu", "24"]) == 0
    out = capsys.readouterr().out
    assert "max micro-batch" in out


def test_fit_json(capsys):
    assert run(["fit", "--model", "llama-3-8b", "--gpu", "24",
                "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["max_batch"] >= 1


def test_compare_covers_all_bit_widths(capsys):
    assert run(["compare", "--model", "llama-3-8b", "--gpu", "24"]) == 0
    out = capsys.readouterr().out
    for bits in ("4", "8", "16"):
        assert bits in out


def test_models_lists_presets(capsys):
    assert run(["models"]) == 0
    out = capsys.readouterr().out
    assert "llama-3-8b" in out
    assert "mistral-7b" in out


def test_demo_runs(capsys):
    assert run(["demo"]) == 0
    out = capsys.readouterr().out
    assert "llama-3-8b" in out
    assert "llama-3-70b" in out


def test_unknown_model_errors():
    with pytest.raises(SystemExit) as exc:
        run(["estimate", "--model", "gpt-99"])
    assert exc.value.code == 2


def test_custom_model():
    assert run(["estimate", "--params-b", "13", "--layers", "40",
                "--hidden", "5120", "--ffn", "13824", "--gpu", "40"]) == 0


def test_config_file(tmp_path, capsys):
    cfg = {"model": "mistral-7b", "rank": 32, "seq_len": 4096}
    p = tmp_path / "q.json"
    p.write_text(json.dumps(cfg))
    assert run(["estimate", "--config", str(p), "--gpu", "24",
                "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "mistral-7b"
    assert payload["config"]["rank"] == 32
    assert payload["config"]["seq_len"] == 4096


def test_cli_overrides_config_file(tmp_path, capsys):
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"model": "mistral-7b", "rank": 32}))
    assert run(["estimate", "--config", str(p), "--rank", "8",
                "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["rank"] == 8


def test_bad_config_file_errors(tmp_path):
    p = tmp_path / "q.json"
    p.write_text("{not json")
    with pytest.raises(SystemExit) as exc:
        run(["estimate", "--config", str(p)])
    assert exc.value.code == 2


def test_estimate_gpus1_matches_default(capsys):
    assert run(["estimate", "--model", "llama-3-8b", "--gpu", "24",
                "--gpus", "1"]) == 0
    out1 = capsys.readouterr().out
    assert run(["estimate", "--model", "llama-3-8b", "--gpu", "24"]) == 0
    out2 = capsys.readouterr().out
    assert out1 == out2


def test_estimate_gpus2_text(capsys):
    rc = run(["estimate", "--model", "llama-3-70b", "--batch", "1",
              "--gpu", "rtx4090", "--gpus", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "data-parallel" in out
    assert "tensor-parallel" in out
    assert "recommendation" in out


def test_estimate_gpus2_json(capsys):
    rc = run(["estimate", "--model", "llama-3-8b", "--gpu", "24",
              "--gpus", "2", "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpus"] == 2
    assert set(payload["strategies"]) == {"data-parallel", "tensor-parallel"}
    dp = payload["strategies"]["data-parallel"]
    assert dp["per_gpu_batch"] == pytest.approx(1.0)
    assert dp["fits"] is True
    assert "base weights (replicated)" in dp["per_gpu_components_gb"]
    assert payload["recommendation"] in ("data-parallel", "tensor-parallel")


def test_estimate_gpus_json_neither_fits(capsys):
    rc = run(["estimate", "--model", "llama-3-70b", "--batch", "1",
              "--gpu", "rtx4090", "--gpus", "2", "--format", "json",
              "--strict"])
    assert rc == 0  # tensor-parallel fits, so strict passes
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommendation"] == "tensor-parallel"


def test_estimate_gpus_strict_fails_when_neither_fits(capsys):
    rc = run(["estimate", "--model", "llama-3-70b", "--batch", "1",
              "--gpu", "8", "--gpus", "2", "--strict"])
    assert rc == 1
    assert "neither strategy fits" in capsys.readouterr().out


def test_estimate_gpus_zero_errors(capsys):
    assert run(["estimate", "--model", "llama-3-8b", "--gpus", "0"]) == 2
    assert "--gpus must be >= 1" in capsys.readouterr().err
