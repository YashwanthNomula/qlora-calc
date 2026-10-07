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
