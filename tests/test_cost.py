"""Tests for the run-cost estimator (`qcalc cost`)."""

import argparse
import json

import pytest

from qloracalc.cli import main
from qloracalc.cost import (
    CostEstimate,
    estimate_cost,
    parse_gpu_rate,
    parse_wall_clock,
)


# --- parse_gpu_rate (argparse-level, fail closed) ---

def test_parse_gpu_rate_valid():
    assert parse_gpu_rate("3.50") == pytest.approx(3.50)
    assert parse_gpu_rate("2") == pytest.approx(2.0)
    assert parse_gpu_rate("0.01") == pytest.approx(0.01)


def test_parse_gpu_rate_rejects_zero_and_negative():
    for bad in ("0", "0.0", "-3.50", "-0.01"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_gpu_rate(bad)


def test_parse_gpu_rate_rejects_non_finite():
    for bad in ("nan", "NaN", "inf", "-inf"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_gpu_rate(bad)


def test_parse_gpu_rate_rejects_non_numeric():
    for bad in ("free", "", "3.5/hr"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_gpu_rate(bad)


# --- parse_wall_clock ---

def test_parse_wall_clock_decimal_hours():
    assert parse_wall_clock("3.5") == pytest.approx(3.5)
    assert parse_wall_clock("  12  ") == pytest.approx(12.0)
    assert parse_wall_clock("0.25") == pytest.approx(0.25)


def test_parse_wall_clock_hms():
    assert parse_wall_clock("1:30:00") == pytest.approx(1.5)
    assert parse_wall_clock("0:45:00") == pytest.approx(0.75)
    assert parse_wall_clock("2:00:00") == pytest.approx(2.0)
    assert parse_wall_clock("90:00") == pytest.approx(1.5)  # M:SS form
    assert parse_wall_clock("0:00:36") == pytest.approx(0.01)


def test_parse_wall_clock_rejects():
    for bad in ("", "   ", "abc", "-1", "0", "0.0", "nan", "inf",
                "1:70:00", "1:30:99", "1:2:3:4", "-1:30:00", "12x"):
        with pytest.raises(ValueError):
            parse_wall_clock(bad)


# --- estimate_cost ---

def test_estimate_cost_math():
    est = estimate_cost(8.0, 3.50, gpus=1)
    assert est.total_cost == pytest.approx(28.0)
    est2 = estimate_cost(2.0, 4.00, gpus=2)
    assert est2.total_cost == pytest.approx(16.0)
    assert isinstance(est, CostEstimate)


def test_cost_per_million_tokens():
    est = estimate_cost(10.0, 2.00, total_tokens=30_000_000)
    # $20 over 30M tokens -> $20 / 30 = $0.6667 per M
    assert est.cost_per_million_tokens == pytest.approx(20.0 / 30.0)
    no_tokens = estimate_cost(10.0, 2.00)
    assert no_tokens.total_tokens is None
    assert no_tokens.cost_per_million_tokens is None


def test_estimate_cost_validation():
    with pytest.raises(ValueError):
        estimate_cost(0, 3.50)
    with pytest.raises(ValueError):
        estimate_cost(-2, 3.50)
    with pytest.raises(ValueError):
        estimate_cost(float("nan"), 3.50)
    with pytest.raises(ValueError):
        estimate_cost(8.0, 0)
    with pytest.raises(ValueError):
        estimate_cost(8.0, -1)
    with pytest.raises(ValueError):
        estimate_cost(8.0, float("inf"))
    with pytest.raises(ValueError):
        estimate_cost(8.0, 3.50, gpus=0)
    with pytest.raises(ValueError):
        estimate_cost(8.0, 3.50, total_tokens=-5)


# --- CLI ---

def test_cli_cost_throughput_path(capsys):
    rc = main(["cost", "--model", "llama-3-8b", "--tflops", "165",
               "--dataset-tokens", "10000000", "--epochs", "1",
               "--gpu-rate", "3.50"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "total cost:" in out
    assert "$/M tokens:" in out
    assert "wall-clock" in out
    assert "per-GPU rate:" in out
    assert "qcalc ships no price table" in out


def test_cli_cost_wall_clock_override(capsys):
    rc = main(["cost", "--model", "llama-3-8b",
               "--wall-clock", "2:00:00",
               "--gpu-rate", "4.00", "--gpus", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    # 2 h x $4 x 2 GPUs = $16.00
    assert "total cost:     $16.00" in out
    assert "override" in out
    # no dataset given -> no per-million-tokens line
    assert "$/M tokens:" not in out


def test_cli_cost_override_with_dataset_gets_per_m_line(capsys):
    rc = main(["cost", "--model", "llama-3-8b",
               "--wall-clock", "3.5",
               "--dataset-tokens", "1000000", "--epochs", "2",
               "--gpu-rate", "2.00"])
    assert rc == 0
    out = capsys.readouterr().out
    # $7.00 over 2M tokens -> $3.50/M
    assert "$/M tokens:     $3.50" in out


def test_cli_cost_json(capsys):
    rc = main(["cost", "--model", "llama-3-8b",
               "--wall-clock", "1:00:00",
               "--gpu-rate", "3.00", "--gpus", "2",
               "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["wall_clock_hours"] == pytest.approx(1.0)
    assert payload["gpu_rate_per_hour"] == pytest.approx(3.0)
    assert payload["gpus"] == 2
    assert payload["total_cost"] == pytest.approx(6.0)
    assert payload["total_tokens"] is None
    assert payload["cost_per_million_tokens"] is None


def test_cli_cost_requires_gpu_rate():
    with pytest.raises(SystemExit) as exc:
        main(["cost", "--model", "llama-3-8b",
              "--wall-clock", "1:00:00"])
    assert exc.value.code == 2


def test_cli_cost_rejects_bad_rate():
    for bad in ("0", "-3.50", "nan"):
        with pytest.raises(SystemExit) as exc:
            main(["cost", "--model", "llama-3-8b",
                  "--wall-clock", "1:00:00", "--gpu-rate", bad])
        assert exc.value.code == 2


def test_cli_cost_rejects_bad_wall_clock():
    assert main(["cost", "--model", "llama-3-8b",
                 "--wall-clock", "abc", "--gpu-rate", "3.00"]) == 2


def test_cli_cost_needs_wall_clock_source():
    # no --wall-clock and no dataset -> cannot derive hours
    assert main(["cost", "--model", "llama-3-8b",
                 "--tflops", "165", "--gpu-rate", "3.00"]) == 2


def test_cli_cost_rejects_wall_clock_plus_tflops():
    assert main(["cost", "--model", "llama-3-8b",
                 "--wall-clock", "1:00:00", "--tflops", "165",
                 "--gpu-rate", "3.00"]) == 2
