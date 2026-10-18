"""Tests for the training throughput / wall-clock estimator."""

import json

import pytest

from qloracalc.estimator import TrainConfig
from qloracalc.models import PRESETS, parse_tflops
from qloracalc.throughput import (
    FLOPS_PER_TOKEN_FULL,
    FLOPS_PER_TOKEN_GRAD_CKPT,
    estimate_throughput,
    flops_per_token,
    format_duration,
)
from qloracalc.cli import main


def _cfg(**over):
    base = {"model": PRESETS["llama-3-8b"], "batch": 2, "seq_len": 2048}
    base.update(over)
    return TrainConfig(**base)


def test_flops_per_token_grad_ckpt_vs_plain():
    assert flops_per_token(_cfg(grad_checkpoint=True)) == FLOPS_PER_TOKEN_GRAD_CKPT
    assert flops_per_token(_cfg(grad_checkpoint=False)) == FLOPS_PER_TOKEN_FULL
    assert FLOPS_PER_TOKEN_GRAD_CKPT == 8.0
    assert FLOPS_PER_TOKEN_FULL == 6.0


def test_tokens_per_sec_formula():
    # 8.03B params, 6N, 165 TFLOPS peak x 0.40 MFU, no grad ckpt
    est = estimate_throughput(_cfg(grad_checkpoint=False), 165.0, mfu=0.40)
    expected = 165.0 * 0.40 * 1e12 / (6.0 * 8.03e9)
    assert est.tokens_per_sec == pytest.approx(expected, rel=1e-9)
    assert est.effective_tflops == pytest.approx(66.0)
    assert est.steps_per_sec == pytest.approx(
        expected / (2 * 2048), rel=1e-9)
    assert est.seconds_per_step == pytest.approx(1.0 / est.steps_per_sec)


def test_grad_checkpointing_costs_throughput():
    plain = estimate_throughput(_cfg(grad_checkpoint=False), 165.0)
    ckpt = estimate_throughput(_cfg(grad_checkpoint=True), 165.0)
    # 8N vs 6N -> exactly 3/4 the throughput
    assert ckpt.tokens_per_sec == pytest.approx(
        plain.tokens_per_sec * 6.0 / 8.0, rel=1e-9)


def test_wall_clock_scales_with_tokens_and_epochs():
    one = estimate_throughput(_cfg(), 165.0, dataset_tokens=10_000_000, epochs=1)
    two = estimate_throughput(_cfg(), 165.0, dataset_tokens=10_000_000, epochs=2)
    double = estimate_throughput(_cfg(), 165.0, dataset_tokens=20_000_000, epochs=1)
    assert two.total_seconds == pytest.approx(2 * one.total_seconds)
    assert double.total_seconds == pytest.approx(2 * one.total_seconds)
    assert one.total_steps == pytest.approx(10_000_000 / (2 * 2048))
    assert one.total_hours == pytest.approx(one.total_seconds / 3600)


def test_no_dataset_means_no_wall_clock():
    est = estimate_throughput(_cfg(), 165.0)
    assert est.total_seconds is None
    assert est.total_steps is None
    assert est.total_hours is None


def test_zero_dataset_means_no_wall_clock():
    est = estimate_throughput(_cfg(), 165.0, dataset_tokens=0)
    assert est.total_seconds is None


def test_multi_gpu_scaling():
    one = estimate_throughput(_cfg(), 165.0, gpus=1)
    two = estimate_throughput(_cfg(), 165.0, gpus=2, scaling_eff=0.9)
    assert two.tokens_per_sec == pytest.approx(
        one.tokens_per_sec * 2 * 0.9, rel=1e-9)
    perfect = estimate_throughput(_cfg(), 165.0, gpus=4, scaling_eff=1.0)
    assert perfect.tokens_per_sec == pytest.approx(
        one.tokens_per_sec * 4, rel=1e-9)


def test_validation():
    with pytest.raises(ValueError):
        estimate_throughput(_cfg(), 0)
    with pytest.raises(ValueError):
        estimate_throughput(_cfg(), -5)
    with pytest.raises(ValueError):
        estimate_throughput(_cfg(), 165.0, mfu=0)
    with pytest.raises(ValueError):
        estimate_throughput(_cfg(), 165.0, mfu=1.5)
    with pytest.raises(ValueError):
        estimate_throughput(_cfg(), 165.0, gpus=0)
    with pytest.raises(ValueError):
        estimate_throughput(_cfg(), 165.0, scaling_eff=0)
    with pytest.raises(ValueError):
        estimate_throughput(_cfg(), 165.0, epochs=0)
    with pytest.raises(ValueError):
        estimate_throughput(_cfg(), 165.0, dataset_tokens=-1)


def test_format_duration():
    assert format_duration(12.5) == "12.5s"
    assert format_duration(59.9) == "59.9s"
    assert format_duration(90) == "1m 30.0s"
    assert format_duration(3 * 3600 + 12 * 60) == "3h 12m"
    assert format_duration(2 * 86400 + 4 * 3600) == "2d 04h"


def test_parse_tflops():
    assert parse_tflops("rtx4090") == (165.0, "rtx4090")
    assert parse_tflops("165") == (165.0, "165 TFLOPS")
    assert parse_tflops("165tflops") == (165.0, "165 TFLOPS")
    with pytest.raises(ValueError):
        parse_tflops("nope")


def test_cli_throughput_text(capsys):
    rc = main(["throughput", "--model", "llama-3-8b", "--gpu", "rtx4090",
               "--dataset-tokens", "10000000", "--epochs", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "tokens/sec" in out
    assert "wall-clock" in out
    assert "10,000,000 tokens/epoch" in out


def test_cli_throughput_examples_avg_len(capsys):
    rc = main(["throughput", "--model", "llama-3-8b", "--tflops", "165",
               "--examples", "5000", "--avg-len", "2000"])
    assert rc == 0
    assert "10,000,000 tokens/epoch" in capsys.readouterr().out


def test_cli_throughput_json(capsys):
    rc = main(["throughput", "--model", "llama-3-8b", "--tflops", "165",
               "--dataset-tokens", "1000000", "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpu_tflops"] == 165.0
    assert payload["total_seconds"] is not None
    assert payload["flops_per_token"] == 8.0  # grad ckpt on by default


def test_cli_throughput_requires_tflops_or_known_gpu():
    assert main(["throughput", "--model", "llama-3-8b"]) == 2


def test_cli_throughput_rejects_both_dataset_forms():
    assert main(["throughput", "--model", "llama-3-8b", "--tflops", "165",
                 "--dataset-tokens", "1000", "--examples", "10",
                 "--avg-len", "100"]) == 2


def test_cli_throughput_rejects_half_examples():
    assert main(["throughput", "--model", "llama-3-8b", "--tflops", "165",
                 "--examples", "10"]) == 2


def test_cli_throughput_rejects_bad_mfu():
    assert main(["throughput", "--model", "llama-3-8b", "--tflops", "165",
                 "--mfu", "1.5"]) == 2
