"""Tests for the qlora-calc VRAM estimator."""

import json

import pytest

from qloracalc import estimator as E
from qloracalc.models import PRESETS, parse_gpu, parse_size


def llama8b(**kw):
    base = dict(model=PRESETS["llama-3-8b"], quant_bits=4, rank=16,
                seq_len=2048, batch=2)
    base.update(kw)
    return E.TrainConfig(**base)


# ---- model/parse helpers -------------------------------------------------

def test_parse_size():
    assert parse_size("7b") == 7.0
    assert parse_size("8.03B") == pytest.approx(8.03)
    assert parse_size("800m") == pytest.approx(0.8)
    assert parse_size("13") == 13.0


def test_parse_gpu_names_and_numbers():
    assert parse_gpu("rtx4090") == (24.0, "rtx4090")
    assert parse_gpu("24") == (24.0, "24GB")
    assert parse_gpu("80gb") == (80.0, "80GB")
    with pytest.raises(ValueError):
        parse_gpu("flux-capacitor")


def test_config_validation():
    with pytest.raises(ValueError):
        llama8b(quant_bits=2)
    with pytest.raises(ValueError):
        llama8b(target_modules="everything")
    with pytest.raises(ValueError):
        llama8b(optimizer="sgd")
    with pytest.raises(ValueError):
        llama8b(rank=0)


# ---- weights math --------------------------------------------------------

def test_base_weights_llama3_8b_4bit():
    est = E.estimate(llama8b())
    # 8.03B params x 4.4 bits / 8
    expected_gb = 8.03e9 * 4.4 / 8 / E.GIB
    assert est.gb("base weights (4-bit)") == pytest.approx(expected_gb, rel=1e-9)
    # 8.03B params at an effective 4.4 bits: ~4.1 GiB, the known QLoRA figure
    assert est.gb("base weights (4-bit)") == pytest.approx(4.11, abs=0.05)


def test_quantization_ordering():
    gb4 = E.estimate(llama8b(quant_bits=4)).total_gb
    gb8 = E.estimate(llama8b(quant_bits=8)).total_gb
    gb16 = E.estimate(llama8b(quant_bits=16)).total_gb
    assert gb4 < gb8 < gb16


# ---- LoRA math ------------------------------------------------------------

def test_lora_param_count_attention():
    # 4 modules x 2 x r x h x layers
    n = E.lora_param_count(PRESETS["llama-3-8b"], 16, "attention")
    assert n == 4 * 2 * 16 * 4096 * 32


def test_all_linear_trains_more_than_attention():
    a = E.lora_param_count(PRESETS["llama-3-8b"], 16, "attention")
    b = E.lora_param_count(PRESETS["llama-3-8b"], 16, "all-linear")
    assert b > a


def test_rank_scales_adapters_grads_optimizer():
    e16 = E.estimate(llama8b(rank=16))
    e64 = E.estimate(llama8b(rank=64))
    assert e64.lora_params == 4 * e16.lora_params
    assert e64.total_gb > e16.total_gb


def test_8bit_optimizer_smaller_than_fp32():
    e8 = E.estimate(llama8b(optimizer="adamw-8bit"))
    efp = E.estimate(llama8b(optimizer="adamw"))
    assert e8.gb("optimizer states (adamw-8bit)") < \
        efp.gb("optimizer states (adamw)")
    assert e8.total_gb < efp.total_gb


# ---- activation math ------------------------------------------------------

def test_grad_checkpoint_saves_memory():
    on = E.estimate(llama8b(grad_checkpoint=True))
    off = E.estimate(llama8b(grad_checkpoint=False))
    assert off.total_gb > on.total_gb


def test_flash_attn_saves_memory_long_seq():
    cfg = dict(seq_len=8192, batch=1)
    on = E.estimate(llama8b(flash_attn=True, **cfg))
    off = E.estimate(llama8b(flash_attn=False, **cfg))
    assert off.total_gb > on.total_gb


def test_batch_and_seq_monotonic():
    b1 = E.estimate(llama8b(batch=1)).total_gb
    b4 = E.estimate(llama8b(batch=4)).total_gb
    s1 = E.estimate(llama8b(seq_len=1024)).total_gb
    s2 = E.estimate(llama8b(seq_len=4096)).total_gb
    assert b1 < b4
    assert s1 < s2


def test_components_sum_to_total():
    est = E.estimate(llama8b())
    parts = sum(nbytes for _, nbytes in est.components)
    assert parts == pytest.approx(est.total_bytes, rel=1e-9)


# ---- fit / verdict ----------------------------------------------------------

def test_classic_recipe_fits_24gb():
    est = E.estimate(llama8b())
    assert E.fits(est, 24.0)


def test_70b_does_not_fit_24gb():
    cfg = E.TrainConfig(model=PRESETS["llama-3-70b"], quant_bits=4,
                        rank=16, seq_len=2048, batch=1)
    est = E.estimate(cfg)
    assert not E.fits(est, 24.0)
    assert E.fits(est, 80.0)


def test_max_batch_brackets_gpu():
    cfg = llama8b(batch=2)
    mb = E.max_batch(cfg, 24.0)
    assert mb >= 1
    from dataclasses import replace
    assert E.estimate(replace(cfg, batch=mb)).total_gb <= 24.0
    assert E.estimate(replace(cfg, batch=mb + 1)).total_gb > 24.0


def test_max_batch_zero_when_batch1_overflows():
    cfg = E.TrainConfig(model=PRESETS["llama-3-70b"], quant_bits=16,
                        rank=64, seq_len=8192, batch=1)
    assert E.max_batch(cfg, 16.0) == 0


def test_max_batch_shrinks_with_seq_len():
    cfg = llama8b()
    assert E.max_batch(cfg, 24.0) >= E.max_batch(
        E.TrainConfig(**{**cfg.__dict__, "seq_len": 8192}), 24.0)


def test_suggest_mentions_batch_when_overflowing():
    cfg = E.TrainConfig(model=PRESETS["llama-3-70b"], quant_bits=4,
                        rank=16, seq_len=2048, batch=4)
    est = E.estimate(cfg)
    tips = E.suggest(cfg, est, 24.0)
    assert any("batch" in t for t in tips)


def test_suggest_mentions_headroom_when_comfortable():
    cfg = llama8b(batch=1)
    est = E.estimate(cfg)
    tips = E.suggest(cfg, est, 80.0)
    assert any("headroom" in t for t in tips)


def test_describe_config_mentions_key_settings():
    desc = E.describe_config(llama8b())
    for token in ("4-bit", "rank 16", "attention", "2048", "batch 2",
                  "adamw-8bit", "bf16"):
        assert token in desc
