"""Terminal and JSON reports for qlora-calc."""

from __future__ import annotations

import json

from qloracalc.estimator import Estimate, describe_config, fits, suggest

BAR_W = 28


def _bar(frac: float) -> str:
    filled = int(round(max(0.0, min(1.0, frac)) * BAR_W))
    return "#" * filled + "-" * (BAR_W - filled)


def render_estimate(est: Estimate, gpu_gb: float, gpu_label: str) -> str:
    cfg = est.config
    lines = [
        f"QLoRA VRAM estimate: {cfg.model.name} ({cfg.model.params_b:g}B params)",
        f"  config: {describe_config(cfg)}",
        "",
        f"  {'component':<34}{'GB':>8}  {'share'}",
    ]
    for label, nbytes in est.components:
        gb = nbytes / (1024 ** 3)
        frac = nbytes / est.total_bytes if est.total_bytes else 0
        lines.append(f"  {label:<34}{gb:>8.2f}  [{_bar(frac)}]")
    lines += [
        f"  {'-' * 34}{'--------':>8}",
        f"  {'total':<34}{est.total_gb:>8.2f}",
        "",
        f"  GPU budget: {gpu_gb:g} GB ({gpu_label})",
    ]
    ok = fits(est, gpu_gb)
    if ok:
        headroom = gpu_gb - est.total_gb
        verdict = f"FITS - {headroom:.1f} GB headroom"
        if headroom < 2.0:
            verdict += " (tight: leave a safety margin)"
    else:
        over = est.total_gb - gpu_gb
        verdict = f"EXCEEDS by {over:.1f} GB"
    lines.append(f"  verdict: {verdict}")
    tips = suggest(cfg, est, gpu_gb)
    if tips:
        lines.append("")
        lines.append("  tips:")
        for t in tips:
            lines.append(f"    - {t}")
    return "\n".join(lines)


def render_estimate_json(est: Estimate, gpu_gb: float, gpu_label: str) -> str:
    cfg = est.config
    payload = {
        "model": cfg.model.name,
        "params_b": cfg.model.params_b,
        "config": {
            "quant_bits": cfg.quant_bits,
            "rank": cfg.rank,
            "target_modules": cfg.target_modules,
            "seq_len": cfg.seq_len,
            "batch": cfg.batch,
            "grad_checkpoint": cfg.grad_checkpoint,
            "flash_attn": cfg.flash_attn,
            "optimizer": cfg.optimizer,
            "train_dtype": cfg.train_dtype,
        },
        "lora_params": est.lora_params,
        "components_gb": {label: nbytes / (1024 ** 3)
                          for label, nbytes in est.components},
        "total_gb": est.total_gb,
        "gpu_gb": gpu_gb,
        "gpu_label": gpu_label,
        "fits": fits(est, gpu_gb),
        "headroom_gb": gpu_gb - est.total_gb,
        "tips": suggest(cfg, est, gpu_gb),
    }
    return json.dumps(payload, indent=2)


def render_compare(rows: list[tuple[int, Estimate]], gpu_gb: float,
                   gpu_label: str) -> str:
    lines = [
        f"quantization comparison on {gpu_gb:g} GB ({gpu_label}):",
        "",
        f"  {'bits':<6}{'total GB':>10}  {'verdict':<22}largest batch",
    ]
    for bits, est in rows:
        ok = fits(est, gpu_gb)
        verdict = "FITS" if ok else f"EXCEEDS by {est.total_gb - gpu_gb:.1f} GB"
        from qloracalc.estimator import max_batch
        mb = max_batch(est.config, gpu_gb)
        lines.append(f"  {bits:<6}{est.total_gb:>10.2f}  {verdict:<22}{mb}")
    return "\n".join(lines)


def render_compare_json(rows: list[tuple[int, Estimate]], gpu_gb: float,
                        gpu_label: str) -> str:
    from qloracalc.estimator import max_batch
    payload = {
        "gpu_gb": gpu_gb,
        "gpu_label": gpu_label,
        "options": [
            {
                "quant_bits": bits,
                "total_gb": est.total_gb,
                "fits": fits(est, gpu_gb),
                "max_batch": max_batch(est.config, gpu_gb),
            }
            for bits, est in rows
        ],
    }
    return json.dumps(payload, indent=2)


def render_models(models) -> str:
    lines = [f"  {'model':<16}{'params':>8}  {'layers':>6}  {'hidden':>6}  "
             f"{'ffn':>6}  {'vocab':>7}"]
    for m in models:
        lines.append(
            f"  {m.name:<16}{m.params_b:>7.2f}B  {m.layers:>6}  "
            f"{m.hidden:>6}  {m.ffn:>6}  {m.vocab:>7}"
        )
    lines.append("")
    lines.append("  use --model <name>, or a custom size with "
                 "--params-b/--layers/--hidden/--ffn")
    return "\n".join(lines)
