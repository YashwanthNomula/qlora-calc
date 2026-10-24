"""Terminal and JSON reports for qlora-calc."""

from __future__ import annotations

import json

from qloracalc.estimator import (
    Estimate,
    StrategyPlan,
    describe_config,
    fits,
    suggest,
)
from qloracalc.cost import CostEstimate
from qloracalc.throughput import ThroughputEstimate, format_duration

GIB = 1024 ** 3

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


def _strategy_verdict(plan: StrategyPlan, gpu_gb: float) -> str:
    if plan.fits(gpu_gb):
        return f"FITS - {plan.headroom(gpu_gb):.1f} GB headroom"
    return f"EXCEEDS by {plan.per_gpu_gb - gpu_gb:.1f} GB"


def render_shard_plan(shards: dict[str, StrategyPlan], cfg,
                      gpu_gb: float, gpu_label: str,
                      recommendation: tuple[str | None, str]) -> str:
    n = next(iter(shards.values())).gpus
    lines = [
        f"multi-GPU sharding plan: {cfg.model.name} "
        f"({cfg.model.params_b:g}B params) on {n} x {gpu_gb:g} GB "
        f"({gpu_label})",
        f"  global config: {describe_config(cfg)}",
        "",
        f"  {'strategy':<16}{'per-GPU GB':>12}  verdict",
    ]
    for key, plan in shards.items():
        lines.append(f"  {key:<16}{plan.per_gpu_gb:>12.2f}  "
                     f"{_strategy_verdict(plan, gpu_gb)}")
    for key, plan in shards.items():
        lines += ["", f"  {key} per-GPU breakdown (GB):"]
        for label, nbytes in plan.components:
            lines.append(f"    {label:<34}{nbytes / GIB:>8.2f}")
        lines.append(f"    {'total':<34}{plan.per_gpu_gb:>8.2f}")
    lines += ["", f"  recommendation: {recommendation[1]}"]
    return "\n".join(lines)


def render_throughput(est: ThroughputEstimate) -> str:
    cfg = est.config
    lines = [
        f"training throughput: {cfg.model.name} ({cfg.model.params_b:g}B params)",
        f"  config: {describe_config(cfg)}",
        f"  GPU: {est.gpu_label} @ {est.gpu_tflops:g} peak TFLOPS "
        f"(BF16/FP16 tensor), MFU {est.mfu:.2f} -> "
        f"{est.effective_tflops:.1f} effective TFLOPS",
        f"  compute: {est.flops_per_token:.0f} FLOPs/token "
        f"({'with' if cfg.grad_checkpoint else 'without'} grad checkpointing)",
        "",
        f"  tokens/sec:     {est.tokens_per_sec:,.0f}",
        f"  steps/sec:      {est.steps_per_sec:.4f} "
        f"(step = batch {cfg.batch} x seq {cfg.seq_len} = "
        f"{cfg.batch * cfg.seq_len:,} tokens)",
        f"  sec/step:       {est.seconds_per_step:.2f}",
    ]
    if est.gpus > 1:
        lines.append(f"  scaled:         {est.gpus} GPUs x "
                     f"{est.scaling_eff:.2f} scaling efficiency")
    if est.total_seconds is not None:
        lines += [
            "",
            f"  dataset:        {est.dataset_tokens:,.0f} tokens/epoch "
            f"x {est.epochs} epoch(s)",
            f"  total steps:    {est.total_steps:,.0f}",
            f"  wall-clock:     {format_duration(est.total_seconds)} "
            f"({est.total_hours:.2f} h)",
        ]
    else:
        lines.append("")
        lines.append("  (pass --dataset-tokens or --examples/--avg-len "
                     "for a wall-clock estimate)")
    return "\n".join(lines)


def render_throughput_json(est: ThroughputEstimate) -> str:
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
        "gpu_tflops": est.gpu_tflops,
        "gpu_label": est.gpu_label,
        "mfu": est.mfu,
        "gpus": est.gpus,
        "scaling_eff": est.scaling_eff,
        "flops_per_token": est.flops_per_token,
        "effective_tflops": est.effective_tflops,
        "tokens_per_sec": est.tokens_per_sec,
        "steps_per_sec": est.steps_per_sec,
        "seconds_per_step": est.seconds_per_step,
        "dataset_tokens": est.dataset_tokens,
        "epochs": est.epochs,
        "total_steps": est.total_steps,
        "total_seconds": est.total_seconds,
        "total_hours": est.total_hours,
    }
    return json.dumps(payload, indent=2)


def render_cost(est: CostEstimate) -> str:
    lines = [
        "run cost estimate",
        f"  wall-clock:     {format_duration(est.wall_clock_hours * 3600)} "
        f"({est.wall_clock_hours:.2f} h) - {est.wall_clock_source}",
        f"  per-GPU rate:   ${est.gpu_rate:.2f}/GPU-hour (you supplied it)",
        f"  GPUs:           {est.gpus}",
        "",
        f"  total cost:     ${est.total_cost:,.2f}",
    ]
    per_m = est.cost_per_million_tokens
    if per_m is not None:
        lines.append(
            f"  $/M tokens:     ${per_m:.2f} "
            f"(over {est.total_tokens:,.0f} training tokens)")
    lines += [
        "",
        "  qcalc ships no price table: cloud rates move too fast and vary "
        "by",
        "  region, commitment and spot vs on-demand. Re-run with your own "
        "--gpu-rate",
        "  to compare on-demand vs spot vs reserved quotes.",
    ]
    return "\n".join(lines)


def render_cost_json(est: CostEstimate) -> str:
    per_m = est.cost_per_million_tokens
    payload = {
        "wall_clock_hours": round(est.wall_clock_hours, 4),
        "wall_clock_source": est.wall_clock_source,
        "gpu_rate_per_hour": est.gpu_rate,
        "gpus": est.gpus,
        "total_cost": round(est.total_cost, 2),
        "total_tokens": est.total_tokens,
        "cost_per_million_tokens": (round(per_m, 2) if per_m is not None
                                    else None),
    }
    return json.dumps(payload, indent=2)


def render_shard_plan_json(shards: dict[str, StrategyPlan], cfg,
                           gpu_gb: float, gpu_label: str,
                           recommendation: tuple[str | None, str]) -> str:
    n = next(iter(shards.values())).gpus
    payload = {
        "model": cfg.model.name,
        "params_b": cfg.model.params_b,
        "gpus": n,
        "gpu_gb": gpu_gb,
        "gpu_label": gpu_label,
        "global_batch": cfg.batch,
        "strategies": {
            key: {
                "per_gpu_batch": plan.per_gpu_batch,
                "per_gpu_gb": plan.per_gpu_gb,
                "per_gpu_components_gb": {
                    label: nbytes / GIB
                    for label, nbytes in plan.components
                },
                "fits": plan.fits(gpu_gb),
                "headroom_gb": plan.headroom(gpu_gb),
            }
            for key, plan in shards.items()
        },
        "recommendation": recommendation[0],
        "recommendation_reason": recommendation[1],
    }
    return json.dumps(payload, indent=2)
