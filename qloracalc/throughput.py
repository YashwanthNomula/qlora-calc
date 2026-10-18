"""Training throughput and wall-clock estimates.

Companion to the VRAM estimator: "it fits" is only half the planning
question. This module answers "how long will it take?" from the same
training config, using the standard Kaplan-style FLOPs accounting:

* forward pass: ~2 FLOPs per parameter per token
* backward pass: ~4 FLOPs per parameter per token
* total: ~6N FLOPs/token for a full training step

Gradient checkpointing recomputes the forward pass during backward, so it
adds one extra forward pass: ~8N FLOPs/token. That is the same ~33%
slowdown the VRAM estimator's own tips already promise ("trades ~20%
speed" is the optimistic end; 6N -> 8N is the compute-bound bound).

LoRA adapter FLOPs are folded into the 6N/8N figure: for typical ranks
(r <= 64) adapters are under 2% of the parameter count, well inside the
error bar of the MFU assumption below.

Everything is then scaled by model FLOPs utilization (MFU): the fraction
of the GPU's peak tensor-core throughput a real training run sustains.
0.40 is a realistic default for single-GPU QLoRA; large well-tuned runs
reach 0.50-0.60, small runs with tiny batches often sit below 0.30.
Treat the wall-clock as a planning estimate with a +/-2x band, not a
measurement: data loading, eval, checkpointing, and optimizer overhead
are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

from qloracalc.estimator import TrainConfig

# FLOPs per parameter per token: forward 2N + backward 4N. Gradient
# checkpointing recomputes the forward pass, adding 2N.
FLOPS_PER_TOKEN_FULL = 6.0
FLOPS_PER_TOKEN_GRAD_CKPT = 8.0

DEFAULT_MFU = 0.40
DEFAULT_SCALING_EFF = 0.90  # per-GPU efficiency when scaling past 1 GPU


@dataclass
class ThroughputEstimate:
    config: TrainConfig
    gpu_tflops: float       # peak dense BF16/FP16 tensor-core TFLOPS
    gpu_label: str
    mfu: float
    gpus: int
    scaling_eff: float
    flops_per_token: float
    effective_tflops: float
    tokens_per_sec: float
    steps_per_sec: float
    seconds_per_step: float
    dataset_tokens: float | None
    epochs: int
    total_steps: float | None
    total_seconds: float | None

    @property
    def total_hours(self) -> float | None:
        return None if self.total_seconds is None else self.total_seconds / 3600.0


def flops_per_token(cfg: TrainConfig) -> float:
    """6N normally, 8N with gradient checkpointing (recomputed forward)."""
    return (FLOPS_PER_TOKEN_GRAD_CKPT if cfg.grad_checkpoint
            else FLOPS_PER_TOKEN_FULL)


def estimate_throughput(
    cfg: TrainConfig,
    gpu_tflops: float,
    mfu: float = DEFAULT_MFU,
    *,
    gpus: int = 1,
    scaling_eff: float = DEFAULT_SCALING_EFF,
    dataset_tokens: float | None = None,
    epochs: int = 1,
) -> ThroughputEstimate:
    """Estimate training throughput for one config on one GPU type.

    gpu_tflops is the GPU's peak dense BF16/FP16 tensor-core throughput
    (see GPU_TFLOPS in models.py, or pass --tflops explicitly). dataset_tokens
    is the number of training tokens per epoch; omit it to get rates only
    (tokens/sec, steps/sec) with no wall-clock total.
    """
    if not gpu_tflops > 0:
        raise ValueError("gpu_tflops must be > 0")
    if not 0 < mfu <= 1.0:
        raise ValueError("mfu must be in (0, 1]")
    if gpus < 1:
        raise ValueError("gpus must be >= 1")
    if not 0 < scaling_eff <= 1.0:
        raise ValueError("scaling_eff must be in (0, 1]")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if dataset_tokens is not None and dataset_tokens < 0:
        raise ValueError("dataset_tokens must be >= 0")

    fpt = flops_per_token(cfg)
    effective = gpu_tflops * mfu
    # Multi-GPU: near-linear scaling, discounted by the scaling efficiency
    # (communication + stragglers). Single-GPU runs scale at 1.0.
    gpu_factor = float(gpus) if gpus == 1 else gpus * scaling_eff
    tokens_per_sec = effective * 1e12 / (fpt * cfg.model.params) * gpu_factor

    tokens_per_step = cfg.batch * cfg.seq_len
    steps_per_sec = tokens_per_sec / tokens_per_step
    seconds_per_step = 1.0 / steps_per_sec

    total_steps = total_seconds = None
    if dataset_tokens:
        total_tokens = dataset_tokens * epochs
        total_steps = total_tokens / tokens_per_step
        total_seconds = total_tokens / tokens_per_sec

    return ThroughputEstimate(
        config=cfg,
        gpu_tflops=gpu_tflops,
        gpu_label="",
        mfu=mfu,
        gpus=gpus,
        scaling_eff=scaling_eff,
        flops_per_token=fpt,
        effective_tflops=effective,
        tokens_per_sec=tokens_per_sec,
        steps_per_sec=steps_per_sec,
        seconds_per_step=seconds_per_step,
        dataset_tokens=dataset_tokens,
        epochs=epochs,
        total_steps=total_steps,
        total_seconds=total_seconds,
    )


def format_duration(seconds: float) -> str:
    """Human wall-clock: '12.5s', '45m 10s', '3h 12m', '2d 4h'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{int(hours)}h {int(minutes):02d}m"
    days, hours = divmod(hours, 24)
    return f"{int(days)}d {int(hours):02d}h"
