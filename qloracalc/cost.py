"""Run-cost estimator: turn wall-clock hours into dollars.

Companion to the throughput estimator: "how long will it take" is only
useful when you also know what the cloud bill looks like. This module
answers "what will this fine-tune cost?" as

    total_cost = wall_clock_hours x gpu_rate x gpus

The wall-clock hours come either from the throughput estimator (same
--dataset-tokens/--examples/--avg-len, --epochs, --tflops, --mfu, --gpus,
--scaling-eff inputs as `qcalc throughput`) or from a direct
--wall-clock override in decimal hours ("3.5") or H:MM:SS ("1:30:00").

The per-GPU hourly rate is always supplied by the user (--gpu-rate).
qcalc ships no price table: cloud prices move too fast and vary by
region, commitment and spot vs on-demand, so any built-in table would be
stale or wrong. Fail closed: a missing or non-positive rate is an error,
never a guess.

Cost per million training tokens is reported when the total training
token count is known (dataset tokens x epochs): it lets you compare the
run against per-token API pricing.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass
class CostEstimate:
    wall_clock_hours: float     # training wall-clock on the given GPUs
    wall_clock_source: str      # how the hours were obtained (for the report)
    gpu_rate: float             # dollars per GPU-hour, user-supplied
    gpus: int
    total_cost: float           # hours x rate x gpus
    total_tokens: float | None  # dataset_tokens x epochs, when known

    @property
    def cost_per_million_tokens(self) -> float | None:
        if self.total_tokens:
            return self.total_cost / (self.total_tokens / 1e6)
        return None


def parse_gpu_rate(text: str) -> float:
    """argparse type for --gpu-rate: $/GPU-hour. Fail closed.

    Rejects anything that is not a finite positive number: missing,
    non-numeric, zero, negative, NaN and inf all raise, so argparse
    exits 2 instead of running with a made-up rate.
    """
    try:
        rate = float(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"invalid --gpu-rate {text!r}: give a number like 3.50")
    if not math.isfinite(rate):
        raise argparse.ArgumentTypeError(
            f"invalid --gpu-rate {text!r}: must be a finite number")
    if rate <= 0:
        raise argparse.ArgumentTypeError(
            f"invalid --gpu-rate {rate}: must be > 0 $/GPU-hour")
    return rate


def parse_wall_clock(text: str) -> float:
    """Parse --wall-clock into hours.

    Accepts decimal hours ("3.5") or clock time ("1:30:00", "90:00").
    Fail closed: empty, non-numeric, NaN/inf, negative, zero, or minutes
    and seconds outside [0, 60) all raise ValueError.
    """
    s = text.strip()
    if not s:
        raise ValueError("empty --wall-clock: give hours like 3.5 or H:MM:SS")
    if ":" in s:
        parts = s.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"bad --wall-clock {text!r}: use hours (3.5) or H:MM:SS")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            raise ValueError(
                f"bad --wall-clock {text!r}: use hours (3.5) or H:MM:SS")
        if any(n < 0 for n in nums) or not all(math.isfinite(n) for n in nums):
            raise ValueError(
                f"bad --wall-clock {text!r}: negative or non-finite")
        hours, minutes, seconds = (0.0, *nums) if len(nums) == 2 else nums
        if len(parts) == 3 and (minutes >= 60 or seconds >= 60):
            raise ValueError(
                f"bad --wall-clock {text!r}: minutes and seconds must be < 60")
        if len(parts) == 2 and seconds >= 60:
            raise ValueError(
                f"bad --wall-clock {text!r}: seconds must be < 60")
        value = hours + minutes / 60.0 + seconds / 3600.0
    else:
        try:
            value = float(s)
        except ValueError:
            raise ValueError(
                f"bad --wall-clock {text!r}: use hours (3.5) or H:MM:SS")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"bad --wall-clock {text!r}: duration must be a positive number "
            "of hours")
    return value


def estimate_cost(
    wall_clock_hours: float,
    gpu_rate: float,
    *,
    gpus: int = 1,
    total_tokens: float | None = None,
    wall_clock_source: str = "throughput estimate",
) -> CostEstimate:
    """Dollars for a run: hours x rate x GPUs. Everything validated.

    total_tokens (dataset tokens x epochs) enables the cost-per-million
    line; pass None when only the wall-clock override is known.
    """
    if not math.isfinite(wall_clock_hours) or wall_clock_hours <= 0:
        raise ValueError("wall_clock_hours must be a positive number of hours")
    if not math.isfinite(gpu_rate) or gpu_rate <= 0:
        raise ValueError("gpu_rate must be a positive $/GPU-hour")
    if not isinstance(gpus, int) or isinstance(gpus, bool) or gpus < 1:
        raise ValueError("gpus must be an integer >= 1")
    if total_tokens is not None and (
            not math.isfinite(total_tokens) or total_tokens < 0):
        raise ValueError("total_tokens must be >= 0 or None")

    return CostEstimate(
        wall_clock_hours=wall_clock_hours,
        wall_clock_source=wall_clock_source,
        gpu_rate=gpu_rate,
        gpus=gpus,
        total_cost=wall_clock_hours * gpu_rate * gpus,
        total_tokens=total_tokens,
    )
