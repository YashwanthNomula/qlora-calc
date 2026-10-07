"""Known model architectures and GPU memory sizes.

Parameter counts and shapes are the public architecture figures for each
model family. They feed the estimator; nothing here is measured at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float      # billions of parameters
    layers: int          # transformer blocks
    hidden: int          # hidden size
    ffn: int             # MLP intermediate size
    heads: int           # query attention heads
    kv_heads: int        # key/value heads (GQA)
    vocab: int           # vocabulary size

    @property
    def params(self) -> float:
        return self.params_b * 1e9


PRESETS: dict[str, ModelSpec] = {
    "llama-3.2-1b": ModelSpec("llama-3.2-1b", 1.24, 16, 2048, 8192, 32, 8, 128256),
    "llama-3.2-3b": ModelSpec("llama-3.2-3b", 3.21, 28, 3072, 8192, 24, 8, 128256),
    "llama-3-8b": ModelSpec("llama-3-8b", 8.03, 32, 4096, 14336, 32, 8, 128256),
    "llama-3-70b": ModelSpec("llama-3-70b", 70.6, 80, 8192, 28672, 64, 8, 128256),
    "mistral-7b": ModelSpec("mistral-7b", 7.25, 32, 4096, 14336, 32, 8, 32000),
    "mixtral-8x7b": ModelSpec("mixtral-8x7b", 46.7, 32, 4096, 14336, 32, 8, 32000),
    "qwen2.5-7b": ModelSpec("qwen2.5-7b", 7.61, 28, 3584, 18944, 28, 4, 152064),
    "qwen2.5-14b": ModelSpec("qwen2.5-14b", 14.7, 48, 5120, 13824, 40, 8, 152064),
    "qwen2.5-32b": ModelSpec("qwen2.5-32b", 32.5, 64, 5120, 27648, 40, 8, 152064),
    "gemma-2-9b": ModelSpec("gemma-2-9b", 9.24, 42, 3584, 28672, 16, 8, 256000),
    "phi-3-mini": ModelSpec("phi-3-mini", 3.82, 32, 3072, 9216, 32, 32, 32064),
}

# Common GPUs by marketing name -> usable VRAM in GiB.
GPU_SIZES: dict[str, float] = {
    "rtx4060ti": 16.0,
    "rtx3090": 24.0,
    "rtx4090": 24.0,
    "a10": 24.0,
    "l4": 24.0,
    "a100-40": 40.0,
    "a100": 40.0,
    "a6000": 48.0,
    "l40s": 48.0,
    "a100-80": 80.0,
    "h100": 80.0,
    "h200": 141.0,
    "b200": 192.0,
}


def parse_size(text: str) -> float:
    """Parse '7b', '8.03B', '800m' into billions of parameters."""
    t = text.strip().lower().replace("_", "")
    for suffix, mult in (("b", 1.0), ("m", 1e-3), ("k", 1e-6)):
        if t.endswith(suffix):
            return float(t[: -len(suffix)]) * mult
    return float(t)


def parse_gpu(text: str) -> tuple[float, str]:
    """Parse '24', '24gb', 'rtx4090' into (GiB, display label)."""
    t = text.strip().lower().replace(" ", "")
    if t in GPU_SIZES:
        return GPU_SIZES[t], t
    t = t.removesuffix("gb").removesuffix("gib")
    try:
        return float(t), f"{float(t):g}GB"
    except ValueError:
        raise ValueError(
            f"unknown GPU '{text}': use a size like 24 or a name like "
            + ", ".join(sorted(GPU_SIZES))
        )
