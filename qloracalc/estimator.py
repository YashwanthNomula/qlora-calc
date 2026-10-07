"""The VRAM math behind QLoRA fine-tuning.

Every component is computed from the model architecture and the training
config. The formulas follow the standard accounting used by the QLoRA paper
and the bitsandbytes / PEFT / unsloth stacks:

* base weights: params x effective bits / 8. NF4 double quantization adds
  roughly 0.4 bits/param of quantization constants; 8-bit adds ~0.1.
* LoRA adapters: per targeted linear module, A (in x r) + B (r x out).
  "attention" targets q/k/v/o; "all-linear" also targets the MLP
  gate/up/down projections. Adapters train in fp32 by default.
* gradients: one fp32 copy of every trainable (adapter) parameter.
* optimizer states: AdamW keeps 2 fp32 states per trainable param;
  the 8-bit paged variant keeps 2 int8 states instead.
* activations: with gradient checkpointing only each layer's input is kept
  (L x b x s x h), plus one layer's full working set while it recomputes.
  Without checkpointing every layer's full working set is kept. Flash
  attention removes the s^2 attention-matrix term.
* logits: b x s x vocab in the train dtype (the cross-entropy input).
* CUDA overhead: ~1 GiB context plus ~5% fragmentation reserve.

This is an estimate, not a measurement: real runs vary with the framework,
allocator behavior, and eval/data-loader overhead. Treat the verdict as a
planning signal with a safety margin, not a guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qloracalc.models import ModelSpec

GIB = 1024 ** 3

# Per-layer activation working set, in units of (b x s x hidden) elements.
# Roughly: q/k/v/o projections + attention output + MLP gate/up/down +
# residuals + norms. Flash attention keeps the s^2 matrix out of this.
ACTIVATION_FACTOR = 30.0
# Extra per-layer s^2 term without flash attention: 5 x b x heads x s^2.
ATTN_S2_FACTOR = 5.0

DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}

VALID_BITS = (4, 8, 16)
VALID_TARGETS = ("attention", "all-linear")
VALID_OPTIMIZERS = ("adamw", "adamw-8bit")
VALID_DTYPES = ("bf16", "fp16")


@dataclass
class TrainConfig:
    model: ModelSpec
    quant_bits: int = 4
    rank: int = 16
    target_modules: str = "attention"  # or "all-linear"
    seq_len: int = 2048
    batch: int = 2
    grad_checkpoint: bool = True
    flash_attn: bool = True
    optimizer: str = "adamw-8bit"      # or "adamw"
    train_dtype: str = "bf16"          # or "fp16"

    def __post_init__(self):
        if self.quant_bits not in VALID_BITS:
            raise ValueError(f"quant_bits must be one of {VALID_BITS}")
        if self.target_modules not in VALID_TARGETS:
            raise ValueError(f"target_modules must be one of {VALID_TARGETS}")
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {VALID_OPTIMIZERS}")
        if self.train_dtype not in VALID_DTYPES:
            raise ValueError(f"train_dtype must be one of {VALID_DTYPES}")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if self.seq_len < 1 or self.batch < 1:
            raise ValueError("seq_len and batch must be >= 1")


@dataclass
class Estimate:
    config: TrainConfig
    components: list[tuple[str, float]] = field(default_factory=list)  # (label, bytes)
    lora_params: int = 0
    total_bytes: float = 0.0

    @property
    def total_gb(self) -> float:
        return self.total_bytes / GIB

    def gb(self, label: str) -> float:
        for name, nbytes in self.components:
            if name == label:
                return nbytes / GIB
        raise KeyError(label)


def effective_bits(quant_bits: int) -> float:
    """Bits per param including quantization-constant overhead."""
    if quant_bits == 4:
        return 4.4   # NF4 + double quantization constants
    if quant_bits == 8:
        return 8.1   # LLM.int8() block-wise constants
    return 16.0


def lora_param_count(model: ModelSpec, rank: int, target_modules: str) -> int:
    """Trainable adapter parameters across all layers."""
    h, f = model.hidden, model.ffn
    # q, k, v, o: each contributes A (h x r) + B (r x h) = 2*r*h
    attn = 4 * 2 * rank * h
    if target_modules == "attention":
        per_layer = attn
    else:
        # gate, up: r*(h+f) each; down: r*(f+h)
        mlp = 3 * rank * (h + f)
        per_layer = attn + mlp
    return per_layer * model.layers


def estimate(cfg: TrainConfig) -> Estimate:
    m = cfg.model
    dt_bytes = DTYPE_BYTES[cfg.train_dtype]
    b, s, h = cfg.batch, cfg.seq_len, m.hidden

    # 1. Frozen base model weights, quantized.
    weights = m.params * effective_bits(cfg.quant_bits) / 8.0

    # 2/3/4. Trainable side: adapters (fp32), gradients (fp32), optimizer.
    lora_params = lora_param_count(m, cfg.rank, cfg.target_modules)
    adapters = lora_params * 4.0
    grads = lora_params * 4.0
    optim_bytes_per_param = 8.0 if cfg.optimizer == "adamw" else 2.0
    optimizer = lora_params * optim_bytes_per_param

    # 5. Activations.
    bsh = b * s * h
    per_layer = ACTIVATION_FACTOR * bsh * dt_bytes
    if not cfg.flash_attn:
        per_layer += ATTN_S2_FACTOR * b * m.heads * s * s * dt_bytes
    if cfg.grad_checkpoint:
        # keep each layer's input; one layer's working set recomputes at a time
        activations = m.layers * bsh * dt_bytes + per_layer
    else:
        activations = m.layers * per_layer

    # 6. Logits for the cross-entropy loss.
    logits = b * s * m.vocab * dt_bytes

    subtotal = weights + adapters + grads + optimizer + activations + logits
    overhead = 1.0 * GIB + 0.05 * subtotal
    total = subtotal + overhead

    comp = [
        (f"base weights ({cfg.quant_bits}-bit)", weights),
        (f"LoRA adapters ({lora_params / 1e6:.1f}M params)", adapters),
        ("gradients (fp32)", grads),
        (f"optimizer states ({cfg.optimizer})", optimizer),
        ("activations" + (" (grad ckpt)" if cfg.grad_checkpoint else ""), activations),
        ("output logits", logits),
        ("CUDA overhead", overhead),
    ]
    return Estimate(config=cfg, components=comp,
                    lora_params=lora_params, total_bytes=total)


def fits(est: Estimate, gpu_gb: float) -> bool:
    return est.total_gb <= gpu_gb


def max_batch(cfg: TrainConfig, gpu_gb: float, cap: int = 1024) -> int:
    """Largest micro-batch that fits in gpu_gb. Returns 0 if batch 1 overflows."""
    from dataclasses import replace
    if estimate(replace(cfg, batch=1)).total_gb > gpu_gb:
        return 0
    lo, hi = 1, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate(replace(cfg, batch=mid)).total_gb <= gpu_gb:
            lo = mid
        else:
            hi = mid - 1
    return lo


def suggest(cfg: TrainConfig, est: Estimate, gpu_gb: float) -> list[str]:
    """Actionable tips ordered by typical impact."""
    tips: list[str] = []
    ok = fits(est, gpu_gb)
    headroom = gpu_gb - est.total_gb
    if not ok or headroom < 2.0:
        mb = max_batch(cfg, gpu_gb)
        if mb < cfg.batch:
            tips.append(
                f"drop micro-batch to {max(mb, 1)} "
                f"(largest that fits this GPU)"
                if mb >= 1 else "even batch 1 overflows: shrink the model or the GPU budget is too small"
            )
        if not cfg.grad_checkpoint:
            tips.append("enable gradient checkpointing (default): it trades ~20% speed for most of the activation memory")
        if cfg.quant_bits > 4:
            tips.append("use 4-bit NF4 quantization for the base model: the standard QLoRA recipe")
        if cfg.optimizer == "adamw":
            tips.append("switch to paged adamw-8bit: identical training, 4x smaller optimizer states")
        if cfg.target_modules == "all-linear":
            tips.append("target attention-only LoRA modules (q/k/v/o) instead of all-linear: fewer trainable params")
        if cfg.seq_len > 1024:
            tips.append(f"shorten sequence length below {cfg.seq_len}: activations and logits scale with it")
    elif headroom > 6.0:
        mb = max_batch(cfg, gpu_gb)
        if mb > cfg.batch:
            tips.append(f"you have {headroom:.1f} GB headroom: batch {mb} would still fit")
    return tips


def describe_config(cfg: TrainConfig) -> str:
    ckpt = "on" if cfg.grad_checkpoint else "off"
    flash = "on" if cfg.flash_attn else "off"
    return (
        f"{cfg.quant_bits}-bit, rank {cfg.rank} ({cfg.target_modules}), "
        f"seq {cfg.seq_len}, batch {cfg.batch}, grad-ckpt {ckpt}, "
        f"flash-attn {flash}, {cfg.optimizer}, {cfg.train_dtype}"
    )
