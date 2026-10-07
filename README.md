# qlora-calc

**Will this fine-tune fit on my GPU?** A QLoRA VRAM estimator that answers
that question in under a second, before you burn an hour launching a training
run that OOMs at step 3.

Give it a model, a LoRA rank, a sequence length and a batch size. It breaks
down exactly where the VRAM goes (quantized weights, adapters, gradients,
optimizer states, activations, logits, CUDA overhead) and tells you whether
the run fits on your card, plus the largest batch that would.

Pure Python. Zero dependencies.

## Install

```bash
pip install .
# or just run it from the repo:
python -m qloracalc demo
```

## Demo (real output)

```bash
$ qcalc demo
demo 1: the classic QLoRA recipe (llama-3-8b, 4-bit, r=16) on a 24GB card

QLoRA VRAM estimate: llama-3-8b (8.03B params)
  config: 4-bit, rank 16 (attention), seq 2048, batch 2, grad-ckpt on, flash-attn on, adamw-8bit, bf16

  component                               GB  share
  base weights (4-bit)                  4.11  [#############---------------]
  LoRA adapters (16.8M params)          0.06  [----------------------------]
  gradients (fp32)                      0.06  [----------------------------]
  optimizer states (adamw-8bit)         0.03  [----------------------------]
  activations (grad ckpt)               1.94  [######----------------------]
  output logits                         0.98  [###-------------------------]
  CUDA overhead                         1.36  [####------------------------]
  ------------------------------------------
  total                                 8.54

  GPU budget: 24 GB (rtx4090)
  verdict: FITS - 15.5 GB headroom

  tips:
    - you have 15.5 GB headroom: batch 12 would still fit

demo 2: same recipe on llama-3-70b

QLoRA VRAM estimate: llama-3-70b (70.6B params)
  ...
  total                                43.91

  GPU budget: 24 GB (rtx4090)
  verdict: EXCEEDS by 19.9 GB

  tips:
    - even batch 1 overflows: shrink the model or the GPU budget is too small
    - shorten sequence length below 2048: activations and logits scale with it
```

The 70B number is the well known QLoRA result: a 70B model in 4-bit needs a
~48GB card, not a 24GB one.

## Usage

```bash
# estimate one config
qcalc estimate --model llama-3-8b --bits 4 --rank 16 --seq-len 2048 --batch 2 --gpu rtx4090

# largest micro-batch that fits
qcalc fit --model qwen2.5-7b --rank 32 --seq-len 4096 --gpu 24

# compare 4/8/16-bit quantization for the same run
qcalc compare --model mistral-7b --rank 16 --batch 2 --gpu 24

# list preset models and known GPUs
qcalc models

# machine-readable output for scripts and CI
qcalc estimate --model llama-3-8b --gpu 24 --format json

# fail CI when the config would OOM (exit 1)
qcalc estimate --model llama-3-70b --gpu 24 --strict

# load a config from a file (see examples/qlora-config.json); CLI flags override it
qcalc estimate --config examples/qlora-config.json

# custom architecture instead of a preset
qcalc estimate --params-b 13 --layers 40 --hidden 5120 --ffn 13824 --gpu 40
```

## How the math works

| Component | Formula |
|---|---|
| Base weights | params x effective bits / 8 (4-bit NF4 counts as 4.4 with double-quant constants, 8-bit as 8.1) |
| LoRA adapters | per layer: q/k/v/o (or all-linear incl. MLP): A(in x r) + B(r x out), fp32 |
| Gradients | one fp32 copy of every trainable (adapter) parameter |
| Optimizer | AdamW: 2 fp32 states/param; paged AdamW 8-bit: 2 int8 states/param |
| Activations | grad checkpointing: L x b x s x h kept + one layer's working set; without it: every layer's full working set. Flash attention drops the s^2 term |
| Logits | b x s x vocab in the train dtype (the cross-entropy input; large for 128k vocabs) |
| CUDA overhead | ~1 GB context + 5% fragmentation reserve |

## Caveats

This is an **estimate**, not a measurement. Real runs vary with the
framework (transformers vs unsloth), the allocator, eval overhead and the
data loader. Treat the verdict as a planning signal and keep a safety
margin. MoE models (mixtral-8x7b) are estimated by total params, which
overstates dense-equivalent training memory.

## Tests

```bash
python -m pytest tests/ -q   # 35 tests
```

## License

MIT
