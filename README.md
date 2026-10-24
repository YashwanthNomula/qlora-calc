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

# plan across N GPUs: data-parallel vs tensor-parallel per-GPU VRAM
qcalc estimate --model llama-3-70b --batch 1 --gpus 2 --gpu rtx4090

# training speed + wall-clock: 10M tokens/epoch, 3 epochs, RTX 4090
qcalc throughput --model llama-3-8b --gpu rtx4090 --dataset-tokens 10000000 --epochs 3
# same, from dataset size: 5,000 examples x ~2,000 tokens each, 2 GPUs
qcalc throughput --model llama-3-8b --tflops 165 --examples 5000 --avg-len 2000 --gpus 2
# rates only (no wall-clock) with an explicit MFU
qcalc throughput --model llama-3-8b --tflops 312 --mfu 0.5

# dollar cost of the run: wall-clock from the throughput inputs above
qcalc cost --model llama-3-8b --tflops 165 --dataset-tokens 10000000 --epochs 3 --gpu-rate 3.50
# same, but you already know the run takes 90 minutes on 2 GPUs
qcalc cost --model llama-3-8b --wall-clock 1:30:00 --gpus 2 --gpu-rate 2.49
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

## Multi-GPU sharding

`qcalc estimate --gpus N` plans the same config across N identical GPUs and
compares two sharding strategies, with a per-GPU verdict for each:

| Strategy | Per-GPU VRAM |
|---|---|
| data-parallel | base weights, adapters and gradients replicated; optimizer states and micro-batch sharded /N (ZeRO-1 style) |
| tensor-parallel | weights, adapters, gradients, optimizer states, activations and logits all sharded /N, plus a ~2% NCCL/collective buffer |

qcalc recommends the strategy that fits with the most per-GPU headroom.
With `--format json` the full per-strategy plan (components, fits,
headroom) is machine-readable; with `--strict`, exit 1 means neither
strategy fits the cards.

## Training throughput

`qcalc throughput` answers the second planning question: how long will
the run take? It uses the standard 6N FLOPs/token accounting (forward 2N
+ backward 4N), rising to 8N with gradient checkpointing (the forward
pass is recomputed), scaled by model FLOPs utilization (MFU, default
0.40) against the GPU's peak BF16/FP16 tensor-core throughput.

```bash
$ qcalc throughput --model llama-3-8b --gpu rtx4090 --dataset-tokens 10000000 --epochs 3
training throughput: llama-3-8b (8.03B params)
  config: 4-bit, rank 16 (attention), seq 2048, batch 2, grad-ckpt on, flash-attn on, adamw-8bit, bf16
  GPU: rtx4090 @ 165 peak TFLOPS (BF16/FP16 tensor), MFU 0.40 -> 66.0 effective TFLOPS
  compute: 8 FLOPs/token (with grad checkpointing)

  tokens/sec:     1,027
  steps/sec:      0.2508 (step = batch 2 x seq 2048 = 4,096 tokens)
  sec/step:       3.99

  dataset:        10,000,000 tokens/epoch x 3 epoch(s)
  total steps:    7,324
  wall-clock:     8h 06m (8.11 h)
```

`--tflops` takes an explicit peak figure (or a known card name) and
overrides the `--gpu` lookup; `--mfu` tunes the utilization assumption;
`--gpus N` scales near-linearly with a `--scaling-eff` discount (default
0.90) for communication. Treat the wall-clock as a planning estimate
with a +/-2x band: data loading, eval, checkpointing and optimizer
overhead are not modelled.

## Run cost

`qcalc cost` turns the wall-clock into dollars:
`total = hours x $/GPU-hour x GPUs`.

```bash
$ qcalc cost --model llama-3-8b --tflops 165 --dataset-tokens 10000000 --epochs 3 --gpu-rate 3.50
run cost estimate
  wall-clock:     8h 06m (8.11 h) - throughput estimate: 165 TFLOPS @ 165 TFLOPS, MFU 0.40
  per-GPU rate:   $3.50/GPU-hour (you supplied it)
  GPUs:           1

  total cost:     $28.39
  $/M tokens:     $0.95 (over 30,000,000 training tokens)

  qcalc ships no price table: cloud rates move too fast and vary by
  region, commitment and spot vs on-demand. Re-run with your own --gpu-rate
  to compare on-demand vs spot vs reserved quotes.
```

`--gpu-rate` is **required** and never guessed: cloud prices move too
fast for a built-in table, so a missing, zero or negative rate is an
error. The wall-clock hours come from the same throughput inputs as
`qcalc throughput`, or from a direct `--wall-clock` override in decimal
hours (`3.5`) or `H:MM:SS` (`1:30:00`) when you already know how long the
run takes. When the training token count is known you also get a
`$/M tokens` line for comparing the run against per-token API pricing.
`--format json` gives the same numbers machine-readable.

## Caveats

This is an **estimate**, not a measurement. Real runs vary with the
framework (transformers vs unsloth), the allocator, eval overhead and the
data loader. Treat the verdict as a planning signal and keep a safety
margin. MoE models (mixtral-8x7b) are estimated by total params, which
overstates dense-equivalent training memory.

## Tests

```bash
python -m pytest tests/ -q   # 87 tests
```

## License

MIT
