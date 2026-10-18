"""Command-line interface for qlora-calc."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

from qloracalc import __version__
from qloracalc import estimator as E
from qloracalc import report as R
from qloracalc import throughput as T
from qloracalc.models import (
    GPU_SIZES,
    GPU_TFLOPS,
    PRESETS,
    ModelSpec,
    parse_gpu,
    parse_size,
    parse_tflops,
)


def _model_from_args(args) -> ModelSpec:
    if args.model:
        key = args.model.strip().lower()
        if key not in PRESETS:
            print(f"error: unknown model '{args.model}'. "
                  f"See `qcalc models`.", file=sys.stderr)
            sys.exit(2)
        return PRESETS[key]
    if args.params_b is None:
        print("error: give --model <preset> or --params-b <size> "
              "(plus --layers/--hidden/--ffn for a custom model)",
              file=sys.stderr)
        sys.exit(2)
    return ModelSpec(
        name=args.name or f"custom-{args.params_b}",
        params_b=parse_size(args.params_b),
        layers=args.layers,
        hidden=args.hidden,
        ffn=args.ffn,
        heads=args.heads,
        kv_heads=args.kv_heads,
        vocab=args.vocab,
    )


def _config_from_args(args) -> E.TrainConfig:
    model = _model_from_args(args)
    return E.TrainConfig(
        model=model,
        quant_bits=args.bits,
        rank=args.rank,
        target_modules=args.targets,
        seq_len=args.seq_len,
        batch=args.batch,
        grad_checkpoint=not args.no_grad_checkpoint,
        flash_attn=not args.no_flash_attn,
        optimizer=args.optimizer,
        train_dtype=args.dtype,
    )


def _load_config_file(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read config {path}: {exc}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(data, dict):
        print(f"error: config {path} must be a JSON object", file=sys.stderr)
        sys.exit(2)
    return data


def _apply_config_file(args, data: dict):
    """Fill unset CLI options from a JSON config file."""
    mapping = {
        "model": "model", "params_b": "params_b", "name": "name",
        "layers": "layers", "hidden": "hidden", "ffn": "ffn",
        "heads": "heads", "kv_heads": "kv_heads", "vocab": "vocab",
        "quant_bits": "bits", "rank": "rank",
        "target_modules": "targets", "seq_len": "seq_len", "batch": "batch",
        "optimizer": "optimizer", "train_dtype": "dtype", "gpu": "gpu",
        "gpus": "gpus",
        "tflops": "tflops", "mfu": "mfu",
        "dataset_tokens": "dataset_tokens", "examples": "examples",
        "avg_len": "avg_len", "epochs": "epochs",
        "scaling_eff": "scaling_eff",
    }
    for key, attr in mapping.items():
        if key in data and getattr(args, attr, None) is None:
            setattr(args, attr, data[key])
    if data.get("grad_checkpoint") is False and not args.no_grad_checkpoint:
        args.no_grad_checkpoint = True
    if data.get("flash_attn") is False and not args.no_flash_attn:
        args.no_flash_attn = True


def _add_train_args(p):
    # Defaults are None so a --config file can fill them; _fill_defaults
    # applies the real defaults afterwards. Explicit CLI flags always win.
    p.add_argument("--model", default=None,
                   help="preset model (see `qcalc models`)")
    p.add_argument("--params-b", default=None,
                   help="custom model size, e.g. 7b, 8.03B, 800m")
    p.add_argument("--name", default=None, help="display name for custom model")
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--ffn", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--vocab", type=int, default=None)
    p.add_argument("--bits", type=int, choices=[4, 8, 16], default=None,
                   help="base model quantization (default: 4)")
    p.add_argument("--rank", type=int, default=None,
                   help="LoRA rank (default: 16)")
    p.add_argument("--targets", choices=["attention", "all-linear"],
                   default=None,
                   help="LoRA target modules (default: attention)")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch", type=int, default=None,
                   help="micro-batch size (default: 2)")
    p.add_argument("--no-grad-checkpoint", action="store_true",
                   help="disable gradient checkpointing")
    p.add_argument("--no-flash-attn", action="store_true",
                   help="disable flash attention (adds the s^2 term)")
    p.add_argument("--optimizer", choices=["adamw", "adamw-8bit"],
                   default=None)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default=None)
    p.add_argument("--gpu", default=None,
                   help="GPU VRAM: GB like 24, or a name like rtx4090 (default: 24)")
    p.add_argument("--config", default=None,
                   help="JSON file with any of the above options")
    p.add_argument("--format", choices=["text", "json"], default="text")


_CLI_DEFAULTS = {
    "bits": 4, "rank": 16, "targets": "attention", "seq_len": 2048,
    "batch": 2, "optimizer": "adamw-8bit", "dtype": "bf16",
    "layers": 32, "hidden": 4096, "ffn": 14336, "heads": 32,
    "kv_heads": 8, "vocab": 32000, "gpu": "24",
}


def _fill_defaults(args):
    for attr, value in _CLI_DEFAULTS.items():
        if getattr(args, attr, None) is None:
            setattr(args, attr, value)


def _prepare(args):
    """Merge --config file, then fill defaults. Returns (cfg, gpu_gb, label)."""
    if args.config:
        _apply_config_file(args, _load_config_file(args.config))
    _fill_defaults(args)
    cfg = _config_from_args(args)
    gpu_gb, gpu_label = parse_gpu(args.gpu)
    return cfg, gpu_gb, gpu_label


def cmd_estimate(args):
    cfg, gpu_gb, gpu_label = _prepare(args)
    gpus = 1 if args.gpus is None else args.gpus
    if gpus < 1:
        raise ValueError("--gpus must be >= 1")
    if gpus > 1:
        return _cmd_estimate_multi(args, cfg, gpu_gb, gpu_label, gpus)
    est = E.estimate(cfg)
    if args.format == "json":
        print(R.render_estimate_json(est, gpu_gb, gpu_label))
    else:
        print(R.render_estimate(est, gpu_gb, gpu_label))
    if args.strict and not E.fits(est, gpu_gb):
        return 1
    return 0


def _cmd_estimate_multi(args, cfg, gpu_gb, gpu_label, gpus):
    shards = E.shard_plan(cfg, gpus)
    rec = E.recommend_strategy(shards, gpu_gb)
    if args.format == "json":
        print(R.render_shard_plan_json(shards, cfg, gpu_gb, gpu_label, rec))
    else:
        print(R.render_shard_plan(shards, cfg, gpu_gb, gpu_label, rec))
    if args.strict and rec[0] is None:
        return 1
    return 0


def cmd_fit(args):
    cfg, gpu_gb, gpu_label = _prepare(args)
    mb = E.max_batch(cfg, gpu_gb)
    base = E.estimate(replace(cfg, batch=1))
    if args.format == "json":
        print(json.dumps({
            "model": cfg.model.name,
            "gpu_gb": gpu_gb,
            "gpu_label": gpu_label,
            "batch_1_gb": base.total_gb,
            "max_batch": mb,
            "at_requested_batch_gb": E.estimate(cfg).total_gb,
        }, indent=2))
    else:
        print(f"max micro-batch for {cfg.model.name} on "
              f"{gpu_gb:g} GB ({gpu_label}): {mb}")
        print(f"  batch 1 uses {base.total_gb:.2f} GB; "
              f"requested batch {cfg.batch} uses "
              f"{E.estimate(cfg).total_gb:.2f} GB")
        if mb == 0:
            print("  even batch 1 overflows: shrink the model or pick a bigger GPU")
    return 0


def cmd_compare(args):
    base_cfg, gpu_gb, gpu_label = _prepare(args)
    rows = [(bits, E.estimate(replace(base_cfg, quant_bits=bits)))
            for bits in (4, 8, 16)]
    if args.format == "json":
        print(R.render_compare_json(rows, gpu_gb, gpu_label))
    else:
        print(f"{base_cfg.model.name}: {E.describe_config(base_cfg)}")
        print(R.render_compare(rows, gpu_gb, gpu_label))
    return 0


def _resolve_tflops(args) -> tuple[float, str]:
    """--tflops wins; else look up --gpu in the TFLOPS table."""
    if args.tflops is not None:
        return parse_tflops(str(args.tflops))
    gpu_key = (args.gpu or "").strip().lower()
    if gpu_key in GPU_TFLOPS:
        return GPU_TFLOPS[gpu_key], gpu_key
    raise ValueError(
        "throughput needs a GPU throughput figure: pass --tflops N "
        "(e.g. --tflops 165) or --gpu <name> with a known card "
        f"({', '.join(sorted(GPU_TFLOPS))})"
    )


def cmd_throughput(args):
    cfg, _gpu_gb, _gpu_label = _prepare(args)
    tflops, tflops_label = _resolve_tflops(args)
    if args.dataset_tokens is not None and (
            args.examples is not None or args.avg_len is not None):
        raise ValueError(
            "pass --dataset-tokens OR --examples + --avg-len, not both")
    if (args.examples is None) != (args.avg_len is None):
        raise ValueError("--examples and --avg-len must be given together")
    dataset_tokens = args.dataset_tokens
    if dataset_tokens is None and args.examples is not None:
        if args.examples < 0 or args.avg_len < 0:
            raise ValueError("--examples and --avg-len must be >= 0")
        dataset_tokens = float(args.examples) * float(args.avg_len)
    est = T.estimate_throughput(
        cfg, tflops, mfu=args.mfu, gpus=args.gpus,
        scaling_eff=args.scaling_eff,
        dataset_tokens=dataset_tokens, epochs=args.epochs)
    est.gpu_label = tflops_label
    if args.format == "json":
        print(R.render_throughput_json(est))
    else:
        print(R.render_throughput(est))
    return 0


def cmd_models(_args):
    print("preset models:")
    print(R.render_models(PRESETS.values()))
    print("known GPUs: " + ", ".join(
        f"{k} ({v:g}GB)" for k, v in sorted(GPU_SIZES.items())))
    return 0


def cmd_demo(_args):
    model = PRESETS["llama-3-8b"]
    print("demo 1: the classic QLoRA recipe (llama-3-8b, 4-bit, r=16) on a 24GB card")
    print()
    cfg = E.TrainConfig(model=model, quant_bits=4, rank=16, seq_len=2048, batch=2)
    print(R.render_estimate(E.estimate(cfg), 24.0, "rtx4090"))
    print()
    print("demo 2: same recipe on llama-3-70b")
    print()
    big = E.TrainConfig(model=PRESETS["llama-3-70b"], quant_bits=4, rank=16,
                        seq_len=2048, batch=1)
    print(R.render_estimate(E.estimate(big), 24.0, "rtx4090"))
    print()
    print("demo 3: what is the biggest batch for the 8B run on 24GB?")
    print()
    print(f"  max micro-batch: {E.max_batch(cfg, 24.0)}")
    print()
    print("demo 4: how long does one epoch over 10M tokens take on an RTX 4090?")
    print()
    import qloracalc.throughput as T
    test = T.estimate_throughput(cfg, 165.0, mfu=0.40,
                                 dataset_tokens=10_000_000, epochs=1)
    test.gpu_label = "rtx4090"
    print(R.render_throughput(test))
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="qcalc",
        description="QLoRA VRAM estimator: will this fine-tune fit on your GPU?",
    )
    p.add_argument("--version", action="version",
                   version=f"qcalc {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("estimate", help="estimate VRAM for one training config")
    _add_train_args(pe)
    pe.add_argument("--strict", action="store_true",
                    help="exit 1 when the config exceeds the GPU budget")
    pe.add_argument("--gpus", type=int, default=None,
                    help="plan sharding across N GPUs: compares data-parallel "
                         "vs tensor-parallel per-GPU VRAM (default: 1)")
    pe.set_defaults(func=cmd_estimate)

    pf = sub.add_parser("fit", help="find the largest micro-batch that fits")
    _add_train_args(pf)
    pf.set_defaults(func=cmd_fit)

    pc = sub.add_parser("compare",
                        help="compare 4/8/16-bit quantization for one config")
    _add_train_args(pc)
    pc.set_defaults(func=cmd_compare)

    pt = sub.add_parser("throughput",
                        help="estimate training speed and wall-clock time")
    _add_train_args(pt)
    pt.add_argument("--tflops", default=None,
                    help="GPU peak BF16/FP16 tensor TFLOPS: a number like 165, "
                         "or a known card name; overrides --gpu lookup")
    pt.add_argument("--mfu", type=float, default=0.40,
                    help="model FLOPs utilization 0-1 (default: 0.40)")
    pt.add_argument("--dataset-tokens", type=float, default=None,
                    help="training tokens per epoch (for wall-clock)")
    pt.add_argument("--examples", type=float, default=None,
                    help="dataset size in examples (needs --avg-len)")
    pt.add_argument("--avg-len", type=float, default=None,
                    help="average tokens per example (needs --examples)")
    pt.add_argument("--epochs", type=int, default=1)
    pt.add_argument("--gpus", type=int, default=1,
                    help="number of GPUs (default: 1)")
    pt.add_argument("--scaling-eff", type=float, default=0.90,
                    help="per-GPU efficiency when gpus > 1 (default: 0.90)")
    pt.set_defaults(func=cmd_throughput)

    pm = sub.add_parser("models", help="list preset models and known GPUs")
    pm.set_defaults(func=cmd_models)

    pd = sub.add_parser("demo", help="run the canned demo")
    pd.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
