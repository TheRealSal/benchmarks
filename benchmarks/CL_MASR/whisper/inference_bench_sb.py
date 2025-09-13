
#!/usr/bin/env python3
"""
SpeechBrain Whisper Adapter Inference Benchmark

Run:
  python inference_bench_sb.py \
    --yaml AB_Bn.yaml \
    --device cuda \
    --precision bf16 \
    --iters 30 --warmup 10 \
    --batch-sizes 1 2 4 \
    --seconds 5 10 20 \
    --out results.json

What it does:
- Loads your YAML with hyperpyyaml (same as training)
- Instantiates the Whisper model from hparams["modules"]["whisper"]
- Inserts adapters exactly per hparams["adapter_config"]
- Synthesizes dummy audio + BOS tokens (like the training forward)
- Times:
    A) encode+decode forward (teacher-forced), returning logits
    B) optional generation from encoder features (toggle with --gen)

Outputs:
- Prints latency stats per (batch_size, seconds)
- Writes a JSON file with raw timings and summaries
"""

import argparse, time, json, importlib, os, statistics
from pathlib import Path

import torch
torch.set_grad_enabled(False)

from hyperpyyaml import load_hyperpyyaml

# Utilities to import adapter class string like "speechbrain.nnet.adapters.HoulsbyAdapterLinear"
def get_class_from_str(path: str):
    mod, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def _infer_insertion(adapter_cfg) -> str:
    """Infer adapter insertion location: 'encoder', 'decoder', or 'enc_dec'."""
    if not adapter_cfg:
        return "none"
    tl = adapter_cfg.get("target_layers", [])
    # Support both strings and dict/list structures; flatten to strings
    parts = []
    def _collect(x):
        if isinstance(x, str):
            parts.append(x.lower())
        elif isinstance(x, dict):
            for v in x.values():
                _collect(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                _collect(v)
    _collect(tl)
    has_enc = any("enc" in p or "encoder" in p for p in parts)
    has_dec = any("dec" in p or "decoder" in p for p in parts)
    if has_enc and has_dec: return "enc_dec"
    if has_enc: return "encoder"
    if has_dec: return "decoder"
    return "unknown"

def _dtype_from_str(s):
    s = s.lower()
    if s in ("fp32", "float32"): return torch.float32
    if s in ("fp16", "float16"): return torch.float16
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    raise ValueError(f"Unknown precision '{s}' (choose fp32|fp16|bf16).")

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def summarize(times_ms):
    p50 = statistics.median(times_ms)
    p90 = statistics.quantiles(times_ms, n=10)[8] if len(times_ms) >= 10 else max(times_ms)
    mean = statistics.fmean(times_ms)
    stdev = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    return {"mean_ms": mean, "p50_ms": p50, "p90_ms": p90, "stdev_ms": stdev}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", type=str, default="fp16", help="fp32|fp16|bf16")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1,2,4])
    ap.add_argument("--seconds", type=float, nargs="+", default=[5.0,10.0,20.0])
    ap.add_argument("--gen", action="store_true", help="also time generate() after encoding")
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--scratch_folder", action="store_true", default="")
    ap.add_argument("--data_folder", action="store_true", default="")
    args = ap.parse_args()

    # ===== Load YAML (same as training) =====
    with open(args.yaml, "r") as f:
        hparams = load_hyperpyyaml(f)

    device = torch.device(args.device)
    dtype = _dtype_from_str(args.precision)

    # ===== Build Whisper from YAML modules =====
    # Training file makes it available as hparams["modules"]["whisper"]
    whisper = hparams["modules"]["whisper"]
    whisper = whisper.to(device)
    try:
        # Cast model to requested dtype when possible
        whisper = whisper.to(dtype=dtype)
    except Exception:
        pass
    whisper.eval()

    # ===== Insert Adapters per YAML =====
    adapter_cfg = hparams.get("adapter_config", None)
    insertion = _infer_insertion(adapter_cfg)
    if adapter_cfg:
        adapter_cls = get_class_from_str(adapter_cfg["adapter_class"])
        AdaptedModel = get_class_from_str("speechbrain.nnet.adapters.AdaptedModel")
        AdaptedModel(
            model_to_adapt=whisper,
            adapter_class=adapter_cls,
            target_layers=adapter_cfg["target_layers"],
            adapter_kwargs=adapter_cfg.get("adapter_kwargs", {}),
            adapter_name=adapter_cfg.get("adapter_name", "default"),
        )
    # Precision knobs for speed
    if dtype in (torch.float16, torch.bfloat16):
        autocast_dtype = dtype
    else:
        autocast_dtype = None

    sample_rate = int(hparams.get("sample_rate", 16000))
    max_len = int(getattr(whisper.model.config, "max_length", 448))
    tokenizer = whisper.tokenizer  # used only to find BOS/PAD ids
    bos_id = getattr(tokenizer, "bos_token_id", 1)
    pad_id = getattr(tokenizer, "pad_token_id", 0)

    @torch.inference_mode()
    def make_inputs(batch: int, seconds: float):
        T = int(seconds * sample_rate)
        wavs = torch.randn(batch, T, device=device, dtype=torch.float32)
        # teacher-forced tokens for forward()
        bos_tokens = torch.full((batch, max_len), fill_value=pad_id, device=device, dtype=torch.long)
        bos_tokens[:, 0] = bos_id
        return wavs, bos_tokens

    @torch.inference_mode()
    def forward_once(wavs, bos_tokens):
        if autocast_dtype is not None and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                enc_out, logits, _ = whisper(wavs, bos_tokens)
        else:
            enc_out, logits, _ = whisper(wavs, bos_tokens)
        return enc_out, logits

    @torch.inference_mode()
    def generate_once(enc_out):
        # Mirrors training code's generate() usage
        hyps, _ = whisper.generate(
            audio_features=enc_out,
            forced_decoder_locale=hparams.get("forced_decoder_locale", None),
            max_gen_tokens=hparams.get("max_gen_tokens", 80),
        )
        return hyps

    results = {
        "device": str(device),
        "precision": args.precision,
        "iters": args.iters,
        "warmup": args.warmup,
        "batch_sizes": args.batch_sizes,
        "seconds": args.seconds,
        "gen": args.gen,
        "meta": {
            "adapter": hparams.get("adapter_type", None),
            "projection_size": hparams.get("projection_size", None),
            "whisper_variant": hparams.get("whisper_variant", None),
            "insertion": insertion,
            "sample_rate": sample_rate,
        },
        "runs": []
    }

    # Warmup
    for _ in range(args.warmup):
        wavs, bos = make_inputs(1, args.seconds[0])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        enc_out, logits = forward_once(wavs, bos)
        if args.gen:
            _ = generate_once(enc_out)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Timed
    for bs in args.batch_sizes:
        for sec in args.seconds:
            times_ms = []
            for _ in range(args.iters):
                wavs, bos = make_inputs(bs, sec)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = time.perf_counter()
                enc_out, logits = forward_once(wavs, bos)
                if args.gen:
                    _ = generate_once(enc_out)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

            summary = summarize(times_ms)
            items_per_s = (1000.0 / summary["mean_ms"]) * bs
            row = {
                "insertion": insertion,
                "batch_size": bs,
                "seconds": sec,
                "summary": summary,
                "throughput_items_per_s": items_per_s,
                "throughput_audio_seconds_per_s": items_per_s * sec,
            }
            results["runs"].append(row)
            mode = "fwd+gen" if args.gen else "fwd"
            print(f"[{mode}] {insertion}  bs={bs} sec={sec}  mean={summary['mean_ms']:.2f}ms  p50={summary['p50_ms']:.2f}  p90={summary['p90_ms']:.2f}  items/s={items_per_s:.2f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
