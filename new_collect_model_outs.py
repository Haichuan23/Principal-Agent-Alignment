#!/usr/bin/env python3
"""
Collect generations using *controlled decoding* (value-guided) and save results.

This is a drop-in replacement for the original collector, but it uses your
ControlledDecoder (greedy / beam / sample) instead of plain .generate().
Writes JSONL (one json object per line) to args.out_file.

Example:
python collect_controlled_outs.py \
  --dataset Dahoas/full-hh-rlhf --split test --run_percent 10 \
  --out_file outs.jsonl \
  --value_ckpt checkpoints_value/value_agent_epoch03.pt \
  --model_id meta-llama/Llama-3.1-8B \
  --mode sample --sample_temperature 0.9 --top_k 30 --lambda_coef 1.0
"""

import argparse
import json
import time
from pathlib import Path
import sys
import os

import torch
from datasets import load_dataset

# Ensure we can import from the current directory even if launched elsewhere
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

# Import ControlledDecoder from util_decode.py (same folder)
from util_decode_new import ControlledDecoder


def _resolve_prompts(ds, dataset_name: str):
    """Normalize to a simple list[str] of prompts."""
    if dataset_name == "Dahoas/full-hh-rlhf":
        return ds["prompt"]
    elif dataset_name == "stanfordnlp/SHP":
        unique_prompts, seen_posts = [], set()
        for post_id, histr in zip(ds["post_id"], ds["history"]):
            if post_id in seen_posts:
                continue
            unique_prompts.append(f" Human: {histr} Assistant: ")
            seen_posts.add(post_id)
        return unique_prompts
    else:
        # If the split already yields plain strings, try that:
        first = ds[0]
        if isinstance(first, str):
            return list(ds)
        # Otherwise, try common columns:
        for col in ("prompt", "question", "inputs", "text"):
            if col in ds.column_names:
                return ds[col]
        raise ValueError(
            "Could not resolve prompts for this dataset; "
            "specify a supported dataset or adapt the code."
        )


def main():
    ap = argparse.ArgumentParser("Collect outputs with Controlled Decoding")
    # Data
    ap.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--run_percent", type=float, default=2.0)
    ap.add_argument("--out_file", type=str, required=True)

    # ControlledDecoder config
    ap.add_argument("--value_ckpt", type=str, required=True, help="Path to value_agent_epochXX.pt")
    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--lambda_coef", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)

    # Decoding mode
    ap.add_argument("--mode", type=str, default="greedy", choices=["greedy", "beam", "sample"])
    ap.add_argument("--num_beams", type=int, default=4)                 # for beam
    ap.add_argument("--sample_temperature", type=float, default=1.0)    # for sample
    ap.add_argument("--sample_top_k", type=int, default=None)           # optional override for sample

    # Generation control
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--skip_if_prompt_too_long", action="store_true",
                    help="Skip prompts that exceed model context instead of attempting anyway.")

    args = ap.parse_args()

    # --- setup output path (JSONL) ---
    out_path = Path(args.out_file)
    if out_path.exists():
        raise SystemExit(f"ERROR: out_file already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- load dataset ---
    print(f"[INFO] Loading dataset={args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    prompts = _resolve_prompts(ds, args.dataset)

    end_idx = int(len(prompts) * (args.run_percent / 100.0))
    prompts = prompts[:end_idx]
    print(f"[INFO] Will run on {len(prompts)} prompts (run_percent={args.run_percent}%).")

    # --- build controlled decoder ---
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    print(f"[INFO] Loading ControlledDecoder model_id={args.model_id}")
    dec = ControlledDecoder(
        value_ckpt_path=args.value_ckpt,
        model_id=args.model_id,
        device=args.device,
        dtype=dtype_map[args.dtype],
        lambda_coef=args.lambda_coef,
        top_k=args.top_k,
        temperature=args.temperature,
        debug = True,
        debug_max_steps = 12
    )

    # --- helper for one prompt ---
    def run_one(prompt: str):
        t0 = time.time()
        if args.mode == "greedy":
            full_text = dec.decode_greedy(
                prompt,
                max_new_tokens=args.max_new_tokens,
                stop_on_eos=True,
            )
        elif args.mode == "beam":
            full_text = dec.decode_beam(
                prompt,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                stop_on_eos=True,
            )
        else:  # "sample"
            full_text = dec.decode_sample(
                prompt,
                max_new_tokens=args.max_new_tokens,
                stop_on_eos=True,
                sample_temperature=args.sample_temperature,
                top_k=args.sample_top_k,  # falls back to self.top_k if None
            )
        elapsed = time.time() - t0

        # Derive continuation robustly (decoder may return full text including prompt)
        if isinstance(full_text, str) and full_text.startswith(prompt):
            continuation = full_text[len(prompt):]
        else:
            # Fall back to treating the returned text as the continuation itself
            continuation = full_text

        return continuation, elapsed

    # --- iterate & write JSONL incrementally ---
    num_skipped = 0
    num_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            try:
                result_text, elapsed = run_one(prompt)
                obj = {
                    "prompt": prompt,
                    "result": result_text,                   # continuation only
                    "response": f"{prompt}{result_text}",    # full string for convenience
                    "elapsed": elapsed,
                    # provenance
                    "mode": args.mode,
                    "model_id": args.model_id,
                    "lambda_coef": args.lambda_coef,
                    "top_k": args.top_k,
                    "temperature": args.temperature,
                    "num_beams": args.num_beams if args.mode == "beam" else None,
                    "sample_temperature": args.sample_temperature if args.mode == "sample" else None,
                    "sample_top_k": args.sample_top_k if args.mode == "sample" else None,
                    "max_new_tokens": args.max_new_tokens,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                num_written += 1
            except RuntimeError as e:
                # Optionally skip on context overflow or CUDA OOM
                print(f"[WARN] Skipped idx={i}: {e}")
                num_skipped += 1
                continue

    print(f"[INFO] Done. Wrote: {out_path}  (written={num_written}, skipped={num_skipped})")


if __name__ == "__main__":
    main()