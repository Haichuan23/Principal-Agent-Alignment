#!/usr/bin/env python3
import os, json, random, argparse, re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def model_tag(model_id: str) -> str:
    """Safe folder name for a HF id or local path."""
    base = model_id.rstrip("/").split("/")[-2:] if "/" in model_id else [model_id]
    base = "_".join(base)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)

## Input:
## ex: a single example, which is a Python dictionary from Dahoas/full-hh-rlhf
##
def extract_prompt(ex):
    """
    For Dahoas/full-hh-rlhf, the prompt is already provided as a clean conversation
    prefix ending with 'Assistant:'. Just return it.
    """
    v = ex.get("prompt")
    if isinstance(v, str) and v.strip():
        return v
    raise KeyError(f"Could not find 'prompt' in example. Keys: {list(ex.keys())}")

def render_chat_prompt(tokenizer, prompt_text: str, for_generation: bool):
    # If it already looks like a transcript, pass through unchanged.
    if isinstance(prompt_text, str) and ("Human:" in prompt_text or "Assistant:" in prompt_text):
        return prompt_text, False

    # Otherwise, if the tokenizer has a chat template, use it.
    has_template = getattr(tokenizer, "chat_template", None)
    if has_template:
        msgs = [{"role": "user", "content": prompt_text}]
        rendered = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=for_generation
        )
        return rendered, True

    return prompt_text, False

@torch.no_grad()
def get_prompt_last_hidden(model, tokenizer, prompt_text, device):
    """
    Compute last hidden state vector for the (rendered) prompt.
    We render with for_generation=True to match tokenization used right before generation.
    """
    rendered, chat_used = render_chat_prompt(tokenizer, prompt_text, for_generation=True)
    enc = tokenizer(rendered, return_tensors="pt", padding=False, truncation=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    attn = torch.ones_like(enc["input_ids"], device=device)

    outs = model(
        input_ids=enc["input_ids"],
        attention_mask=attn,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    last_hidden = outs.hidden_states[-1]          # [1, T_prompt, H]
    prompt_len = enc["input_ids"].size(1)
    hidden_prompt_last = last_hidden[:, -1, :].contiguous()[0].detach().cpu()  # [H]
    return enc["input_ids"][0].detach().cpu().numpy(), int(prompt_len), hidden_prompt_last, rendered, chat_used

@torch.no_grad()
def generate_many(model, tokenizer, prompt_text, M, max_new_tokens, temperature, top_p, device, base_seed):
    """
    Generate M samples for one prompt; returns list of dicts for responses (no prompt hidden duplication).
    Uses the SAME rendering as in get_prompt_last_hidden (for_generation=True) so prompt_len matches.
    """
    model.eval()
    out = []

    # Pre-render once (same for all m)
    rendered, chat_used = render_chat_prompt(tokenizer, prompt_text, for_generation=True)

    for m in range(M):
        set_seed(base_seed + m)

        enc = tokenizer(rendered, return_tensors="pt", padding=False, truncation=False)
        enc = {k: v.to(device) for k, v in enc.items()}

        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        seq = gen.sequences                              # [1, T_total]
        attn_mask = torch.ones_like(seq, device=device)  # no padding at save-time

        outs = model(
            input_ids=seq,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = outs.hidden_states[-1]             # [1, T_total, H]
        T_total = last_hidden.size(1)
        prompt_len = enc["input_ids"].size(1)
        gen_len = T_total - prompt_len
        hidden_gen = last_hidden[:, prompt_len:, :].contiguous()  # [1, T_gen, H]

        # Decode only the generated span
        response_text = tokenizer.decode(
            seq[0, prompt_len:], skip_special_tokens=True
        )

        out.append({
            "prompt_text": prompt_text,
            "rendered_prompt": rendered,
            "chat_formatted": bool(chat_used),
            "response_text": response_text,
            "input_ids_full": seq[0].detach().cpu().numpy(),            # [T_total]
            "attention_mask_full": attn_mask[0].detach().cpu().numpy(), # [T_total]
            "hidden_last_gen": hidden_gen[0].detach().cpu(),            # [T_gen, H]
            "prompt_len": int(prompt_len),
            "gen_len": int(gen_len),
            "T_total": int(T_total),
        })
    return out

# =========================
# Reward model additions ↓
# =========================

@torch.no_grad()
def score_with_rm(rm_tok, rm_model, prompt_text: str, response_text: str, max_length: int = None) -> float:
    """
    Skywork Reward (Llama-3.1-8B) expects a chat-formatted conversation:
      [{"role":"user","content": prompt}, {"role":"assistant","content": response}]
    It outputs a single scalar in `logits[0][0]`.
    """
    # 1) Build chat conversation
    conv = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": response_text},
    ]

    # 2) Render with the tokenizer's chat template (no BOS added here)
    rendered = rm_tok.apply_chat_template(conv, tokenize=False)

    # 3) Tokenize; tokenizer will add BOS as needed. Truncate to model max if not provided.
    if max_length is None:
        max_length = getattr(rm_tok, "model_max_length", 4096)

    enc = rm_tok(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_length),
    )
    enc = {k: v.to(rm_model.device) for k, v in enc.items()}

    # 4) Forward pass → scalar reward in logits[0][0]
    out = rm_model(**enc)
    score = out.logits[0][0].item()
    return float(score)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True,
                    help="Baseline CausalLM (e.g. Qwen/Qwen3-8B or meta-llama/Meta-Llama-3-8B)")
    # switched to Dahoas dataset; no 'subset' needed
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--num_problems", type=int, default=100)
    ap.add_argument("--samples_per_prompt", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--output_dir", type=str, default="datasets/hh_offline")
    ap.add_argument("--fp16_hidden", action="store_true",
                    help="Store hidden states in float16 (else bfloat16).")
    # ---- reward model flags (added) ----
    ap.add_argument("--reward_model_name_or_path", type=str, default=None,
                    help="HF id or local path of a SequenceClassification reward model.")
    ap.add_argument("--rm_max_length", type=int, default=4096,
                    help="Max tokens when encoding prompt+response for the RM.")
    ap.add_argument("--write_jsonl", action="store_true",
                    help="Also write compact samples.jsonl with prompt/response/reward.")
    # ------------------------------------
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset Dahoas/full-hh-rlhf [{args.split}] …")
    ds = load_dataset("Dahoas/full-hh-rlhf", split=args.split)

    print(f"Loading model/tokenizer {args.model_name_or_path} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=(torch.float16 if args.fp16_hidden else torch.bfloat16),
        device_map=None,
        trust_remote_code=True
    ).to(device)
    model.eval()
    model.requires_grad_(False)

    # ---- load reward model if provided (added) ----
    rm_tok = rm_model = None
    if args.reward_model_name_or_path:
        print(f"Loading reward model {args.reward_model_name_or_path} …")
        rm_tok = AutoTokenizer.from_pretrained(args.reward_model_name_or_path, use_fast=True, trust_remote_code=True)
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model_name_or_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(device)
        rm_model.eval(); rm_model.requires_grad_(False)
    # -----------------------------------------------

    tag = model_tag(args.model_name_or_path)
    root = Path(args.output_dir) / tag
    shards_dir = root / "shards"
    prompts_dir = root / "prompts"
    ensure_dir(shards_dir); ensure_dir(prompts_dir)
    index_path = root / "index.jsonl"
    index_f = open(index_path, "a", buffering=1)

    # optional compact JSONL of results (added)
    samples_jsonl_path = root / "samples.jsonl"
    samples_jsonl_f = open(samples_jsonl_path, "a", buffering=1) if args.write_jsonl else None

    dtype_str = "float16" if args.fp16_hidden else "bfloat16"

    N = min(args.num_problems, len(ds))
    for i in range(N):
        ex = ds[i]
        prompt = extract_prompt(ex)  # now reads ex["prompt"]

        # Prompt-only last hidden (rendered with chat template when needed)
        input_ids_prompt, prompt_len_true, hidden_prompt_last, rendered_prompt, chat_used = get_prompt_last_hidden(
            model, tokenizer, prompt, device
        )
        hidden_prompt_last = hidden_prompt_last.to(
            torch.float16 if args.fp16_hidden else torch.bfloat16
        )

        # Per-prompt folder with artifacts
        pd = prompts_dir / f"prompt_{i:07d}"
        ensure_dir(pd)
        np.save(pd / "input_ids_prompt.npy", input_ids_prompt.astype(np.int64))
        with open(pd / "prompt.txt", "w") as f:
            f.write(prompt)
        with open(pd / "rendered_prompt.txt", "w") as f:
            f.write(rendered_prompt)
        torch.save(hidden_prompt_last, pd / "hidden_prompt_last.pt")

        with open(pd / "meta.json", "w") as f:
            json.dump({
                "prompt_id": i,
                "prompt_len": int(prompt_len_true),
                "chat_formatted": bool(chat_used),
                "arrays": {
                    "input_ids_prompt": str(pd / "input_ids_prompt.npy"),
                    "hidden_prompt_last": str(pd / "hidden_prompt_last.pt"),
                },
                "dtypes": {"hidden_prompt_last": dtype_str},
                "model_id": args.model_name_or_path,
                "dataset": "Dahoas/full-hh-rlhf",
                "split": args.split
            }, f, ensure_ascii=False)

        # Generate multiple responses for the SAME prompt
        samples = generate_many(
            model, tokenizer, prompt,
            M=args.samples_per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            base_seed=args.seed + i * 100000
        )

        # Per-prompt sample root (hierarchical naming)
        samples_root_for_prompt = shards_dir / f"prompt_{i:07d}"
        ensure_dir(samples_root_for_prompt)

        for m, s in enumerate(samples):
            # ---- score with reward model (added) ----
            reward_val = None
            if rm_model is not None:
                reward_val = score_with_rm(
                    rm_tok, rm_model, s["prompt_text"], s["response_text"], max_length=args.rm_max_length
                )
            # ----------------------------------------

            # shards/prompt_XXXXXXX/sample_YYY
            d = samples_root_for_prompt / f"sample_{m:03d}"
            ensure_dir(d)

            np.save(d / "input_ids.npy", s["input_ids_full"].astype(np.int64))
            np.save(d / "attention_mask.npy", s["attention_mask_full"].astype(np.uint8))

            hg = s["hidden_last_gen"].to(
                torch.float16 if args.fp16_hidden else torch.bfloat16
            )
            torch.save(hg, d / "hidden_last.pt")

            meta = {
                "prompt_id": i,
                "sample_id": m,
                "prompt_ref": str(pd),
                "prompt_text": s["prompt_text"],
                "rendered_prompt": s["rendered_prompt"] if "rendered_prompt" in s else rendered_prompt,
                "chat_formatted": s.get("chat_formatted", chat_used),
                "response_text": s["response_text"],
                "reward": reward_val,
                "reward_model_id": args.reward_model_name_or_path,
                "prompt_len": s["prompt_len"],
                "gen_len": s["gen_len"],
                "T_total": s["T_total"],
                "model_id": args.model_name_or_path,
                "tokenizer_id": args.model_name_or_path,
                "gen_params": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed
                },
                "arrays": {
                    "input_ids": str(d / "input_ids.npy"),
                    "attention_mask": str(d / "attention_mask.npy"),
                    "hidden_last": str(d / "hidden_last.pt"),
                    "input_ids_prompt_ref": str(pd / "input_ids_prompt.npy"),
                    "hidden_prompt_last_ref": str(pd / "hidden_prompt_last.pt"),
                },
                "dtypes": {
                    "hidden_last": dtype_str,
                    "hidden_prompt_last_ref": dtype_str
                },
                "dataset": "Dahoas/full-hh-rlhf",
                "split": args.split,
                "indices": {"dataset_idx": i, "sample_idx_for_prompt": m},
            }
            with open(d / "meta.json", "w") as f:
                json.dump(meta, f, ensure_ascii=False)

            # optional compact JSONL row (added)
            if samples_jsonl_f is not None:
                samples_jsonl_f.write(json.dumps({
                    "id": f"{i}:{m}",
                    "prompt": s["prompt_text"],
                    "response": s["response_text"],
                    "reward": reward_val,
                    "lm": args.model_name_or_path,
                    "rm": args.reward_model_name_or_path,
                    "dataset": "Dahoas/full-hh-rlhf",
                    "split": args.split
                }, ensure_ascii=False) + "\n")

            index_entry = {
                "id": f"{i}:{m}",
                "path": str(d),
                "prompt_id": i,
                "prompt_ref": str(pd),
                "prompt_len": s["prompt_len"],
                "gen_len": s["gen_len"],
                "model_id": args.model_name_or_path,
                "dataset": "Dahoas/full-hh-rlhf",
                "split": args.split,
            }
            index_f.write(json.dumps(index_entry) + "\n")

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{N} prompts …")

    index_f.close()
    if samples_jsonl_f is not None:
        samples_jsonl_f.close()
    print(f"Done. Wrote {N * args.samples_per_prompt} samples to {root}")

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# import os, json, random, argparse, re
# from pathlib import Path
# import numpy as np
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import load_dataset

# def set_seed(seed: int):
#     random.seed(seed); np.random.seed(seed)
#     torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# def ensure_dir(p: Path):
#     p.mkdir(parents=True, exist_ok=True)

# def model_tag(model_id: str) -> str:
#     """Safe folder name for a HF id or local path."""
#     base = model_id.rstrip("/").split("/")[-2:] if "/" in model_id else [model_id]
#     base = "_".join(base)
#     return re.sub(r"[^A-Za-z0-9._-]+", "_", base)

# def extract_prompt(ex):
#     v = ex.get("prompt")
#     if isinstance(v, str) and v.strip():
#         return v
#     for k in ("chosen", "rejected"):
#         s = ex.get(k)
#         if isinstance(s, str) and s.strip():
#             pre = s.split("Assistant:", 1)[0]
#             if "Human:" in pre:
#                 pre = pre.split("Human:", 1)[1]
#             prompt = pre.strip()
#             if prompt:
#                 return prompt
#     raise KeyError(f"Could not find a prompt in example. Keys: {list(ex.keys())}")

# def render_chat_prompt(tokenizer, prompt_text: str, for_generation: bool):
#     """
#     Returns (rendered_text:str, used_chat_template:bool).
#     If the tokenizer has a chat template, use it; else return the raw prompt.
#     - for_generation=True adds the assistant prefix so generation starts correctly.
#     """
#     if hasattr(tokenizer, "apply_chat_template"):
#         msgs = [{"role": "user", "content": prompt_text}]
#         rendered = tokenizer.apply_chat_template(
#             msgs, tokenize=False, add_generation_prompt=for_generation
#         )
#         return rendered, True
#     return prompt_text, False

# @torch.no_grad()
# def get_prompt_last_hidden(model, tokenizer, prompt_text, device):
#     """
#     Compute last hidden state vector for the (rendered) prompt.
#     IMPORTANT: we render with for_generation=True to match the exact tokenization
#     used right before generation (assistant prefix included).
#     Returns:
#       input_ids_prompt [T_prompt],
#       prompt_len (int),
#       hidden_prompt_last [H] (torch CPU tensor),
#       rendered_text (str),
#       chat_used (bool)
#     """
#     rendered, chat_used = render_chat_prompt(tokenizer, prompt_text, for_generation=True)
#     enc = tokenizer(rendered, return_tensors="pt", padding=False, truncation=False)
#     enc = {k: v.to(device) for k, v in enc.items()}
#     attn = torch.ones_like(enc["input_ids"], device=device)

#     outs = model(
#         input_ids=enc["input_ids"],
#         attention_mask=attn,
#         output_hidden_states=True,
#         use_cache=False,
#         return_dict=True,
#     )
#     last_hidden = outs.hidden_states[-1]          # [1, T_prompt, H]
#     prompt_len = enc["input_ids"].size(1)
#     hidden_prompt_last = last_hidden[:, -1, :].contiguous()[0].detach().cpu()  # [H]
#     return enc["input_ids"][0].detach().cpu().numpy(), int(prompt_len), hidden_prompt_last, rendered, chat_used

# @torch.no_grad()
# def generate_many(model, tokenizer, prompt_text, M, max_new_tokens, temperature, top_p, device, base_seed):
#     """
#     Generate M samples for one prompt; returns list of dicts for responses (no prompt hidden duplication).
#     Uses the SAME chat rendering as in get_prompt_last_hidden (for_generation=True) so prompt_len matches.
#     """
#     model.eval()
#     out = []

#     # Pre-render once (same for all m)
#     rendered, chat_used = render_chat_prompt(tokenizer, prompt_text, for_generation=True)

#     for m in range(M):
#         set_seed(base_seed + m)

#         enc = tokenizer(rendered, return_tensors="pt", padding=False, truncation=False)
#         enc = {k: v.to(device) for k, v in enc.items()}

#         gen = model.generate(
#             **enc,
#             do_sample=True,
#             temperature=float(temperature),
#             top_p=float(top_p),
#             max_new_tokens=int(max_new_tokens),
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
#             return_dict_in_generate=True,
#         )
#         seq = gen.sequences                              # [1, T_total]
#         attn_mask = torch.ones_like(seq, device=device)  # no padding at save-time

#         outs = model(
#             input_ids=seq,
#             attention_mask=attn_mask,
#             output_hidden_states=True,
#             use_cache=False,
#             return_dict=True,
#         )
#         last_hidden = outs.hidden_states[-1]             # [1, T_total, H]
#         T_total = last_hidden.size(1)
#         prompt_len = enc["input_ids"].size(1)
#         gen_len = T_total - prompt_len
#         hidden_gen = last_hidden[:, prompt_len:, :].contiguous()  # [1, T_gen, H]

#         # Decode only the generated span
#         response_text = tokenizer.decode(
#             seq[0, prompt_len:], skip_special_tokens=True
#         )

#         out.append({
#             "prompt_text": prompt_text,
#             "rendered_prompt": rendered,
#             "chat_formatted": bool(chat_used),
#             "response_text": response_text,
#             "input_ids_full": seq[0].detach().cpu().numpy(),            # [T_total]
#             "attention_mask_full": attn_mask[0].detach().cpu().numpy(), # [T_total]
#             "hidden_last_gen": hidden_gen[0].detach().cpu(),            # [T_gen, H]
#             "prompt_len": int(prompt_len),
#             "gen_len": int(gen_len),
#             "T_total": int(T_total),
#         })
#     return out

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model_name_or_path", type=str, required=True,
#                     help="Baseline CausalLM (e.g. Qwen/Qwen3-8B or meta-llama/Meta-Llama-3-8B)")
#     ap.add_argument("--subset", type=str, default="helpful-base",
#                     choices=["helpful-base","harmless-base"])
#     ap.add_argument("--split", type=str, default="train")
#     ap.add_argument("--num_problems", type=int, default=100)
#     ap.add_argument("--samples_per_prompt", type=int, default=4)
#     ap.add_argument("--max_new_tokens", type=int, default=128)
#     ap.add_argument("--temperature", type=float, default=0.7)
#     ap.add_argument("--top_p", type=float, default=0.9)
#     ap.add_argument("--seed", type=int, default=1234)
#     ap.add_argument("--device", type=str, default="cuda:0")
#     ap.add_argument("--output_dir", type=str, default="datasets/hh_offline")
#     ap.add_argument("--fp16_hidden", action="store_true",
#                     help="Store hidden states in float16 (else bfloat16).")
#     args = ap.parse_args()

#     set_seed(args.seed)
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")

#     print(f"Loading dataset Anthropic/hh-rlhf [{args.subset}/{args.split}] …")
#     ds = load_dataset("Anthropic/hh-rlhf", args.subset, split=args.split)

#     print(f"Loading model/tokenizer {args.model_name_or_path} …")
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_name_or_path,
#         torch_dtype=(torch.float16 if args.fp16_hidden else torch.bfloat16),
#         device_map=None,
#         trust_remote_code=True
#     ).to(device)
#     model.eval()
#     model.requires_grad_(False)

#     tag = model_tag(args.model_name_or_path)
#     root = Path(args.output_dir) / tag
#     shards_dir = root / "shards"
#     prompts_dir = root / "prompts"
#     ensure_dir(shards_dir); ensure_dir(prompts_dir)
#     index_path = root / "index.jsonl"
#     index_f = open(index_path, "a", buffering=1)

#     dtype_str = "float16" if args.fp16_hidden else "bfloat16"

#     global_id = 0
#     N = min(args.num_problems, len(ds))
#     for i in range(N):
#         ex = ds[i]
#         prompt = extract_prompt(ex)

#         # Prompt-only last hidden (rendered with chat template, for_generation=True)
#         input_ids_prompt, prompt_len_true, hidden_prompt_last, rendered_prompt, chat_used = get_prompt_last_hidden(
#             model, tokenizer, prompt, device
#         )
#         hidden_prompt_last = hidden_prompt_last.to(
#             torch.float16 if args.fp16_hidden else torch.bfloat16
#         )

#         pd = prompts_dir / f"prompt_{i:07d}"
#         ensure_dir(pd)
#         np.save(pd / "input_ids_prompt.npy", input_ids_prompt.astype(np.int64))
#         with open(pd / "prompt.txt", "w") as f:
#             f.write(prompt)
#         with open(pd / "rendered_prompt.txt", "w") as f:
#             f.write(rendered_prompt)
#         torch.save(hidden_prompt_last, pd / "hidden_prompt_last.pt")

#         with open(pd / "meta.json", "w") as f:
#             json.dump({
#                 "prompt_id": i,
#                 "prompt_len": int(prompt_len_true),
#                 "chat_formatted": bool(chat_used),
#                 "arrays": {
#                     "input_ids_prompt": str(pd / "input_ids_prompt.npy"),
#                     "hidden_prompt_last": str(pd / "hidden_prompt_last.pt"),
#                 },
#                 "dtypes": {"hidden_prompt_last": dtype_str},
#                 "model_id": args.model_name_or_path,
#             }, f, ensure_ascii=False)

#         # Generate multiple responses for the SAME (chat-rendered) prompt
#         samples = generate_many(
#             model, tokenizer, prompt,
#             M=args.samples_per_prompt,
#             max_new_tokens=args.max_new_tokens,
#             temperature=args.temperature,
#             top_p=args.top_p,
#             device=device,
#             base_seed=args.seed + i * 100000
#         )

#         for m, s in enumerate(samples):
#             d = shards_dir / f"sample_{global_id:07d}"
#             ensure_dir(d)

#             np.save(d / "input_ids.npy", s["input_ids_full"].astype(np.int64))
#             np.save(d / "attention_mask.npy", s["attention_mask_full"].astype(np.uint8))

#             hg = s["hidden_last_gen"].to(
#                 torch.float16 if args.fp16_hidden else torch.bfloat16
#             )
#             torch.save(hg, d / "hidden_last.pt")

#             meta = {
#                 "prompt_id": i,
#                 "prompt_ref": str(pd),
#                 "prompt_text": s["prompt_text"],
#                 "rendered_prompt": s["rendered_prompt"] if "rendered_prompt" in s else rendered_prompt,
#                 "chat_formatted": s.get("chat_formatted", chat_used),
#                 "response_text": s["response_text"],
#                 "prompt_len": s["prompt_len"],
#                 "gen_len": s["gen_len"],
#                 "T_total": s["T_total"],
#                 "model_id": args.model_name_or_path,
#                 "tokenizer_id": args.model_name_or_path,
#                 "gen_params": {
#                     "temperature": args.temperature,
#                     "top_p": args.top_p,
#                     "max_new_tokens": args.max_new_tokens,
#                     "seed": args.seed
#                 },
#                 "arrays": {
#                     "input_ids": str(d / "input_ids.npy"),
#                     "attention_mask": str(d / "attention_mask.npy"),
#                     "hidden_last": str(d / "hidden_last.pt"),
#                     "input_ids_prompt_ref": str(pd / "input_ids_prompt.npy"),
#                     "hidden_prompt_last_ref": str(pd / "hidden_prompt_last.pt"),
#                 },
#                 "dtypes": {
#                     "hidden_last": dtype_str,
#                     "hidden_prompt_last_ref": dtype_str
#                 },
#                 "subset": args.subset,
#                 "split": args.split,
#                 "indices": {"dataset_idx": i, "sample_idx_for_prompt": m},
#             }
#             with open(d / "meta.json", "w") as f:
#                 json.dump(meta, f, ensure_ascii=False)

#             index_entry = {
#                 "id": global_id,
#                 "path": str(d),
#                 "prompt_id": i,
#                 "prompt_ref": str(pd),
#                 "prompt_len": s["prompt_len"],
#                 "gen_len": s["gen_len"],
#                 "model_id": args.model_name_or_path,
#                 "subset": args.subset,
#                 "split": args.split,
#             }
#             index_f.write(json.dumps(index_entry) + "\n")
#             global_id += 1

#         if (i + 1) % 10 == 0:
#             print(f"Processed {i+1}/{N} prompts …")

#     index_f.close()
#     print(f"Done. Wrote {global_id} samples to {root}")

# if __name__ == "__main__":
#     main()
