from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import torch
import json
import re
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--out_file", type=str, required=True)
parser.add_argument("--rm", type=str, required=True)
parser.add_argument("--rm_gpu", type=str, default="cuda:0")
parser.add_argument("--tokenizer", type=str, required=True)
parser.add_argument("--npout", type=str, default="")
parser.add_argument("--experiment", type=str, default="hhrlhf")
args = parser.parse_args()

# -------- Load tokenizer/model --------
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
rm_model = AutoModelForSequenceClassification.from_pretrained(
    args.rm, num_labels=1, torch_dtype=torch.float16
).to(args.rm_gpu)
rm_model.eval()

# Try to infer max length safely
rm_max_len = getattr(rm_model.config, "max_position_embeddings", None)
if rm_max_len is None or rm_max_len <= 0:
    rm_max_len = getattr(tokenizer, "model_max_length", 2048)
# Put a sane ceiling to avoid absurd large values like 1e12 sometimes set by tokenizers
if rm_max_len is None or rm_max_len > 8192:
    rm_max_len = 2048

# -------- Load JSON or JSONL --------
with open(args.out_file, "r") as f:
    try:
        data = json.load(f)
        if isinstance(data, dict):
            lines = [data]
        elif isinstance(data, list):
            lines = data
        else:
            raise ValueError("Unsupported JSON structure in --out_file")
        print("✅ Loaded as standard JSON")
    except json.JSONDecodeError:
        print("⚠️ JSON decode failed — trying JSONL format")
        f.seek(0)
        lines = [json.loads(line) for line in f if line.strip()]
        print("✅ Loaded as JSONL (line-by-line)")

print(f"📦 Total entries: {len(lines)}")

def extract_out(item):
    """
    Return the text to score. For HH-RLHF, your code concatenated prompt + cleaned response.
    Keep that behavior, but be robust to missing keys.
    """
    # Prefer 'response', fallback to 'output', then 'result'
    output = item.get("response") or item.get("output") or item.get("result")
    if output is None:
        return ""  # nothing to score

    if args.experiment.lower() == "hhrlhf":
        prompt = item.get("prompt", "")
        output_np = output.removeprefix(prompt)

        # fix bug: use output_np consistently
        if output_np.startswith(": "):
            output_np = output_np[2:]

        # strip simulated new "human:" turns
        output_np = re.split(r"human:", output_np, flags=re.IGNORECASE)[0]
        return prompt + output_np
    elif args.experiment.lower() == "shp":
        return output
    else:
        # default: just return output
        return output

def get_rm(text):
    # Truncate to model’s max length to avoid skipping everything
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=rm_max_len)
    input_ids = enc.input_ids.to(args.rm_gpu)
    # Log length for debugging
    print(f"len(tokens)={input_ids.shape[1]}")
    with torch.no_grad():
        out = rm_model(input_ids)
        score = out.logits.flatten().item()
    return score

rm_scores = []
num_skip = 0
for i, item in enumerate(tqdm(lines)):
    text = extract_out(item)
    if not text:
        rm_scores.append(np.nan)  # mark missing as NaN
        continue
    try:
        score = get_rm(text)
        rm_scores.append(score)
    except Exception as e:
        print(f"[WARN] Skipping idx={i} due to RM error: {e}")
        rm_scores.append(np.nan)
        num_skip += 1

rm_scores = np.array(rm_scores, dtype=np.float32)

if args.npout:
    np.save(args.npout, rm_scores)
    print(f"💾 Saved per-sample rewards to: {args.npout}")

valid = np.isfinite(rm_scores)
num_valid = int(valid.sum())
if num_valid == 0:
    print("❗ No valid scores computed (all NaN). Check token lengths and RM settings.")
else:
    mean_reward = np.nanmean(rm_scores)
    print(f"✅ Mean reward over {num_valid} valid samples = {mean_reward:.6f}")

print(f"Skipped/NaN samples = {int((~valid).sum())}")


# from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaTokenizer, LlamaForSequenceClassification
# import argparse
# import torch
# import json
# import re

# parser = argparse.ArgumentParser()
# parser.add_argument("--out_file", type=str)
# parser.add_argument("--rm", type=str)
# parser.add_argument("--rm_gpu", type=str, default="cuda:0")
# parser.add_argument("--tokenizer", type=str)
# parser.add_argument("--npout", type=str, default="")
# parser.add_argument("--experiment", type=str, default="hhrlhf")

# args = parser.parse_args()

# tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

# # with open(args.out_file, "r") as out_f:
# #     lines = json.load(out_f)

# with open(args.out_file, "r") as out_f:
#     try:
#         lines = json.load(out_f)
#         if isinstance(data, dict):
#             lines = [data]        # wrap single object
#         elif isinstance(data, list):
#             lines = data
#         else:
#             raise ValueError("Unsupported JSON structure")
#     except json.JSONDecodeError:
#         print("Detected JSONL format, parsing line by line...")
#         lines = [json.loads(line) for line in out_f if line.strip()]

# rm_model = AutoModelForSequenceClassification.from_pretrained(args.rm, num_labels=1, torch_dtype=torch.float16).to(args.rm_gpu)

# def extract_out(output_data):
#     # output = output_data["result"]
#     # if output.startswith(": "): output = output[2:]
#     # output = re.split("human:", output, flags=re.IGNORECASE)[0]
#     # return output_data["prompt"] + output
#     if "response" in output_data:
#         output = output_data["response"]
#     elif "output" in output_data:
#         output = output_data["output"]

#     if args.experiment == "hhrlhf":
#         output_np = output.removeprefix(output_data["prompt"])
#         if output_np.startswith(": "): output = output_np[2:]
#         output_np = re.split("human:", output_np, flags=re.IGNORECASE)[0]
#         return output_data["prompt"]+output_np
#     elif args.experiment == "shp":
#         return output

#     # return output_data["output"]

# def get_rm(text):
#     tokens = tokenizer(text, return_tensors="pt").input_ids.to(args.rm_gpu)
#     print(f"{tokens.shape=}")
#     # 1966 1819 1813
#     if tokens.shape[1] >= 1334: return None
#     rm_out = rm_model(tokens)

#     rm_val = rm_out.logits.flatten().item()

#     del rm_out
#     del tokens
#     return rm_val

# def get_rm_from_tokens(tokens):
#     return rm_model(torch.tensor(tokens).unsqueeze(0).to(args.rm_gpu)).logits.flatten().item()

# from tqdm import tqdm

# rm_scores = []
# num_skip = 0
# for line in tqdm(lines):
#     outp = extract_out(line)
#     if len(outp) == 0: rm_scores.append(0.)
#     # print(f"{get_rm(outp)}")
#     rm_score = get_rm(outp)
#     if rm_score == None: 
#         print("skipped one")
#         num_skip += 1
#         continue
#     else: rm_scores.append(rm_score)

# import numpy as np
# if args.npout != "": np.save(f"{args.npout}", np.array(rm_scores))
# print(f"{np.mean(rm_scores)=}")
# print(f"{num_skip=}")
