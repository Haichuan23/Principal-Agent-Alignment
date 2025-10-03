import json
import argparse
import os
from tqdm import tqdm
from nltk import word_tokenize

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="Path prefix (without .jsonl)")
    return parser.parse_args()

def compute_rep_n(text, n):
    tokens = word_tokenize(text, preserve_line=True)
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    rep_n = 100 * (1.0 - len(set(ngrams)) / (len(ngrams) + 1))
    return rep_n

def compute_diversity(text):
    diversity = 1.0
    for n in range(2, 5):
        rep_n_val = compute_rep_n(text, n)
        diversity *= 1.0 - rep_n_val / 100
    return diversity

def average(lst):
    return sum(lst) / len(lst) if lst else 0.0

if __name__ == "__main__":
    args = get_args()
    path = os.path.join("outputs", f"{args.run_name}.jsonl")
    # path = f"{args.run_name}.jsonl"

    # Try loading JSON or JSONL
    with open(path, "r") as f:
        try:
            data = json.load(f)
            generations = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            f.seek(0)
            generations = [json.loads(line) for line in f if line.strip()]

    entries = []
    for g in tqdm(generations):
        prompt = g.get("prompt", "")
        full_resp = g.get("response", "")
        response = full_resp[len(prompt):].strip() or " "
        rep2 = compute_rep_n(response, 2)
        rep3 = compute_rep_n(response, 3)
        rep4 = compute_rep_n(response, 4)
        div = compute_diversity(response)
        entries.append({"rep_2": rep2, "rep_3": rep3, "rep_4": rep4, "diversity": div})

    evaluations = {
        "rep_2": average([e["rep_2"] for e in entries]),
        "rep_3": average([e["rep_3"] for e in entries]),
        "rep_4": average([e["rep_4"] for e in entries]),
        "diversity": average([e["diversity"] for e in entries]),
        "count": len(entries),
    }

    os.makedirs("evaluations", exist_ok=True)
    out_path = os.path.join("evaluations", f"{args.run_name}_diversity.json")
    with open(out_path, "w") as f:
        json.dump(evaluations, f, indent=2)

    print("✅ Saved diversity metrics to:", out_path)
    print(json.dumps(evaluations, indent=2))


# from tqdm import tqdm
# import json
# import argparse
# from nltk import word_tokenize
# import os
# os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
# from simcse import SimCSE
# import numpy as np

# def get_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--run_name", default="llama-7b-greedy", type=str)

#     parser.set_defaults(bottleneck=True)
#     parser.set_defaults(augment=True)
#     args = parser.parse_args()

#     return args


# def compute_rep_n(text, n):
#     tokens = word_tokenize(text, preserve_line=True)
#     ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
#     rep_n = 100 * (1.0 - len(set(ngrams)) / (len(ngrams) + 1))
#     return rep_n


# def compute_diversity(text):
#     diversity = 1.0
#     for n in range(2, 5):
#         rep_n_val = compute_rep_n(text, n)
#         diversity *= 1.0 - rep_n_val / 100
#     return diversity


# def clean(text, sep="###"):
#     return text.split(sep)[0]


# def average(entries):
#     return sum(entries) / len(entries)


# def compute_coherence(prompts, responses):
#     model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
#     similarities = np.array(model.similarity(prompts, responses))
#     return similarities.trace() / len(similarities)


# if __name__ == "__main__":
#     args = get_args()

#     # path = os.path.join("final_outputs", f"{args.run_name}.jsonl")
#     path = f"{args.run_name}.jsonl"
#     # generations = json.load(open(path, "r"))

#     with open(path, "r") as f:
#         try:
#             data = json.load(f)  # try standard JSON
#             if isinstance(data, dict):
#                 generations = [data]
#             elif isinstance(data, list):
#                 generations = data
#             else:
#                 raise ValueError("Unsupported JSON structure in file")
#             print(f"✅ Loaded JSON: {len(generations)} entries")
#         except json.JSONDecodeError:
#             print("⚠️ JSON decode failed — trying JSONL format")
#             f.seek(0)
#             generations = [json.loads(line) for line in f if line.strip()]
#             print(f"✅ Loaded JSONL: {len(generations)} lines")

#     # path = os.path.join("outputs", f"{args.run_name}.jsonl")
#     # with open(path, "r") as f:
#     #     generations = [json.loads(line) for line in f if line.strip()]

#     entries = []
#     for generation in tqdm(generations):
#         prompt = generation["prompt"]
#         response = clean(clean(generation["response"][len(prompt) :], "###Human:"), "\n\nHuman:")
#         if len(response) == 0:
#             response = " "
#         rep_2 = compute_rep_n(response, 2)
#         rep_3 = compute_rep_n(response, 3)
#         rep_4 = compute_rep_n(response, 4)
#         diversity = compute_diversity(response)
#         entries.append(
#             {
#                 "prompt": prompt,
#                 "response": response,
#                 "original_response": generation["response"][len(prompt) :],
#                 "rep_2": rep_2,
#                 "rep_3": rep_3,
#                 "rep_4": rep_4,
#                 "diversity": diversity,
#                 "response_length": len(response),
#                 "elapsed": generation["elapsed"],
#             }
#         )

#     evaluations = {
#         "rep_2": average([entry["rep_2"] for entry in entries]),
#         "rep_3": average([entry["rep_3"] for entry in entries]),
#         "rep_4": average([entry["rep_4"] for entry in entries]),
#         "diversity": average([entry["diversity"] for entry in entries]),
#         "coherence": compute_coherence(
#             [entry["prompt"] for entry in entries], [entry["response"] for entry in entries]
#         ),
#         "response_length": average([entry["response_length"] for entry in entries]),
#         "elapsed": average([entry["elapsed"] for entry in entries]),
#         "entries": entries,
#     }

#     eval_path = os.path.join("evaluations", f"{args.run_name}.json")
#     json.dump(evaluations, open(eval_path, "w"), indent=2)
