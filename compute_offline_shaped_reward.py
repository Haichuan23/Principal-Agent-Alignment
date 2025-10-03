import os, json
from pathlib import Path
import numpy as np
from scipy.optimize import bisect

# ---------- your helper + cutoff (as you already wrote) ----------
def helper_function(m, rewards, B=1.0, beta=1.0):
    rewards = np.array(rewards, dtype=float)
    weights = np.where(rewards > m, np.exp(B / beta), 1.0)
    return np.sum(weights * (rewards - m))

def compute_cutoff(rewards, B=1.0, beta=1.0):
    rewards = np.array(rewards, dtype=float)
    lo, hi = rewards.min() - 10.0, rewards.max() + 10.0
    try:
        root = bisect(helper_function, lo, hi, args=(rewards, B, beta))
        used_default = False
    except ValueError:
        root = rewards.max()
        used_default = True
        print(f"[Warning] No root found. Defaulting cutoff to max reward {root:.4f}")
    num_above = int(np.sum(rewards > root))
    num_below = int(np.sum(rewards <= root))
    return float(root), num_above, num_below, used_default

def compute_shaped_rewards_for_prompt(prompt_dir, B=1.0, beta=1.0):
    """
    Given a single prompt directory, return shaped rewards for its samples.
    """
    prompt_dir = Path(prompt_dir)
    sample_dirs = sorted([d for d in prompt_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])

    rewards = []
    for sd in sample_dirs:
        meta_path = sd / "meta.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            r = meta.get("reward", None)
            if r is not None:
                rewards.append(float(r))
        except Exception as e:
            print(f"[Warn] Failed to read {meta_path}: {e}")

    if not rewards:
        return []

    m_star, _, _, _ = compute_cutoff(rewards, B=B, beta=beta)
    shaped = np.where(np.array(rewards) > m_star, float(B), 0.0)
    return shaped.tolist()

# ---------- main traversal over dataset ----------
def batch_compute_shaped_rewards(root_dir, max_prompts=4000, B=1.0, beta=1.0, num_responses=10):
    """
    Traverse up to `max_prompts` prompt folders under root_dir/shards,
    compute shaped rewards, and save them to JSON files.
    """
    root_dir = Path(root_dir)
    shard_dir = root_dir / "shards"
    prompt_dirs = sorted([d for d in shard_dir.iterdir() if d.is_dir() and d.name.startswith("prompt_")])

    for idx, prompt_dir in enumerate(prompt_dirs[:max_prompts]):
        shaped = compute_shaped_rewards_for_prompt(prompt_dir, B=B, beta=beta)
        if not shaped:
            continue

        out_name = f"shaped_reward_prompt{idx:05d}_response{num_responses}_B_{B}_beta_{beta}.json"
        out_path = prompt_dir / out_name
        with open(out_path, "w") as f:
            json.dump({"shaped_rewards": shaped}, f, indent=2)

    print(f"Up to {max_prompts} shaped rewards are completed")
        # print(f"[{idx}] Saved shaped rewards → {ou t_path}")

# Example usage
# batch_compute_shaped_rewards(
#     root_dir="datasets/models_llama3-8b",
#     max_prompts=4000,
#     B=2.0,
#     beta=1.0,
#     num_responses=10
# )

import os, json, torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader

class FlatSampleDataset(Dataset):
    def __init__(self, shards_dir, response_num=10, B=2.0, beta=1.0, max_len=128, device="cpu"):
        """
        shards_dir/
          prompt_0000000/
            sample_000/hidden_last.pt
            ...
            shaped_reward_prompt00000_response10_B_2.0_beta_1.0.json
        """
        self.samples = []
        self.max_len = max_len
        self.device  = device

        shards = Path(shards_dir)
        prompt_dirs = sorted(d for d in shards.iterdir() if d.is_dir() and d.name.startswith("prompt_"))

        file_pattern = f"shaped_reward_prompt*_response{response_num}_B_{B}_beta_{beta}.json"
        for pdir in prompt_dirs:
            shaped_files = list(pdir.glob(file_pattern))
            if not shaped_files:
                continue

            with open(shaped_files[0], "r") as f:
                shaped_rewards = json.load(f)["shaped_rewards"]

            sample_dirs = sorted(d for d in pdir.iterdir() if d.is_dir() and d.name.startswith("sample_"))

            for i, sd in enumerate(sample_dirs):
                if i >= len(shaped_rewards):
                    break
                hidden_path = sd / "hidden_last.pt"
                if hidden_path.exists():
                    self.samples.append({
                        "hidden_path": hidden_path,
                        "reward": float(shaped_rewards[i]),
                        "prompt_id": pdir.name,
                        "sample_id": sd.name,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        h = torch.load(item["hidden_path"])           # [T_gen, H]
        T, H = h.shape
        length = min(T, self.max_len)

        x = h[:self.max_len]
        if x.size(0) < self.max_len:
            pad = torch.zeros(self.max_len - x.size(0), H, dtype=h.dtype)
            x = torch.cat([x, pad], dim=0)

        # mask: 1 for real tokens, 0 for padding
        mask = torch.zeros(self.max_len, dtype=torch.float32)
        mask[:length] = 1.0

        return {
            "hidden": x.to(self.device),                                      # [max_len, H]
            "mask":   mask.to(self.device),                                   # [max_len]
            "reward": torch.tensor(item["reward"], dtype=torch.float32, device=self.device),
            "length": torch.tensor(length, dtype=torch.long, device=self.device),
            "prompt_id": item["prompt_id"],
            "sample_id": item["sample_id"],
        }

root = Path("./datasets/models_llama3-8b/shards")
ds = FlatSampleDataset(root, response_num=10, B=2.0, beta=1.0, max_len=128)
loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

batch = next(iter(loader))
print("hidden:", batch["hidden"].shape)   # [2, 128, 4096]
print("mask:  ", batch["mask"].shape, "valid per item:", batch["mask"].sum(dim=1).tolist())
print("reward:", batch["reward"].tolist())
print("ids:   ", list(zip(batch["prompt_id"], batch["sample_id"])))


