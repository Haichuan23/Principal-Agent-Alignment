'''
Things to notice:
1. each hidden_last.pt has size [T_gen, H], so they might differ
    by size, and we need to do padding
2. reward needs to be loaded as shaped reward
'''

import os
import json
import time
import random
import argparse
import math  # <-- needed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import wandb

# --- optional/unused imports removed for clarity ---
# from torch.distributions import Normal
# from transformers import (
#     LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM,
#     AutoModelForSequenceClassification, AutoTokenizer,
# )
# from intervented_model.gllama import Intervented_LlamaForCausalLM
# import intervented_model.gllama as llama
# from get_llm_hiddenstates import get_llm_activations
# from get_reward import get_score
# from utils import load_data
# from gpt_eval import get_gpt_score
# from reward_model import RewardModel

class FlatSampleDataset(Dataset):
    def __init__(self, shards_dir, response_num=10, B=2.0, beta=1.0, max_len=128, device="cpu", max_prompt=None):
        self.samples = []
        self.max_len = max_len
        self.device  = device

        shards = Path(shards_dir)
        prompt_dirs = sorted(d for d in shards.iterdir() if d.is_dir() and d.name.startswith("prompt_"))

        if max_prompt is not None:
            prompt_dirs = prompt_dirs[:max_prompt]

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
        h = torch.load(item["hidden_path"], map_location="cpu")  # [T_gen, H]
        T, H = h.shape
        length = min(T, self.max_len)

        x = h[:self.max_len]
        if x.size(0) < self.max_len:
            pad = torch.zeros(self.max_len - x.size(0), H, dtype=h.dtype)
            x = torch.cat([x, pad], dim=0)

        mask = torch.zeros(self.max_len, dtype=torch.float32)
        mask[:length] = 1.0

        return {
            "hidden": x.to(self.device),  # [max_len, H]
            "mask":   mask.to(self.device),  # [max_len]
            "reward": torch.tensor(item["reward"], dtype=torch.float32, device=self.device),
            "length": torch.tensor(length, dtype=torch.long, device=self.device),
            "prompt_id": item["prompt_id"],
            "sample_id": item["sample_id"],
        }

def layer_init(layer, std=1.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer

class Agent(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 8192)),
            nn.Tanh(),
            layer_init(nn.Linear(8192, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
    def get_value(self, x):
        x = x.float()
        return self.critic(x).squeeze(-1)  # <-- ensure shape [N]

def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(
    agent, loader, device, optimizer,
    use_pairwise=True, pairwise_coef=1.0,
    use_global=False, global_coef=0.0,
):
    agent.train()
    ep_final_sum = 0.0; ep_final_den = 0
    ep_pair_sum  = 0.0; ep_pair_den  = 0
    ep_glob_sum  = 0.0; ep_glob_den  = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        hidden = batch["hidden"].to(device)   # [B,T,H]
        mask   = batch["mask"].to(device)     # [B,T]
        reward = batch["reward"].to(device)   # [B]
        length = batch["length"].to(device)   # [B]

        B, T, H = hidden.shape
        V = agent.get_value(hidden.view(-1, H)).view(B, T)  # [B,T]

        last_idx = (length - 1).clamp(min=0)
        rows = torch.arange(B, device=device)
        last_V = V[rows, last_idx]
        final_sum = F.mse_loss(last_V, reward, reduction="sum")
        final_den = int(B)

        if use_pairwise:
            valid_pairs = (mask[:, :-1] * mask[:, 1:]).bool()
            if valid_pairs.any():
                v1 = V[:, :-1][valid_pairs]
                v2 = V[:,  1:][valid_pairs]
                pair_sum = F.mse_loss(v1, v2, reduction="sum")
                pair_den = int(valid_pairs.sum().item())
            else:
                pair_sum, pair_den = torch.tensor(0.0, device=device), 0
        else:
            pair_sum, pair_den = torch.tensor(0.0, device=device), 0

        if use_global and global_coef > 0.0:
            target = reward[:, None].expand_as(V)
            glob_err = (V - target) ** 2
            glob_sum = (glob_err * mask).sum()
            glob_den = int(mask.sum().item())
        else:
            glob_sum, glob_den = torch.tensor(0.0, device=device), 0

        batch_loss = 0.0
        if final_den > 0: batch_loss = batch_loss + (final_sum / final_den)
        if pair_den  > 0: batch_loss = batch_loss + pairwise_coef * (pair_sum / pair_den)
        if glob_den  > 0: batch_loss = batch_loss + global_coef   * (glob_sum / glob_den)

        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        ep_final_sum += float(final_sum.item()); ep_final_den += final_den
        ep_pair_sum  += float(pair_sum.item());  ep_pair_den  += pair_den
        ep_glob_sum  += float(glob_sum.item());  ep_glob_den  += glob_den

    final_avg = (ep_final_sum / ep_final_den) if ep_final_den > 0 else 0.0
    pair_avg  = (ep_pair_sum  / ep_pair_den)  if ep_pair_den  > 0 else 0.0
    glob_avg  = (ep_glob_sum  / ep_glob_den)  if ep_glob_den  > 0 else 0.0
    epoch_loss = final_avg + pairwise_coef * pair_avg + global_coef * glob_avg
    return epoch_loss, {"final": final_avg, "pair": pair_avg, "global": glob_avg}

@torch.no_grad()
# def evaluate(agent, loader, device):
#     agent.eval()
#     running_loss = 0.0
#     running_items = 0

#     for batch in tqdm(loader, desc="Eval", leave=False):
#         hidden = batch["hidden"].to(device)   # [B, T, H]
#         mask   = batch["mask"].to(device)     # [B, T]
#         reward = batch["reward"].to(device)   # [B]

#         B, T, H = hidden.shape
#         V = agent.get_value(hidden.view(-1, H)).view(B, T)  # [B, T]
#         target = reward[:, None].expand_as(V)
#         mse = (V - target) ** 2
#         loss = (mse * mask).sum() / mask.sum().clamp_min(1.0)

#         running_loss += loss.item() * B
#         running_items += B

#     return running_loss / max(running_items, 1)

def parse_args():
    p = argparse.ArgumentParser("Offline value training on shaped rewards")
    p.add_argument("--shards_dir", type=str, required=True)
    p.add_argument("--response_num", type=int, default=10)
    p.add_argument("--B", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=None)  # auto-detect if None

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_prompt", type=int, default=4000)

    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--use_pairwise", action="store_true")
    p.add_argument("--pairwise_coef", type=float, default=1.0)

    p.add_argument("--ckpt_dir", type=str, default="checkpoints_value")
    p.add_argument("--save_every", type=int, default=1)

    p.add_argument("--model_name", type=str, default="llama3-8b", help="Base model to use.")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Build a clean config
    run_cfg = {
        "model_name":args.model_name,
        "shards_dir": args.shards_dir,
        "response_num": args.response_num,
        "B": args.B,
        "beta": args.beta,
        "max_len": args.max_len,
        "hidden_size": args.hidden_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "device": str(args.device),
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "seed": args.seed,
        "max_prompt": args.max_prompt,
        "val_split": args.val_split,
        "use_pairwise": args.use_pairwise,
        "pairwise_coef": args.pairwise_coef,
    }

    wandb.init(
        project="Principal_Agent_Alignment",
        name=f"{args.model_name}_value_train_resp{args.response_num}_B{args.B}_beta{args.beta}_seed{args.seed}",
        config=run_cfg,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")

    # Dataset & split
    full_ds = FlatSampleDataset(
        shards_dir=args.shards_dir,
        response_num=args.response_num,
        B=args.B,
        beta=args.beta,
        max_len=args.max_len,
        device="cpu",
        max_prompt=args.max_prompt,
    )
    n_total = len(full_ds)
    if n_total == 0:
        raise RuntimeError("Dataset is empty. Check shards_dir and shaped reward filenames.")

    # Auto-detect hidden size H if not provided
    if args.hidden_size is None:
        probe = full_ds[0]["hidden"]  # [T,H]
        args.hidden_size = int(probe.shape[-1])
        wandb.config.update({"hidden_size": args.hidden_size}, allow_val_change=True)
        print(f"[Info] Auto-detected hidden_size = {args.hidden_size}")

    n_val = max(1, int(math.floor(args.val_split * n_total)))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )
    val_loader = DataLoader(  # kept in case you re-enable evaluate()
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )



    device = torch.device(args.device)
    agent = Agent(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    # Train
    # best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, comps = train_one_epoch(
            agent, train_loader, device, optimizer,
            use_pairwise=args.use_pairwise, pairwise_coef=args.pairwise_coef
        )
         # print(f"[Epoch {epoch:02d}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.6f} | final={comps['final']:.6f} pair={comps['pair']:.6f} global={comps['global']:.6f}")
        wandb.log({"epoch": epoch, "train/loss": train_loss,
                   "train/final": comps["final"], "train/pair": comps["pair"], "train/global": comps["global"]})

        # if epoch % args.save_every == 1 or val_loss < best_val:
        if epoch % args.save_every == 0:  # <-- fixed colon and simple policy
            ckpt_path = os.path.join(args.ckpt_dir, f"value_agent_epoch{epoch:02d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": float(train_loss),
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved checkpoint → {ckpt_path}")

if __name__ == "__main__":
    main()



