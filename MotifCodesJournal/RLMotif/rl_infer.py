"""rl_infer.py
===========

Load a trained RL model saved by rl_main.py and select a seed set for ANY budget,
with NO retraining. The model file already stores the exact config it was trained
with, so the network is rebuilt to match automatically.

Set the graph/motif/model below and the budget you want, then run:
    python rl_infer.py
"""

import os
import torch
import numpy as np

import rl_motif_algorithm as R
from rl_main import load_dict, to_compact_matrix   # reuse the same loaders

# ------------------------------ what to infer ------------------------------
NAME = "trivalency"                       # which graph version the model was trained on
GRAPH_FILE = "facebook_trivalency.txt"
COST_FILE = "cost.txt"
BENEFIT_FILE = "benefit.txt"
MOTIF_FILE = "motifs_size3.txt"
SIZE = 3
THRESHOLD = 2
MODEL_DIR = "rl_models"
BUDGET = 35                               # <-- any budget, need not be one you trained/logged
EVAL_SIMS = 1000                          # sims for the reported profit


def main():
    model_path = os.path.join(MODEL_DIR, f"rl_{NAME}_size{SIZE}_tau{THRESHOLD}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model at {model_path}. Run rl_main.py first.")

    # rebuild the EXACT config the model was trained with (stored in the checkpoint)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("cfg")
    if saved_cfg is not None:
        cfg = R.RLConfig(**saved_cfg)
    else:  # fallback if an older model without stored cfg
        cfg = R.RLConfig(threshold=THRESHOLD, reach_hops=4, emb_dim=64, t_layers=3,
                         verbose=False)

    costs = load_dict(COST_FILE)
    benefits = load_dict(BENEFIT_FILE)
    motifs = R.load_motifs(MOTIF_FILE)
    adj, prob, n = to_compact_matrix(GRAPH_FILE)

    env = R.MotifProfitEnv(adj, prob, costs, benefits, motifs, BUDGET, cfg)
    agent = R.build_agent(adj, prob, env, cfg)
    agent.load(model_path)                    # load the trained weights

    if cfg.hybrid_topk:
        seed_set, seed_cost = agent.rollout_hybrid(env, cfg.hybrid_topk, cfg.hybrid_sims)
    else:
        seed_set, seed_cost = agent.rollout(env)

    phi = env.phi_mc(seed_set, EVAL_SIMS)
    profit = phi - seed_cost
    print(f"model      : {model_path}")
    print(f"budget     : {BUDGET}")
    print(f"seed set   : {seed_set}")
    print(f"seed size  : {len(seed_set)}   cost {seed_cost:.2f}   remaining {BUDGET - seed_cost:.2f}")
    print(f"motif value: {phi:.2f}")
    print(f"profit     : {profit:.2f}")


if __name__ == "__main__":
    main()
