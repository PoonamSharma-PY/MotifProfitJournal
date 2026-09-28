"""rl_main.py  (sweep + full logging)
==================================

Runs the RL (S2V-DQN) motif-profit method over EVERY motif size and threshold,
for every graph version and budget, and records a full row per run:

  Model, Motif_Size, Motif_File, Threshold, Num_Motifs, Budget,
  Seed_Set, Seed_Size, Seed_Cost, Remaining_Budget,
  Motif_Value, Motif_Profit, Max_Motif_Value, Profit_Fraction,
  Train_Time_s (one-time, per size/tau/graph), Inference_Time_s (per-budget query),
  + every RL hyperparameter (cfg_*) so each row is fully reproducible.

Because a motif is "influenced" at >= threshold activated nodes, the reward --
and therefore the trained agent -- depends on (motif size, threshold). So a
fresh agent is trained for each (size, threshold, graph). Results are written
incrementally, so a crash mid-sweep keeps everything computed so far.

Reads your existing files only (nothing is generated here):
  graph  : facebook_*.txt ("u v w")   cost/benefit: dict literals
  motifs : motifs_size{k}.txt (one Python set per line)
"""

import os
import ast
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

import rl_motif_algorithm as R

# ------------------------------ config ------------------------------------
GRAPH_VERSIONS = {
    "trivalency": "facebook_trivalency.txt",
    "uniform": "facebook_uniform.txt",
    "weighted": "facebook_weighted.txt",
}
COST_FILE = "cost.txt"
BENEFIT_FILE = "benefit.txt"

# motif size -> file. Only sizes whose file exists are used.
MOTIF_FILES = {
    2: "motifs_size2.txt",
    3: "motifs_size3.txt",
    4: "motifs_size4.txt",
}
# thresholds per size. Empty dict -> auto = 1..size. Override any size here,
# e.g. THRESHOLDS_PER_SIZE = {4: [1, 2, 3]} to skip tau=4 for size-4 motifs.
THRESHOLDS_PER_SIZE = {}

BUDGETS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
OBJECTIVE = "profit"                   # "profit" or "count"
REACH_HOPS = 4
OUTPUT_FILE = "RLMotif_Results_AllSizesThresholds.xlsx"
MODEL_DIR = "rl_models"                 # trained agents saved here (one per size/tau/graph)


# --------------------------- IO helpers -----------------------------------
def load_dict(path):
    with open(path) as f:
        raw = ast.literal_eval(f.read().strip())
    return {int(k): float(v) for k, v in raw.items()}


def to_compact_matrix(edgelist_path):
    """Weighted directed edgelist -> compact out-adjacency (adj, prob), n."""
    src, dst, wt, nodes = [], [], [], set()
    with open(edgelist_path) as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#%/;":
                continue
            p = s.replace(",", " ").split()
            if len(p) < 2:
                continue
            try:
                u, v = int(p[0]), int(p[1])
            except ValueError:
                continue
            w = float(p[2]) if len(p) >= 3 else 0.1
            nodes.add(u)
            nodes.add(v)
            if u == v:
                continue
            src.append(u)
            dst.append(v)
            wt.append(w)
    n = (max(nodes) + 1) if nodes else 1
    outdeg = np.zeros(n, dtype=np.int64)
    for u in src:
        outdeg[u] += 1
    max_deg = max(int(outdeg.max()), 1)
    adj = -np.ones((n, max_deg), dtype=np.int32)
    prob = np.zeros((n, max_deg), dtype=np.float32)
    fill = np.zeros(n, dtype=np.int64)
    for u, v, w in zip(src, dst, wt):
        adj[u, fill[u]] = v
        prob[u, fill[u]] = w
        fill[u] += 1
    return adj, prob, n


def motif_ceiling(motifs, benefits):
    """Max attainable motif value = benefit over the union of all motif nodes."""
    if not motifs:
        return 0.0
    nodes = set().union(*motifs)
    return float(sum(benefits.get(i, 0.0) for i in nodes))


def main():
    t0 = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    costs = load_dict(COST_FILE)
    benefits = load_dict(BENEFIT_FILE)

    # ---- resume: reload any results already computed, so a restart continues ----
    rows, done = [], set()
    if os.path.exists(OUTPUT_FILE):
        prev = pd.read_excel(OUTPUT_FILE)
        rows = prev.to_dict("records")
        done = {(r["Model"], int(r["Motif_Size"]), int(r["Threshold"]), int(r["Budget"]))
                for r in rows}
        print(f"Resuming from {OUTPUT_FILE}: {len(rows)} rows already done.")

    # which (size, file) pairs actually exist
    sizes = [(k, f) for k, f in sorted(MOTIF_FILES.items()) if os.path.exists(f)]
    if not sizes:
        raise FileNotFoundError("None of the motif files in MOTIF_FILES were found.")
    print("Motif files found:", ", ".join(f for _, f in sizes))

    # graph matrices once per graph (reused across every size/threshold)
    graphs = {name: to_compact_matrix(gfile) for name, gfile in GRAPH_VERSIONS.items()}

    for size, mfile in sizes:
        motifs = R.load_motifs(mfile)
        cap = motif_ceiling(motifs, benefits)
        thresholds = THRESHOLDS_PER_SIZE.get(size, list(range(1, size + 1)))
        thresholds = [t for t in thresholds if 1 <= t <= size]   # tau>size is vacuous
        print(f"\n########## motif size {size} ({mfile}): "
              f"{len(motifs)} motifs, ceiling {cap:.2f}, thresholds {thresholds} ##########")

        for threshold in thresholds:
            cfg = R.RLConfig(
                threshold=threshold,
                objective=OBJECTIVE,
                reach_hops=REACH_HOPS,
                episodes=400,       # raise for real graphs; lower to speed the sweep
                reward_sims=32,
                eval_sims=1000,
                emb_dim=64,
                t_layers=3,
                verbose=True,
            )
            cfg_cols = {f"cfg_{k}": v for k, v in asdict(cfg).items()}

            for name, (adj, prob, n) in graphs.items():
                # skip a fully-finished combo without even loading the model
                if all((name, size, threshold, b) in done for b in BUDGETS):
                    print(f"=== size {size} | tau {threshold} | {name} : already done, skip ===")
                    continue

                print(f"\n=== size {size} | tau {threshold} | {name} ===")
                model_path = os.path.join(
                    MODEL_DIR, f"rl_{name}_size{size}_tau{threshold}.pt")
                train_env = R.MotifProfitEnv(adj, prob, costs, benefits, motifs,
                                             budget=max(BUDGETS), cfg=cfg)
                agent = R.build_agent(adj, prob, train_env, cfg)

                # reuse a saved model if present, else train once and save it
                if os.path.exists(model_path):
                    try:
                        ckpt = agent.load(model_path)
                        train_time = ckpt.get("train_time", float("nan"))
                        print(f"    loaded trained model {model_path} (skipped training)")
                    except Exception as e:
                        print(f"    model load failed ({e}); retraining")
                        t_train = time.time(); agent.train(train_env)
                        train_time = round(time.time() - t_train, 1)
                        agent.save(model_path, train_time=train_time, cfg=asdict(cfg))
                else:
                    t_train = time.time(); agent.train(train_env)
                    train_time = round(time.time() - t_train, 1)
                    agent.save(model_path, train_time=train_time, cfg=asdict(cfg))
                    print(f"    saved trained model -> {model_path}")

                for budget in BUDGETS:
                    if (name, size, threshold, budget) in done:
                        continue                              # already computed, skip
                    env = R.MotifProfitEnv(adj, prob, costs, benefits, motifs, budget, cfg)
                    t_infer = time.time()                     # time ONLY the seed selection
                    if cfg.hybrid_topk:                       # RL-guided greedy inference
                        seed_set, seed_cost = agent.rollout_hybrid(
                            env, cfg.hybrid_topk, cfg.hybrid_sims)
                    else:
                        seed_set, seed_cost = agent.rollout(env)
                    infer_time = time.time() - t_infer        # inference (query) time
                    phi = env.phi_mc(seed_set, cfg.eval_sims)     # evaluation, not inference
                    profit = phi - seed_cost
                    row = {
                        "Model": name,
                        "Motif_Size": size,
                        "Motif_File": mfile,
                        "Threshold": threshold,
                        "Num_Motifs": len(motifs),
                        "Objective": OBJECTIVE,
                        "Budget": budget,
                        "Seed_Set": str(seed_set),
                        "Seed_Size": len(seed_set),
                        "Seed_Cost": round(seed_cost, 2),
                        "Remaining_Budget": round(budget - seed_cost, 2),
                        "Motif_Value": round(phi, 2),
                        "Motif_Profit": round(profit, 2),
                        "Max_Motif_Value": round(cap, 2),
                        "Profit_Fraction": round(profit / cap, 4) if cap > 0 else 0.0,
                        "Train_Time_s": train_time,           # ONE-TIME cost per (size,tau,graph)
                        "Inference_Time_s": round(infer_time, 4),  # per-budget query cost
                        "Model_Path": model_path,
                    }
                    row.update(cfg_cols)          # RL hyperparameters, one column each
                    rows.append(row)
                    done.add((name, size, threshold, budget))
                    # LIVE save after EVERY row -> a crash loses at most one budget
                    pd.DataFrame(rows).to_excel(OUTPUT_FILE, index=False)
                    print(f"  budget {budget:3d} | seeds {len(seed_set):2d} | "
                          f"cost {seed_cost:6.2f} | remaining {budget - seed_cost:6.2f} | "
                          f"profit {profit:9.2f} | {100*row['Profit_Fraction']:.1f}% of ceiling")

    print(f"\n✅ Saved {len(rows)} rows to {OUTPUT_FILE}  "
          f"({round(time.time() - t0, 2)}s total)")


if __name__ == "__main__":
    main()
 