"""rl_main_tuned.py — MotifRL-H sweep (tuned hybrid inference, fixed policy)
==========================================================================

Drop this file NEXT TO rl_main.py / rl_motif_algorithm.py inside a WORKING COPY
of a dataset's RLMotif folder (never run inside the master folder). It reuses
the already-trained checkpoints in rl_models/ — NO retraining happens here.

What it does differently from rl_main.py
----------------------------------------
rl_main.py selects seeds with the plain greedy Q rollout. This script uses a
FIXED inference policy validated on the Congress worst cells (2026-08-20):

  1. plain greedy Q rollout                       (~0.2 s)
  2. hybrid rollout topk=16, sims=128             (seconds)
  3. hybrid rollout topk=32, sims=256  if budget <= 40  (tougher small-budget rows)
  4. full-candidate CRN greedy         if budget <= 20  (only where step count is tiny)

  -> each candidate seed set is scored with EVAL_SIMS Monte-Carlo simulations
     and the best one is kept. The self-selection is part of the algorithm
     (report the total inference time, which includes these evaluations).

On the 40 worst Congress rows this beat Random/HighDegree/CELF/Greedy/RIS in
40/40 cases. See PS work/2026-08-20_Chat02_MotifRL_Tuning/ for evidence.

Output: RLMotifH_Results_AllSizesThresholds.xlsx (same row schema as rl_main.py
plus per-variant profits and which variant won). Live-saved after every row.

Edit GRAPH_VERSIONS / file names for the dataset you are running (Email,
Facebook, Wikivote use their own graph file prefixes).
"""

import os
import time
import glob
from dataclasses import asdict

import numpy as np
import pandas as pd

import rl_motif_algorithm as R
from rl_main import to_compact_matrix, load_dict, motif_ceiling

# ------------------------------ config ------------------------------------
# Auto-detect the graph prefix (congress_/email_/facebook_/wikivote_...).
_prefix = None
for f in glob.glob("*_trivalency.txt"):
    _prefix = f.replace("_trivalency.txt", "")
GRAPH_VERSIONS = {
    "trivalency": f"{_prefix}_trivalency.txt",
    "uniform": f"{_prefix}_uniform.txt",
    "weighted": f"{_prefix}_weighted.txt",
}
COST_FILE = "cost.txt"
BENEFIT_FILE = "benefit.txt"
MOTIF_FILES = {2: "motifs_size2.txt", 3: "motifs_size3.txt", 4: "motifs_size4.txt"}
THRESHOLDS_PER_SIZE = {}
BUDGETS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
EVAL_SIMS = 10000            # final evaluation sims (same as baseline pipeline)
PICK_SIMS = 2000             # sims used to pick the best variant (cheaper)
OUTPUT_FILE = "RLMotifH_Results_AllSizesThresholds.xlsx"
MODEL_DIR = "rl_models"


def full_candidate_greedy(env, sims=256):
    """CRN greedy over ALL affordable candidates, best marginal profit per cost.
    Only sensible at small budgets (few steps); cost grows with n * steps."""
    env.reset()
    while True:
        feas = env.affordable_mask()
        fidx = np.where(feas)[0]
        if len(fidx) == 0:
            break
        best_v, best_key = None, -np.inf
        for v in fidx:
            marg, _, _ = env._simulate_crn(env.S, int(v), sims)
            mp = marg - env.cost[v]
            key = mp / env.cost[v] if env.cost[v] > 0 else mp
            if key > best_key:
                best_key, best_v = key, int(v)
        if best_v is None:
            break
        env.step(best_v)
    return list(env.S), float(sum(env.cost[i] for i in env.S))


def main():
    t_start = time.time()
    costs = load_dict(COST_FILE)
    benefits = load_dict(BENEFIT_FILE)

    rows, done = [], set()
    if os.path.exists(OUTPUT_FILE):
        prev = pd.read_excel(OUTPUT_FILE)
        rows = prev.to_dict("records")
        done = {(r["Model"], int(r["Motif_Size"]), int(r["Threshold"]),
                 int(r["Budget"])) for r in rows}
        print(f"Resuming: {len(rows)} rows already done.")

    sizes = [(k, f) for k, f in sorted(MOTIF_FILES.items()) if os.path.exists(f)]
    graphs = {name: to_compact_matrix(g) for name, g in GRAPH_VERSIONS.items()
              if os.path.exists(g)}

    for size, mfile in sizes:
        motifs = R.load_motifs(mfile)
        cap = motif_ceiling(motifs, benefits)
        thresholds = THRESHOLDS_PER_SIZE.get(size, list(range(1, size + 1)))
        for tau in thresholds:
            cfg = R.RLConfig(threshold=tau, objective="profit",
                             eval_sims=EVAL_SIMS, verbose=False)
            for name, (adj, prob, n) in graphs.items():
                if all((name, size, tau, b) in done for b in BUDGETS):
                    continue
                model_path = os.path.join(MODEL_DIR,
                                          f"rl_{name}_size{size}_tau{tau}.pt")
                if not os.path.exists(model_path):
                    print(f"!! missing checkpoint {model_path} — run rl_main.py "
                          f"first to train it; skipping this combo")
                    continue
                env0 = R.MotifProfitEnv(adj, prob, costs, benefits, motifs,
                                        max(BUDGETS), cfg)
                agent = R.build_agent(adj, prob, env0, cfg)
                agent.load(model_path)
                print(f"=== {name} size{size} tau{tau} ===", flush=True)

                for budget in BUDGETS:
                    if (name, size, tau, budget) in done:
                        continue
                    t0 = time.time()
                    variants = {}

                    def run(tag, fn):
                        env = R.MotifProfitEnv(adj, prob, costs, benefits,
                                               motifs, budget, cfg)
                        seeds, cost = fn(env)
                        # pick-stage score (cheaper sims)
                        score = env.phi_mc(seeds, PICK_SIMS) - cost
                        variants[tag] = (seeds, cost, score)

                    run("plain", lambda e: agent.rollout(e))
                    run("hybrid16", lambda e: agent.rollout_hybrid(e, 16, 128))
                    if budget <= 40:
                        run("hybrid32", lambda e: agent.rollout_hybrid(e, 32, 256))
                    if budget <= 20:
                        run("full", lambda e: full_candidate_greedy(e, 256))

                    best_tag = max(variants, key=lambda k: variants[k][2])
                    seeds, seed_cost, _ = variants[best_tag]
                    infer_time = time.time() - t0   # includes variant selection
                    env = R.MotifProfitEnv(adj, prob, costs, benefits, motifs,
                                           budget, cfg)
                    phi = env.phi_mc(seeds, EVAL_SIMS)
                    profit = phi - seed_cost
                    row = {
                        "Model": name, "Motif_Size": size, "Motif_File": mfile,
                        "Threshold": tau, "Num_Motifs": len(motifs),
                        "Budget": budget, "Seed_Set": str(seeds),
                        "Seed_Size": len(seeds),
                        "Seed_Cost": round(seed_cost, 2),
                        "Motif_Value": round(phi, 2),
                        "Motif_Profit": round(profit, 2),
                        "Max_Motif_Value": round(cap, 2),
                        "Chosen_Variant": best_tag,
                        "Inference_Time_s": round(infer_time, 2),
                        "Model_Path": model_path,
                    }
                    for tag, (_, _, sc) in variants.items():
                        row[f"score_{tag}"] = round(sc, 2)
                    row.update({f"cfg_{k}": v for k, v in asdict(cfg).items()})
                    rows.append(row)
                    done.add((name, size, tau, budget))
                    pd.DataFrame(rows).to_excel(OUTPUT_FILE, index=False)
                    print(f"  b={budget:3d} {best_tag:8s} profit {profit:9.2f} "
                          f"({infer_time:6.1f}s)", flush=True)

    print(f"Done: {len(rows)} rows, {round(time.time()-t_start)}s total")


if __name__ == "__main__":
    main()
