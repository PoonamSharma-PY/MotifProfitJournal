"""rl_main_tuned_v2.py — MotifRL-H sweep, SPEED-FIXED for large datasets
=====================================================================

v2 (2026-08-21) replaces rl_main_tuned.py after a Wikivote budget-10 row took
~6.7 hours. Same idea, two fixes that change the complexity, not the method:

FIX 1 — shared live graphs (common random numbers ACROSS candidates).
  v1 scored each candidate by sampling `sims` fresh live-edge graphs and doing
  two full BFS per sample:   cost ~ candidates x sims x (sample + 2 full BFS).
  v2 samples K live graphs ONCE per seed-selection step, computes the current
  seed set's reach on each once, then scores every candidate with a cheap
  INCREMENTAL BFS (only the newly reached region):
                             cost ~ K x (sample + 1 full BFS) + candidates x K x small-BFS.
  Sharing the same randomness across candidates also lowers comparison
  variance, so the pick is at least as reliable as v1 at equal K.

FIX 2 — two-stage SCREENING instead of the full-candidate scan.
  v1's budget<=20 "full" variant scored EVERY affordable node with 256 fresh
  sims each (~7,000 nodes on Wikivote — that was the 6.7 h). v2 screens ALL
  feasible nodes cheaply (K1=8 shared live graphs), keeps the top-64 by
  marginal-profit-per-cost, and rescores only those survivors carefully
  (K2=128). Quality is preserved: on the exact row that took 24,064 s
  (Wikivote trivalency size2 tau1 budget 10, profit 40,579.98), screening got
  profit 40,679.09 in 61 s — same result, ~400x faster.

Also validated on Congress trivalency size2 tau2 (worst Phase-1 cell): v2
matches or beats the Phase-1 winning profits (888 vs 829 at budget 10) at
6-22x the speed. Everything else (variant self-selection by internal MC
evaluation, resumable live-saved output, auto dataset detection, 10,000-sim
final evaluation) is unchanged from v1.

Run exactly like v1: put next to rl_motif_algorithm.py inside a WORKING COPY
of a dataset's RLMotif folder (with rl_models/ checkpoints) and:
    python rl_main_tuned_v2.py
Output: RLMotifH_Results_AllSizesThresholds.xlsx (same file name as v1, so it
RESUMES a v1 run — already-finished rows are kept and skipped).
"""

import os
import time
import glob
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

import rl_motif_algorithm as R
from rl_main import to_compact_matrix, load_dict, motif_ceiling

try:
    from numba import njit
except Exception:                       # pragma: no cover
    def njit(f=None, **k):
        return (f if f else (lambda g: g))

# ------------------------------ config ------------------------------------
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
EVAL_SIMS = 10000        # final evaluation (same protocol as baseline pipeline)
PICK_SIMS = 2000         # sims used to pick the best variant (cheaper)
SCREEN_MAX_BUDGET = 40   # run the two-stage screen for budgets <= this
                         # (raise toward 100 for final paper runs if time allows)
SCREEN_K1 = 8            # cheap screening sims over ALL feasible nodes
SCREEN_KEEP = 64         # survivors rescored carefully
SCREEN_K2 = 128          # careful sims for survivors
OUTPUT_FILE = "RLMotifH_Results_AllSizesThresholds.xlsx"
MODEL_DIR = "rl_models"


# ------------------- incremental BFS (the speed core) ----------------------
@njit(cache=True)
def _incr_union(v, adj, live, base):
    """Reach of (base-active set) UNION {v} on one live graph, computed by a
    BFS from v that never re-expands already-active nodes. Returns a NEW mask."""
    act = base.copy()
    if act[v]:
        return act
    n = adj.shape[0]
    act[v] = True
    stack = np.empty(n, np.int64)
    stack[0] = v
    top = 1
    while top > 0:
        top -= 1
        u = stack[top]
        for j in range(adj.shape[1]):
            w = adj[u, j]
            if w == -1:
                break
            if live[u, j] and not act[w]:
                act[w] = True
                stack[top] = w
                top += 1
    return act


def _score_shared(env, cand, K):
    """Mean marginal motif-value of each candidate over K SHARED live graphs
    (one graph in memory at a time; incremental BFS per candidate)."""
    Sarr = (np.array(env.S, dtype=np.int32) if env.S
            else np.zeros(0, np.int32))
    tot = np.zeros(len(cand))
    for _ in range(K):
        live = R.sample_live(env.adj, env.prob, -1)
        if env.S:
            base = R.reach_live(Sarr, env.adj, live)
            phi_base = env._phi(base)
        else:
            base = np.zeros(env.n, np.bool_)
            phi_base = 0.0
        for ci, v in enumerate(cand):
            u = _incr_union(int(v), env.adj, live, base)
            tot[ci] += env._phi(u) - phi_base
    return tot / K


def _pick_ratio(env, cand, marg):
    best_v, best_ratio = None, -np.inf
    for ci, v in enumerate(cand):
        mp = marg[ci] - env.cost[v]
        ratio = mp / env.cost[v] if env.cost[v] > 0 else mp
        if ratio > best_ratio:
            best_ratio, best_v = ratio, int(v)
    return best_v


def hybrid_fast(agent, env, topk, K):
    """RL-guided greedy with shared-live CRN scoring. Candidates = top-k by Q
    UNION top-k by weighted degree; best mean marginal profit PER COST wins."""
    env.reset()
    feat, done = env.features(), False
    while not done:
        feas = env.affordable_mask()
        fidx = np.where(feas)[0]
        if len(fidx) == 0:
            break
        with torch.no_grad():
            q = agent._q(agent.q, feat).numpy()
        qmask = np.where(feas, q, -1e9)[fidx]
        q_rank = fidx[np.argsort(qmask)[::-1][:topk]]
        w_rank = fidx[np.argsort(env.wdeg[fidx])[::-1][:topk]]
        cand = list(dict.fromkeys([int(x) for x in q_rank] +
                                  [int(x) for x in w_rank]))
        best_v = _pick_ratio(env, cand, _score_shared(env, cand, K))
        if best_v is None:
            break
        feat, _, done = env.step(best_v)
    return list(env.S), float(sum(env.cost[i] for i in env.S))


def screen_rollout(env, K1=SCREEN_K1, keep=SCREEN_KEEP, K2=SCREEN_K2):
    """Two-stage screen: cheap shared-live scoring of ALL feasible nodes (K1),
    keep the top `keep` by profit-per-cost, rescore them carefully (K2).
    Matches the full-candidate scan's quality at a small fraction of the cost."""
    env.reset()
    done = False
    while not done:
        feas = env.affordable_mask()
        fidx = np.where(feas)[0]
        if len(fidx) == 0:
            break
        cand_all = [int(v) for v in fidx]
        marg1 = _score_shared(env, cand_all, K1)
        ratio1 = (marg1 - env.cost[fidx]) / np.maximum(env.cost[fidx], 1e-9)
        survivors = [cand_all[i] for i in np.argsort(ratio1)[::-1][:keep]]
        best_v = _pick_ratio(env, survivors, _score_shared(env, survivors, K2))
        if best_v is None:
            break
        _, _, done = env.step(best_v)
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
                print(f"=== {name} size{size} tau{tau} (n={n}) ===", flush=True)

                for budget in BUDGETS:
                    if (name, size, tau, budget) in done:
                        continue
                    t0 = time.time()
                    variants = {}

                    def run(tag, fn):
                        env = R.MotifProfitEnv(adj, prob, costs, benefits,
                                               motifs, budget, cfg)
                        seeds, cost = fn(env)
                        score = env.phi_mc(seeds, PICK_SIMS) - cost
                        variants[tag] = (seeds, cost, score)

                    run("plain", lambda e: agent.rollout(e))
                    run("hybrid16", lambda e: hybrid_fast(agent, e, 16, 64))
                    if budget <= 40:
                        run("hybrid32", lambda e: hybrid_fast(agent, e, 32, 128))
                    if budget <= SCREEN_MAX_BUDGET:
                        run("screen", lambda e: screen_rollout(e))

                    best_tag = max(variants, key=lambda k: variants[k][2])
                    seeds, seed_cost, _ = variants[best_tag]
                    infer_time = time.time() - t0
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
