"""pipeline_common.py
====================

Everything that is IDENTICAL across the five baselines (Random, HighDegree,
CELF, Greedy, RIS) lives here. Each algorithm file only has to answer one
question: "given a graph and a budget, which seed set do I pick?"  Everything
else -- loading data, building matrices, the 10,000-simulation evaluation,
seeding, checkpointing, writing Excel, motif processing, plotting -- is shared.

The contract each algorithm implements is a single function:

    def select_seeds(ctx) -> (seed_set, seed_cost, extra_scalars)

where `ctx` (a SelectionContext) hands the algorithm the graph, budget, costs,
benefits, the adjacency matrices, and a couple of helpers. `extra_scalars` is a
dict of any algorithm-specific numbers to record (RIS uses it for KPT/Theta;
everyone else returns {}).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import ast
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import networkx as nx
import joblib
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from motif_influence import process_motif_profits


# ---------------------------------------------------------------------------
# 1. One place for every knob.
# ---------------------------------------------------------------------------
@dataclass
class Config:
    num_cpus: int = 24
    simulations: int = 10000          # final evaluation simulations
    candidate_sims: int = 100         # Monte-Carlo sims per candidate (CELF/Greedy)
    budgets: tuple = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    random_seed: int | None = None    # int -> reproducible; None -> nondeterministic
    motif_file: str = "motifs_size3.txt"
    thresholds: tuple = (1, 2, 3)
    graph_versions: dict = field(default_factory=lambda: {
        "trivalency": "facebook_trivalency.txt",
        "uniform": "facebook_uniform.txt",
        "weighted": "facebook_weighted.txt",
    })


# ---------------------------------------------------------------------------
# 2. The ICM simulator + benefit helpers (identical for all five algorithms).
#    adj/prob are the compact adjacency-list form: adj[u] lists u's out-
#    neighbours padded with -1; prob[u] the matching edge probabilities.
# ---------------------------------------------------------------------------
@njit
def simulate_icm_numba(seed_set, adj_matrix, prob_matrix, seed):
    if seed >= 0:
        np.random.seed(seed)
    n = len(adj_matrix)
    activated = np.zeros(n, dtype=np.bool_)
    newly_activated = np.zeros(n, dtype=np.bool_)
    for node in seed_set:
        activated[node] = True
        newly_activated[node] = True
    steps = 0
    while np.any(newly_activated):
        next_new = np.zeros(n, dtype=np.bool_)
        for u in range(n):
            if newly_activated[u]:
                for j in range(adj_matrix.shape[1]):
                    v = adj_matrix[u, j]
                    if v == -1:
                        break
                    if not activated[v] and np.random.rand() < prob_matrix[u, j]:
                        activated[v] = True
                        next_new[v] = True
        newly_activated = next_new
        steps += 1
    return activated, steps, np.sum(activated)


@njit
def _benefit_batch(seed_arr, adj, prob, barr, batch_size, seed):
    """Average benefit of a seed set over `batch_size` ICM runs (CELF/Greedy)."""
    if seed >= 0:
        np.random.seed(seed)
    total = 0.0
    for _ in range(batch_size):
        activated, _, _ = simulate_icm_numba(seed_arr, adj, prob, -1)
        total += np.sum(activated * barr)
    return total / batch_size


def compute_benefit(mask, benefits_array):
    return float(np.sum(mask * benefits_array))


def to_matrix(graph):
    """Compact adjacency-list matrices sized by max OUT-degree."""
    n = max(graph.nodes()) + 1
    max_deg = max((d for _, d in graph.out_degree()), default=0)
    max_deg = max(max_deg, 1)
    adj = -np.ones((n, max_deg), dtype=np.int32)
    prob = np.zeros((n, max_deg), dtype=np.float32)
    deg = np.zeros(n, dtype=np.int32)
    for u, v, data in graph.edges(data=True):
        idx = deg[u]
        adj[u, idx] = v
        prob[u, idx] = data.get("weight", 0.1)
        deg[u] += 1
    return adj, prob


def aligned_benefits_array(benefits, n):
    """Benefit vector of length n (must match the ICM activated-array length)."""
    barr = np.zeros(n, dtype=np.float32)
    for k, v in benefits.items():
        if 0 <= k < n:
            barr[k] = v
    return barr


def load_data():
    with open("cost.txt") as f:
        costs = ast.literal_eval(f.read())
    with open("benefit.txt") as f:
        benefits = ast.literal_eval(f.read())
    return ({int(k): float(v) for k, v in costs.items()},
            {int(k): float(v) for k, v in benefits.items()})


# ---------------------------------------------------------------------------
# 3. Seeding. One well-mixed, reproducible seed stream (or -1 = "don't seed").
# ---------------------------------------------------------------------------
def make_seeds(cfg, count, *key):
    if cfg.random_seed is None:
        return np.full(count, -1, dtype=np.int64)
    ss = np.random.SeedSequence([cfg.random_seed, *[int(x) for x in key]])
    return ss.generate_state(count, dtype=np.uint32).astype(np.int64)


def set_seed(cfg, seed_list):
    """A stable seed derived from a seed SET (so benefit(S) is reproducible)."""
    if cfg.random_seed is None:
        return -1
    ss = np.random.SeedSequence([cfg.random_seed, len(seed_list), *sorted(seed_list)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def benefit_batch(seed_list, adj, prob, barr, sims, cfg):
    """Picklable benefit-of-a-set helper (used by Greedy's parallel workers)."""
    if not seed_list:
        return 0.0
    arr = np.array(sorted(seed_list), dtype=np.int32)
    return float(_benefit_batch(arr, adj, prob, barr, sims, set_seed(cfg, seed_list)))


# ---------------------------------------------------------------------------
# 4. What an algorithm's select_seeds() receives.
# ---------------------------------------------------------------------------
@dataclass
class SelectionContext:
    G: object
    name: str
    model_idx: int
    budget: float
    costs: dict
    benefits: dict
    benefits_array: np.ndarray
    n: int
    adj: np.ndarray
    prob: np.ndarray
    cfg: Config
    cache: dict = field(default_factory=dict)   # per-(model,budget) benefit cache

    def make_seeds(self, count, *key):
        return make_seeds(self.cfg, count, self.model_idx, self.budget, *key)

    def benefit_of(self, seed_list):
        """Cached expected benefit of a seed set (CELF/Greedy)."""
        key = frozenset(seed_list)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        val = benefit_batch(list(seed_list), self.adj, self.prob,
                            self.benefits_array, self.cfg.candidate_sims, self.cfg)
        self.cache[key] = val
        return val


# ---------------------------------------------------------------------------
# 5. Result assembly + Excel-safe columns.
# ---------------------------------------------------------------------------
def summary_columns(algo, extra_columns=()):
    exec_key = f"{algo}_Execution_Time"
    base = ["Model", "Budget"] + list(extra_columns) + [
        "Seed_Set", "Seed_Size", "Seed_Cost", "Remaining_Budget",
        "Avg_Benefit", "Profit", "Avg_Timestep", "Avg_Activated_Nodes",
        exec_key, "Motif_Execution_Time", "Total_Execution_Time",
        "Simulations", "Threshold",
    ]
    return base


def summary_row(result, cols):
    return {k: result.get(k) for k in cols}


# ---------------------------------------------------------------------------
# 6. THE shared runner. This is the loop that used to be copy-pasted five times.
# ---------------------------------------------------------------------------
def run_pipeline(algo, select_seeds, cfg=None, extra_columns=()):
    cfg = cfg or Config()
    exec_key = f"{algo}_Execution_Time"
    full_file = f"{algo}_full_results.pkl"
    cols = summary_columns(algo, extra_columns)
    costs, benefits = load_data()

    # Resume from checkpoint (full objects, so motif processing still works).
    results, completed = [], set()
    if os.path.exists(full_file):
        try:
            results = joblib.load(full_file)
            completed = {(r["Model"], r["Budget"]) for r in results}
            print(f"📂 Resuming {algo} from {full_file} ({len(completed)} done)")
        except Exception as e:
            print(f"⚠️ Could not read checkpoint, starting fresh: {e}")

    pbar = tqdm(total=len(cfg.graph_versions) * len(cfg.budgets),
                desc=f"Processing {algo}", unit="task")

    for model_idx, (name, file) in enumerate(cfg.graph_versions.items()):
        G = nx.read_weighted_edgelist(file, create_using=nx.DiGraph(), nodetype=int)
        adj, prob = to_matrix(G)
        n = adj.shape[0]
        barr = aligned_benefits_array(benefits, n)
        log_file = f"live_log_{algo.lower()}_{name}.csv"

        for budget in cfg.budgets:
            if (name, budget) in completed:
                pbar.update(1)
                continue

            t0 = time.time()
            ctx = SelectionContext(G, name, model_idx, budget, costs, benefits,
                                   barr, n, adj, prob, cfg)
            seed_set, seed_cost, extra = select_seeds(ctx)

            # ---- the one, uniform evaluation every algorithm is judged by ----
            sim_seeds = make_seeds(cfg, cfg.simulations, model_idx, budget, 777)
            sa = np.array(seed_set, dtype=np.int32)
            sims = Parallel(n_jobs=cfg.num_cpus)(
                delayed(simulate_icm_numba)(sa, adj, prob, int(s)) for s in sim_seeds
            )
            avg_benefit = float(np.mean([compute_benefit(s[0], barr) for s in sims]))
            avg_steps = float(np.mean([s[1] for s in sims]))
            avg_activated = float(np.mean([int(s[2]) for s in sims]))
            profit = avg_benefit - seed_cost

            result = {
                "Model": name, "Budget": budget,
                "Seed_Set": str(seed_set), "Seed_Size": len(seed_set),
                "Seed_Cost": float(seed_cost),
                "Remaining_Budget": float(budget - seed_cost),
                "Avg_Benefit": avg_benefit, "Profit": float(profit),
                "Avg_Timestep": math.ceil(avg_steps),
                "Avg_Activated_Nodes": avg_activated,
                exec_key: round(time.time() - t0, 2),
                "Motif_Execution_Time": 0.0, "Total_Execution_Time": 0.0,
                "Simulations": cfg.simulations, "Threshold": 0,
                **extra,                       # e.g. RIS's KPT / Theta
                "Simulation_Results": sims,    # heavy -> pickled, never in Excel
                "Benefits": benefits,          # heavy -> needed by motif stage
            }
            results.append(result)
            completed.add((name, budget))
            pbar.update(1)

            write_live_log([summary_row(result, cols)], cols, log_file)
            joblib.dump(results, full_file, compress=3)   # resumable checkpoint

        model_rows = [summary_row(r, cols) for r in results if r["Model"] == name]
        if model_rows:
            pd.DataFrame(model_rows, columns=cols).to_excel(
                f"{algo}_Results_{name}.xlsx", index=False)

    pbar.close()
    return results


def write_live_log(rows, columns, log_file):
    df = pd.DataFrame(rows, columns=columns)
    header = not os.path.exists(log_file)
    df.to_csv(log_file, mode="a", header=header, index=False)


# ---------------------------------------------------------------------------
# 7. The shared "main": summary Excel + plots + motif workbook.
# ---------------------------------------------------------------------------
def finalize(results, algo, cfg=None, plot_specs=None, extra_columns=()):
    cfg = cfg or Config()
    cols = summary_columns(algo, extra_columns)
    summary_df = pd.DataFrame([summary_row(r, cols) for r in results], columns=cols)
    summary_df.to_excel(f"{algo}_Final_Results.xlsx", index=False)
    print(f"✅ Summary saved to {algo}_Final_Results.xlsx")

    plot_specs = plot_specs or [("Profit", "^", "Profit"),
                                ("Avg_Activated_Nodes", "o", "Avg Activated Nodes")]
    for model in summary_df["Model"].unique():
        subset = summary_df[summary_df["Model"] == model].sort_values("Budget")
        for col, marker, ylabel in plot_specs:
            plt.figure()
            plt.plot(subset["Budget"], subset[col], marker=marker)
            plt.title(f"{ylabel} vs Budget ({model})")
            plt.xlabel("Budget")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.savefig(f"{col}_vs_Budget_{model}_{algo}.png")
            plt.close()

    threshold_results = {}
    for threshold in cfg.thresholds:
        print(f"🎯 Motif profits for threshold = {threshold}")
        threshold_results[threshold] = process_motif_profits(
            results, motif_file=cfg.motif_file, threshold=threshold)

    with pd.ExcelWriter(f"{algo}_Motif_Results_DualThreshold.xlsx") as writer:
        for threshold, res in threshold_results.items():
            pd.DataFrame(res).to_excel(writer, sheet_name=f"Threshold_{threshold}",
                                       index=False)
    print(f"✅ Motif results saved to {algo}_Motif_Results_DualThreshold.xlsx")
    return threshold_results
