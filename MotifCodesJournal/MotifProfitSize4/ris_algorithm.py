"""RIS with KPT/Theta and profit-aware RR-set selection.

RIS is the one algorithm that needs a *dense* graph (its RR-set builder walks
predecessors), and it reports extra numbers (KPT, Theta). All of that is
confined to this file's select_seeds and the RIS-local njit helpers. The final
seed set is still evaluated by the shared, uniform ICM in pipeline_common, so
RIS is judged the same way as every other baseline.
"""
import math
import numpy as np
import networkx as nx
from numba import njit
from joblib import Parallel, delayed

import pipeline_common as pc

ALGO = "RIS"
EXTRA_COLUMNS = ("KPT", "Theta")
EPSILON = 0.3
L = 1


def _dense_matrix(G):
    n = max(G.nodes()) + 1
    adj = np.zeros((n, n), dtype=np.int32)
    prob = np.zeros((n, n), dtype=np.float64)
    for u, v, d in G.edges(data=True):
        adj[u][v] = 1
        prob[u][v] = d.get("weight", 0.1)
    return adj, prob


@njit
def generate_rr_set(start_node, adj, prob, seed):
    if seed >= 0:
        np.random.seed(seed)
    n = adj.shape[0]
    rr = np.zeros(n, dtype=np.bool_)
    queue = [start_node]
    rr[start_node] = True
    while queue:
        cur = queue.pop()
        for pred in range(n):
            if adj[pred, cur] and not rr[pred]:
                if np.random.rand() < prob[pred, cur]:   # same unrounded coin as the ICM
                    rr[pred] = True
                    queue.append(pred)
    return rr


def _batch_rr(nodes, seeds, adj, prob):
    return [generate_rr_set(int(nodes[i]), adj, prob, int(seeds[i]))
            for i in range(len(nodes))]


def _generate_rr_sets(start_nodes, seeds, adj, prob, cfg, batch_size=20):
    batches = [(start_nodes[i:i + batch_size], seeds[i:i + batch_size])
               for i in range(0, len(start_nodes), batch_size)]
    out = Parallel(n_jobs=cfg.num_cpus)(          # process backend -> independent RNG
        delayed(_batch_rr)(nb, sb, adj, prob) for nb, sb in batches)
    return [rr for b in out for rr in b]


def _log_binomial(n, k):
    if k < 0 or k > n:
        return -float("inf")
    if k == 0 or k == n:
        return 0.0
    r = 0.0
    for i in range(1, k + 1):
        r += math.log(n - i + 1) - math.log(i)
    return r


def _kpt(ctx, k, adj, prob, indeg, probs):
    n, m = ctx.n, ctx.G.number_of_edges()
    eta = n
    if eta < 4:
        return 1.0
    for i in range(1, int(math.log2(eta)) + 1):
        c_i = max(int((6 * L * math.log(eta) + 6 * math.log(math.log2(eta))) * (2 ** i)), 1)
        seeds = ctx.make_seeds(c_i, 100 + i)
        s = 0.0
        for t in range(c_i):
            node = int(np.random.choice(n, p=probs))
            rr = generate_rr_set(node, adj, prob, int(seeds[t]))
            wr = int(indeg[np.where(rr)[0]].sum())
            s += (1 - (1 - wr / m) ** k) if m > 0 else 0
        if s / c_i > 1 / (2 ** i):
            return max((eta * s) / (2 * c_i), 1.0)
    return 1.0


def _greedy_over_rr(rr_sets, cost_dict, budget, n):
    if not rr_sets:
        return []
    covered = np.zeros(len(rr_sets), dtype=np.bool_)
    seed_set, seed_lookup, remaining = [], set(), budget
    while True:
        score = np.zeros(n)
        for i, rr in enumerate(rr_sets):
            if not covered[i]:
                for node in np.where(rr)[0]:
                    score[node] += 1
        ratios = [(i, score[i] / cost_dict.get(i, 1e9)) for i in range(n)
                  if i not in seed_lookup and cost_dict.get(i, 1e9) <= remaining]
        if not ratios:
            break
        sel, best = max(ratios, key=lambda x: x[1])
        if best <= 0:
            break
        seed_set.append(sel)
        seed_lookup.add(sel)
        remaining -= cost_dict.get(sel, 0)
        for i, rr in enumerate(rr_sets):
            if rr[sel]:
                covered[i] = True
    return seed_set


def select_seeds(ctx):
    if ctx.cfg.random_seed is not None:
        np.random.seed(int(ctx.make_seeds(1, 0)[0]))     # for np.random.choice

    adj, prob = _dense_matrix(ctx.G)
    indeg = adj.sum(axis=0)
    n = ctx.n
    positive = [v for v in ctx.costs.values() if v > 0]
    k = max(1, int(ctx.budget / (min(positive) if positive else 1.0)))

    ratios = np.array([ctx.benefits.get(v, 0.0) / max(ctx.costs.get(v, 1e-5), 1e-5)
                       for v in range(n)])
    probs = ratios / ratios.sum() if ratios.sum() > 0 else np.full(n, 1 / n)

    kpt = _kpt(ctx, k, adj, prob, indeg, probs)
    theta = max(int(((8 + 2 * EPSILON) * n *
                     (L * math.log(n) + _log_binomial(n, k) + math.log(2)))
                    / (kpt * EPSILON ** 2)), 1)
    print(f"Computed theta: {theta} RR sets  (KPT={kpt:.4f})")

    start_nodes = np.random.choice(n, size=theta, p=probs)
    rr_seeds = ctx.make_seeds(theta, 999)
    rr_sets = _generate_rr_sets(start_nodes, rr_seeds, adj, prob, ctx.cfg)

    seed_set = _greedy_over_rr(rr_sets, ctx.costs, ctx.budget, n)
    seed_cost = sum(ctx.costs.get(i, 0) for i in seed_set)
    return seed_set, seed_cost, {"KPT": round(kpt, 4), "Theta": theta}


def run_ris(cfg=None):
    return pc.run_pipeline(ALGO, select_seeds, cfg, extra_columns=EXTRA_COLUMNS)
