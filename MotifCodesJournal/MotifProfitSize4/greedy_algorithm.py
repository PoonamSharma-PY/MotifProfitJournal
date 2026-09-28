"""Naive greedy: every round, re-evaluate all affordable candidates in parallel
and add the best profit-per-cost node. Selection only; the rest is shared.
"""
import pipeline_common as pc
from joblib import Parallel, delayed

ALGO = "Greedy"
BATCH_SIZE = 100


def _evaluate_batch(candidates, seed_set, current_profit, total_cost,
                    adj, prob, barr, cost_dict, sims, cfg):
    out = []
    for node in candidates:
        b = pc.benefit_batch(seed_set + [node], adj, prob, barr, sims, cfg)
        cost_node = cost_dict[node]
        profit = b - (total_cost + cost_node)
        gain = (profit - current_profit) / cost_node if cost_node > 0 else 0.0
        out.append((node, gain))
    return out


def select_seeds(ctx):
    seed_set, seed_lookup, total_cost, remaining = [], set(), 0.0, ctx.budget
    sims = ctx.cfg.candidate_sims

    while remaining > 0:
        candidates = [i for i in range(ctx.n)
                      if i not in seed_lookup
                      and ctx.costs.get(i, float("inf")) <= remaining]
        if not candidates:
            break
        current_benefit = ctx.benefit_of(seed_set) if seed_set else 0.0
        current_profit = current_benefit - total_cost

        batches = [candidates[i:i + BATCH_SIZE]
                   for i in range(0, len(candidates), BATCH_SIZE)]
        batch_results = Parallel(n_jobs=ctx.cfg.num_cpus)(
            delayed(_evaluate_batch)(b, seed_set, current_profit, total_cost,
                                     ctx.adj, ctx.prob, ctx.benefits_array,
                                     ctx.costs, sims, ctx.cfg)
            for b in batches
        )
        results = [item for sub in batch_results for item in sub]
        best_node, best_gain = max(results, key=lambda x: x[1])
        if best_gain <= 0:
            break
        seed_set.append(best_node)
        seed_lookup.add(best_node)
        total_cost += ctx.costs[best_node]
        remaining -= ctx.costs[best_node]

    return seed_set, total_cost, {}


def run_greedy_algorithm(cfg=None):
    return pc.run_pipeline(ALGO, select_seeds, cfg)
