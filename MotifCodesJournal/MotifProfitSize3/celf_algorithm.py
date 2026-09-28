"""CELF lazy-greedy on the profit-per-cost ratio.

Only the selection is here; the benefit estimator, seeding, evaluation and I/O
all come from pipeline_common. ctx.benefit_of(S) returns a cached, seed-stable
expected benefit, so the marginal gain is derived without any separate cache
(this is what avoids the old "two values under one key" collision).
"""
import heapq
import pipeline_common as pc

ALGO = "CELF"


def _marginal_gain(ctx, node, seed_list, cost_node):
    if cost_node <= 0:
        return 0.0
    b_s = ctx.benefit_of(seed_list)
    b_s1 = ctx.benefit_of(seed_list + [node])
    return ((b_s1 - b_s) - cost_node) / cost_node


def select_seeds(ctx):
    seed_set, seed_lookup, total_cost, remaining = [], set(), 0.0, ctx.budget
    last_size, pq = {}, []

    for node in range(ctx.n):
        c = ctx.costs.get(node, float("inf"))
        if c <= ctx.budget:
            mg = _marginal_gain(ctx, node, [], c)
            heapq.heappush(pq, (-mg, node))
            last_size[node] = 0

    while pq and remaining > 0:
        neg_gain, node = heapq.heappop(pq)
        mg = -neg_gain
        c = ctx.costs.get(node, float("inf"))
        if node in seed_lookup or c > remaining:
            continue
        if last_size[node] != len(seed_set):          # stale -> re-evaluate
            mg = _marginal_gain(ctx, node, seed_set, c)
            last_size[node] = len(seed_set)
            heapq.heappush(pq, (-mg, node))
            continue
        if mg > 0:                                     # fresh top -> select
            seed_set.append(node)
            seed_lookup.add(node)
            total_cost += c
            remaining -= c

    return seed_set, total_cost, {}


def run_celf_algorithm(cfg=None):
    return pc.run_pipeline(ALGO, select_seeds, cfg)
