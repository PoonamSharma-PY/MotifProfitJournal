"""High-degree baseline: take highest out-degree nodes the budget affords."""
import pipeline_common as pc

ALGO = "HighDegree"


def select_seeds(ctx):
    out = dict(ctx.G.out_degree())
    nodes = sorted(out, key=lambda x: out[x], reverse=True)
    seed_set, total, missing = [], 0.0, 0
    for node in nodes:
        c = ctx.costs.get(node)
        if c is None:                 # missing cost -> skip, never treat as free
            missing += 1
            continue
        if total + c <= ctx.budget:
            seed_set.append(node)
            total += c
    if missing:
        print(f"  ⚠️ {missing} node(s) had no cost entry and were skipped.")
    return seed_set, total, {}


def run_high_degree_algorithm(cfg=None):
    return pc.run_pipeline(ALGO, select_seeds, cfg)
