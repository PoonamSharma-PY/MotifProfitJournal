"""Random baseline: shuffle nodes, add until the budget is spent."""
import random
import pipeline_common as pc

ALGO = "Random"


def select_seeds(ctx):
    nodes = list(ctx.G.nodes())
    rng = (random.Random(int(ctx.make_seeds(1, 42)[0]))
           if ctx.cfg.random_seed is not None else random)
    rng.shuffle(nodes)
    seed_set, total = [], 0.0
    for node in nodes:
        c = ctx.costs[node]
        if total + c <= ctx.budget:
            seed_set.append(node)
            total += c
    return seed_set, total, {}


def run_random_algorithm(cfg=None):
    return pc.run_pipeline(ALGO, select_seeds, cfg)
