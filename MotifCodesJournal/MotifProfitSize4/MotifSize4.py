"""generate_motifs_size4.py
==========================

Build a set of size-4 motifs that are pairwise VERTEX-DISJOINT (each vertex is
used by at most one motif) and save them, one Python set per line, to
motifs_size4.txt -- the exact format your load_motifs() already parses:

    {0, 5, 12, 88}
    {3, 4, 9, 40}
    ...

A "motif" here is any CONNECTED 4-vertex subgraph of the UNDIRECTED PROJECTION
of the directed graph: we treat an edge u->v (or v->u) as simply joining u and
v, and require the 4 chosen vertices to induce a connected subgraph.

Two build modes (set MODE below):

  "fast"     : streaming greedy growth. For each still-unused vertex (processed
               in benefit order), grow a connected block of 4 unused vertices,
               emit it, mark them used. O(m)-ish, single pass, tiny memory.
               This is the one to use on Slashdot / Epinions-scale graphs.

  "quality"  : enumerate connected 4-subgraph candidates with ESU (each set
               once, rooted at its min-index vertex, parallelised by root),
               weight each by the summed benefit of its 4 vertices, then take a
               weighted greedy disjoint packing. Better packing, more memory /
               time -- fine for Email-Eu-Core, watch memory on huge graphs.

Both write the same file format, so the rest of your pipeline is unchanged.
"""

import os
import ast
import time
from collections import defaultdict

# ----------------------------- config -------------------------------------
GRAPH_FILE = "wikivote_uniform.txt"   # topology only; all 3 weight versions share it
BENEFIT_FILE = "benefit.txt"          # optional; used to prioritise high-benefit vertices
OUTPUT_FILE = "motifs_size4.txt"
MODE = "fast"                          # "fast" or "quality"
NUM_CPUS = 24                          # only used by "quality" enumeration
MOTIF_SIZE = 4


# -------------------------- graph loading ---------------------------------
def load_undirected_projection(edgelist_path):
    """Directed edgelist ('u v [w]' per line) -> undirected adjacency (dict of sets)."""
    adj = defaultdict(set)
    nodes = set()
    with open(edgelist_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            nodes.add(u)
            nodes.add(v)
            if u == v:                 # drop self-loops
                continue
            adj[u].add(v)
            adj[v].add(u)              # <-- the projection: forget direction
    for x in nodes:
        adj.setdefault(x, set())
    return adj, nodes


def load_benefits(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = ast.literal_eval(f.read())
    return {int(k): float(v) for k, v in raw.items()}


# --------------------- mode 1: fast greedy growth -------------------------
def grow_block(start, adj, used, benefit, size=MOTIF_SIZE):
    """Grow a connected set of `size` UNUSED vertices starting from `start`.

    Each added vertex is adjacent to the current block, so the block is
    connected by construction. When several unused neighbours are available we
    take the highest-benefit one. Returns a frozenset of `size` vertices, or
    None if the block can't reach `size` (start is stranded among used nodes).
    """
    block = [start]
    blockset = {start}
    while len(block) < size:
        best, best_b = None, float("-inf")
        for node in block:
            for nb in adj[node]:
                if nb not in used and nb not in blockset:
                    b = benefit.get(nb, 0.0)
                    if b > best_b:
                        best, best_b = nb, b
        if best is None:
            return None
        block.append(best)
        blockset.add(best)
    return frozenset(block)


def greedy_growth_pack(adj, nodes, benefit):
    used = set()
    motifs = []
    # process high-benefit vertices first so they land inside motifs
    order = sorted(nodes, key=lambda x: (benefit.get(x, 0.0), len(adj[x])), reverse=True)
    for start in order:
        if start in used:
            continue
        block = grow_block(start, adj, used, benefit)
        if block is not None:
            motifs.append(block)
            used.update(block)
    return motifs


# ------------------- mode 2: ESU enumerate + weighted pack ----------------
def esu_from_root(root, adj, k=MOTIF_SIZE):
    """All connected k-subgraphs whose minimum-index vertex is `root` (each once)."""
    results = []
    v_ext = [u for u in adj[root] if u > root]

    def extend(sub, subset, ext):
        if len(sub) == k:
            results.append(frozenset(sub))
            return
        ext = list(ext)
        while ext:
            w = ext.pop()
            # exclusive neighbourhood of w: neighbours > root, not in sub,
            # and not adjacent to any vertex already in sub
            excl = [u for u in adj[w]
                    if u > root and u not in subset
                    and not any(u in adj[x] for x in sub)]
            extend(sub + [w], subset | {w}, ext + excl)

    extend([root], {root}, v_ext)
    return results


def enumerate_connected4(adj, nodes, num_cpus=NUM_CPUS):
    roots = list(nodes)
    try:
        from joblib import Parallel, delayed
        batches = Parallel(n_jobs=num_cpus)(
            delayed(esu_from_root)(r, adj) for r in roots
        )
    except Exception:
        batches = [esu_from_root(r, adj) for r in roots]
    return [m for batch in batches for m in batch]


def weighted_greedy_pack(candidates, benefit):
    scored = sorted(candidates,
                    key=lambda m: sum(benefit.get(x, 0.0) for x in m),
                    reverse=True)
    used = set()
    motifs = []
    for m in scored:
        if used.isdisjoint(m):
            motifs.append(m)
            used.update(m)
    return motifs


# ------------------------------ output ------------------------------------
def save_motifs(motifs, path):
    with open(path, "w") as f:
        for m in motifs:
            f.write("{" + ", ".join(str(x) for x in sorted(m)) + "}\n")


def is_connected(vertices, adj):
    """Sanity check: do these vertices induce a connected subgraph?"""
    vs = set(vertices)
    seen = {next(iter(vs))}
    stack = [next(iter(vs))]
    while stack:
        x = stack.pop()
        for nb in adj[x]:
            if nb in vs and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen == vs


def main():
    t0 = time.time()
    adj, nodes = load_undirected_projection(GRAPH_FILE)
    benefit = load_benefits(BENEFIT_FILE)
    print(f"Projection: {len(nodes)} vertices, "
          f"{sum(len(a) for a in adj.values()) // 2} undirected edges")

    if MODE == "fast":
        motifs = greedy_growth_pack(adj, nodes, benefit)
    elif MODE == "quality":
        candidates = enumerate_connected4(adj, nodes)
        print(f"Enumerated {len(candidates)} connected 4-subgraph candidates")
        motifs = weighted_greedy_pack(candidates, benefit)
    else:
        raise ValueError(f"Unknown MODE: {MODE!r} (use 'fast' or 'quality')")

    # verify disjointness + connectivity before writing
    seen = set()
    for m in motifs:
        assert len(m) == MOTIF_SIZE, f"motif not size {MOTIF_SIZE}: {m}"
        assert seen.isdisjoint(m), f"overlap detected: {m}"
        assert is_connected(m, adj), f"disconnected motif: {m}"
        seen.update(m)

    save_motifs(motifs, OUTPUT_FILE)
    covered = len(motifs) * MOTIF_SIZE
    print(f"[{MODE}] {len(motifs)} disjoint size-4 motifs "
          f"covering {covered}/{len(nodes)} vertices "
          f"({100 * covered / max(len(nodes), 1):.1f}%)")
    print(f"Saved to {OUTPUT_FILE} in {round(time.time() - t0, 2)}s")


if __name__ == "__main__":
    main()