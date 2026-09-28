"""generate_motifs.py
===================

Build vertex-DISJOINT motifs of a chosen size and save them, one Python set per
line, to motifs_size{MOTIF_SIZE}.txt -- the exact format load_motifs() parses:

    {0, 5, 12}
    {3, 4, 9}
    ...

One file for every size: set MOTIF_SIZE = 3 (or 4, ...). A "motif" is a set of
MOTIF_SIZE vertices of the UNDIRECTED PROJECTION of your directed graph (an edge
u->v or v->u simply joins u and v). Disjoint = each vertex used by at most one
motif, which removes cross-motif double counting in compute_motif_influence.

SHAPE controls what the vertices must form:
  "connected"  : any connected subgraph on MOTIF_SIZE vertices (default).
                 For size 3 that's triangles OR open wedges (paths).
  "clique"     : all vertices mutually adjacent. For size 3 -> triangles (K3);
                 for size 4 -> 4-cliques (K4).

MODE controls how they're found:
  "fast"     : streaming greedy growth in benefit order. O(m)-ish, tiny memory.
               Use this at scale (Slashdot / Epinions / wikiRFA).
  "quality"  : ESU-enumerate candidates (parallel, capped), weight by summed
               benefit, weighted-greedy disjoint pack. Small graphs only --
               connected-subgraph counts explode with density.
"""

import os
import ast
import time
from itertools import combinations
from collections import defaultdict

# ----------------------------- config -------------------------------------
GRAPH_FILE = "facebook_uniform.txt"    # topology only; all weight versions share it
BENEFIT_FILE = "benefit.txt"           # optional; prioritises high-benefit vertices
MOTIF_SIZE = 3                         # <-- 3 here; set 4 to regenerate size-4
SHAPE = "connected"                    # "connected" or "clique"
MODE = "fast"                          # "fast" or "quality"
NUM_CPUS = 24                          # only used by "quality" enumeration
MAX_CANDIDATES = 2_000_000             # "quality" safety cap so enumeration can't OOM

OUTPUT_FILE = f"motifs_size{MOTIF_SIZE}.txt"


# -------------------------- graph loading ---------------------------------
def load_undirected_projection(edgelist_path):
    """Directed edgelist -> undirected adjacency (dict of sets).

    Tolerant of real dataset formats: SNAP-style '#'/'%'/'//' comment or header
    lines, blank lines, tab/space/comma separation, and a missing weight column
    (only the first two tokens are read). Non-edge lines are skipped and counted.
    """
    adj = defaultdict(set)
    nodes = set()
    skipped = 0
    with open(edgelist_path) as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#%/;":
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 2:
                skipped += 1
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                skipped += 1
                continue
            nodes.add(u)
            nodes.add(v)
            if u == v:
                continue
            adj[u].add(v)
            adj[v].add(u)                     # <-- projection: forget direction
    for x in nodes:
        adj.setdefault(x, set())
    if skipped:
        print(f"  (skipped {skipped} non-edge line(s): comments/headers)")
    return adj, nodes


def load_benefits(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = ast.literal_eval(f.read())
    return {int(k): float(v) for k, v in raw.items()}


# --------------------------- shape helpers --------------------------------
def is_connected(vertices, adj):
    vs = set(vertices)
    start = next(iter(vs))
    seen, stack = {start}, [start]
    while stack:
        x = stack.pop()
        for nb in adj[x]:
            if nb in vs and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen == vs


def is_clique(vertices, adj):
    vs = list(vertices)
    return all(b in adj[a] for a, b in combinations(vs, 2))


# --------------------- mode 1: fast greedy growth -------------------------
def grow_connected(start, adj, used, benefit, size):
    """Grow a connected block of `size` UNUSED vertices (best-benefit neighbour)."""
    block, blockset = [start], {start}
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


def grow_clique(start, adj, used, benefit, size):
    """Grow a clique of `size` UNUSED vertices (common-neighbour intersection)."""
    block, blockset = [start], {start}
    cand = {x for x in adj[start] if x not in used}
    while len(block) < size:
        cand -= blockset
        if not cand:
            return None
        nxt = max(cand, key=lambda x: benefit.get(x, 0.0))
        block.append(nxt)
        blockset.add(nxt)
        cand = {x for x in cand if x in adj[nxt]}   # keep only common neighbours
    return frozenset(block)


def greedy_growth_pack(adj, nodes, benefit, size, shape):
    grow = grow_clique if shape == "clique" else grow_connected
    used, motifs = set(), []
    order = sorted(nodes, key=lambda x: (benefit.get(x, 0.0), len(adj[x])), reverse=True)
    for start in order:
        if start in used:
            continue
        block = grow(start, adj, used, benefit, size)
        if block is not None:
            motifs.append(block)
            used.update(block)
    return motifs


# ------------------- mode 2: ESU enumerate + weighted pack ----------------
def esu_from_root(root, adj, k):
    """All connected k-subgraphs whose minimum-index vertex is `root` (each once)."""
    results = []

    def extend(sub, subset, ext):
        if len(sub) == k:
            results.append(frozenset(sub))
            return
        ext = list(ext)
        while ext:
            w = ext.pop()
            excl = [u for u in adj[w]
                    if u > root and u not in subset
                    and not any(u in adj[x] for x in sub)]
            extend(sub + [w], subset | {w}, ext + excl)

    extend([root], {root}, [u for u in adj[root] if u > root])
    return results


def _esu_chunk(chunk, adj, k):
    out = []
    for r in chunk:
        out.extend(esu_from_root(r, adj, k))
    return out


def enumerate_candidates(adj, nodes, k, num_cpus, max_candidates):
    roots = list(nodes)
    chunk = max(1, len(roots) // (num_cpus * 4) or 1)
    chunks = [roots[i:i + chunk] for i in range(0, len(roots), chunk)]
    try:
        from joblib import Parallel, delayed
        parallel = Parallel(n_jobs=num_cpus)
        results = []
        for w in range(0, len(chunks), num_cpus):
            wave = chunks[w:w + num_cpus]
            for batch in parallel(delayed(_esu_chunk)(c, adj, k) for c in wave):
                results.extend(batch)
                if len(results) >= max_candidates:
                    print(f"  ⚠️ candidate cap {max_candidates:,} reached; packing a "
                          f"subset. Use MODE='fast' for a graph this size.")
                    return results
        return results
    except Exception as e:
        print(f"  (parallel enumeration unavailable: {e}; running sequentially)")
        results = []
        for r in roots:
            results.extend(esu_from_root(r, adj, k))
            if len(results) >= max_candidates:
                break
        return results


def weighted_greedy_pack(candidates, benefit, adj, shape):
    if shape == "clique":
        candidates = [m for m in candidates if is_clique(m, adj)]
    scored = sorted(candidates,
                    key=lambda m: sum(benefit.get(x, 0.0) for x in m),
                    reverse=True)
    used, motifs = set(), []
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


def main():
    t0 = time.time()
    adj, nodes = load_undirected_projection(GRAPH_FILE)
    benefit = load_benefits(BENEFIT_FILE)
    print(f"Projection: {len(nodes)} vertices, "
          f"{sum(len(a) for a in adj.values()) // 2} undirected edges | "
          f"size={MOTIF_SIZE} shape={SHAPE} mode={MODE}")

    if MODE == "fast":
        motifs = greedy_growth_pack(adj, nodes, benefit, MOTIF_SIZE, SHAPE)
    elif MODE == "quality":
        cand = enumerate_candidates(adj, nodes, MOTIF_SIZE, NUM_CPUS, MAX_CANDIDATES)
        print(f"Enumerated {len(cand)} connected {MOTIF_SIZE}-subgraph candidates")
        motifs = weighted_greedy_pack(cand, benefit, adj, SHAPE)
    else:
        raise ValueError(f"Unknown MODE: {MODE!r} (use 'fast' or 'quality')")

    # verify size + disjointness + shape before writing
    seen = set()
    for m in motifs:
        assert len(m) == MOTIF_SIZE, f"wrong size: {m}"
        assert seen.isdisjoint(m), f"overlap: {m}"
        if SHAPE == "clique":
            assert is_clique(m, adj), f"not a clique: {m}"
        else:
            assert is_connected(m, adj), f"disconnected: {m}"
        seen.update(m)

    save_motifs(motifs, OUTPUT_FILE)
    covered = len(motifs) * MOTIF_SIZE
    print(f"[{MODE}/{SHAPE}] {len(motifs)} disjoint size-{MOTIF_SIZE} motifs "
          f"covering {covered}/{len(nodes)} vertices "
          f"({100 * covered / max(len(nodes), 1):.1f}%)")
    print(f"Saved to {OUTPUT_FILE} in {round(time.time() - t0, 2)}s")


if __name__ == "__main__":
    main()
