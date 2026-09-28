import networkx as nx

def extract_disjoint_directed_3cycles(G, output_file):
    if not isinstance(G, nx.DiGraph):
        raise TypeError("This function is for directed graphs only (nx.DiGraph).")

    used_nodes = set()
    disjoint_cycles = []

    # Step 1: Find all directed 3-cycles (u→v→w→u)
    all_cycles = set()
    for u in G.nodes():
        for v in G.successors(u):
            if v == u:
                continue
            for w in G.successors(v):
                if w in {u, v}:
                    continue
                if G.has_edge(w, u):  # completes the cycle
                    cycle = frozenset((u, v, w))
                    if len(cycle) == 3:
                        all_cycles.add(cycle)

    # Step 2: Select disjoint cycles only
    for cycle in sorted(all_cycles, key=lambda x: tuple(sorted(x))):
        if cycle.isdisjoint(used_nodes):
            disjoint_cycles.append(cycle)
            used_nodes.update(cycle)

    # Step 3: Save to file
    with open(output_file, 'w') as f:
        for cycle in disjoint_cycles:
            sorted_cycle = sorted(cycle)
            f.write(f"{{{sorted_cycle[0]}, {sorted_cycle[1]}, {sorted_cycle[2]}}}\n")

    return disjoint_cycles

# --- Main Execution ---
if __name__ == "__main__":
    G = nx.read_edgelist("facebook.txt", create_using=nx.DiGraph(), nodetype=int)
    output_file = "motifs_size3.txt"
    motifs = extract_disjoint_directed_3cycles(G, output_file)
    print(f"{len(motifs)} disjoint directed 3-node cycles written to {output_file}")
