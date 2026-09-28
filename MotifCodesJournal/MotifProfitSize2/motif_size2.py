import networkx as nx

def extract_disjoint_size2_motifs(G, output_file):
    used_nodes = set()
    disjoint_edges = []

    for u, v in G.edges():
        if u == v:  # 🚫 Skip self-loops
            continue
        if u not in used_nodes and v not in used_nodes:
            disjoint_edges.append((u, v))
            used_nodes.update([u, v])

    # Save to file
    with open(output_file, 'w') as f:
        for u, v in disjoint_edges:
            f.write(f"{{{u}, {v}}}\n")

    return disjoint_edges

# Example usage
G = nx.read_edgelist("facebook.txt", create_using=nx.DiGraph(), nodetype=int)
output_file = "motifs_size2.txt"
motifs = extract_disjoint_size2_motifs(G, output_file)

print(f"{len(motifs)} disjoint size-2 motifs written to {output_file}")
