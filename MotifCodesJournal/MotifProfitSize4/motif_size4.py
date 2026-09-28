# import networkx as nx
# from itertools import combinations, islice
# from joblib import Parallel, delayed
# from tqdm import tqdm
# import multiprocessing

# def check_and_return_motif(G, nodes, used_global, min_pairs):
#     a, b, c, d = nodes
#     if any(n in used_global for n in (a, b, c, d)):
#         return None

#     pair_count = 0
#     pairs = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
#     for u, v in pairs:
#         if G.has_edge(u, v) or G.has_edge(v, u):
#             pair_count += 1
#         if 6 - pair_count > (6 - min_pairs):
#             return None

#     return frozenset((a, b, c, d)) if pair_count >= min_pairs else None

# def chunked_iterable(iterable, size):
#     """Yield successive chunks from an iterable."""
#     it = iter(iterable)
#     while True:
#         chunk = list(islice(it, size))
#         if not chunk:
#             break
#         yield chunk

# def parallel_disjoint_motifs_streaming(G, output_file, min_pairs=6, chunk_size=10000, n_jobs=-1):
#     used_nodes = set()
#     all_motifs = []

#     node_list = list(G.nodes())
#     total_combos = (len(node_list) * (len(node_list)-1) * (len(node_list)-2) * (len(node_list)-3)) // 24

#     combo_gen = combinations(node_list, 4)

#     with tqdm(total=total_combos, desc="Scanning 4-node combos") as pbar:
#         for chunk in chunked_iterable(combo_gen, chunk_size):
#             results = Parallel(n_jobs=n_jobs, backend='loky')(
#                 delayed(check_and_return_motif)(G, combo, used_nodes, min_pairs)
#                 for combo in chunk
#             )
#             for motif in results:
#                 if motif and motif.isdisjoint(used_nodes):
#                     all_motifs.append(motif)
#                     used_nodes.update(motif)
#             pbar.update(len(chunk))

#     with open(output_file, 'w') as f:
#         for motif in all_motifs:
#             f.write(f"{{{', '.join(map(str, sorted(motif)))}}}\n")

#     return all_motifs

# # --- Main Execution ---
# if __name__ == "__main__":
#     G = nx.read_edgelist("wikivote.txt", create_using=nx.DiGraph(), nodetype=int)
#     output_file = "wikivote_stream_disjoint_motifs_size4.txt"
#     motifs = parallel_disjoint_motifs_streaming(G, output_file, min_pairs=6, chunk_size=10000, n_jobs=multiprocessing.cpu_count())
#     print(f"{len(motifs)} disjoint motifs written to {output_file}")



# import networkx as nx
# import numpy as np
# from itertools import combinations, islice
# from joblib import Parallel, delayed
# from tqdm import tqdm
# from numba import njit
# import multiprocessing

# # --- Numba-accelerated edge checker ---
# @njit
# def count_connected_pairs(adj_matrix, node_ids):
#     count = 0
#     for i in range(4):
#         for j in range(i + 1, 4):
#             u, v = node_ids[i], node_ids[j]
#             if adj_matrix[u, v] or adj_matrix[v, u]:
#                 count += 1
#     return count

# # --- Per-motif structure check ---
# def check_motif_structure(adj_matrix, node_map, combo, min_pairs):
#     node_ids = [node_map[n] for n in combo]
#     count = count_connected_pairs(adj_matrix, np.array(node_ids))
#     return frozenset(combo) if count >= min_pairs else None

# # --- Chunk generator ---
# def chunked_iterable(iterable, size):
#     it = iter(iterable)
#     while True:
#         chunk = list(islice(it, size))
#         if not chunk:
#             break
#         yield chunk

# # --- Main motif pipeline ---
# def run_numba_optimized_disjoint_pipeline(G, output_file, min_pairs=6, chunk_size=10000, n_jobs=-1):
#     node_list = list(G.nodes())
#     node_map = {node: idx for idx, node in enumerate(node_list)}
#     n = len(node_list)

#     # Build adjacency matrix
#     adj_matrix = np.zeros((n, n), dtype=np.bool_)
#     for u, v in G.edges():
#         if u in node_map and v in node_map:
#             adj_matrix[node_map[u], node_map[v]] = True

#     combo_gen = combinations(node_list, 4)
#     total_combos = (n * (n - 1) * (n - 2) * (n - 3)) // 24

#     all_motifs = []
#     with tqdm(total=total_combos, desc="Scanning 4-node combos") as pbar:
#         for chunk in chunked_iterable(combo_gen, chunk_size):
#             results = Parallel(n_jobs=n_jobs, backend='loky')(
#                 delayed(check_motif_structure)(adj_matrix, node_map, combo, min_pairs)
#                 for combo in chunk
#             )
#             all_motifs.extend(filter(None, results))
#             pbar.update(len(chunk))

#     # Filter disjoint motifs
#     used_nodes = set()
#     disjoint_motifs = []
#     for motif in all_motifs:
#         if motif.isdisjoint(used_nodes):
#             disjoint_motifs.append(motif)
#             used_nodes.update(motif)

#     # Save results
#     with open(output_file, 'w') as f:
#         for motif in disjoint_motifs:
#             sorted_nodes = sorted(motif)
#             f.write(f"{{{', '.join(map(str, sorted_nodes))}}}\n")

#     return disjoint_motifs

# # --- Execution Entry Point ---
# if __name__ == "__main__":
#     # ✅ Replace with correct path to your data file
#     data_file = "wikivote.txt"
#     output_file = "wikivote_disjoint_motifs_size4.txt"

#     G = nx.read_edgelist(data_file, create_using=nx.DiGraph(), nodetype=int)
#     motifs = run_numba_optimized_disjoint_pipeline(
#         G,
#         output_file=output_file,
#         min_pairs=6,                  # Set to 5 if you want relaxed motifs
#         chunk_size=10000,             # Chunk size can be tuned
#         n_jobs=multiprocessing.cpu_count()  # Max parallelism
#     )

#     print(f"{len(motifs)} disjoint size-4 motifs written to: {output_file}")


# Re-executing the same code after environment reset
import networkx as nx
import numpy as np
from itertools import combinations
from tqdm import tqdm
from numba import njit

# Numba-accelerated pairwise edge checker
@njit
def count_connected_pairs(adj_matrix, node_ids):
    count = 0
    for i in range(4):
        for j in range(i + 1, 4):
            u, v = node_ids[i], node_ids[j]
            if adj_matrix[u, v] or adj_matrix[v, u]:
                count += 1
    return count

# Main motif extraction function
def extract_disjoint_motifs_inline(G, output_file, min_pairs=6):
    node_list = list(G.nodes())
    node_map = {node: idx for idx, node in enumerate(node_list)}
    n = len(node_list)

    # Build adjacency matrix
    adj_matrix = np.zeros((n, n), dtype=np.bool_)
    for u, v in G.edges():
        if u in node_map and v in node_map:
            adj_matrix[node_map[u], node_map[v]] = True

    used_nodes = set()
    motifs = []

    total_combos = (n * (n - 1) * (n - 2) * (n - 3)) // 24

    for combo in tqdm(combinations(node_list, 4), total=total_combos, desc="Scanning motifs"):
        if any(n in used_nodes for n in combo):
            continue  # Skip reused nodes

        node_ids = [node_map[n] for n in combo]
        if count_connected_pairs(adj_matrix, np.array(node_ids)) >= min_pairs:
            motifs.append(combo)
            used_nodes.update(combo)

    # Save motifs to file
    with open(output_file, 'w') as f:
        for motif in motifs:
            f.write(f"{{{', '.join(map(str, sorted(motif)))}}}\n")

    return motifs

# This block is not executable without the actual file "wikivote.txt"
# So it is commented out, and you can uncomment it locally to run.
G = nx.read_edgelist("wikivote.txt", create_using=nx.DiGraph(), nodetype=int)
output_file = "motifs_size4.txt"
motifs = extract_disjoint_motifs_inline(G, output_file, min_pairs=6)
print(f"{len(motifs)} disjoint motifs saved to {output_file}")
print("Sample motifs:", motifs[:5])

