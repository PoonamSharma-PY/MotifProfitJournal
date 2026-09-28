import ast
import numpy as np
import time
import joblib
from joblib import Parallel, delayed

# Config
NUM_CPUS = 24


def load_motifs(motif_file):
    motifs = []
    try:
        with open(motif_file, "r") as f:
            for line in f:
                motif = set(ast.literal_eval(line.strip()))
                motifs.append(motif)
    except FileNotFoundError:
        print(f"Error: {motif_file} not found.")
        return []
    except Exception as e:
        print(f"Error parsing {motif_file}: {e}")
        return []
    return motifs


def compute_motif_influence(activated_nodes, motifs, benefits, threshold):
    motif_influence_simulation = set()
    activated_set = set(activated_nodes)
    for motif in motifs:
        common_nodes = motif & activated_set
        if len(common_nodes) >= threshold:
            motif_influence_simulation.update(motif)
    return motif_influence_simulation


def compute_motif_profit(activated_nodes, motifs, benefits, threshold, seed_cost):
    influenced_nodes = compute_motif_influence(activated_nodes, motifs, benefits, threshold)
    motif_profit = sum(benefits.get(i, 0) for i in influenced_nodes) - seed_cost
    return motif_profit


def _base_exec_time(row):
    """Algorithm runtime, regardless of pipeline (Random_/HighDegree_/...)."""
    for k, v in row.items():
        if (k.endswith("_Execution_Time")
                and k not in ("Motif_Execution_Time", "Total_Execution_Time")):
            return v or 0.0
    return 0.0


def process_motif_profits(algorithm_results, motif_file, threshold):
    motifs = load_motifs(motif_file)
    final_results = []
    distributions = {}  # (Model, Budget) -> full per-simulation motif-profit array

    for result in algorithm_results:
        # Drop heavy objects up front so the output row is Excel-safe.
        new_result = {k: v for k, v in result.items()
                      if k not in ("Simulation_Results", "Benefits")}

        if not motifs:
            print("No motifs loaded, adding default motif metrics.")
            new_result.update({
                "Avg_Motif_Profit": 0.0,
                "Max_Motif_Profit": 0.0,
                "Min_Motif_Profit": 0.0,
                "Std_Motif_Profit": 0.0,
                "Motif_Execution_Time": 0.0,
                "Threshold": threshold,
            })
            final_results.append(new_result)
            continue

        start_time = time.time()
        sims = result["Simulation_Results"]
        benefits = result["Benefits"]
        seed_cost = result["Seed_Cost"]

        motif_profits = Parallel(n_jobs=NUM_CPUS)(
            delayed(compute_motif_profit)(
                np.where(sim_result[0])[0], motifs, benefits, threshold, seed_cost
            ) for sim_result in sims
        )
        motif_profits = np.asarray(motif_profits, dtype=float)
        motif_execution_time = time.time() - start_time
        total_time = _base_exec_time(new_result) + motif_execution_time

        has_data = motif_profits.size > 0
        new_result.update({
            "Avg_Motif_Profit": float(motif_profits.mean()) if has_data else 0.0,
            "Max_Motif_Profit": float(motif_profits.max()) if has_data else 0.0,
            "Min_Motif_Profit": float(motif_profits.min()) if has_data else 0.0,
            "Std_Motif_Profit": float(motif_profits.std()) if has_data else 0.0,
            "Motif_Execution_Time": round(motif_execution_time, 2),
            "Total_Execution_Time": round(total_time, 2),
            "Threshold": threshold,
        })
        distributions[(new_result["Model"], new_result["Budget"])] = motif_profits
        final_results.append(new_result)

    # The full per-simulation distributions can't live in Excel (a single cell
    # would be a ~10k-value string, past Excel's 32,767-char limit), so persist
    # them separately for plotting/analysis.
    if distributions:
        joblib.dump(distributions,
                    f"Motif_Profits_Threshold{threshold}.pkl", compress=3)

    return final_results
