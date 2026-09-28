"""One main for all five baselines.

    python main.py            # runs the algorithm named in ALGO below
    python main.py ris        # or pick one on the command line

Each algorithm module exposes: ALGO, its run_* function, and (RIS only) an
EXTRA_COLUMNS tuple. pipeline_common does the rest.
"""
import sys
import time
import importlib
import pipeline_common as pc

# Which baseline to run by default. Override on the command line.
ALGO = "greedy"

# name -> (module, run-function attribute)
REGISTRY = {
    "random":     ("random_algorithm",      "run_random_algorithm"),
    "highdegree": ("high_degree_algorithm",  "run_high_degree_algorithm"),
    "celf":       ("celf_algorithm",         "run_celf_algorithm"),
    "greedy":     ("greedy_algorithm",       "run_greedy_algorithm"),
    "ris":        ("ris_algorithm",          "run_ris"),
}


def main(which):
    module_name, run_attr = REGISTRY[which]
    algo_module = importlib.import_module(module_name)
    algo = algo_module.ALGO
    extra_columns = getattr(algo_module, "EXTRA_COLUMNS", ())

    # RIS plots KPT/Theta; everyone else plots Profit/Activated nodes.
    plot_specs = ([("KPT", "o", "KPT"), ("Theta", "s", "Theta")]
                  if extra_columns else
                  [("Profit", "^", "Profit"),
                   ("Avg_Activated_Nodes", "o", "Avg Activated Nodes")])

    cfg = pc.Config()                 # tweak knobs here, e.g. cfg.random_seed = 42
    t0 = time.time()

    results = getattr(algo_module, run_attr)(cfg)
    pc.finalize(results, algo, cfg, plot_specs=plot_specs, extra_columns=extra_columns)

    print(f"\n⏱️ Total Execution Time: {round(time.time() - t0, 2)} s")


if __name__ == "__main__":
    which = sys.argv[1].lower() if len(sys.argv) > 1 else ALGO
    main(which)
