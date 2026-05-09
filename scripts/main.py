# Load libraries
import os
import argparse

from scripts.utils.data_loader import create_configs
from scripts.utils.train_tune_val import run_experiment

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_idx", type=int, default=None, help="0: QBF/M, 1: JF/M, etc. Defaults to SLURM_ARRAY_TASK_ID or 8.")
    parser.add_argument("--model_name", type=str, default="XGB", choices=["SVC", "DT", "RF", "GB", "XGB", "LGBM", "MLP", "NB", "KNN"])
    parser.add_argument("--feature_selector_method", type=str, default="rfe", choices=["mi_based", "corr_based", "rfe", "passthrough"])
    args = parser.parse_args()
    
    # Get and set configuration
    if args.case_idx is not None:
        case_idx = args.case_idx
    else:
        case_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "8"))

    model_name = args.model_name
    feature_selector_method = args.feature_selector_method
    
    N_REPEATS = 7
    OUTER_SPLITS = 5
    INNER_SPLITS = 5
    N_ITER = 150
    TOTAL_OUTER_FITS = N_REPEATS * OUTER_SPLITS
    ALLOCATED_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    n_dict = {
        "n_repeats": N_REPEATS,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "n_iter": N_ITER,
        "outer_verbose": 20,
        "inner_verbose": 1,
        "outer_n_jobs": min(ALLOCATED_CPUS, TOTAL_OUTER_FITS),
        "inner_n_jobs": 1
    }

    config = create_configs(case_idx, model_name, feature_selector_method, n_dict)
    out_dir = f"../results/{model_name}_{feature_selector_method}/array={case_idx}"
    config.update({"out_dir": out_dir})
    os.makedirs(out_dir, exist_ok=True)

    # Run the experiment with nested cross-validation
    print(
        f"\n*** Running {config['outer_splits']} outer folds and {config['n_repeats']} repeats "
        f"for {model_name} | sex={config['sexes_key']} | task={config['tasks_key']} ***\n"
    )
    run_experiment(config)

if __name__ == "__main__":
    main()
