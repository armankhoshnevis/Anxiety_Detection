# Load libraries
import os
import argparse

from pathlib import Path

from utils.data_loader import make_data_grid
from utils.train_tune_val import run_experiment

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", type=int, default=None, help="0: QBF/M, 1: JF/M, etc.")
    parser.add_argument("--model_name", type=str, default="SVC", choices=["SVC", "DT", "RF", "GB", "XGB"])
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=10000)
    parser.add_argument("--outer_verbose", type=int, default=20)
    parser.add_argument("--inner_verbose", type=int, default=1)
    parser.add_argument("--outer_n_jobs", type=int, default=-1)
    parser.add_argument("--inner_n_jobs", type=int, default=1)
    args = parser.parse_args(args=["--case_id", "0", "--model_name", "DT"])
    
    # Get and set data configuration
    cnfgs = make_data_grid()
    if args.case_id is not None:
        case_id = args.case_id
    else:
        case_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    cnfg = cnfgs[case_id]

    # Set base directory for results
    base_dir = Path("../Results")
    out_dir = base_dir / args.model_name / f"sex={cnfg['sexes_key']}" / f"task={cnfg['tasks_key']}"

    cnfg.update({
        "model_name": args.model_name,
        "out_dir": out_dir,
        "n_repeats": args.n_repeats,
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "n_iter": args.n_iter,
        "outer_verbose": args.outer_verbose,
        "inner_verbose": args.inner_verbose,
        "outer_n_jobs": args.outer_n_jobs,
        "inner_n_jobs": args.inner_n_jobs,
    })

    # Run the experiment with nested cross-validation
    print(
        f"\n*** Running {cnfg['outer_splits']} outer folds and {cnfg['n_repeats']} repeats "
        f"for {args.model_name} | sex={cnfg['sexes_key']} | task={cnfg['tasks_key']} ***\n"
    )
    run_experiment(cnfg)

if __name__ == "__main__":
    main()