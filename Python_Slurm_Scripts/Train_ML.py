# Load libraries
import os
import argparse
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    StratifiedGroupKFold,
    RandomizedSearchCV,
    cross_validate
)

from utils.data_loader import get_training_data, make_data_grid
from utils.model_factory import build_pipeline, param_space

# Run the experiment with nested cross-validation
def run_experiment(cnfg: dict):
    """
    Runs nested cross-validation with hyperparameter tuning and saves the results.
    """
    # Load data
    X, y, groups = get_training_data(
        countries=cnfg["countries"],
        tasks=cnfg["tasks"],
        sexes=cnfg["sexes"]
    )

    # Define scoring metrics
    scoring = {
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
        "average_precision": "average_precision",
        "f1_score": "f1"
    }

    # Build pipeline
    pipeline = build_pipeline(model_name=cnfg["model_name"])

    # Get hyperparameter space
    param_distributions = param_space(model_name=cnfg["model_name"])

    # Setup cross-validation
    outer_splits = []
    for i in range(cnfg["n_repeats"]):
        sgkf = StratifiedGroupKFold(
            n_splits=cnfg["outer_splits"],
            shuffle=True,
            random_state=42+i
        )
        outer_splits.extend(list(sgkf.split(X, y, groups)))
    
    inner_cv = StratifiedGroupKFold(
        n_splits=cnfg["inner_splits"],
        shuffle=True,
        random_state=42
    )

    # Setup hyperparameter tuning
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=cnfg["n_iter"],
        scoring="roc_auc",
        n_jobs=cnfg["inner_n_jobs"],
        cv=inner_cv,
        verbose=cnfg["inner_verbose"],
        random_state=42,
        refit=True,
        error_score='raise'
    )
    
    # Execute nested cross-validation
    results = cross_validate(
        search,
        X=X,
        y=y,
        params={'groups': groups},
        cv=outer_splits,
        scoring=scoring,
        return_estimator=True,
        n_jobs=cnfg["outer_n_jobs"],
        verbose=cnfg["outer_verbose"],
        error_score='raise'
    )

    # Save results
    out_dir = cnfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame({k: v for k, v in results.items() if k != "estimator"})
    results_df["best_params"] = [est.best_params_ for est in results["estimator"]]
    params_df = results_df["best_params"].apply(pd.Series)
    if cnfg["model_name"] == "GB":
        params_df["classifier__n_estimators"] = [
            est.best_estimator_.named_steps["classifier"].n_estimators_ for est in results["estimator"]
        ]
    results_df = pd.concat([results_df.drop(columns=["best_params"]), params_df], axis=1)
    results_df = results_df.sort_values("test_roc_auc", ascending=False)
    results_df.to_csv(out_dir / "results.csv", index=False)
    
    results_summary_df = pd.DataFrame({
        m: results[f"test_{m}"] for m in list(scoring.keys())
    }).agg(["mean", "std"]).T
    results_summary_df.to_csv(out_dir / "results_summary.csv")

    n_outer = cnfg["outer_splits"]
    n_total = cnfg["outer_splits"] * cnfg["n_repeats"]
    outer_df = pd.DataFrame({
        "repeat": (np.arange(n_total) // n_outer) + 1,
        "outer_fold": (np.arange(n_total) % n_outer) + 1,
        **{m: results[f"test_{m}"] for m in list(scoring.keys())}
    })
    outer_df.to_csv(out_dir / "outer_cv_results.csv", index=False)

    inner_df = pd.DataFrame([
        {
            "repeat": (i // cnfg["outer_splits"]) + 1,
            "outer_fold": (i % cnfg["outer_splits"]) + 1,
            "inner_best_score": est.best_score_,
            "inner_best_params": est.best_params_,
            "n_candidates": len(est.cv_results_["params"]),
        }
        for i, est in enumerate(results["estimator"])
    ])
    inner_df.to_csv(out_dir / "inner_cv_results.csv", index=False)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", type=int, default=None, help="0: QBF/M, 1: JF/M, etc.")
    parser.add_argument("--model_name", type=str, default="SVC", choices=["SVC", "DT", "RF", "GB"])
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=10)
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
    base_dir = Path("../Results_Tests")
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

    print(
        f"\n*** Running {cnfg['outer_splits']} outer folds and {cnfg['n_repeats']} repeats "
        f"for {args.model_name} | sex={cnfg['sexes_key']} | task={cnfg['tasks_key']} ***\n"
    )
    run_experiment(cnfg)

if __name__ == "__main__":
    main()