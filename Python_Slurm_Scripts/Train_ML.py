# Load libraries
import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform

from itertools import product
from pathlib import Path

from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    StratifiedGroupKFold,
    RandomizedSearchCV,
    cross_validate
)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load the data
def get_training_data(countries, tasks, sexes, base_path="../Datasets/"):
    """
    Loads and combines data based on selected countries, tasks, and sexes.
    
    Parameters:
    - countries: List of strings (e.g., ["Botswana", "Ghana"])
    - tasks: List of strings (e.g., ["QBF", "JF"])
    - sexes: List of strings (e.g., ["Male"] or ["Male", "Female"])
    
    Returns:
    - X_arr, y_arr: Numpy arrays for features and target
    - groups: Numpy array of participant SessionIDs
    """
    
    df_list = []
    
    for country in countries:
        for task in tasks:
            file_name = f"{base_path}{country}_GAD_eGeMAPS_{task}.csv"
            
            try:
                temp_df = pd.read_csv(file_name)
                temp_df = temp_df[temp_df["Sex"].isin(sexes)]
                temp_df["Anxiety_Binary"] = temp_df["GAD7_Total"].apply(lambda x: 1 if x >= 5 else 0)
                df_list.append(temp_df)
            
            except FileNotFoundError:
                print(f"Warning: File not found: {file_name}")
            except KeyError as e:
                print(f"Warning: Missing column in {file_name}: {e}")

    if not df_list:
        raise ValueError("No data loaded. Check your file paths and parameters.")

    combined_df = pd.concat(df_list, axis=0, ignore_index=True)

    metadata_cols = [
        "SessionID", "QBF_Name", "JohnFarm_Name", "Sex", "Age", "Health", "Health_Binary",
        "Country", "GAD7_Total", "Anxiety_Category", "Anxiety_Binary"
    ]
    
    groups = combined_df["SessionID"].astype(str).to_numpy()
    X = combined_df.drop(columns=metadata_cols, errors='ignore').to_numpy()
    y = combined_df["Anxiety_Binary"].to_numpy()

    return X, y, groups

# Outlier removal using Local Outlier Factor
def lof_outlier_removal(X, y, n_neighbors=20, contamination=0.05, algorithm='auto', metric='manhattan'):
    """
    Removes outliers from the dataset using Local Outlier Factor (LOF).
    Args:
        X (numpy array): Feature matrix.
        y (numpy array): Target vector.
        n_neighbors (int, optional): Number of neighbors to use. Defaults to 20.
        contamination (float, optional): Proportion of outliers in the data set. Defaults to 0.05.
        algorithm (str, optional): Algorithm to compute nearest neighbors. Defaults to 'auto'.
        metric (str, optional): Distance metric to use. Defaults to 'manhattan'.

    Returns:
        tuple: Filtered feature matrix and target vector as numpy arrays.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        algorithm=algorithm,
        metric=metric,
        leaf_size=30,
        novelty=False)
    
    y_pred = lof.fit_predict(X_arr)
    mask_inliers = y_pred == 1

    return X_arr[mask_inliers], y_arr[mask_inliers]

# Build the machine learning pipeline
def build_pipeline(model_name: str) -> ImbPipeline:
    """
    Builds a machine learning pipeline with preprocessing, outlier removal, feature selection,
    oversampling, and classification steps.
    """
    lof_sampler = FunctionSampler(
        func=lof_outlier_removal,
        kw_args={
            "contamination": 0.05,
            "n_neighbors": 20,
            "algorithm": "auto",
            "metric": "manhattan",
        },
        validate=False,
    )
    
    if model_name == "SVC":
        feature_step = PCA(svd_solver="full")
        clf = SVC(probability=False, random_state=42)
    elif model_name == "DT":
        # feature_step = "passthrough"
        feature_step = PCA(svd_solver="full")
        clf = DecisionTreeClassifier(random_state=42)
    elif model_name == "RF":
        feature_step = "passthrough"
        clf = RandomForestClassifier(
            class_weight=None,
            n_jobs=1,
            random_state=42,
            )
    elif model_name == "GB":
        # feature_step = "passthrough"
        feature_step = PCA(svd_solver="full")
        clf = GradientBoostingClassifier(
            n_estimators=1500,
            n_iter_no_change=10,
            tol=0.0001,
            validation_fraction=0.10,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    steps = [
        ("yjpt", PowerTransformer(method="yeo-johnson", standardize=True)),
        ("outlier_removal", lof_sampler),
        ("feature_selection", feature_step),
        ("oversampling", SMOTE(random_state=42)),
        ("classifier", clf),
    ]

    return ImbPipeline(steps=steps)

# Define hyperparameter search space
def param_space(model_name: str) -> dict:
    """
    Returns the hyperparameter search space for the specified model.
    """
    if model_name == "SVC":
        param_grid = {
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "feature_selection__n_components": uniform(0.75, 0.20),  # Variance explained ratio [0.75, 0.95]
            "classifier__C": loguniform(1e-3, 1e6),
            "classifier__gamma": loguniform(1e-6, 1e2),
            "classifier__kernel": ["rbf"],
            # "outlier_removal__kw_args": [
            #     {
            #         "contamination": c,
            #         "n_neighbors": 20,
            #         "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            #         "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
            #     }
            #     for c in [0.025, 0.05, 0.075, 0.1]
            # ]
        }
        return param_grid
    elif model_name == "DT":
        param_grid = {
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "feature_selection__n_components": uniform(0.75, 0.20),  # [0.75, 0.95]
            "classifier__max_depth": randint(3, 20),  # [3, 19]
            "classifier__max_features": ["sqrt", "log2", None],
            "classifier__min_samples_split": uniform(0.05, 0.35),  # Fraction [0.05, 0.4]
            "classifier__min_samples_leaf": uniform(0.01, 0.09),  # Fraction [0.01, 0.1]
            "classifier__ccp_alpha": loguniform(1e-6, 1e-1),
        }
        return param_grid
    elif model_name == "RF":
        param_grid = {
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "classifier__n_estimators": randint(200, 1001),  # [200, 1000]
            "classifier__max_depth": randint(3, 20),  # [3, 19]
            "classifier__max_features": ["sqrt", "log2", 0.2, 0.3, 0.4, None],
            "classifier__min_samples_split": uniform(0.05, 0.45),  # [0.05, 0.5]
            "classifier__min_samples_leaf": uniform(0.01, 0.19),  # [0.01, 0.2]
            "classifier__ccp_alpha": loguniform(1e-6, 1e-1),
        }
        return param_grid
    elif model_name == "GB":
        param_grid = {
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "classifier__learning_rate": loguniform(1e-4, 1e-1),
            "classifier__max_depth": randint(3, 8),  # [3, 7]
            "classifier__max_features": ["sqrt", "log2", 0.2, 0.3, 0.4, None],
            "classifier__min_samples_split": uniform(0.05, 0.30),  # [0.05, 0.35]
            "classifier__min_samples_leaf": uniform(0.01, 0.09),  # [0.01, 0.1]
            "classifier__subsample": uniform(0.75, 0.25),  # [0.75, 1.0]
        }
        return param_grid
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

# Create data grid for experiments
def make_data_grid() -> list[dict]:
    """
    Creates a grid of data selection parameters for experiments.
    """
    countries = ["Botswana", "Ghana", "Nigeria", "Tanzania"]
    tasks_dict = {
        "QBF": ["QBF"],
        "JohnFarm": ["JohnFarm"],
        "Both": ["QBF", "JohnFarm"]
    }
    sexes_dict = {
        "Male": ["Male"],
        "Female": ["Female"],
        "Both": ["Male", "Female"]
    }

    grid = []
    for sex_key, task_key in product(sexes_dict.keys(), tasks_dict.keys()):
        grid.append({
            "countries": countries,
            "tasks": tasks_dict[task_key],
            "sexes": sexes_dict[sex_key],
            "tasks_key": task_key,
            "sexes_key": sex_key
        }
        )
    return grid

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
        refit=True
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
        verbose=cnfg["outer_verbose"]
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
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--outer_verbose", type=int, default=10)
    parser.add_argument("--inner_verbose", type=int, default=1)
    parser.add_argument("--outer_n_jobs", type=int, default=-1)
    parser.add_argument("--inner_n_jobs", type=int, default=1)
    args = parser.parse_args()
    
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

    print(
        f"\n*** Running {cnfg['outer_splits']} outer folds and {cnfg['n_repeats']} repeats "
        f"for {args.model_name} | sex={cnfg['sexes_key']} | task={cnfg['tasks_key']} ***\n"
    )
    run_experiment(cnfg)

if __name__ == "__main__":
    main()