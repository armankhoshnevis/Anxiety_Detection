import numpy as np
import pandas as pd

from pathlib import Path

from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory

from sklearn.model_selection import (
    StratifiedGroupKFold,
    RandomizedSearchCV,
    cross_validate
)

from utils.data_loader import get_training_data
from utils.model_factory import build_pipeline, param_space
from utils.post_processing import save_results, plot_shap_summary, compute_fold_shap

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

    # Setup caching for pipeline
    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)
    
    try:
        # Build pipeline
        pipeline = build_pipeline(model_name=cnfg["model_name"], memory=memory)

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
        save_results(cnfg, results, scoring)
        
        # Compute SHAP values across all outer folds
        _, _, shap_df_avg = compute_fold_shap(outer_splits, results, cnfg["model_name"], X, y, cnfg)

        # Plot SHAP summary
        plot_shap_summary(shap_df_avg, X, cnfg)
    
    finally:
        # Clean up temporary cache
        rmtree(cachedir)