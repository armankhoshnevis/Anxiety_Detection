import os
from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory

from sklearn.model_selection import (
    StratifiedGroupKFold, GroupKFold,
    RandomizedSearchCV, cross_validate
)

from scripts.utils.data_loader import load_data
from scripts.utils.model_factory import build_pipeline, param_space
from scripts.utils.post_processing import save_results, plot_shap_summary, compute_fold_shap, save_and_plot_gad_predictions

# Run the experiment with nested cross-validation
def run_experiment(config: dict):
    """
    Runs nested cross-validation with hyperparameter tuning and saves the results.
    """
    # Load data
    X, y, groups, num_cols, cat_cols = load_data(config)

    # Define scoring metrics
    if config["prediction_task"] == "classification-binary":
        scoring = {
            "roc_auc": "roc_auc",
            "balanced_accuracy": "balanced_accuracy",
            "average_precision": "average_precision",
            "f1": "f1"
        }
        tuning_scoring = "roc_auc"
    
    else:
        scoring = {
            "neg_root_mean_squared_error": "neg_root_mean_squared_error",
            "neg_mean_absolute_error": "neg_mean_absolute_error",
            "r2": "r2",
        }
        tuning_scoring = "neg_root_mean_squared_error"

    # Setup caching for pipeline
    cache_root = os.environ.get("PIPELINE_CACHE_DIR")
    if cache_root:
        os.makedirs(cache_root, exist_ok=True)
        cachedir = mkdtemp(prefix="pipeline_cache_", dir=cache_root)
    else:
        cachedir = mkdtemp(prefix="pipeline_cache_")

    memory = Memory(location=cachedir, verbose=0)
    
    try:
        # Build pipeline
        pipeline = build_pipeline(
            config=config, 
            num_cols=num_cols,
            cat_cols=cat_cols,
            memory=memory
        )

        # Get hyperparameter space
        param_distributions = param_space(config=config)

        # Setup cross-validation
        outer_splits = []
        if config["prediction_task"] == "classification-binary":
            for i in range(config["n_repeats"]):
                sgkf = StratifiedGroupKFold(
                    n_splits=config["outer_splits"],
                    shuffle=True,
                    random_state=42+i
                )
                outer_splits.extend(list(sgkf.split(X, y, groups)))
            
            inner_cv = StratifiedGroupKFold(
                n_splits=config["inner_splits"],
                shuffle=True,
                random_state=42
            )
        
        else:
            for i in range(config["n_repeats"]):
                gkf = GroupKFold(
                    n_splits=config["outer_splits"],
                    shuffle=True,
                    random_state=42+i
                )
                outer_splits.extend(list(gkf.split(X, y, groups)))
            
            inner_cv = GroupKFold(
                n_splits=config["inner_splits"],
                shuffle=True,
                random_state=42
            )
        
        # Setup hyperparameter tuning
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=config["n_iter"],
            scoring=tuning_scoring,
            n_jobs=config["inner_n_jobs"],
            cv=inner_cv,
            verbose=config["inner_verbose"],
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
            n_jobs=config["outer_n_jobs"],
            pre_dispatch=config["outer_n_jobs"],
            verbose=config["outer_verbose"],
            error_score='raise'
        )

        # Save results
        results_df, scoring_statistics_df, outer_df, inner_df = save_results(config, results, scoring)
        
        # Compute SHAP values across all outer folds
        all_shap_dfs, total_shap_df, shap_df_avg = compute_fold_shap(outer_splits, results, config["model_name"], X, config)
        
        # Plot SHAP summary
        plot_shap_summary(shap_df_avg, X, config)

        # Plot and save GAD regression predictions
        if config["prediction_task"] == "regression":
            save_and_plot_gad_predictions(outer_splits, results, X, y, groups, config)
    
    finally:
        # Clean up temporary cache
        rmtree(cachedir, ignore_errors=True)
