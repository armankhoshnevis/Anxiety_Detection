from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory

from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, cross_validate

from scripts.utils.data_loader import load_data
from scripts.utils.model_factory import build_pipeline, param_space
from scripts.utils.post_processing import save_results, plot_shap_summary, compute_fold_shap

# Run the experiment with nested cross-validation
def run_experiment(config: dict):
    """
    Runs nested cross-validation with hyperparameter tuning and saves the results.
    """
    # Load data
    X, y, groups = load_data(config)

    # Define scoring metrics
    scoring = {
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
        "average_precision": "average_precision",
        "f1": "f1"
    }

    # Setup caching for pipeline
    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)
    
    try:
        # Build pipeline
        pipeline = build_pipeline(config=config, memory=memory)

        # Get hyperparameter space
        param_distributions = param_space(config=config)

        # Setup cross-validation
        outer_splits = []
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

        # Setup hyperparameter tuning
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=config["n_iter"],
            scoring="roc_auc",
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
            verbose=config["outer_verbose"],
            error_score='raise'
        )

        # Save results
        results_df, scoring_statistics_df, outer_df, inner_df = save_results(config, results, scoring)
        
        # Compute SHAP values across all outer folds
        all_shap_dfs, total_shap_df, shap_df_avg = compute_fold_shap(outer_splits, results, config["model_name"], X, y, config)
        
        # Plot SHAP summary
        plot_shap_summary(shap_df_avg, X, config)
    
    finally:
        # Clean up temporary cache
        rmtree(cachedir)
