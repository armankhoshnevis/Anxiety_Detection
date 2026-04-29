from pathlib import Path
import pandas as pd
import numpy as np

import shap
import matplotlib.pyplot as plt

def save_results(config, results, scoring):
    """Save the results of the model training and evaluation to CSV files.

    Args:
        config (dict): Configuration dictionary.
        results (dict): Dictionary containing results and estimators from model training.
        scoring (dict): Dictionary of scoring metrics used for evaluation.
    
    Returns:
        tuple: A tuple containing dataframes of results with best parameters, scoring statistics, outer CV results, and inner CV results.
    """
    out_dir = config["out_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # DataFrame of results with best parameters
    results_df = pd.DataFrame(results).drop(columns=["estimator"])
    params_df = pd.DataFrame.from_records(est.best_params_ for est in results["estimator"])
    if config["model_name"] == "GB":
        params_df["classifier__n_estimators"] = [
            est.best_estimator_.named_steps["classifier"].n_estimators_ for est in results["estimator"]
        ]
    results_df = pd.concat([results_df.reset_index(drop=True), params_df.reset_index(drop=True)], axis=1)
    results_df = results_df.sort_values("test_roc_auc", ascending=False)
    results_df.to_csv(f"{out_dir}/results.csv", index=False)

    # Summary statistics of metrics
    scoring_statistics_df = pd.DataFrame({
        k: results[f"test_{v}"] for k, v in scoring.items()
    }).agg(["mean", "std"]).T
    scoring_statistics_df.to_csv(f"{out_dir}/scoring_statistics.csv", index=True)

    # Outer CV results
    n_outer = config["outer_splits"]
    n_total = config["outer_splits"] * config["n_repeats"]
    outer_df = pd.DataFrame({
        "repeat": (np.arange(n_total) // n_outer) + 1,
        "outer_fold": (np.arange(n_total) % n_outer) + 1,
        **{k: results[f"test_{v}"] for k, v in scoring.items()}
    })
    outer_df.to_csv(f"{out_dir}/outer_cv_results.csv", index=False)

    if config["feature_selector_method"] == "corr_based":
        inner_df = pd.DataFrame([
            {
                "repeat": (i // config["outer_splits"]) + 1,
                "outer_fold": (i % config["outer_splits"]) + 1,
                "inner_best_score": est.best_score_,
                "inner_best_params": est.best_params_,
                "selected_features": list(est.best_estimator_.named_steps["feature_selector"].selected_features_),
                "n_selected_features": len(list(est.best_estimator_.named_steps["feature_selector"].selected_features_))
            }
            for i, est in enumerate(results["estimator"])
        ])
    else:
        inner_df = pd.DataFrame([
            {
                "repeat": (i // config["outer_splits"]) + 1,
                "outer_fold": (i % config["outer_splits"]) + 1,
                "inner_best_score": est.best_score_,
                "inner_best_params": est.best_params_,
                "selected_features": list(est.best_estimator_.named_steps["feature_selector"].get_feature_names_out()),
                "n_selected_features": len(list(est.best_estimator_.named_steps["feature_selector"].get_feature_names_out()))
            }
            for i, est in enumerate(results["estimator"])
        ])
    inner_df.to_csv(f"{out_dir}/inner_cv_results.csv", index=False)
    
    return results_df, scoring_statistics_df, outer_df, inner_df

def compute_fold_shap(outer_splits, results, model_name, X, y, config):
    """Compute SHAP values per outer fold and tuned model

    Args:
        outer_splits (list): List of outer fold train/validation indices.
        results (dict): Dictionary containing results and estimators from inner cv.
        model_name (str): Name of the model used.
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target data.
        config (dict): Configuration dictionary.

    Raises:
        ValueError: If the model_name is unsupported for SHAP computation.

    Returns:
        tuple: A tuple containing a list of SHAP dataframes for each fold, a dataframe of all SHAP values, and a dataframe of average SHAP values.
    """

    all_shap_dfs = []
    for fold_idx, ((train_idx, val_idx), search_estimator) in enumerate(zip(outer_splits, results['estimator'])):
        search_estimator = results['estimator'][fold_idx]
        best_estimator = search_estimator.best_estimator_

        X_train_fold = X.iloc[train_idx]  # .reset_index(drop=True)
        X_val_fold = X.iloc[val_idx]  # .reset_index(drop=True)
        
        preprocessor = best_estimator[:-2]  # Exclude oversampling and classifier
        classifier = best_estimator[-1]
        
        X_train_trans = preprocessor.transform(X_train_fold)
        X_val_trans = preprocessor.transform(X_val_fold)

        selected_features_names = X_train_trans.columns.tolist()
        all_features_names = X.columns.tolist()

        # Decide on background data for SHAP based on model type
        if model_name not in ["SVC"]:
            X_val_trans_sampled = X_val_trans.copy()
            background = X_train_trans.copy()
        
        else:
            X_val_trans_sampled = X_val_trans.sample(n=int(np.floor(len(X_val_trans)/3)), random_state=42).reset_index(drop=True)
            background = shap.kmeans(X_train_trans, 50)

        # Compute SHAP values based on model type
        if model_name in ["DT", "RF"]:
            explainer = shap.TreeExplainer(classifier, background)
            shap_values = explainer.shap_values(X_val_trans_sampled)[:, :, 1]
        
        elif model_name in ["GB", "XGB"]:
            explainer = shap.TreeExplainer(classifier, background)
            shap_values = explainer.shap_values(X_val_trans_sampled)
        
        elif model_name == "SVC":
            explainer = shap.KernelExplainer(classifier.decision_function, background)
            shap_values = explainer.shap_values(X_val_trans_sampled)
        
        elif model_name == "MLP":
            explainer = shap.DeepExplainer(classifier, background)
            shap_values = explainer.shap_values(X_val_trans_sampled)
        
        else:
            raise ValueError(f"Unsupported model_name for SHAP: {model_name}")
        
        shap_df = pd.DataFrame(0.0, index=X_val_trans_sampled.index, columns=all_features_names)
        shap_df.update(pd.DataFrame(shap_values, index=X_val_trans_sampled.index, columns=selected_features_names))
        all_shap_dfs.append(shap_df)
    
    total_shap_df = pd.concat(all_shap_dfs, axis=0)
    shap_df_avg = total_shap_df.groupby(total_shap_df.index).mean()

    shap_df_avg.to_csv(f"{config['out_dir']}/shap_values_avg_{config['model_name']}.csv")
    total_shap_df.to_csv(f"{config['out_dir']}/shap_values_all_{config['model_name']}.csv")
    shap_df_avg.to_csv(f"{config['out_dir']}/shap_values_avg_{config['model_name']}.csv")

    return all_shap_dfs, total_shap_df, shap_df_avg

def plot_shap_summary(shap_df_avg, X, config):
    """
    Plot SHAP summary plots (bar and dot) and save them to files.
    Args:
        shap_df_avg (pandas DataFrame): Average SHAP values dataframe.
        X (pandas DataFrame): Original feature dataframe.
        config (dict): Configuration dictionary containing output directory and model name.
    """
    # Bar plot
    fig = shap.summary_plot(
        shap_df_avg.values,
        X,
        plot_type="bar",
        show=False,
        max_display=10,
        plot_size=(15, 10)
    )
    ax = plt.gca()
    ax.set_xlabel("Mean Absolute SHAP Value", fontsize=20)
    ax.set_ylabel(ax.get_ylabel(), fontsize=20)
    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)
    plt.savefig(f"{config['out_dir']}/SHAP_summary_bar_plot_{config['model_name']}.png", bbox_inches='tight')
    plt.show()

    # Dot plot
    fig = shap.summary_plot(
        shap_df_avg.values,
        X,
        plot_type="dot",
        show=False,
        max_display=10,
        plot_size=(15, 8)
    )
    ax = plt.gca()
    ax.set_xlabel("Mean Absolute SHAP Value", fontsize=20)
    ax.set_ylabel(ax.get_ylabel(), fontsize=20)
    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)
    plt.savefig(f"{config['out_dir']}/SHAP_summary_dot_plot_{config['model_name']}.png", bbox_inches='tight')
    plt.show()
