import os
import pandas as pd
import numpy as np

import shap
import matplotlib

try:
    from IPython import get_ipython
except ImportError:
    get_ipython = None

if get_ipython is None or get_ipython() is None:
    matplotlib.use("Agg")

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
    os.makedirs(config["out_dir"], exist_ok=True)

    # DataFrame of results with best parameters
    results_df = pd.DataFrame(results).drop(columns=["estimator"])
    params_df = pd.DataFrame.from_records(est.best_params_ for est in results["estimator"])
    
    if config["model_name"] == "GB":
        params_df["classifier__n_estimators"] = [
            est.best_estimator_.named_steps["classifier"].n_estimators_ for est in results["estimator"]
        ]
    
    elif config["model_name"] == "MLP":
        classifiers = [
            est.best_estimator_.named_steps["classifier"]
            for est in results["estimator"]
        ]
        params_df["classifier__loss"] = [
            getattr(clf, "loss_", None)
            for clf in classifiers
        ]
        params_df["classifier__best_loss"] = [
            getattr(clf, "best_loss_", None)
            for clf in classifiers
        ]
        params_df["classifier__best_validation_score"] = [
            getattr(clf, "best_validation_score_", None)
            for clf in classifiers
        ]
        params_df["classifier__n_iter"] = [
            getattr(clf, "n_iter_", None)
            for clf in classifiers
        ]
    
    results_df = pd.concat([results_df.reset_index(drop=True), params_df.reset_index(drop=True)], axis=1)
    results_df = results_df.sort_values("test_roc_auc", ascending=False)
    results_df.to_csv(f"{config['out_dir']}/results.csv", index=False)

    # Summary statistics of metrics
    scoring_statistics_df = pd.DataFrame({
        k: results[f"test_{v}"] for k, v in scoring.items()
    }).agg(["mean", "std"]).T
    scoring_statistics_df.to_csv(f"{config['out_dir']}/scoring_statistics.csv", index=True)

    # Outer CV results
    n_outer = config["outer_splits"]
    n_total = config["outer_splits"] * config["n_repeats"]
    outer_df = pd.DataFrame({
        "repeat": (np.arange(n_total) // n_outer) + 1,
        "outer_fold": (np.arange(n_total) % n_outer) + 1,
        **{k: results[f"test_{v}"] for k, v in scoring.items()}
    })
    outer_df.to_csv(f"{config['out_dir']}/outer_cv_results.csv", index=False)

    if config["model_name"] != "MLP":
        inner_df = pd.DataFrame([
            {
                "repeat": (i // config["outer_splits"]) + 1,
                "outer_fold": (i % config["outer_splits"]) + 1,
                "inner_best_score": est.best_score_,
                "inner_best_params": est.best_params_,
                "selected_features": list(est.best_estimator_.named_steps["feature_selector"].get_feature_names_out()) if config["feature_selector_method"] != "passthrough" else "passthrough",
                "n_selected_features": len(list(est.best_estimator_.named_steps["feature_selector"].get_feature_names_out())) if config["feature_selector_method"] != "passthrough" else "passthrough"
            }
            for i, est in enumerate(results["estimator"])
        ])
        inner_df.to_csv(f"{config['out_dir']}/inner_cv_results.csv", index=False)

    else:
        inner_df = pd.DataFrame([
            {
                "repeat": (i // config["outer_splits"]) + 1,
                "outer_fold": (i % config["outer_splits"]) + 1,
                "inner_best_score": est.best_score_,
                "inner_best_params": est.best_params_,
                "selected_features": list(est.best_estimator_.named_steps["feature_selector"].get_feature_names_out()) if config["feature_selector_method"] != "passthrough" else "passthrough",
                "n_selected_features": len(list(est.best_estimator_.named_steps["feature_selector"].get_feature_names_out())) if config["feature_selector_method"] != "passthrough" else "passthrough"
            }
            for i, est in enumerate(results["estimator"])
        ])

        loss_validation_df = pd.DataFrame([
            {
                "repeat": (i // config["outer_splits"]) + 1,
                "outer_fold": (i % config["outer_splits"]) + 1,
                "iteration": iteration + 1,
                "loss": loss,
                "validation_score": score
            }
            for i, clf in enumerate(classifiers)
            for iteration, (loss, score) in enumerate(zip(getattr(clf, "loss_curve_", []), getattr(clf, "validation_scores_", [])))
        ])
        
        inner_df.to_csv(f"{config['out_dir']}/inner_cv_results.csv", index=False)
        loss_validation_df.to_csv(f"{config['out_dir']}/loss_validation_curves.csv", index=False)
    
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
    os.makedirs(config["out_dir"], exist_ok=True)
    
    shap_background_fraction = 1/4
    shap_eval_fraction = 1/3

    shap_background_min = 100 # 25 or 100 or 200
    shap_background_max = 200 # 50 or 200 or 1000

    shap_eval_min = 50 # 12 or 50
    shap_eval_max = 100 # 25 or 100 or 500

    all_shap_dfs = []
    for fold_idx, ((train_idx, val_idx), search_estimator) in enumerate(zip(outer_splits, results['estimator'])):
        best_estimator = search_estimator.best_estimator_

        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]

        preprocessor = best_estimator[:-2]  # Exclude oversampling and classifier
        classifier = best_estimator[-1]

        X_train_trans = preprocessor.transform(X_train_fold)
        X_val_trans = preprocessor.transform(X_val_fold)

        selected_features_names = X_train_trans.columns.tolist()
        all_features_names = X.columns.tolist()

        # Determine the number of background samples for SHAP
        n_background = min(
            len(X_train_trans),
            max(shap_background_min, int(np.ceil(len(X_train_trans) * shap_background_fraction))),
            shap_background_max
        )

        # Sample evaluation data for SHAP from the transformed validation set
        n_eval = min(
            len(X_val_trans),
            max(shap_eval_min, int(np.ceil(len(X_val_trans) * shap_eval_fraction))),
            shap_eval_max,
        )
        X_val_trans_sampled = X_val_trans.sample(
            n=n_eval,
            random_state=42 + fold_idx,
        )

        # Compute SHAP values based on the model type
        if model_name in ["DT", "RF", "GB", "XGB", "LGBM"]:    
            background = X_train_trans.sample(
                n=n_background,
                random_state=42 + fold_idx
            )
            explainer = shap.TreeExplainer(
                classifier,
                data=background,
                model_output="probability",
                feature_perturbation="interventional",
            )
            shap_values = explainer.shap_values(X_val_trans_sampled, check_additivity=False)
        
        elif model_name == "SVC":
            background = shap.kmeans(
                X_train_trans,
                n_background
            )
            explainer = shap.KernelExplainer(
                classifier.decision_function, 
                background
            )
            shap_values = explainer.shap_values(
                X_val_trans_sampled,
                nsamples=2*X_val_trans_sampled.shape[1]+512
            )
        
        elif model_name == "MLP":
            background = shap.kmeans(
                X_train_trans,
                n_background
            )
            explainer = shap.KernelExplainer(
                lambda X_batch: classifier.predict_proba(
                    pd.DataFrame(X_batch, columns=selected_features_names)
                )[:, 1],
                background,
            )
            shap_values = explainer.shap_values(
                X_val_trans_sampled,
                nsamples=2 * X_val_trans_sampled.shape[1] + 512,
            )
        
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
        
        shap_df = pd.DataFrame(0.0, index=X_val_trans_sampled.index, columns=all_features_names)
        shap_df.loc[:, selected_features_names] = pd.DataFrame(
            shap_values,
            index=X_val_trans_sampled.index,
            columns=selected_features_names,
        )
        all_shap_dfs.append(shap_df)

    total_shap_df = pd.concat(all_shap_dfs, axis=0)
    shap_df_avg = total_shap_df.groupby(total_shap_df.index).mean()

    shap_df_avg.to_csv(f"{config['out_dir']}/shap_values_avg_{config['model_name']}.csv")
    total_shap_df.to_csv(f"{config['out_dir']}/shap_values_all_{config['model_name']}.csv")

    return all_shap_dfs, total_shap_df, shap_df_avg

def plot_shap_summary(shap_df_avg, X, config):
    """
    Plot SHAP summary plots (bar and dot) and save them to files.
    Args:
        shap_df_avg (pandas DataFrame): Average SHAP values dataframe.
        X (pandas DataFrame): Original feature dataframe.
        config (dict): Configuration dictionary containing output directory and model name.
    """
    os.makedirs(config["out_dir"], exist_ok=True)
    X_shap = X.loc[shap_df_avg.index]

    # Bar plot
    fig = shap.summary_plot(
        shap_df_avg.values,
        X_shap,
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
    plt.close()

    # Dot plot
    fig = shap.summary_plot(
        shap_df_avg.values,
        X_shap,
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
    plt.close()
