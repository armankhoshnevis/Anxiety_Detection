import pandas as pd
import numpy as np

import shap
import matplotlib.pyplot as plt

def save_results(cnfg, results, scoring):
    """Save the results of the model training and evaluation to CSV files.

    Args:
        cnfg (dict): Configuration dictionary.
        results (dict): Dictionary containing results and estimators from model training.
        scoring (dict): Dictionary of scoring metrics used for evaluation.
    """
    out_dir = cnfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full results with best parameters
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

    # Summary statistics of metrics
    results_summary_df = pd.DataFrame({
        m: results[f"test_{m}"] for m in list(scoring.keys())
    }).agg(["mean", "std"]).T
    results_summary_df.to_csv(out_dir / "results_summary.csv")

    # Outer CV results
    n_outer = cnfg["outer_splits"]
    n_total = cnfg["outer_splits"] * cnfg["n_repeats"]
    outer_df = pd.DataFrame({
        "repeat": (np.arange(n_total) // n_outer) + 1,
        "outer_fold": (np.arange(n_total) % n_outer) + 1,
        **{m: results[f"test_{m}"] for m in list(scoring.keys())}
    })
    outer_df.to_csv(out_dir / "outer_cv_results.csv", index=False)

    # Inner CV results
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

def compute_fold_shap(outer_splits, results, model_name, X, y):
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

        if model_name not in ["SVC"]:
            X_val_trans_sampled = X_val_trans.copy()
            background = X_train_trans.copy()
        else:
            X_val_trans_sampled = X_val_trans.sample(n=int(np.floor(len(X_val_trans)/2)), random_state=42).reset_index(drop=True)
            background = shap.kmeans(X_train_trans, 50)

        if model_name in ["DT", "RF"]:
            explainer = shap.TreeExplainer(classifier, background)
            shap_values = explainer.shap_values(X_val_trans_sampled)[:, :, 1]
        elif model_name == "GB":
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

    return all_shap_dfs, total_shap_df, shap_df_avg


def plot_shap_summary(shap_df_avg, X, cnfg):
    """
    Plot SHAP summary plots (bar and dot) and save them to files.
    Args:
        shap_df_avg (pandas DataFrame): Average SHAP values dataframe.
        X (pandas DataFrame): Original feature dataframe.
        cnfg (dict): Configuration dictionary containing output directory and model name.
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
    ax.set_xlabel(ax.get_xlabel(), fontsize=24)
    ax.set_ylabel(ax.get_ylabel(), fontsize=24)
    plt.setp(ax.get_xticklabels(), fontsize=24)
    plt.setp(ax.get_yticklabels(), fontsize=24)
    plt.savefig(f"{cnfg['out_dir']}/SHAP_summary_bar_plot_{cnfg['model_name']}.png", bbox_inches='tight')
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
    ax.set_xlabel(ax.get_xlabel(), fontsize=24)
    ax.set_ylabel(ax.get_ylabel(), fontsize=24)
    plt.setp(ax.get_xticklabels(), fontsize=24)
    plt.setp(ax.get_yticklabels(), fontsize=24)
    plt.savefig(f"{cnfg['out_dir']}/SHAP_summary_dot_plot_{cnfg['model_name']}.png", bbox_inches='tight')
    plt.show()
