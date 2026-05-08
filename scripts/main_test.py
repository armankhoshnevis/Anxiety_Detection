import os

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform

from sklearn.preprocessing import PowerTransformer

from imblearn import FunctionSampler

from functools import partial
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_classif

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, cross_validate

from xgboost import XGBClassifier

from scripts.utils.data_loader import create_configs, load_data
from scripts.utils.preprocessing import lof_outlier_removal
from scripts.utils.post_processing import save_results, compute_fold_shap, plot_shap_summary

class CorrelationBasedFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, intercorr_threshold=0.90, target_corr_threshold=0.25):
        self.intercorr_threshold = intercorr_threshold
        self.target_corr_threshold = target_corr_threshold
        self.to_drop_intercorrelated_ = []
        self.to_drop_target_corr_ = []
        self.to_drop_ = []
        self.selected_features_ = []

    def fit(self, X, y):
        X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        y_series = pd.Series(y) if isinstance(y, np.ndarray) else y
        
        corr_matrix = X_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        target_corr = X_df.apply(lambda col: col.corr(y_series)).abs()
        
        drop_intercorr_set = set()
        for col in upper_tri.columns:
            for row in upper_tri.index:
                if upper_tri.loc[row, col] > self.intercorr_threshold:
                    if row not in drop_intercorr_set and col not in drop_intercorr_set:
                        if target_corr[row] >= target_corr[col]:
                            drop_intercorr_set.add(col)
                        else:
                            drop_intercorr_set.add(row)
        
        self.to_drop_intercorrelated_ = list(drop_intercorr_set)

        X_reduced = X_df.drop(columns=self.to_drop_intercorrelated_, errors='ignore')
        target_corr_reduced = target_corr.loc[X_reduced.columns]
        n_reduced = len(target_corr_reduced)
        n_keep = int(np.ceil(self.target_corr_threshold * n_reduced))

        self.selected_features_ = (
            target_corr_reduced
            .sort_values(ascending=False)
            .head(n_keep)
            .index
            .tolist()
        )

        to_drop_target_corr_ = [
            col for col in X_reduced.columns
            if col not in self.selected_features_
        ]

        self.to_drop_target_corr_ = to_drop_target_corr_
        self.to_drop_ = self.to_drop_intercorrelated_ + self.to_drop_target_corr_

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
        X_selected = X_df.drop(columns=self.to_drop_, errors='ignore')
        return X_selected.values if isinstance(X, np.ndarray) else X_selected

    def set_output(self, transform):
        return self

# Setup experiment configurations
case_idx = -1
model_name = "XGB"
feature_selector_method = "mi_based"
N_REPEATS = 7
OUTER_SPLITS = 5
INNER_SPLITS = 5
N_ITER = 10
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
out_dir = f"../results_tests/{model_name}_{feature_selector_method}/sex={config['sexes_key']}/task={config['tasks_key']}"
config.update({"out_dir": out_dir})

# Load data
X, y, groups = load_data(config)

# Define scoring metrics
scoring = {
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy",
    "average_precision": "average_precision",
    "f1": "f1"
}

# Build pipeline
yj_pt = PowerTransformer(method="yeo-johnson", standardize=True)

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

if config["feature_selector_method"] == "mi_based":
    feature_selector = SelectPercentile(score_func=partial(mutual_info_classif, n_neighbors=5, random_state=42))

elif config["feature_selector_method"] == "corr_based":
    feature_selector = CorrelationBasedFeatureSelection()

elif config["feature_selector_method"] == "rfe":
    feature_selector = RFE(estimator=RandomForestClassifier(n_estimators=25, random_state=42), step = 0.1)

smote = SMOTE(random_state=42)

clf = XGBClassifier(
    booster="gbtree",
    tree_method="hist",
    n_jobs=1,
    verbosity=3,
    random_state=42
)

steps = [
    ("yjpt", yj_pt),
    ("outlier_removal", lof_sampler),
    ("feature_selector", feature_selector),
    ("oversampling", smote),
    ("classifier", clf),
]

pipeline = ImbPipeline(steps=steps).set_output(transform="pandas")

# Param distributions for RandomizedSearchCV
param_distributions = {}
if feature_selector_method == "mi_based":
    param_distributions.update({
        "feature_selector__percentile": randint(50, 91),  # [50, 90]
    })

elif feature_selector_method == "corr_based":
    param_distributions.update({
        "feature_selector__intercorr_threshold": uniform(0.85, 0.1),  # [0.85, 0.95]
        "feature_selector__target_corr_threshold": uniform(0.2, 0.1),  # [0.2, 0.3]
    })

elif feature_selector_method == "rfe":
    param_distributions.update({
        "feature_selector__n_features_to_select": uniform(0.1, 0.9),  # [0.1, 1.0]
    })

param_distributions.update({
    "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
    "classifier__n_estimators": randint(200, 1001),  # [200, 1000]
    "classifier__learning_rate": loguniform(1e-3, 3e-1),  # [0.001, 0.3]
    "classifier__min_child_weight": uniform(1, 7),  # [1, 8]
    "classifier__max_depth": randint(1, 10),  # [1, 9]
    "classifier__gamma": uniform(0, 1),  # [0, 1]
    "classifier__subsample": uniform(0.6, 0.4),  # [0.6, 1.0]
    "classifier__colsample_bytree": uniform(0.6, 0.4),  # [0.6, 1.0]
    "classifier__reg_lambda": loguniform(1e-3, 1e1),  # [0.001, 10]
})

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
    pre_dispatch=config["outer_n_jobs"],
    verbose=config["outer_verbose"],
    error_score='raise'
)

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

if feature_selector_method == "corr_based":
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