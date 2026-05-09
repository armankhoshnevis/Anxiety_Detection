import numpy as np
import pandas as pd

from functools import partial
from scipy.stats import randint, uniform, loguniform

from .preprocessing import lof_outlier_removal

from sklearn.preprocessing import PowerTransformer

from imblearn import FunctionSampler

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_classif

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Define the custom correlation-based feature selection class
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

# Build the machine learning pipeline
def build_pipeline(config, memory=None):
    """Builds and returns a machine learning pipeline with power transformation and standard scaling,
    outlier removal, feature selection, oversampling, and classification steps.
    
    Args:
        config (dict): Configuration dictionary containing model and feature selection parameters.
        memory: Optional memory parameter for caching transformers in the pipeline.
    
    Returns:
        pipeline (ImbPipeline): An imbalanced-learn pipeline with the specified steps.
    """
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
        feature_selector = CorrelationBasedFeatureSelection(intercorr_threshold=0.90, target_corr_threshold=0.25)
    
    elif config["feature_selector_method"] == "rfe":
        feature_selector = RFE(estimator=RandomForestClassifier(n_estimators=25, random_state=42), step = 0.1)
    
    elif config["feature_selector_method"] == "passthrough":
        feature_selector = "passthrough"

    smote = SMOTE(random_state=42)

    model_name = config["model_name"]
    if model_name == "SVC":
        clf = SVC(probability=False, random_state=42)
    
    elif model_name == "DT":
        clf = DecisionTreeClassifier(random_state=42)
    
    elif model_name == "RF":
        clf = RandomForestClassifier(
            class_weight=None,
            n_jobs=1,
            random_state=42,
            )
    
    elif model_name == "GB":
        clf = GradientBoostingClassifier(
            n_estimators=1500,
            n_iter_no_change=10,
            tol=1e-4,
            validation_fraction=0.10,
            random_state=42,
        )
    
    elif model_name == "XGB":
        clf = XGBClassifier(
            booster="gbtree",
            tree_method="hist",
            n_jobs=1,
            verbosity=3,
            random_state=42
        )
    
    elif model_name == "LGBM":
        clf = LGBMClassifier(
            boosting_type="gbdt",
            n_jobs=1,
            random_state=42,
        )
    
    elif model_name == "MLP":
        clf = MLPClassifier(
            random_state=42,
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.15
        )
    
    elif model_name == "NB":
        clf = GaussianNB()
    
    elif model_name == "KNN":
        clf = KNeighborsClassifier()
    
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    steps = [
        ("yj_pt", yj_pt),
        ("outlier_removal", lof_sampler),
        ("feature_selector", feature_selector),
        ("oversampling", smote),
        ("classifier", clf),
    ]
    
    pipeline = ImbPipeline(steps=steps, memory=memory).set_output(transform="pandas")
    return pipeline

# Define search space for hyperparameter tuning
def param_space(config):
    """Returns the hyperparameter search space for the specified model.
    
    Args:
        config (dict): Configuration dictionary containing model and feature selection method.
    
    Returns:
        param_distributions (dict): A dictionary containing the hyperparameter search space for the specified model
    """
    model_name = config["model_name"]
    feature_selector_method = config["feature_selector_method"]
    
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
    
    if model_name == "SVC":
        param_distributions.update({
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "classifier__C": loguniform(1e-3, 1e6),
            "classifier__gamma": loguniform(1e-6, 1e2),
            "classifier__kernel": ["rbf"],
        })
        return param_distributions
    
    elif model_name == "DT":
        param_distributions.update({
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "classifier__max_depth": randint(3, 21),  # [3, 20]
            "classifier__max_features": ["sqrt", "log2", 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            "classifier__min_samples_split": uniform(0.05, 0.35),  # Fraction [0.05, 0.4]
            "classifier__min_samples_leaf": uniform(0.01, 0.09),  # Fraction [0.01, 0.1]
            "classifier__ccp_alpha": loguniform(1e-6, 1e-1),
        })
        return param_distributions
    
    elif model_name == "RF":
        param_distributions.update({
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "classifier__n_estimators": randint(200, 1001),  # [200, 1000]
            "classifier__max_depth": randint(3, 21),  # [3, 20]
            "classifier__max_features": ["sqrt", "log2", 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            "classifier__min_samples_split": uniform(0.05, 0.45),  # [0.05, 0.5]
            "classifier__min_samples_leaf": uniform(0.01, 0.19),  # [0.01, 0.2]
            "classifier__ccp_alpha": loguniform(1e-6, 1e-1),
        })
        return param_distributions
    
    elif model_name == "GB":
        param_distributions.update({
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "classifier__learning_rate": loguniform(5e-3, 5e-1),
            "classifier__max_depth": randint(3, 8),  # [3, 7]
            "classifier__max_features": ["sqrt", "log2", 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            "classifier__min_samples_split": uniform(0.05, 0.30),  # [0.05, 0.35]
            "classifier__min_samples_leaf": uniform(0.01, 0.09),  # [0.01, 0.1]
            "classifier__subsample": uniform(0.75, 0.25),  # [0.75, 1.0]
        })
        return param_distributions
    
    elif model_name == "XGB":
        param_distributions.update({
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "classifier__n_estimators": randint(200, 1001),  # [200, 1000]
            "classifier__learning_rate": loguniform(1e-3, 3e-1),  # [0.001, 0.3]
            "classifier__min_child_weight": uniform(1, 7),  # [1, 8]
            "classifier__max_depth": randint(3, 11),  # [3, 10]
            "classifier__gamma": uniform(0, 1),  # [0, 1]
            "classifier__subsample": uniform(0.6, 0.4),  # [0.6, 1.0]
            "classifier__colsample_bytree": uniform(0.6, 0.4),  # [0.6, 1.0]
            "classifier__reg_lambda": loguniform(1e-3, 1e1),  # [0.001, 10]
        })
        return param_distributions
    
    elif model_name == "LGBM":
        param_distributions.update({
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "classifier__num_leaves": randint(24, 91),  # [24, 90]
            "classifier__min_child_samples": randint(15, 101),  # [15, 100]
            "classifier__max_depth": randint(3, 11),  # [3, 10]
            "classifier__min_split_gain": loguniform(1e-3, 1e1),  # [0.001, 10]
            "classifier__n_estimators": randint(200, 1001),  # [200, 1000]
            "classifier__learning_rate": loguniform(1e-3, 3e-1),  # [0.001, 0.3]
            "classifier__subsample": uniform(0.6, 0.4),  # [0.6, 1.0]
            "classifier__colsample_bytree": uniform(0.6, 0.4),  # [0.6, 1.0]
            "classifier__reg_lambda": loguniform(1e-3, 1e1),  # [0.001, 10]
        })
        return param_distributions
    
    elif model_name == "MLP":
        hidden_layers_counts = [3, 4, 5, 6]
        hidden_layer_widths = [16, 32, 64, 128]
        hidden_layer_sizes = [
            tuple([width] * n_layers)
            for n_layers in hidden_layers_counts
            for width in hidden_layer_widths
        ]

        param_distributions.update({
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "classifier__hidden_layer_sizes": hidden_layer_sizes,
            "classifier__learning_rate_init": loguniform(1e-5, 1e-2),  # [0.00001, 0.01]
            "classifier__batch_size": [16, 32, 64, 128, "auto"],
            "classifier__activation": ["relu", "tanh", "logistic"],
            "classifier__solver": ["adam", "sgd", "lbfgs"],
            "classifier__alpha": loguniform(1e-5, 1e-2),  # [0.00001, 0.01]
        })
        return param_distributions
    
    elif model_name == "NB":
        param_distributions.update({
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "classifier__var_smoothing": loguniform(1e-11, 1e-7),  # [1e-11, 1e-7]
        })
        return param_distributions
    
    elif model_name == "KNN":
        param_distributions.update({
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "classifier__n_neighbors": list(range(3, 22, 2)),  # [3, 5, 7, ..., 21]
            "classifier__weights": ["uniform", "distance"],
            "classifier__algorithm": ["ball_tree", "kd_tree", "brute"],
            "classifier__p": [1, 2],
        })
        return param_distributions
    
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
