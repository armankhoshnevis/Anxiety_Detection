import numpy as np
import pandas as pd

from scipy.stats import randint, uniform, loguniform

from .preprocessing import lof_outlier_removal

from sklearn.preprocessing import PowerTransformer

from imblearn import FunctionSampler

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from imblearn.over_sampling import SMOTE, SMOTENC

from sklearn.compose import ColumnTransformer

from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

# Build the machine learning pipeline
def build_pipeline(config, num_cols, cat_cols, memory=None):
    """Builds and returns a machine learning pipeline with power transformation and standard scaling,
    outlier removal, feature selection, oversampling, and classification steps.
    
    Args:
        config (dict): Configuration dictionary containing model and feature selection parameters.
        num_cols (list): List of numerical column names.
        cat_cols (list): List of categorical column names.
        memory: Optional memory parameter for caching transformers in the pipeline.
    
    Returns:
        pipeline (ImbPipeline): An imbalanced-learn pipeline with the specified steps.
    """
    is_classification = config["prediction_task"] == "classification-binary"
    
    yj_pt = PowerTransformer(method="yeo-johnson", standardize=True)
    if config["feature_set"] == "eGeMAPS_Demographics":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", yj_pt, num_cols),
                ("cat", "passthrough", cat_cols)
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )
        
        cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
        oversampler = (
            SMOTENC(categorical_features=cat_indices, random_state=42)
            if is_classification else "passthrough"
        )
    
    else:
        preprocessor = yj_pt
        oversampler = (
            SMOTE(random_state=42) 
            if is_classification else "passthrough"
        )
    
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
    
    if config["feature_selector_method"] == "rfe":
        feature_selector = (
            RFE(estimator=RandomForestClassifier(n_estimators=25, random_state=42), step = 0.1)
            if is_classification 
            else RFE(estimator=RandomForestRegressor(n_estimators=25, random_state=42), step = 0.1)
        )
    
    elif config["feature_selector_method"] == "passthrough":
        feature_selector = "passthrough"

    model_name = config["model_name"]
    if model_name == "SVC":
        model = (
            SVC(probability=True, random_state=42)
            if is_classification
            else SVR()
        )
    
    elif model_name == "DT":
        model = (
            DecisionTreeClassifier(random_state=42)
            if is_classification
            else DecisionTreeRegressor(random_state=42)
        )
    
    elif model_name == "RF":
        model = (
            RandomForestClassifier(class_weight=None, n_jobs=1, random_state=42)
            if is_classification
            else RandomForestRegressor(n_jobs=1, random_state=42)
        )
    
    elif model_name == "GB":
        model = (
            GradientBoostingClassifier(n_estimators=1500, n_iter_no_change=10, tol=1e-4,
                                       validation_fraction=0.10, random_state=42)
            if is_classification
            else GradientBoostingRegressor(n_estimators=1500, n_iter_no_change=10, tol=1e-4,
                                           validation_fraction=0.10, random_state=42)
        )
    
    elif model_name == "XGB":
        model = (
            XGBClassifier(booster="gbtree", tree_method="hist", n_jobs=1,
                          verbosity=0, random_state=42)
            if is_classification
            else XGBRegressor(booster="gbtree", tree_method="hist", n_jobs=1,
                              verbosity=0, random_state=42)
        )
    
    elif model_name == "LGBM":
        model = (
            LGBMClassifier(boosting_type="gbdt", n_jobs=1, random_state=42)
            if is_classification
            else LGBMRegressor(boosting_type="gbdt", n_jobs=1, random_state=42)
        )
    
    elif model_name == "MLP":
        model = (
            MLPClassifier(random_state=42, max_iter=2000, early_stopping=True,
                            n_iter_no_change=15, validation_fraction=0.15)
            if is_classification
            else MLPRegressor(random_state=42, max_iter=2000, early_stopping=True,
                              n_iter_no_change=15, validation_fraction=0.15)
        )
    
    elif model_name == "NB":
        if is_classification:
            model = GaussianNB()
        else:
            raise ValueError("Naive Bayes is not supported for regression tasks.")
    
    elif model_name == "KNN":
        model = (
            KNeighborsClassifier()
            if is_classification
            else KNeighborsRegressor()
        )
    
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    steps = [
        ("preprocessor", preprocessor),
        ("outlier_removal", lof_sampler),
        ("oversampler", oversampler),
        ("feature_selector", feature_selector),
        ("model", model),
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

    if feature_selector_method == "rfe":
        param_distributions.update({
            "feature_selector__n_features_to_select": uniform(0.1, 0.9),  # [0.1, 1.0]
        })
    
    if config["prediction_task"] == "classification-binary":
        param_distributions["oversampler__k_neighbors"] = randint(3, 8)  # [3, 7]
    
    if model_name == "SVC":
        param_distributions.update({
            "model__C": loguniform(1e-3, 1e6),
            "model__gamma": loguniform(1e-6, 1e2),
            "model__kernel": ["rbf"],
        })
    
    elif model_name == "DT":
        param_distributions.update({
            "model__max_depth": randint(3, 21),  # [3, 20]
            "model__max_features": ["sqrt", "log2", 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            "model__min_samples_split": uniform(0.05, 0.35),  # Fraction [0.05, 0.4]
            "model__min_samples_leaf": uniform(0.01, 0.09),  # Fraction [0.01, 0.1]
            "model__ccp_alpha": loguniform(1e-6, 1e-1),
        })
    
    elif model_name == "RF":
        param_distributions.update({
            "model__n_estimators": randint(200, 1001),  # [200, 1000]
            "model__max_depth": randint(3, 21),  # [3, 20]
            "model__max_features": ["sqrt", "log2", 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            "model__min_samples_split": uniform(0.05, 0.45),  # [0.05, 0.5]
            "model__min_samples_leaf": uniform(0.01, 0.19),  # [0.01, 0.2]
            "model__ccp_alpha": loguniform(1e-6, 1e-1),
        })
    
    elif model_name == "GB":
        param_distributions.update({
            "model__learning_rate": loguniform(5e-3, 5e-1),
            "model__max_depth": randint(3, 8),  # [3, 7]
            "model__max_features": ["sqrt", "log2", 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            "model__min_samples_split": uniform(0.05, 0.30),  # [0.05, 0.35]
            "model__min_samples_leaf": uniform(0.01, 0.09),  # [0.01, 0.1]
            "model__subsample": uniform(0.75, 0.25),  # [0.75, 1.0]
        })
    
    elif model_name == "XGB":
        param_distributions.update({
            "model__n_estimators": randint(200, 1001),  # [200, 1000]
            "model__learning_rate": loguniform(1e-3, 3e-1),  # [0.001, 0.3]
            "model__min_child_weight": uniform(1, 7),  # [1, 8]
            "model__max_depth": randint(3, 11),  # [3, 10]
            "model__gamma": uniform(0, 1),  # [0, 1]
            "model__subsample": uniform(0.6, 0.4),  # [0.6, 1.0]
            "model__colsample_bytree": uniform(0.6, 0.4),  # [0.6, 1.0]
            "model__reg_lambda": loguniform(1e-3, 1e1),  # [0.001, 10]
        })
    
    elif model_name == "LGBM":
        param_distributions.update({
            "model__num_leaves": randint(24, 91),  # [24, 90]
            "model__min_child_samples": randint(15, 101),  # [15, 100]
            "model__max_depth": randint(3, 11),  # [3, 10]
            "model__min_split_gain": loguniform(1e-3, 1e1),  # [0.001, 10]
            "model__n_estimators": randint(200, 1001),  # [200, 1000]
            "model__learning_rate": loguniform(1e-3, 3e-1),  # [0.001, 0.3]
            "model__subsample": uniform(0.6, 0.4),  # [0.6, 1.0]
            "model__colsample_bytree": uniform(0.6, 0.4),  # [0.6, 1.0]
            "model__reg_lambda": loguniform(1e-3, 1e1),  # [0.001, 10]
        })
    
    elif model_name == "MLP":
        hidden_layers_counts = [3, 4, 5, 6]
        hidden_layer_widths = [16, 32, 64, 128]
        hidden_layer_sizes = [
            tuple([width] * n_layers)
            for n_layers in hidden_layers_counts
            for width in hidden_layer_widths
        ]

        param_distributions.update({
            "model__hidden_layer_sizes": hidden_layer_sizes,
            "model__learning_rate_init": loguniform(1e-5, 1e-2),  # [0.00001, 0.01]
            "model__batch_size": [16, 32, 64, 128, "auto"],
            "model__activation": ["relu", "tanh", "logistic"],
            "model__solver": ["adam", "sgd", "lbfgs"],
            "model__alpha": loguniform(1e-5, 1e-2),  # [0.00001, 0.01]
        })
    
    elif model_name == "NB":
        param_distributions.update({
            "model__var_smoothing": loguniform(1e-11, 1e-7),  # [1e-11, 1e-7]
        })
    
    elif model_name == "KNN":
        param_distributions.update({            
            "model__n_neighbors": list(range(3, 22, 2)),  # [3, 5, 7, ..., 21]
            "model__weights": ["uniform", "distance"],
            "model__algorithm": ["ball_tree", "kd_tree", "brute"],
            "model__p": [1, 2],
        })
    
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    
    return param_distributions
