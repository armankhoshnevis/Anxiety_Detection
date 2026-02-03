from functools import partial
from scipy.stats import randint, uniform, loguniform

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import PowerTransformer

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .preprocessing import lof_outlier_removal, mi_score_func

# Build the machine learning pipeline
def build_pipeline(model_name: str, memory=None) -> ImbPipeline:
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
    
    mi_score_partial_func = partial(mi_score_func, n_neighbors=5, random_state=42)
    feature_step = SelectKBest(score_func=mi_score_partial_func)
    
    if model_name == "SVC":
        # feature_step = PCA(svd_solver="full")
        clf = SVC(probability=False, random_state=42)
    elif model_name == "DT":
        # feature_step = "passthrough"
        # feature_step = PCA(svd_solver="full")
        clf = DecisionTreeClassifier(random_state=42)
    elif model_name == "RF":
        # feature_step = "passthrough"
        clf = RandomForestClassifier(
            class_weight=None,
            n_jobs=1,
            random_state=42,
            )
    elif model_name == "GB":
        # feature_step = "passthrough"
        # feature_step = PCA(svd_solver="full")
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
    
    return ImbPipeline(steps=steps, memory=memory)

# Define hyperparameter search space
def param_space(model_name: str) -> dict:
    """
    Returns the hyperparameter search space for the specified model.
    """
    if model_name == "SVC":
        param_grid = {
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "feature_selection__k": randint(10, 67),  # [10, 66]
            "classifier__C": loguniform(1e-3, 1e6),
            "classifier__gamma": loguniform(1e-6, 1e2),
            "classifier__kernel": ["rbf"],
        }
        return param_grid
    elif model_name == "DT":
        param_grid = {
            "oversampling__k_neighbors": randint(3, 8),  # [3, 7]
            "feature_selection__k": randint(10, 67),  # [10, 66]
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
            "feature_selection__k": randint(10, 67),  # [10, 66]
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
            "feature_selection__k": randint(10, 67),  # [10, 66]
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