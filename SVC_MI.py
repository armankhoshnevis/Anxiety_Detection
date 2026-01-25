import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from functools import partial
import tempfile
import shutil

from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    RepeatedStratifiedKFold,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE

from outlier_removal import lof_outlier_removal

# Load the dataset
df = pd.read_csv("Datasets/Botswana_GAD_eGeMAPS_QBF.csv")

# Modify GAD7 target for binary classification
df['Anxiety_Binary'] = df['GAD7_Total'].apply(lambda x: 1 if x >= 5 else 0)

# Define Metadata to Drop
metadata_cols = [
    'SessionID', 'QBF_Name', 'Sex', 'Age', 'Health', 'Health_Binary',
    'Country', 'GAD7_Total', 'Anxiety_Category', 'Anxiety_Binary'
]

# Define feature and target
X = df.drop(columns=metadata_cols, errors='ignore')
X = X.select_dtypes(include=[np.number])
y = df['Anxiety_Binary']

X_arr = X.to_numpy()
y_arr = y.to_numpy()

# Create sampler for outlier removal
lof_sampler = FunctionSampler(
    func=lof_outlier_removal,
    kw_args={
        'contamination': 0.05,
        'n_neighbors': 20,
        'algorithm': 'auto',
        'metric': 'manhattan'
    },
    validate=False)

# Define the pipeline
cachedir = tempfile.mkdtemp()
mi_score_func = partial(mutual_info_classif, random_state=42)
pipeline = ImbPipeline([
    ("yjpt", PowerTransformer(method='yeo-johnson', standardize=True)),
    ("outlier_removal", lof_sampler),
    ("feature_selection", SelectPercentile(score_func=mi_score_func)),
    ("oversampling", SMOTE(k_neighbors=5, random_state=42)),
    ("clf", SVC(probability=False, kernel='rbf', random_state=42))
], memory=cachedir)

# mi_score_funcs = [
#     partial(mutual_info_classif, random_state=42, n_neighbors=3),
#     partial(mutual_info_classif, random_state=42, n_neighbors=5),
#     partial(mutual_info_classif, random_state=42, n_neighbors=10),
# ]
# param_grid_rs = {
#     "dim_reduc__score_func": mi_funcs,
#     "dim_reduc__k": randint(...),
#     ...
# }

# Define the hyperparameter search space
param_grid_rs = {
    'oversampling__k_neighbors': randint(3, 8),
    'feature_selection__percentile': randint(5, 51),
    'clf__C': loguniform(1e-1, 1e6),
    'clf__gamma': loguniform(1e-5, 1e1),
}

# Define multiple scoring metrics
scoring = {
    'roc_auc': 'roc_auc',
    'balanced_accuracy': 'balanced_accuracy',
    'average_precision': 'average_precision',
    'f1_score': 'f1'
}

# Nested Cross-Validation
NUM_REPETITION = 4
OUTER_SPLITS = 5
INNER_SPLITS = 3
N_ITER_SEARCH = 100

# Define outer and inner cross-validation strategies
outer_cv = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=NUM_REPETITION, random_state=42)
inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=42)
# inner_cv = StratifiedShuffleSplit(n_splits=INNER_SPLITS, random_state=42)

# Define the RandomizedSearchCV model
model = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid_rs,
    n_iter=N_ITER_SEARCH,
    scoring=scoring,
    refit='roc_auc',
    cv=inner_cv,
    return_train_score=True,
    n_jobs=1,
    verbose=1,
    random_state=42
)

# Execute nested cross-validation
results = cross_validate(
    model,
    X=X_arr,
    y=y_arr,
    cv=outer_cv,
    scoring=scoring,
    return_estimator=True,
    n_jobs=-1,
    verbose=1
)

shutil.rmtree(cachedir)

# Process and save results
results_df = pd.DataFrame(results).sort_values(by='test_roc_auc', ascending=False)

results_summary_df = pd.DataFrame({
    m: results[f"test_{m}"] for m in list(scoring.keys())
}).agg(["mean", "std"]).T

n_outer = OUTER_SPLITS
n_total = OUTER_SPLITS * NUM_REPETITION

outer_df = pd.DataFrame({
    "repeat": (np.arange(n_total) // n_outer) + 1,
    "outer_fold": (np.arange(n_total) % n_outer) + 1,
    **{m: results[f"test_{m}"] for m in list(scoring.keys())}
})

inner_df = pd.DataFrame([
    {
        "repeat": (i // OUTER_SPLITS) + 1,
        "outer_fold": (i % OUTER_SPLITS) + 1,
        "inner_best_score": est.best_score_,
        "inner_best_params": est.best_params_,
        "n_candidates": len(est.cv_results_["params"]),
    }
    for i, est in enumerate(results["estimator"])
])

# results_df.to_csv("results", index=False)
# results_summary_df.to_csv("results_summary.csv", index=True)
# outer_df.to_csv("outer_cv_results.csv", index=False)
# inner_df.to_csv("inner_cv_results.csv", index=False)