import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform

from sklearn.preprocessing import PowerTransformer

from sklearn.decomposition import PCA
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

from auxiliary_function import get_training_data, lof_outlier_removal

# Load the datasets
countries = ['Botswana', 'Ghana', 'Nigeria', 'Tanzania']
sexes = ['Male']  # Can be ['Male', 'Female'] for both
tasks = ['QBF', 'JohnFarm']   # Can be ['QBF', 'JF'] or both

X_arr, y_arr = get_training_data(countries, tasks, sexes)

print(f"Anxious class balance: {len(y_arr[y_arr == 1])/len(y_arr):.2f}; {len(y_arr[y_arr == 1])}")
print(f"Non-Anxious class balance: {len(y_arr[y_arr == 0])/len(y_arr):.2f}; {len(y_arr[y_arr == 0])}")
print(f"Total samples: {len(y_arr)}")

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
pipeline = ImbPipeline([
    ("yjpt", PowerTransformer(method='yeo-johnson', standardize=True)),
    ("outlier_removal", lof_sampler),
    ("pca", PCA(svd_solver='full')),
    ("oversampling", SMOTE(random_state=42)),
    ("clf", SVC(probability=False, kernel='rbf', random_state=42))
])

# Define the hyperparameter search space
contamination_values = np.linspace(0.01, 0.1, 10)
lof_kw_args_list = [
    {
        'contamination': c,
        'n_neighbors': 20,
        'algorithm': 'auto',
        'metric': 'manhattan'
    } 
    for c in contamination_values
]
param_grid_rs = {
    # 'outlier_removal__kw_args': lof_kw_args_list,
    'oversampling__k_neighbors': randint(3, 8),
    'pca__n_components': uniform(0.5, 0.45),  # Explained variance ratio [0.5, 0.95]
    'clf__C': loguniform(1e-2, 1e6),
    'clf__gamma': loguniform(1e-5, 1e2),
}

# Define multiple scoring metrics
scoring = {
    'roc_auc': 'roc_auc',
    'balanced_accuracy': 'balanced_accuracy',
    'average_precision': 'average_precision',
    'f1_score': 'f1'
}

# Nested Cross-Validation
NUM_REPETITION = 7
OUTER_SPLITS = 5
INNER_SPLITS = 5
N_ITER_SEARCH = 2000

# Define outer and inner cross-validation strategies
outer_cv = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=NUM_REPETITION, random_state=42)
inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=42)

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

# Process and save results
results_df = pd.DataFrame(results).sort_values(by='test_roc_auc', ascending=False)
results_df['best_params'] = [est.best_params_ for est in results['estimator']]

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

results_df.to_csv("results.csv", index=False)
results_summary_df.to_csv("results_summary.csv", index=True)
outer_df.to_csv("outer_cv_results.csv", index=False)
inner_df.to_csv("inner_cv_results.csv", index=False)