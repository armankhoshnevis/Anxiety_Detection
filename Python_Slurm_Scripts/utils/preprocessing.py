import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_classif

# Outlier removal using Local Outlier Factor
def lof_outlier_removal(X, y, n_neighbors=20, contamination=0.05, algorithm='auto', metric='manhattan'):
    """Removes outliers from the dataset using Local Outlier Factor (LOF).
    Args:
        X (pandas DataFrame): Feature matrix.
        y (pandas Series): Target vector.
        n_neighbors (int, optional): Number of neighbors to use. Defaults to 20.
        contamination (float, optional): Proportion of outliers in the data set. Defaults to 0.05.
        algorithm (str, optional): Algorithm to compute nearest neighbors. Defaults to 'auto'.
        metric (str, optional): Distance metric to use. Defaults to 'manhattan'.

    Returns:
        tuple: Filtered feature matrix and target vector as numpy arrays.
    """
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        algorithm=algorithm,
        metric=metric,
        leaf_size=30,
        novelty=False)
    
    y_pred = lof.fit_predict(X)
    mask_inliers = y_pred == 1

    return X.iloc[mask_inliers], y.iloc[mask_inliers]