import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def lof_outlier_removal(X, y, n_neighbors=20, contamination=0.05, algorithm='auto', metric='manhattan'):
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        algorithm=algorithm,
        metric=metric,
        leaf_size=30,
        novelty=False)
    
    y_pred = lof.fit_predict(X_arr)
    mask_inliers = y_pred == 1

    return X_arr[mask_inliers], y_arr[mask_inliers]