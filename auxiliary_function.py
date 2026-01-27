import numpy as np
import pandas as pd

from sklearn.neighbors import LocalOutlierFactor

def get_training_data(countries, tasks, sexes, base_path="Datasets/"):
    """
    Loads and combines data based on selected countries, tasks, and sexes.
    
    Parameters:
    - countries: List of strings (e.g., ['Botswana', 'Ghana'])
    - tasks: List of strings (e.g., ['QBF', 'JF'])
    - sexes: List of strings (e.g., ['Male'] or ['Male', 'Female'])
    
    Returns:
    - X_arr, y_arr: Numpy arrays for features and target
    """
    
    df_list = []
    
    for country in countries:
        for task in tasks:
            # Construct file directory
            file_name = f"{base_path}{country}_GAD_eGeMAPS_{task}.csv"
            
            try:
                # Load the dataset
                temp_df = pd.read_csv(file_name)
                # Filter by sex
                temp_df = temp_df[temp_df['Sex'].isin(sexes)]
                # Create/Confirm binary anxiety target
                temp_df['Anxiety_Binary'] = temp_df['GAD7_Total'].apply(lambda x: 1 if x >= 5 else 0)
                # Append to list
                df_list.append(temp_df)
            
            except FileNotFoundError:
                print(f"Warning: File not found: {file_name}")
            except KeyError as e:
                print(f"Warning: Missing column in {file_name}: {e}")

    if not df_list:
        raise ValueError("No data loaded. Check your file paths and parameters.")

    # Combine all collected dataframes
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)

    
    # Define metadata columns to exclude from features
    metadata_cols = [
        'SessionID', 'QBF_Name', 'JohnFarm_Name', 'Sex', 'Age', 'Health', 'Health_Binary',
        'Country', 'GAD7_Total', 'Anxiety_Category', 'Anxiety_Binary'
    ]
    
    # Prepare features and target
    X = combined_df.drop(columns=metadata_cols, errors='ignore')
    y = combined_df['Anxiety_Binary']

    return X.to_numpy(), y.to_numpy()


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