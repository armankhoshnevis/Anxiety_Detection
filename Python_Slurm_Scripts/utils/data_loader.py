import pandas as pd
from itertools import product

# Load the data
def load_data(config):
    """Prepares the training data based on the provided configuration.

    Args:
        config (dict): Configuration dictionary

    Returns:
        X (pandas DataFrame): Feature matrix
        y (pandas Series): Target vector
        groups (numpy array): Participant SessionIDs
    """
    
    countries = config["countries"]
    tasks = config["tasks"]
    sexes = config["sexes"]

    df_list = []

    for country in countries:
        for task in tasks:
            file_name = f"../Datasets/{country}_GAD_eGeMAPS_{task}.csv"
            
            temp_df = pd.read_csv(file_name)
            temp_df = temp_df[temp_df["Sex"].isin(sexes)]
            temp_df["Anxiety_Binary"] = temp_df["GAD7_Total"].apply(lambda x: 1 if x >= 5 else 0)
            df_list.append(temp_df)

    combined_df = pd.concat(df_list, axis=0, ignore_index=True)

    metadata_cols = [
        "SessionID", "QBF_Name", "JohnFarm_Name", "Sex", "Age", "Health", "Health_Binary",
        "Country", "GAD7_Total", "Anxiety_Category", "Anxiety_Binary"
    ]

    groups = combined_df["SessionID"].to_numpy()
    X = combined_df.drop(columns=metadata_cols, errors='ignore')
    y = combined_df["Anxiety_Binary"]

    return X, y, groups

# Create configuration file for experiments
def create_configs(case_idx, model_name, feature_selector_method, n_dict):
    """Creates a configuration dictionary for the specified case index, model name, feature selector method, and additional parameters.
    
    Args:
        case_idx (int): Index of the case to create configuration for.
        model_name (str): Name of the machine learning model to be used.
        feature_selector_method (str): Method for feature selection.
        n_dict (dict): Additional parameters to be included in the configuration.
    
    Returns:        
        config (dict): Configuration dictionary for the specified case. 
    """
    countries = ["Botswana", "Ghana", "Nigeria", "Tanzania"]
    tasks_dict = {
        "QBF": ["QBF"],
        "JohnFarm": ["JohnFarm"],
        "Both": ["QBF", "JohnFarm"]
    }
    sexes_dict = {
        "Male": ["Male"],
        "Female": ["Female"],
        "Both": ["Male", "Female"]
    }

    grid = []
    for sex_key, task_key in product(sexes_dict.keys(), tasks_dict.keys()):
        grid.append({
            "countries": countries,
            "tasks": tasks_dict[task_key],
            "sexes": sexes_dict[sex_key],
            "tasks_key": task_key,
            "sexes_key": sex_key
        }
        )
    
    config = grid[case_idx]
    config.update({"model_name": model_name})
    config.update({"feature_selector_method": feature_selector_method})
    config.update(n_dict)
    return config