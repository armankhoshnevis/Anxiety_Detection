import pandas as pd
from itertools import product

# Load the data
def get_training_data(countries, tasks, sexes, base_path="../Datasets/"):
    """
    Loads and combines data based on selected countries, tasks, and sexes.
    
    Parameters:
    - countries: List of strings (e.g., ["Botswana", "Ghana"])
    - tasks: List of strings (e.g., ["QBF", "JF"])
    - sexes: List of strings (e.g., ["Male"] or ["Male", "Female"])
    
    Returns:
    - X_arr, y_arr: Numpy arrays for features and target
    - groups: Numpy array of participant SessionIDs
    """
    
    df_list = []
    
    for country in countries:
        for task in tasks:
            file_name = f"{base_path}{country}_GAD_eGeMAPS_{task}.csv"
            
            try:
                temp_df = pd.read_csv(file_name)
                temp_df = temp_df[temp_df["Sex"].isin(sexes)]
                temp_df["Anxiety_Binary"] = temp_df["GAD7_Total"].apply(lambda x: 1 if x >= 5 else 0)
                df_list.append(temp_df)
            
            except FileNotFoundError:
                print(f"Warning: File not found: {file_name}")
            except KeyError as e:
                print(f"Warning: Missing column in {file_name}: {e}")

    if not df_list:
        raise ValueError("No data loaded. Check your file paths and parameters.")

    combined_df = pd.concat(df_list, axis=0, ignore_index=True)

    metadata_cols = [
        "SessionID", "QBF_Name", "JohnFarm_Name", "Sex", "Age", "Health", "Health_Binary",
        "Country", "GAD7_Total", "Anxiety_Category", "Anxiety_Binary"
    ]
    
    groups = combined_df["SessionID"].astype(str).to_numpy()
    X = combined_df.drop(columns=metadata_cols, errors='ignore').to_numpy()
    y = combined_df["Anxiety_Binary"].to_numpy()

    return X, y, groups

# Create data grid for experiments
def make_data_grid() -> list[dict]:
    """
    Creates a grid of data selection parameters for experiments.
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
    return grid