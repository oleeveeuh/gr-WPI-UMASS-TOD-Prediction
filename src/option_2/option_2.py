from enum import Enum
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from datetime import datetime
import pandas as pd
import numpy as np
import os

# Def Classes ------------------------------------------------------------------------
class Target(Enum):
    BA11 = 0
    BA47 = 1
    full = 2

class Split(Enum):
    S60 = 0
    S70 = 1
    S80 = 2

class Normalize_Method(Enum):
    Log = 0
    MM = 1
    NN = 2

class DR_Method(Enum):
    PCA = 0
    ICA = 1
    KPCA = 2
    Isomap = 3

class Variance(Enum):
    V90 = 0
    V95 = 1

# Mapping from Enums to actual values
target_map = {Target.BA11: 'BA11', Target.BA47: 'BA47', Target.full: 'full'}
split_map = {Split.S60: '60', Split.S70: '70', Split.S80: '80'}
n_method_map = {Normalize_Method.Log: 'log', Normalize_Method.MM: 'MM', Normalize_Method.NN: 'nonnormalized'}
#dr_method_map = {DR_Method.PCA: 'PCA', DR_Method.ICA: 'ICA', DR_Method.KPCA: 'KPCA', DR_Method.Isomap: 'Isomap'}
variance_map = {Variance.V90: '90', Variance.V95: '95'}

# get the path to data
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '..', '..', 'data', 'train_test_split_data')
data_dir = os.path.normpath(data_dir)


def filter_combinations(targets=None, splits=None, n_methods=None, variances=None):
    """
    Generate combinations of parameters for filtering datasets.

    Parameters:
    targets (list of Target, optional): List of Target enums to include in the combinations.
                                         Defaults to excluding 'full'.
    splits (list of Split, optional): List of Split enums to include in the combinations.
                                      Defaults to including all splits.
    n_methods (list of Normalize_Method, optional): List of Normalize_Method enums to include in the combinations.
                                                    Defaults to excluding 'nonnormalized'
    variances (list of Variance, optional): List of Variance enums to include in the combinations.
                                            Defaults to including all variances.

    Returns:
    list of tuples: A list of tuples, each containing a unique combination of the specified parameters.
                    Each tuple contains (target, split, n_method, variance).

    Example:
    combinations = filter_combinations(
        targets=[Target.BA11, Target.BA47],
        splits=[Split.S60, Split.S70],
        n_methods=[Normalize_Method.Log, Normalize_Method.MM],
        variances=[Variance.V90]
    )
    """
    # Define default lists excluding 'full' and 'nonnormalized'
    default_targets = [Target.BA11, Target.BA47]
    default_n_methods = [Normalize_Method.Log, Normalize_Method.MM]

    # Use provided arrays or defaults
    effective_targets = targets if targets else default_targets
    effective_splits = splits if splits else list(Split)
    effective_n_methods = n_methods if n_methods else default_n_methods
    #effective_DR_methods = DR_methods if DR_methods else list(DR_Method)
    #effective_variances = variances if variances else list(Variance)

    # Generate all combinations
    all_combinations = product(effective_targets, effective_splits, effective_n_methods)

    # Return the filtered combinations
    return list(all_combinations)


def read_files(target, split, n_method):
    '''
    Input:
        target_data(BA11, BA47, Combine)
        normalize_method(log, MM, NN)
        split(60, 70, 80)

    Output:
        X_train, y_train, X_test, y_test
    '''
    # Get sub-file in /data/ for adding to file path:
    sub_file = target_map[target] if (target_map[target] != "full") else "full_data"
    # Construct the file paths for train
    train_name = f"{sub_file}/{target_map[target]}_{split_map[split]}_{n_method_map[n_method]}_train.csv"
    train_file = os.path.join(data_dir, train_name)

    # file path for test
    test_name = f"{sub_file}/{target_map[target]}_{split_map[split]}_{n_method_map[n_method]}_test.csv"
    test_file = os.path.join(data_dir, test_name)

    # Load data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    return train_data, test_data

def create_windows(combination, w_size = 3, verbose = False):
    # read in the relevant data
    target, split, n_method = combination
    train_data, test_data = read_files(target=target, n_method=n_method, split=split)
    if verbose:
        print(
            f"Creating Windows for: {target_map[target]}_{split_map[split]}_{n_method_map[n_method]}")

    results = {}
    i = 1
    for df in [train_data, test_data]:
        df_name = "train" if (i == 1) else "test"
        df = df.sort_values(by='TOD').reset_index(drop=True)

        # Create the new DataFrame
        new_data = {col: [] for col in df.columns}
        # Iterate through the DataFrame and replace values
        for i in range(w_size+1, len(df) - w_size+1): # For w=3, the range is row 4(inclusive) through the 3rd from end (exclusive)
            for col in df.columns:
                if col in ['Age', 'TOD', 'Sex']:
                    new_data[col].append(df.loc[i, col])
                else:
                    col_index = df.columns.get_loc(col)
                    # Collect values from the preceding and following W rows
                    surrounding_values = df.iloc[i - w_size-1:i + w_size, col_index].tolist()
                    #print("Working on Row: ", i, " of value ",  df.loc[i, col])
                    #print("Extracting values from rows:", i - w_size, " through ", i + w_size)
                    #print("Corresponding to Values: ", surrounding_values)
                    new_data[col].append(surrounding_values)
         #Convert new_data back to a DataFrame
        new_df = pd.DataFrame(new_data)
        results[df_name] = new_df
        i += 1
    return results



def apply_autoencoder(combinations, verbose=False):

    results = []

    for combination in combinations:
        target, split, n_method = combination
        if verbose:
            print(
                f"Processing: {target_map[target]}_{split_map[split]}_{n_method_map[n_method]}")

        #train_data, test_data = read_files(target=target, n_method=n_method, split=split, dr_method=dr_method,
        #                                             variance=variance)
        #for df in [train_data, test_data]:
            #for row_num in range(1, len(df.shape[0])):
                #to_encode = df.iloc[(row_num-1):(row_num+1), ]



    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(data_dir, f'model_results_{timestamp}.csv')
    results_df.to_csv(results_file, index=False)

    if verbose:
        print(f"Results saved to {results_file}")

    return results_df


if __name__ == "__main__":
    # Get all the possible method combinations
    combinations = filter_combinations(
        targets=[Target.BA11],
        splits=[Split.S60],
        n_methods=[Normalize_Method.Log]
    )
    for combo in combinations:
        windows = create_windows(combo, w_size=3, verbose = True)
        for window in windows:
            filename = "/Users/tillieslosser/Downloads/BA11_60_log_"+window+".csv"
            windows[window].to_csv(filename)

    #Initialize a receptacle dictionary (DICT1)
    # For each combination:
        # Read the respective files - generate a warning if the file doesn't exist
        # Assign the files to a dictionary with setup key=filename : value = pd.DataFrame. There will be two files
        # Initialize a receptacle dictionary (DICT2) with setup keys= [TOD, Age, Sex, AE_output]: value=respective values
            # For each row (except first and last) in each file:
                # Generate a 3x235 array containing the gene expressions for each TOD.
                # Feed the 3x235 array into autoencoder, receive a single number output (AE_output).
                # join AE_output to a row with the TOD, Age, and Sex of the second row from the original 3x235 array
                # Concat output row to DICT2
            #Before moving onto next combination, add DICT2 to DICT1 with setup key=filename+"_AE"
