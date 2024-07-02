import sys
import os
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to the sys.path
sys.path.append(parent_dir)
# Now you can import the module
from read_train import *
from datetime import datetime
import pandas as pd
import numpy as np
import os


def create_windows(combination, w_size = 3, verbose = False):
    # read in the relevant data
    target, split, n_method = combination
    train_data, test_data = read_data_file(target=target, n_method=n_method, split=split, split_xy=False)
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
