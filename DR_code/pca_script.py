import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Method arguments in order from 1 to 5:

# (1) path of training csv
# (2) path of where you want the transformed training set to go
# (3) path of testing csv
# (4) path of where you want the transformed testing set to go
# (5) optional target_variance property that by default is 0.9 (90%)

# Returned values in order from 1 to 4:

# (1) transformed training dataframe
# (2) untouched TOD values from training set as a dataframe
# (3) transformed testing dataframe
# (4) untouched TOD values from testing set as a dataframe

def reduce_and_save(input_train_csv, output_train_csv, input_test_csv, output_test_csv, target_variance=0.9):
    df_train = pd.read_csv(input_train_csv)
    df_test = pd.read_csv(input_test_csv)

    # Get TOD values and remove the column from training and testing sets
    TOD_train = df_train.pop('TOD_pos')
    TOD_test = df_test.pop('TOD_pos')

    # Convert dfs to np arrays
    x_train = df_train.values
    x_test = df_test.values

    # Initialize and apply PCA to training
    pca = PCA()
    pca_x = pca.fit_transform(x_train)

    # Get cumulative variances
    cumulative_variences = np.cumsum(pca.explained_variance_ratio_)

    # Get index of cumulative_variences element that is first to be >= target_variance, resulting in the number of components that
    # explain (target_varience * 100)% of the variance
    num_dim = np.argmax(cumulative_variences >= target_variance) + 1

    # Reinitialize and fit PCA on training set with restriction of num_dim and then transform testing set using same PCA
    pca = PCA(n_components=num_dim)
    reduced_x_train = pca.fit_transform(x_train)
    reduced_x_test = pca.transform(x_test)  

    
    # Convert to dataframes
    reduced_x_train_df = pd.DataFrame(reduced_x_train, columns=[f'PC{i + 1}' for i in range(num_dim)])
    reduced_x_test_df = pd.DataFrame(reduced_x_test, columns=[f'PC{i + 1}' for i in range(num_dim)])

    # Concatenate with the 'TOD' column
    final_train_df = pd.concat([TOD_train.reset_index(drop=True), reduced_x_train_df], axis=1)
    final_test_df = pd.concat([TOD_test.reset_index(drop=True), reduced_x_test_df], axis=1)

    # Make output paths
    output_path_train = os.path.join(os.getcwd(), output_train_csv)
    output_path_test = os.path.join(os.getcwd(), output_test_csv)

    # Convert dataframe to csv
    os.makedirs(os.path.dirname(output_path_train), exist_ok=True)
    final_train_df.to_csv(output_path_train, index=False)
    final_test_df.to_csv(output_path_test, index=False)

    GREEN = '\033[92m'
    RESET = '\033[0m'
    print(f'{GREEN}CSV files have been transformed using PCA and saved.{RESET}')

    # return reduced_x_train_df, y_train, reduced_x_test_df, y_test


# get the path to data
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '..', 'data', 'train test split data')
data_dir = os.path.normpath(data_dir)

# folder names
folder_BA11 = 'BA11'
folder_BA47 = 'BA47'
split_60 = '60'
split_70 = '70'
split_80 = '80'
method_log = 'log'
method_MM = 'MM'
method_None = 'nonnormalized'

folders = [folder_BA11, folder_BA47]
splits = [split_60, split_70, split_80]
methods = [method_log, method_MM, method_None]
variance = [0.9, 0.95]

for var in variance:
    postfix_num = int(var * 100)
    for folder in folders:
        for split in splits:
            for method in methods:
                # Construct the file paths for train
                train_name = f"{folder}_{split}_{method}_train.csv"
                train_file = os.path.join(data_dir, folder, train_name)
                output_train_file = os.path.join(data_dir, folder, f'PCA{postfix_num}', f"{folder}_{split}_{method}_DR_train.csv")

                # file path for test
                test_name = f"{folder}_{split}_{method}_test.csv"
                test_file = os.path.join(data_dir, folder, test_name)
                output_test_file = os.path.join(data_dir, folder, f'PCA{postfix_num}', f"{folder}_{split}_{method}_DR_test.csv")
                reduce_and_save(input_train_csv=train_file, output_train_csv=output_train_file, input_test_csv=test_file, output_test_csv=output_test_file)