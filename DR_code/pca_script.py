import pandas as pd
import numpy as np
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

def reduce_and_save(input_train_csv, output_train_csv, input_test_csv, output_test_csv, target_variance):
    df_train = pd.read_csv(input_train_csv)
    df_test = pd.read_csv(input_test_csv)

    # Get TOD values and remove the column from training and testing sets
    TOD_train = df_train.pop('TOD')
    TOD_test = df_test.pop('TOD')

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

    # Convert dataframe to csv
    final_train_df.to_csv(output_train_csv, index=False)
    final_test_df.to_csv(output_test_csv, index=False)

    GREEN = '\033[92m'
    RESET = '\033[0m'
    print(f'{GREEN}CSV files have been transformed using PCA and saved.{RESET}')

    # return reduced_x_train_df, y_train, reduced_x_test_df, y_test


# get the path to data
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '..', 'data', 'encoded')
data_dir = os.path.normpath(data_dir)
reduced_data_dir = os.path.join(data_dir, '..', 'reduced_encoded')
reduced_data_dir = os.path.normpath(reduced_data_dir)

# folder names
folder_BA11 = 'BA11'
folder_BA47 = 'BA47'
folder_full = 'full_data'
folder_encoded = 'encoded'
split_60 = '60'
split_70 = '70'
split_80 = '80'
method_log = 'log'
method_MM = 'MM'
method_None = 'nonnormalized'

folders = [folder_BA11, folder_BA47, folder_full]
splits = [split_60, split_70, split_80]
methods = [method_log, method_MM]
variance = [0.9, 0.95]
windows = ['window1', 'window2', 'window3']

f = ''

for var in variance:
    postfix_num = int(var * 100)
    for split in splits:
        for method in methods:
            for window in windows:
                # Construct the file paths for train
                

                train_file = f"BA47_{split}_{method}_{window}_train.csv" # train file name

               
                train_path = os.path.join(data_dir, train_file) # train file path

                
                output_train_file = f'BA47_{split}_{method}_{window}_PCA_{postfix_num}_train.csv' # reduced train file name

                
                output_train_path = os.path.join(reduced_data_dir, output_train_file) # where reduced train file will go

                # file path for test
                
                test_file = f"BA47_{split}_{method}_{window}_test.csv"

                
                test_path = os.path.join(data_dir, test_file)

                
                output_test_file = f'BA47_{split}_{method}_{window}_PCA_{postfix_num}_test.csv'

                
                output_test_path = os.path.join(reduced_data_dir, output_test_file)

                reduce_and_save(input_train_csv=train_path, output_train_csv=output_train_path, input_test_csv=test_path, output_test_csv=output_test_path, target_variance=var)