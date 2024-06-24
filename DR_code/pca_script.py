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
    y_train = df_train['TOD_pos'].values
    df_train.drop(columns=['TOD_pos'], inplace=True)

    y_test = df_test['TOD_pos'].values
    df_test.drop(columns=['TOD_pos'], inplace=True)

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

    # Make output paths
    output_path_train = os.path.join(os.getcwd(), output_train_csv)
    output_path_test = os.path.join(os.getcwd(), output_test_csv)

    # Convert dataframe to csv
    reduced_x_train_df.to_csv(output_path_train, index=False)
    reduced_x_test_df.to_csv(output_path_test, index=False)

    GREEN = '\033[92m'
    RESET = '\033[0m'
    print(f'{GREEN}CSV files have been transformed using PCA and saved.{RESET}')

    return reduced_x_train_df, y_train, reduced_x_test_df, y_test