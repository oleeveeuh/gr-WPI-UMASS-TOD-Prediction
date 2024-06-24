import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import os

# All datasets should be in one folder (set path to that folder)
# For each type (full, full, Unified), replace all 
# If error or missing file, set index to last position
path = (r"/Users/olivialiau/Downloads/REUDATA/full_data")
train_datasets = ['full_60_MM_train.csv', 'full_60_log_train.csv', 
                  'full_70_MM_train.csv', 'full_70_log_train.csv',
                  'full_80_MM_train.csv', 'full_80_log_train.csv']
test_datasets = ['full_60_MM_test.csv', 'full_60_log_test.csv', 
                 'full_70_MM_test.csv', 'full_70_log_test.csv',
                 'full_80_MM_test.csv', 'full_80_log_test.csv']
i = 0

# Perform for each pair of train/test datasets
for file in train_datasets:

    train_file_in = (train_datasets[i])
    test_file_in = (test_datasets[i])

    test_path_in = os.path.join(path, test_file_in)
    train_path_in = os.path.join(path, train_file_in)


    # if file is missing
    if not os.path.exists(test_path_in):
        print(f"{test_path_in} not found.")
        print(f"Terminated at index {i}.")
        exit()
    if not os.path.exists(train_path_in):
        print(f"{train_path_in} not found.")
        print(f"Terminated at index {i}.")
        exit()


    # Import, drop TOD, convert to array
    df = pd.read_csv(test_path_in)
    df2 = pd.read_csv(train_path_in)
    df_filtered = (df.drop(columns=['TOD_pos']))
    df_filtered2 = (df2.drop(columns=['TOD_pos']))
    X = df_filtered.to_numpy()
    X2 = df_filtered2.to_numpy()

    # Parameter grid for KPCA
    kernels = ['poly']
    gammas = [0.01, 0.1, 1.0, 10.0, 15.0]
    degrees = np.arange(1, 5)
    n_components = np.arange(2, 20)

    # Manual Grid Search (scored by reconstruction error)
    best_score = 1.0000
    best_params = None
    best_model = None
    apply_model = None
    reconstruct = None

    for kernel in kernels:
        for gamma in gammas:
            for degree in degrees:
                for component in n_components:
                    kpca = KernelPCA(kernel=kernel, gamma=gamma, degree=degree, n_components=component, random_state=None, fit_inverse_transform=True)
                    kpca_results = None
                    score = 1
                    try:
                        kpca_results = kpca.fit_transform(X)
                        inverse = kpca.inverse_transform(kpca_results)
                        score = mean_squared_error(X, inverse)
                    except np.linalg.LinAlgError as e:
                        print("Error:", e)
                    except AttributeError as a:
                        print("Error:", a)          

                    if abs(score-.10) < abs(best_score-.10):
                        best_score = score
                        best_model = kpca_results
                        apply_model = kpca.transform(X2)
                        reconstruct = kpca.inverse_transform(kpca_results)
                        best_params = (kernel, gamma, degree, component)

    r2 = r2_score(X, reconstruct)
    rmse = sqrt(mean_squared_error(X, reconstruct))
    nrmse = rmse/sqrt(np.mean(X**2))
    residuals = X - reconstruct
    residual_variance = np.mean(residuals ** 2)

    print("Best parameters found:", best_params)
    print(f'Reconstruction Error (MSE): {best_score}')
    print("Residual Variance:", residual_variance)
    print(f'R2): {r2}')
    print(f'RMSE): {rmse}')
    print(f'NRMSE): {nrmse}')

    finaltraindf = pd.DataFrame(best_model)
    finaltestdf = pd.DataFrame(apply_model)
    finaltraindf['TOD'] = df['TOD_pos']
    finaltestdf['TOD'] = df2['TOD_pos']

    out_directory = (r"/Users/olivialiau/Downloads/KPCADATA/")
    os.makedirs(out_directory, exist_ok=True)
    train_file_out = ("KPCA_" + train_file_in)
    test_file_out = ("KPCA_" + test_file_in)
    train_file_path = os.path.join(out_directory, train_file_out)
    test_file_path = os.path.join(out_directory, test_file_out)
    finaltraindf.to_csv(train_file_path, index=False)
    finaltestdf.to_csv(test_file_path, index=False)
    print(f"Datasets at index {i} saved.")
    i += 1