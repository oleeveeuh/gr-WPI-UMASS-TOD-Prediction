import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from math import sqrt
import os

path = (r"/Users/olivialiau/Downloads/REUDATA/")
train_datasets = ['BA11_60_MM_train.csv', 'BA11_60_log_train.csv']
test_datasets = ['BA11_60_MM_test.csv', 'BA11_60_log_test.csv']
i = 0

for file in train_datasets:
    train_file_in = (train_datasets[i])
    test_file_in = (test_datasets[i])

    test_path_in = os.path.join(path, test_file_in)
    train_path_in = os.path.join(path, train_file_in)


    df = pd.read_csv(test_path_in)
    df2 = pd.read_csv(train_path_in)
    df_filtered = df.drop(columns=['TOD_pos'])
    df_filtered2 = df2.drop(columns=['TOD_pos'])
    X = df_filtered.to_numpy()
    X2 = df_filtered2.to_numpy()

    kernels = ['rbf', 'sigmoid', 'cosine']
    gammas = [0.01, 0.1, 1.0, 10.0]
    degrees = [2, 3, 4]
    n_components = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    eigen_solver = ['auto', 'dense', 'arpack', 'randomized']


    best_score = 1.0000
    best_params = None
    best_model = None
    apply_model = None
    reconstruct = None

    for kernel in kernels:
        for gamma in gammas:
            for degree in degrees:
                for component in n_components:
                    kpca = KernelPCA(kernel=kernel, gamma=gamma, degree=degree, n_components=component, fit_inverse_transform=True)
                    kpca_results = kpca.fit_transform(X)
                    total_variance_reduced = np.var(kpca_results, axis=0).sum()
                    inverse = kpca.inverse_transform(kpca_results)
                    score = mean_squared_error(X, inverse)

                    if score < best_score:
                        best_model = kpca_results
                        apply_model = kpca.transform(X2)
                        reconstruct = kpca.inverse_transform(kpca_results)
                        best_score = score
                        best_params = (kernel, gamma, degree, component)

    r2 = r2_score(X, reconstruct)
    rmse = sqrt(mean_squared_error(X, reconstruct))
    nrmse = rmse/sqrt(np.mean(X**2))

    print("Best parameters found:", best_params)
    print(f'Reconstruction Error (MSE): {best_score}')
    print(f'R2): {r2}')
    print(f'RMSE): {rmse}')
    print(f'NRMSE): {nrmse}')

    finaltraindf = pd.DataFrame(best_model)
    finaltestdf = pd.DataFrame(apply_model)
    finaltraindf['TOD'] = df['TOD_pos']
    finaltestdf['TOD'] = df2['TOD_pos']

    out_directory = (r"/Users/olivialiau/Downloads/KPCADATA/")
    os.makedirs(out_directory, exist_ok=True)
    train_file_out = ("KPCA" + train_file_in)
    test_file_out = ("KPCA" + test_file_in)
    train_file_path = os.path.join(out_directory, train_file_out)
    test_file_path = os.path.join(out_directory, test_file_out)
    finaltraindf.to_csv(train_file_path, index=False)
    finaltestdf.to_csv(test_file_path, index=False)
    print(f"Datasets saved")
    i += 1