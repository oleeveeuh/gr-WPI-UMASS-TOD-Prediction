import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics import mean_squared_error, r2_score, pairwise_distances

from math import sqrt
import os

best_scores = []
# All datasets should be in one folder (set path to that folder)
# For each type (BA11, BA11, Unified), replace all 
# If error or missing file, set index to last position
path = (r"/Users/olivialiau/Documents/gr-WPI-UMASS-TOD-Project/data/train_test_split_data/full_data")
train_datasets = [
                # 'BA11_60_MM_train.csv', 'BA11_60_log_train.csv', 
                #   'BA11_70_MM_train.csv', 'BA11_70_log_train.csv',
                #   'BA11_80_MM_train.csv', 'BA11_80_log_train.csv']
                #   'BA47_60_MM_train.csv', 'BA47_60_log_train.csv', 
                #   'BA47_70_MM_train.csv', 'BA47_70_log_train.csv',
                #   'BA47_80_MM_train.csv', 'BA47_80_log_train.csv']
                  'full_60_MM_train.csv', 'full_60_log_train.csv', 
                  'full_70_MM_train.csv', 'full_70_log_train.csv',
                  'full_80_MM_train.csv', 'full_80_log_train.csv']

test_datasets = [
    # 'BA11_60_MM_test.csv', 'BA11_60_log_test.csv', 
    #              'BA11_70_MM_test.csv', 'BA11_70_log_test.csv',
    #              'BA11_80_MM_test.csv', 'BA11_80_log_test.csv',
    #              'BA47_60_MM_test.csv', 'BA47_60_log_test.csv', 
    #              'BA47_70_MM_test.csv', 'BA47_70_log_test.csv',
    #              'BA47_80_MM_test.csv', 'BA47_80_log_test.csv',
                 'full_60_MM_test.csv', 'full_60_log_test.csv', 
                 'full_70_MM_test.csv', 'full_70_log_test.csv',
                 'full_80_MM_test.csv', 'full_80_log_test.csv']
i = 0

def reconstruction_error(X_high_dim, X_low_dim):
    dist_high_dim = pairwise_distances(X_high_dim, metric='euclidean')
    dist_low_dim = pairwise_distances(X_low_dim, metric='euclidean')
    n_samples = X_high_dim.shape[0]
    error = np.sqrt(np.sum((dist_high_dim - dist_low_dim)**2) / (n_samples * (n_samples - 1)))
    max_possible_error = np.sqrt(np.sum((dist_high_dim - np.mean(dist_high_dim))**2) / (dist_high_dim.shape[0] * (dist_high_dim.shape[0] - 1)))
    reference_value = np.mean(dist_high_dim)

    percentage_error = (error - reference_value/ reference_value) * 100
    return percentage_error

def MAPE(orig, pred):
    n, m = orig.shape
    abs_percentage_errors = np.abs((orig - pred) / (orig+1e-8))
    abs_percentage_errors[np.isnan(abs_percentage_errors)] = 0
    mape = 100 / (n * m) * np.sum(abs_percentage_errors)

    return mape

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
    df_filtered = (df.drop(columns=['TOD']))
    df_filtered2 = (df2.drop(columns=['TOD']))
    X = df_filtered.to_numpy()
    X2 = df_filtered2.to_numpy()

    # Parameter grid for KPCA
    kernels = ['poly']
    gammas = [None, .01, 0.1, 1.0, 5.0, 10.0, 15.0]
    degrees = np.arange(1, 5)
    coef0 = [-1, 1, 5, 10]
    n_components = np.arange(4, 10)

    best_score = 100
    best_params = None
    best_model = None
    apply_model = None
    reconstruct = None

    for kernel in kernels:
        for gamma in gammas:
            for degree in degrees:
                for component in n_components:
                    for coef in coef0:
                        kpca = KernelPCA(kernel=kernel, gamma=gamma, degree=degree, n_components=component, random_state = 42, coef0 = coef, fit_inverse_transform=True)
                        kpca_results = None
                        score = 100
                        try:
                            kpca_results = kpca.fit_transform(X)
                            inverse = kpca.inverse_transform(kpca_results)
                            score = MAPE(X, inverse)
                            # score = mean_squared_error(X, inverse)
                        except np.linalg.LinAlgError as e:
                            print("Error: LinAlg")
                        except AttributeError as a:
                            print("Error: Attribute")
                        except ValueError:
                            print("Error: Value")

                        if (abs(5-score) < abs(5-best_score)):
                            best_score = score
                            best_model = kpca_results
                            apply_model = kpca.transform(X2)
                            reconstruct = kpca.inverse_transform(kpca_results)
                            best_params = (kernel, gamma, degree, coef, component)

    print("Best parameters found:", best_params)
    print(f'Reconstruction Error (%): {best_score}')

    best_scores.append(best_score)

    # print(f'Reconstruction Error (MSE): {residual_variance}')
    # print(f'R2): {r2}')
    # print(f'RMSE): {rmse}')
    # print(f'NRMSE): {nrmse}')

    finaltraindf = pd.DataFrame(best_model)
    finaltestdf = pd.DataFrame(apply_model)
    finaltraindf['TOD'] = df['TOD']
    finaltestdf['TOD'] = df2['TOD']

    out_directory = (r"/Users/olivialiau/Downloads/KPCAData/")
    os.makedirs(out_directory, exist_ok=True)
    train_file_out = (train_file_in[:11] + "_KPCA_95_train.csv")
    test_file_out = (test_file_in[:11] + "_KPCA_95_test.csv")
    train_file_path = os.path.join(out_directory, train_file_out)
    test_file_path = os.path.join(out_directory, test_file_out)
    finaltraindf.to_csv(train_file_path, index=False)
    finaltestdf.to_csv(test_file_path, index=False)
    print(f"Datasets at index {i} saved.")
    i += 1

print(best_scores)