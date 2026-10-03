import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics import mean_squared_error, r2_score, pairwise_distances
from math import sqrt
import os


windows = ['window1', 'window2', 'window3']#,'None', ]
variances = [90, 95]
#95 for window2
# path = (r"/Users/olivialiau/Downloads/gr-WPI-UMASS-TOD-Project/data/encoded")
path = (r"/Users/olivialiau/Downloads")

# Parameter grid for KPCA
kernels = ['poly']
gammas = [0.1, 1.0, .01, 5.0, 10.0, None,]
degrees = [4, 5, 3, 2, 1]
coef0 = [10, 5, -1, 1]
n_components = [10, 9, 8, 7, 6, 5, 4]

datasets = [
            'BA11_60_MM', 'BA11_60_log', 
            'BA11_70_MM', 'BA11_70_log',
            'BA11_80_MM', 'BA11_80_log',
            'BA47_60_MM', 'BA47_60_log', 
            'BA47_70_MM', 'BA47_70_log',
            'BA47_80_MM', 'BA47_80_log'
            # 'full_60_MM', 'full_60_log', 
            # 'full_70_MM', 'full_70_log',
            # 'full_80_MM', 'full_80_log'
            ]


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
for window in windows:       
    for variance in variances:
        for file in datasets:

            # #for CNN
            # train_file_in = (f"{file}_train.csv" )
            # test_file_in = (f"{file}_test.csv" )
            # train_path_in = os.path.join(path, window, train_file_in)
            # test_path_in = os.path.join(path, window, test_file_in)

            if window == 'None':
                train_file_in = (f"{file}_train.csv" )
                test_file_in = (f"{file}_test.csv" )
                train_path_in = os.path.join(path, train_file_in)
                test_path_in = os.path.join(path, test_file_in)

            else:
                train_file_in = (f"{file}_{window}_train.csv" )
                test_file_in = (f"{file}_{window}_test.csv" )
                train_path_in = os.path.join(path, train_file_in)
                test_path_in = os.path.join(path, test_file_in)

            # if file is missing
            if not os.path.exists(test_path_in):
                print(f"{test_path_in} not found.")
                exit()
            if not os.path.exists(train_path_in):
                print(f"{train_path_in} not found.")
                exit()


            # Import, drop TOD, convert to array
            df = pd.read_csv(train_path_in)
            df2 = pd.read_csv(test_path_in)
            X = (df.drop(columns=['TOD'])).to_numpy()
            X2 = (df2.drop(columns=['TOD'])).to_numpy()

            score_not_found = True
            best_score = 100
            best_params = None
            best_model = None
            apply_model = None
            reconstruct = None

            for kernel in kernels:
                for coef in coef0:
                    for degree in degrees:
                        for gamma in gammas:
                            for component in n_components:
                                kpca = KernelPCA(kernel=kernel, gamma=gamma, degree=degree, n_components=component, random_state = 42, coef0 = coef, fit_inverse_transform=True, n_jobs=-1)
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

                                if (abs((100-variance)-score) < abs((100-variance)-best_score)):
                                    best_score = score
                                    best_model = kpca_results
                                    apply_model = kpca.transform(X2)
                                        # reconstruct = kpca.inverse_transform(kpca_results)
                                    best_params = (kernel, gamma, degree, coef, component)

                                if ((abs((100-variance)-best_score)) < 1):
                                    score_not_found = False
                                    print("Values found. Terminated.")
                                    break
                            if score_not_found == False: break
                        if score_not_found == False: break
                    if score_not_found == False: break
                if score_not_found == False: break

            print("Best parameters found:", best_params)
            print(f'Reconstruction Error (%): {best_score}')

                # best_scores.append(best_score)

                # print(f'Reconstruction Error (MSE): {residual_variance}')
                # print(f'R2): {r2}')
                # print(f'RMSE): {rmse}')
                # print(f'NRMSE): {nrmse}')

            finaltraindf = pd.DataFrame(best_model)
            finaltestdf = pd.DataFrame(apply_model)
            finaltraindf['TOD'] = df['TOD']
            finaltestdf['TOD'] = df2['TOD']

            out_directory = (r"/Users/olivialiau/Downloads/OPT3_Flatten_KPCA/")
            os.makedirs(out_directory, exist_ok=True)

            if window == 'None':
                train_file_out = (f"{file}_KPCA_{variance}_train.csv")
                test_file_out = (f"{file}_KPCA_{variance}_test.csv")
                
            else:
                train_file_out = (f"{file}_{window}_KPCA_{variance}_train.csv")
                test_file_out = (f"{file}_{window}_KPCA_{variance}_test.csv")

            train_file_path = os.path.join(out_directory, train_file_out)
            test_file_path = os.path.join(out_directory, test_file_out)
            finaltraindf.to_csv(train_file_path, index=False)
            finaltestdf.to_csv(test_file_path, index=False)
            print(f"{train_file_out, test_file_out} saved.")
