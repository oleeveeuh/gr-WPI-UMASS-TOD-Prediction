from enum import Enum
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from datetime import datetime
import pandas as pd
import numpy as np
import os


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
dr_method_map = {DR_Method.PCA: 'PCA', DR_Method.ICA: 'ICA', DR_Method.KPCA: 'KPCA', DR_Method.Isomap: 'Isomap'}
variance_map = {Variance.V90: '90', Variance.V95: '95'}

# get the path to data
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '..', 'data', 'reduced_data')
data_dir = os.path.normpath(data_dir)


def filter_combinations(targets=None, splits=None, n_methods=None, DR_methods=None, variances=None):
    """
    Generate combinations of parameters for filtering datasets.

    Parameters:
    targets (list of Target, optional): List of Target enums to include in the combinations. 
                                         Defaults to excluding 'full'.
    splits (list of Split, optional): List of Split enums to include in the combinations. 
                                      Defaults to including all splits.
    n_methods (list of Normalize_Method, optional): List of Normalize_Method enums to include in the combinations. 
                                                    Defaults to excluding 'nonnormalized'.
    DR_methods (list of DR_Method, optional): List of DR_Method enums to include in the combinations. 
                                              Defaults to including all DR methods.
    variances (list of Variance, optional): List of Variance enums to include in the combinations. 
                                            Defaults to including all variances.

    Returns:
    list of tuples: A list of tuples, each containing a unique combination of the specified parameters.
                    Each tuple contains (target, split, n_method, DR_method, variance).

    Example:
    combinations = filter_combinations(
        targets=[Target.BA11, Target.BA47],
        splits=[Split.S60, Split.S70],
        n_methods=[Normalize_Method.Log, Normalize_Method.MM],
        DR_methods=[DR_Method.PCA],
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
    effective_DR_methods = DR_methods if DR_methods else list(DR_Method)
    effective_variances = variances if variances else list(Variance)
    
    # Generate all combinations
    all_combinations = product(effective_targets, effective_splits, effective_n_methods, effective_DR_methods, effective_variances)
    
    # Return the filtered combinations
    return list(all_combinations)


def read_file(target, split, n_method, dr_method, variance):
    '''
    Input:
        target_data(BA11, BA47, Combine)
        normalize_method(log, MM, NN)
        split(60, 70, 80)
        DR_method(ICA, KPCA, PCA, Isomap)
        variance(90, 95)

    Output:
        X_train, y_train, X_test, y_test
    '''
    # Construct the file paths for train
    train_name = f"{target_map[target]}_{split_map[split]}_{n_method_map[n_method]}_{dr_method_map[dr_method]}_{variance_map[variance]}_train.csv"
    train_file = os.path.join(data_dir, train_name)
    
    # file path for test
    test_name = f"{target_map[target]}_{split_map[split]}_{n_method_map[n_method]}_{dr_method_map[dr_method]}_{variance_map[variance]}_test.csv"
    test_file = os.path.join(data_dir, test_name)
    
    # Load data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Preprocess data
    y_train = train_data.pop('TOD')
    X_train = train_data

    y_test = test_data.pop('TOD')
    X_test = test_data

    return X_train, y_train, X_test, y_test


def train_test_model(models, param_grids, combinations, n_iter=10, cv=5, random_state=42, verbose = False):
    '''
    train_test_model
    Input:
        array of models
        array of model parameters
        array of data combination wants to train, defualt two brain areas

    Output:
        result, write to csv.
    '''
    results = []
    
    for combination in combinations:
        target, split, n_method, dr_method, variance = combination
        if verbose:
            print(f"Processing: {target_map[target]}_{split_map[split]}_{n_method_map[n_method]}_{dr_method_map[dr_method]}_{variance_map[variance]}")
        
        X_train, y_train, X_test, y_test = read_file(target=target, n_method=n_method, split=split, dr_method=dr_method, variance=variance)
        
        for model_name, model in models.items():
            param_grid = param_grids[model_name]
            
            if verbose:
                print(f"Running {model_name} with parameters: {param_grid}")
            # Initialize RandomizedSearchCV
            tscv = TimeSeriesSplit(n_splits=cv)
            randomized_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter, cv=tscv, random_state=random_state)
            randomized_search.fit(X_train, y_train)
            best_model = randomized_search.best_estimator_
            
            # Predict on test set
            y_pred = best_model.predict(X_test)
            
            # Calculate evaluation metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            rmse = mse ** 0.5
            smape = 100 * (2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred))).mean()
            
            result = {
                'target': target,
                'split': split,
                'n_method': n_method,
                'dr_method': dr_method,
                'variance': variance,
                'model': model_name,
                'parameter_search': list(param_grid.keys()),
                'best_params': randomized_search.best_params_,
                'mse': mse,
                'mae': mae,
                'mape': mape,
                'rmse': rmse,
                'smape': smape
            }
            results.append(result)

            if verbose:
                print(f"Best parameters for {model_name}: {randomized_search.best_params_}")
                print(f"Metrics - MSE: {mse}, MAE: {mae}, MAPE: {mape}, RMSE: {rmse}, SMAPE: {smape}")
    
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(data_dir, f'model_results_{timestamp}.csv')
    results_df.to_csv(results_file, index=False)
    
    if verbose:
        print(f"Results saved to {results_file}")

    return results_df



if __name__ == "__main__":
    # read_file(Target.BA11, Split.S60, Normalize_Method.Log, DR_Method.ICA, Variance.V90)

    # Define models
    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'ExtraTrees': ExtraTreesRegressor(random_state=42)
    }

    # Define parameter grids for RandomizedSearchCV
    param_grids = {
        'RandomForest': {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'ExtraTrees': {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }

    # Specify multiple values for each parameter
    combinations = filter_combinations(
        targets=[Target.BA11],
        splits=[Split.S60],
        n_methods=[Normalize_Method.Log],
        DR_methods=[DR_Method.ICA],
        variances=[Variance.V90]
    )
    
    results_df = train_test_model(models, param_grids, combinations, verbose=True)
    print(results_df)