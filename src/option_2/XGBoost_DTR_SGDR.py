import sys
import os
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to the sys.path
sys.path.append(parent_dir)
# Now you can import the module
from read_train import *
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor

if __name__ == "__main__":
    # read_file(Target.BA11, Split.S60, Normalize_Method.Log, DR_Method.ICA, Variance.V90)

    # Define models
    models = {
        'XGBoost Regressor': XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=RANDOM_STATE),
        'Stochastic Gradient Descent Regressor': SGDRegressor(random_state=RANDOM_STATE)
    }

    # Define parameter grids for RandomizedSearchCV
    param_grids = {
        'XGBoost Regressor': {
            'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [int(x) for x in np.linspace(3, 10, num=8)],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        },
        'Decision Tree Regressor': {
            'criterion': ['mse', 'friedman_mse', 'mae'],
            'splitter': ['best', 'random'],
            'max_depth': [int(x) for x in np.linspace(1, 32, num=32)],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'max_leaf_nodes': [None] + list(range(10, 100, 10)),
            'min_impurity_decrease': [0.0, 0.01, 0.1, 1.0],
            'ccp_alpha': [0.0, 0.01, 0.1, 1.0]
        },
        'Stochastic Gradient Descent Regressor': {
            'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            'l1_ratio': [0, 0.15, 0.5, 0.7, 1],
            'fit_intercept': [True, False],
            'max_iter': [1000, 2000, 3000],
            'tol': [1e-3, 1e-4, 1e-5],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'eta0': [1e-3, 1e-2, 1e-1],
            'power_t': [0.25, 0.5, 0.75]
        }
    }

    # Specify which datasets to use
    combinations = filter_combinations(
        targets=[Target.BA11, Target.BA47],
        splits=[Split.S60, Split.S70, Split.S80],
        n_methods=[Normalize_Method.Log, Normalize_Method.MM],
        DR_methods=[DR_Method.KPCA],
        variances=[Variance.V90, Variance.V95]
    )
    
    results_df = train_test_model(models, param_grids, combinations, data_read_function=read_reduced_encoded_file,verbose=True, save_result=True)
    print(results_df)
    write_results_to_excel(results_df, target_folder='performance_sheets_option2',verbose=True)