import sys
import os
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to the sys.path
sys.path.append(parent_dir)
# Now you can import the module
from read_train import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor

if __name__ == "__main__":
    # read_file(Target.BA11, Split.S60, Normalize_Method.Log, DR_Method.ICA, Variance.V90)

    # Define models
    models = {
        'K-Neighbors Regressor': KNeighborsRegressor(),
        'BaggingRegressor': BaggingRegressor(random_state=42),
        'AdaBoostRegressor': AdaBoostRegressor(random_state=42),
    }

    # Define parameter grids for RandomizedSearchCV
    param_grids = {
        'K-Neighbors Regressor': {
                'n_neighbors': np.arange(1, 10),
                'weights': ['uniform', 'distance']
        },
        'BaggingRegressor': {
                'n_estimators': np.arange(10, 200, 10),
                'max_samples': np.linspace(0.1, 1.0, 10),
                'max_features': np.linspace(0.1, 1.0, 10)
        },
        'AdaBoostRegressor': {
                'n_estimators': np.arange(50, 500, 50),  # Common choices are between 50 and 500
                'learning_rate': [0.01, 0.1, 1.0],  # Typical values are 0.01, 0.1, or 1
                'loss': ['linear', 'square', 'exponential']  # Choices of loss function
        },
    }

    # Specify which datasets to use
    combinations = filter_combinations(
        targets=[Target.BA11, Target.BA47],
        splits=[Split.S60, Split.S70, Split.S80],
        n_methods=[Normalize_Method.Log, Normalize_Method.MM],
        DR_methods=[DR_Method.KPCA, DR_Method.ICA, DR_Method.PCA, DR_Method.Isomap],
        variances=[Variance.V90, Variance.V95]
    )
    
    results_df = train_test_model(models, param_grids, combinations, data_read_function=read_reduced_encoded_file,verbose=True, save_result=True)
    print(results_df)
    write_results_to_excel(results_df, target_folder='performance_sheets_option2',verbose=True)