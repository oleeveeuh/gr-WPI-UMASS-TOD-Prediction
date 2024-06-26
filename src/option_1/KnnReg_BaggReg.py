import sys
import os
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to the sys.path
sys.path.append(parent_dir)
# Now you can import the module
from read_train import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor

if __name__ == "__main__":
    # read_file(Target.BA11, Split.S60, Normalize_Method.Log, DR_Method.ICA, Variance.V90)

    # Define models
    models = {
        'K-Neighbors Regressor': KNeighborsRegressor(),
        'BaggingRegressor': BaggingRegressor(random_state=42),
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
        
    }

    # Specify which datasets to use
    combinations = filter_combinations(
        targets=[Target.BA11, Target.BA47],
        splits=[Split.S60, Split.S70, Split.S80],
        n_methods=[Normalize_Method.Log, Normalize_Method.MM],
        DR_methods=[DR_Method.ICA, DR_Method.KPCA, DR_Method.PCA, DR_Method.Isomap],
        variances=[Variance.V90, Variance.V95]
    )
    
    results_df = train_test_model(models, param_grids, combinations, verbose=True, save_result=True)
    # results_df = pd.read_csv('D:\WPI\DirectedResearch\gr-WPI-UMASS-TOD-Project\data\program_output\model_results_20240625_201447.csv')
    print(results_df)
    # Convert DataFrame to list of dictionaries
    # results_list = results_df.to_dict(orient='records')
    write_results_to_excel(results_df, verbose=True)