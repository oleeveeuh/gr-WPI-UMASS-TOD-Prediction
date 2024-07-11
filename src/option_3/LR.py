import sys
import os
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to the sys.path
sys.path.append(parent_dir)
# Now you can import the module
from read_train import *
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    # read_file(Target.BA11, Split.S60, Normalize_Method.Log, DR_Method.ICA, Variance.V90)

    # Define models
    models = {
        'Linear Regressor': LinearRegression(),
    }

    # Define parameter grids for RandomizedSearchCV
    param_grids = {
        'Linear Regressor': {
        },
        
    }

    # Specify which datasets to use
    combinations = filter_combinations(
        targets=[Target.BA11, Target.BA47],
        splits=[Split.S60, Split.S70, Split.S80],
        n_methods=[Normalize_Method.Log, Normalize_Method.MM],
        DR_methods=[DR_Method.KPCA, DR_Method.ICA, DR_Method.PCA, DR_Method.Isomap],
        variances=[Variance.V90, Variance.V95],
        windows = [WindowSize.W1] #, WindowSize.W2, WindowSize.W3]
    )
    
    results_df = train_test_model(models, param_grids, combinations, data_read_function=read_reduced_CNN_file,verbose=True, save_result=True, windows = True)
    print(results_df)
    write_results_to_excel(results_df, target_folder='performance_sheets_option3/window1',verbose=True)