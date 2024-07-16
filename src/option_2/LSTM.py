import sys
import os
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to the sys.path
sys.path.append(parent_dir)
# Now you can import the module
from read_train import *
from model_definitions import LSTMRegressor, create_sequences, set_seeds
import torch.optim as optim
from skorch import NeuralNetRegressor

set_seeds(RANDOM_STATE)

if __name__ == "__main__":
    # read_file(Target.BA11, Split.S60, Normalize_Method.Log, DR_Method.ICA, Variance.V90)

    # Define models
    models = {
        'Long Short-Term Memory Network': NeuralNetRegressor(
            LSTMRegressor,
            # module__input_size=input_size,
            module__hidden_size=50,
            module__num_layers=1,
            module__output_size=1,
            max_epochs=30,
            lr=0.01,
            iterator_train__shuffle=False,
            train_split=None,
            optimizer=optim.Adam,
            device='cuda'
            )
    }

    # Define parameter grids for RandomizedSearchCV
    param_grids = {
        'Long Short-Term Memory Network': {
            'lr': [0.001, 0.01, 0.1],
            'max_epochs': [10, 20, 30],
            'module__hidden_size': [25, 75],
            'module__num_layers': [1, 3],
            'batch_size': [16, 32],
        },
    }


    # Specify which datasets to use
    combinations = filter_combinations(
        targets=[Target.BA11, Target.BA47],
        splits=[Split.S60, Split.S70, Split.S80],
        n_methods=[Normalize_Method.Log, Normalize_Method.MM],
        DR_methods=[DR_Method.KPCA, DR_Method.ICA, DR_Method.PCA, DR_Method.Isomap],
        variances=[Variance.V90, Variance.V95],
        windows = [WindowSize.W3] #, WindowSize.W2, WindowSize.W3]
    )
    
    results_df = train_test_model(models, param_grids, combinations, data_read_function=read_reduced_encoded_file,verbose=True, save_result=True, use_numpy= True, data_process_function=create_sequences, windows=True)
    print(results_df)
    write_results_to_excel(results_df, target_folder='performance_sheets_option2/window3',verbose=True)