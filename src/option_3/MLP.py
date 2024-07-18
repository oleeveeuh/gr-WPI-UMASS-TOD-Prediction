import sys
import os
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to the sys.path
sys.path.append(parent_dir)
# Now you can import the module
from read_train import *
from model_definitions import MLPRegressor, set_seeds
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor

set_seeds(RANDOM_STATE)

if __name__ == "__main__":
    # read_file(Target.BA11, Split.S60, Normalize_Method.Log, DR_Method.ICA, Variance.V90)

    # Define models
    models = {
        'Multilayer Perceptron': NeuralNetRegressor(
            MLPRegressor,
            # module__input_size=input_size,
            max_epochs=20,  # You can adjust this
            lr=0.1,
            iterator_train__shuffle=False,
            optimizer=optim.Adam,
            device = 'cuda',
            )
    }

    # Define parameter grids for RandomizedSearchCV
    param_grids = {
        'Multilayer Perceptron': {
            'lr': [0.01, 0.02, 0.05, 0.1],
            'module__num_units': [10, 20, 50, 100],
            'module__activation_func': [nn.ReLU(), nn.Tanh()],
        },
    }

    
    for window in sliding_window_sizes:
        # Specify which datasets to use
        combinations = filter_combinations(
            targets=[Target.BA11, Target.BA47],
            splits=[Split.S60, Split.S70, Split.S80],
            n_methods=[Normalize_Method.Log, Normalize_Method.MM],
            DR_methods=[DR_Method.KPCA, DR_Method.ICA, DR_Method.PCA, DR_Method.Isomap],
            variances=[Variance.V90, Variance.V95],
            windows = [window] #, WindowSize.W2, WindowSize.W3]
        )
        #set flatten to True for CNN w/o dense
        results_df = train_test_model(models, param_grids, combinations, data_read_function=read_reduced_CNN_file, verbose=True, save_result=True, use_numpy= True, windows=True, flatten=True)
        print(results_df)
        # write_results_to_excel(results_df, target_folder=f'performance_sheets_option3/{window_size_map[window]}',verbose=True)
        write_results_to_excel(results_df, target_folder=f'performance_sheets_option3_flatten/{window_size_map[window]}',verbose=True)