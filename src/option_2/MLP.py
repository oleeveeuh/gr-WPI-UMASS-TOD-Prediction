import sys
import os
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to the sys.path
sys.path.append(parent_dir)
# Now you can import the module
from read_train import *
import torch.nn as nn
from skorch import NeuralNetRegressor

# Define the MLP model
class MLPRegressor(nn.Module):
    def __init__(self, num_units=10, output_size=1, activation_func=nn.ReLU()):
        super(MLPRegressor, self).__init__()
        self.num_units = num_units
        self.output_size = output_size
        self.first_layer = None  # This will be dynamically initialized
        self.hidden_layers = nn.Sequential(
            nn.Linear(num_units, num_units),
            activation_func,
            nn.Linear(num_units, num_units),
            activation_func
        )
        self.output_layer = nn.Linear(num_units, output_size)

    def forward(self, x):
        if self.first_layer is None or self.first_layer.in_features != x.size(1):
            # Dynamically create the first layer based on the current input size
            self.first_layer = nn.Linear(x.size(1), self.num_units).to(x.device)
        
        x = self.first_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x.squeeze(-1)

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
            device = 'cuda'
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


    # Specify which datasets to use
    combinations = filter_combinations(
        targets=[Target.BA11, Target.BA47],
        splits=[Split.S60, Split.S70, Split.S80],
        n_methods=[Normalize_Method.Log, Normalize_Method.MM],
        DR_methods=[DR_Method.KPCA, DR_Method.ICA, DR_Method.PCA, DR_Method.Isomap],
        variances=[Variance.V90, Variance.V95],
        windows = [WindowSize.W1] #, WindowSize.W2, WindowSize.W3]
    )

    results_df = train_test_model(models, param_grids, combinations, data_read_function=read_reduced_encoded_file, verbose=True, save_result=True, use_numpy= True, windows=True)
    print(results_df)
    write_results_to_excel(results_df, target_folder='performance_sheets_option2/window1',verbose=True)