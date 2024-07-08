import sys
import os
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to the sys.path
sys.path.append(parent_dir)
# Now you can import the module
from read_train import *
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor
from sklearn.ensemble import AdaBoostRegressor
# Define the LSTM model
seq_length = 3
# Create sequences for X and adjust y accordingly
def create_sequences(X, y):
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length].astype(np.float32))
        y_seq.append(y[i + seq_length].astype(np.float32))
    return np.array(X_seq), np.array(y_seq)

class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = None  # Initialized later
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if self.lstm is None or self.lstm.input_size != x.size(2):
            # Dynamically create the LSTM based on the current input size
            self.lstm = nn.LSTM(x.size(2), self.hidden_size, self.num_layers, batch_first=True).to(x.device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

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
            max_epochs=20,
            lr=0.01,
            iterator_train__shuffle=False,
            train_split=None,
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
        DR_methods=[DR_Method.PCA, DR_Method.ICA, DR_Method.Isomap],
        variances=[Variance.V90, Variance.V95]
    )
    
    results_df = train_test_model(models, param_grids, combinations, data_read_function=read_reduced_encoded_file,verbose=True, save_result=True, use_numpy= True, data_process_function=create_sequences)
    print(results_df)
    write_results_to_excel(results_df, target_folder='performance_sheets_option2',verbose=True)