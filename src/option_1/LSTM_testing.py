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
class LSTMModel(nn.Module):
    def __init__(self, lstm_units=100, dense_units=128, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.lstm = None
        self.fc = None
        self.output = nn.Linear(dense_units, 1)
    
    def forward(self, x):
        if self.lstm is None:
            input_dim = x.shape[-1]
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.lstm_units, batch_first=True)
            self.dropout = nn.Dropout(self.dropout_rate)
            self.fc = nn.Linear(self.lstm_units, self.dense_units)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = torch.relu(self.fc(x))
        x = self.output(x)
        return x

if __name__ == "__main__":
    # read_file(Target.BA11, Split.S60, Normalize_Method.Log, DR_Method.ICA, Variance.V90)

    # Define models
    models = {
        'Long Short-Term Memory Network': NeuralNetRegressor(
            module=LSTMModel,
            criterion=nn.MSELoss,
            train_split=None,
            verbose=0,
            )
    }

    # Define parameter grids for RandomizedSearchCV
    param_grids = {
        'Long Short-Term Memory Network': {
            'module__lstm_units': [50, 100, 150],
            'module__dense_units': [64, 128, 256],
            'module__dropout_rate': [0.3, 0.5, 0.7],
            'batch_size': [32, 64, 128],
            'max_epochs': [10, 20, 30],
            'optimizer': [optim.Adam, optim.SGD],
            'lr': [0.001, 0.01, 0.1]
        },
    }


    # Specify which datasets to use
    combinations = filter_combinations(
        targets=[Target.BA11,Target.BA47,],
        splits=[Split.S60, Split.S70, Split.S80],
        n_methods=[Normalize_Method.Log, Normalize_Method.MM],
        DR_methods=[DR_Method.ICA, DR_Method.PCA, DR_Method.KPCA, DR_Method.Isomap],
        variances=[Variance.V90, Variance.V95]
    )
    
    results_df = train_test_model(models, param_grids, combinations, verbose=True, save_result=True)
    # results_df = pd.read_csv('D:\WPI\DirectedResearch\gr-WPI-UMASS-TOD-Project\data\program_output\model_results_20240625_201447.csv')
    print(results_df)
    # Convert DataFrame to list of dictionaries
    # results_list = results_df.to_dict(orient='records')
    write_results_to_excel(results_df, verbose=True)