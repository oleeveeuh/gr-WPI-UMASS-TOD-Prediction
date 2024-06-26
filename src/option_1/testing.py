import sys
import os
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to the sys.path
sys.path.append(parent_dir)
# Now you can import the module
from read_train import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Create sequences for X and adjust y accordingly
def create_sequences(X, y, seq_length):
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length].astype(np.float32))
        y_seq.append(y[i + seq_length].astype(np.float32))
    return np.array(X_seq), np.array(y_seq)

X_train, y_train, X_test, y_test = read_file(Target.BA11, Split.S60, Normalize_Method.Log, DR_Method.ICA, Variance.V90)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

sequence = 3

X_train, y_train = create_sequences(X_train, y_train, sequence)
X_test, y_test = create_sequences(X_test, y_test, sequence)

print(X_train.shape)
print(y_train.shape)
# Define the LSTM model
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressor, self).__init__()
        print(f"Initializing model with hidden_size={hidden_size}, num_layers={num_layers}")
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Wrap the model with Skorch
input_size = 84  # Number of features per time step
hidden_size = 50
num_layers = 2
output_size = 1

net = NeuralNetRegressor(
    LSTMRegressor,
    module__input_size=input_size,
    module__hidden_size=hidden_size,
    module__num_layers=num_layers,
    module__output_size=output_size,
    max_epochs=20,
    lr=0.01,
    iterator_train__shuffle=False,
    train_split=None,
    device='cuda'
)

# Define parameter grid for Random Search
params = {
    'lr': [0.001, 0.01, 0.1],
    'max_epochs': [10, 20, 30],
    'module__hidden_size': [25, 75],
    'module__num_layers': [1, 3],
    'batch_size': [16, 32],
}

random_search = RandomizedSearchCV(net, params, n_iter=10, cv=3, scoring='neg_mean_squared_error', verbose=2)

# Fit the model using Random Search
random_search.fit(X_train, y_train)

# Evaluate the model
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_}")

# Predict on new data
y_pred = random_search.predict(X_test)
