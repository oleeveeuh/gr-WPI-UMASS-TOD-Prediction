import torch
import torch.nn as nn
import random
import numpy as np

def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_processing(X, y):
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X, y

# Define the CNN model
class CNNRegressor(nn.Module):
    def __init__(self, num_channels=1, conv1_filters=16, conv2_filters=32,
                 kernel_size=3, pool_kernel_size=2, stride=1, pool_stride=2, padding=1):
        super(CNNRegressor, self).__init__()

        # Storing parameters as member variables
        self.num_channels = num_channels
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.stride = stride
        self.pool_stride = pool_stride
        self.padding = padding

        # Convolutional layers
        self.conv1 = nn.Conv1d(self.num_channels, self.conv1_filters, kernel_size=self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv1d(self.conv1_filters, self.conv2_filters, kernel_size=self.kernel_size, padding=self.padding)

        # Pooling layers
        self.pool = nn.MaxPool1d(self.pool_kernel_size, stride=self.pool_stride)
        self.pool2 = nn.MaxPool1d(self.pool_kernel_size, stride=self.pool_stride)

        # The fully connected layers will be initialized in the forward pass
        self.fc1 = None
        self.fc2 = None
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            output_size = self._calculate_output_size(x.size(2))  # Calculate from the input feature size
            self.fc1 = nn.Linear(self.conv2_filters * output_size, 100).to(x.device)
            self.fc2 = nn.Linear(100, 1).to(x.device)
            self.initialized = True

        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze()
        return x

    def _calculate_output_size(self, input_size):
        # Adjust calculation based on the layers' configurations
        size = input_size
        size = (size + 2 * self.padding - (self.kernel_size - 1) - 1) // self.stride + 1
        size = size // self.pool_stride
        size = (size + 2 * self.padding - (self.kernel_size - 1) - 1) // self.stride + 1
        size = size // self.pool_stride
        return size
    


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
        return out.squeeze(-1)


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
