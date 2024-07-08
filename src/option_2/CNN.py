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

def calculate_conv1d_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

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
    
if __name__ == "__main__":
    # read_file(Target.BA11, Split.S60, Normalize_Method.Log, DR_Method.ICA, Variance.V90)

    # Define models
    models = {
        'Convolutional neural network': NeuralNetRegressor(
            CNNRegressor,
            # module__input_size=input_size,
            max_epochs=20,  # You can adjust this
            lr=0.1,
            iterator_train__shuffle=False,
            criterion=torch.nn.MSELoss,
            optimizer=optim.Adam,
            device = 'cuda'
            )
    }

    # Define parameter grids for RandomizedSearchCV
    param_grids = {
        'Convolutional neural network': {
            # 'lr': [0.01, 0.02, 0.05, 0.1],
            # 'module__num_units': [10, 20, 50, 100],
            # 'module__activation_func': [nn.ReLU(), nn.Tanh()],
        },
    }


    # Specify which datasets to use
    combinations = filter_combinations(
        targets=[Target.BA11, Target.BA47],
        splits=[Split.S60, Split.S70, Split.S80],
        n_methods=[Normalize_Method.Log, Normalize_Method.MM],
        DR_methods=[DR_Method.KPCA],
        variances=[Variance.V90, Variance.V95]
    )
    
    results_df = train_test_model(models, param_grids, combinations, data_read_function=read_reduced_encoded_file,verbose=True, save_result=True, use_numpy= True, data_process_function=data_processing)
    # results_df = pd.read_csv('D:\WPI\DirectedResearch\gr-WPI-UMASS-TOD-Project\data\program_output\model_results_20240708_004758.csv')
    print(results_df)
    write_results_to_excel(results_df, target_folder='performance_sheets_option2',verbose=True)