import sys
import os
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to the sys.path
sys.path.append(parent_dir)
# Now you can import the module
from read_train import *
import pandas as pd
import numpy as np
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import ast

WINDOW_SIZE = 3
this_script = os.path.dirname(__file__)
encoded_path = os.path.join(this_script, '..', '..', 'data', 'encoded')
encoded_path = os.path.normpath(encoded_path)
window_data_path = os.path.join(this_script, '..', '..', 'data', 'window')
window_data_path = os.path.normpath(window_data_path)

# Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

class Autoencoder(nn.Module):
    def __init__(self, original_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),  # Flatten the input if it's not already 1D
            nn.Linear(original_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.Tanh(),  # Compresses into the specified encoding dimension
        )
        
        # Define the decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, original_dim),
            nn.Sigmoid()  # Assuming the input features were scaled between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def create_windows(combination, w_size = 3, verbose = False):
    # read in the relevant data
    target, split, n_method = combination
    train_data, test_data = read_data_file(target=target, n_method=n_method, split=split, split_xy=False)
    if verbose:
        print(
            f"Creating Windows for: {target_map[target]}_{split_map[split]}_{n_method_map[n_method]}")

    results = {}
    i = 1
    for df in [train_data, test_data]:
        df_name = "train" if (i == 1) else "test"
        df = df.sort_values(by='TOD').reset_index(drop=True)

        # Create the new DataFrame
        new_data = {col: [] for col in df.columns}
        # Iterate through the DataFrame and replace values
        for i in range(w_size+1, len(df) - w_size+1): # For w=3, the range is row 4(inclusive) through the 3rd from end (exclusive)
            for col in df.columns:
                if col in ['Age', 'TOD', 'Sex']:
                    new_data[col].append(df.loc[i, col])
                else:
                    col_index = df.columns.get_loc(col)
                    # Collect values from the preceding and following W rows
                    surrounding_values = df.iloc[i - w_size-1:i + w_size, col_index].tolist()
                    #print("Working on Row: ", i, " of value ",  df.loc[i, col])
                    #print("Extracting values from rows:", i - w_size, " through ", i + w_size)
                    #print("Corresponding to Values: ", surrounding_values)
                    new_data[col].append(surrounding_values)
         #Convert new_data back to a DataFrame
        new_df = pd.DataFrame(new_data)
        results[df_name] = new_df
        i += 1
    return results

encoding_dim = 1

def apply_autoencoder(combinations, verbose=False):
    os.makedirs(encoded_path, exist_ok=True)
    results = []

    for combination in combinations:
        target, split, n_method = combination
        train_file = f"{target_map[target]}_{split_map[split]}_{n_method_map[n_method]}_window{WINDOW_SIZE}_train.csv"
        test_file = f"{target_map[target]}_{split_map[split]}_{n_method_map[n_method]}_window{WINDOW_SIZE}_test.csv"
        if verbose:
            print(f"Processing: {train_file}")
        train_path = os.path.join(window_data_path, train_file)
        test_path = os.path.join(window_data_path, test_file)
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_head_df = train_df[['Age', 'TOD', 'Sex']].copy()
        test_head_df = test_df[['Age', 'TOD', 'Sex']].copy()

        # Drop the columns before processing
        train_df.drop(['Age', 'TOD', 'Sex'], axis=1, inplace=True)
        test_df.drop(['Age', 'TOD', 'Sex'], axis=1, inplace=True)

        # Convert string lists to actual lists
        for column in train_df.columns:
            train_df[column] = train_df[column].apply(ast.literal_eval)
            test_df[column] = test_df[column].apply(ast.literal_eval)

        if verbose:
            print(train_df.shape)
        
        # Stack the DataFrame to collapse the columns into a single column of lists
        stacked_train = train_df.stack()
        stacked_test = test_df.stack()

        # Apply pd.Series to expand each list into its own row
        expanded_train = stacked_train.apply(pd.Series)
        expanded_test = stacked_train.apply(pd.Series)

        # Reset the index to flatten it, and optionally drop the old indices
        reshaped_train_df = expanded_train.reset_index(drop=True)
        reshaped_test_df = expanded_test.reset_index(drop=True)

        if verbose:
            # Now reshaped_df should be in the shape of (20445, 7)
            print(reshaped_train_df.shape)

        # Prepare data for PyTorch
        train_tensor = torch.tensor(reshaped_train_df.values, dtype=torch.float32)
        test_tensor = torch.tensor(reshaped_test_df.values, dtype=torch.float32)

        if verbose:
            print(train_tensor.shape)
            
        num_features = train_tensor.shape[1]

        # Initialize Autoencoder and training essentials
        autoencoder = Autoencoder(num_features, encoding_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=True)

        # Train the Autoencoder with tqdm for progress tracking
        autoencoder.train()
        epochs = 10  # Set the number of epochs
        for epoch in range(epochs):
            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as tepoch:
                for data, in tepoch:
                    data = data.to(device)
                    optimizer.zero_grad()
                    outputs = autoencoder(data)
                    loss = criterion(outputs, data)
                    loss.backward()
                    optimizer.step()

                    # Update tqdm bar showing current batch loss
                    tepoch.set_postfix(loss=loss.item())

        # Encode the train and test data
        autoencoder.eval()
        with torch.no_grad():
            # Ensure train_tensor and test_tensor are on the GPU
            train_tensor = train_tensor.to(device)
            test_tensor = test_tensor.to(device)

            # Generate encoded outputs
            encoded_train = autoencoder.encoder(train_tensor)
            encoded_test = autoencoder.encoder(test_tensor)

            # Move encoded outputs back to CPU and then convert to numpy arrays
            encoded_train = encoded_train.to('cpu').numpy()
            encoded_test = encoded_test.to('cpu').numpy()

        # Combine the encoded data with the kept columns
        encoded_train_df = pd.DataFrame(encoded_train, columns=[f'encoded_feature_{i}' for i in range(encoding_dim)])
        encoded_test_df = pd.DataFrame(encoded_test, columns=[f'encoded_feature_{i}' for i in range(encoding_dim)])
        final_train_df = pd.concat([train_head_df.reset_index(drop=True), encoded_train_df], axis=1)
        final_test_df = pd.concat([test_head_df.reset_index(drop=True), encoded_test_df], axis=1)

        # Save encoded outputs
        final_train_df.to_csv(os.path.join(encoded_path, train_file), index=False)
        final_test_df.to_csv(os.path.join(encoded_path, test_file), index=False)

        if verbose:
            print(f"Encoded data saved for {train_file} and {test_file}")


def create_windowed_files(combinations):
    os.makedirs(window_data_path, exist_ok=True)
# Get all the possible method combinations
    
    for combo in combinations:
        windows = create_windows(combo, w_size=WINDOW_SIZE, verbose = True)
        target, split, n_method = combo
        for window in windows:
            filename = f"{target_map[target]}_{split_map[split]}_{n_method_map[n_method]}_window{WINDOW_SIZE}_{window}.csv"
            path = os.path.join(window_data_path, filename)
            windows[window].to_csv(path, index=False)

if __name__ == "__main__":
    # combinations = filter_combinations(
    #     targets=[Target.BA11, Target.BA47],
    #     splits=[Split.S60, Split.S70, Split.S80],
    #     n_methods=[Normalize_Method.Log, Normalize_Method.MM]
    # )
    combinations = filter_combinations(
        targets=[Target.BA11],
        splits=[Split.S60],
        n_methods=[Normalize_Method.Log]
    )
    # create_windowed_files(combinations)
    apply_autoencoder(combinations, verbose=True)

    #Initialize a receptacle dictionary (DICT1)
    # For each combination:
        # Read the respective files - generate a warning if the file doesn't exist
        # Assign the files to a dictionary with setup key=filename : value = pd.DataFrame. There will be two files
        # Initialize a receptacle dictionary (DICT2) with setup keys= [TOD, Age, Sex, AE_output]: value=respective values
            # For each row (except first and last) in each file:
                # Generate a 3x235 array containing the gene expressions for each TOD.
                # Feed the 3x235 array into autoencoder, receive a single number output (AE_output).
                # join AE_output to a row with the TOD, Age, and Sex of the second row from the original 3x235 array
                # Concat output row to DICT2
            #Before moving onto next combination, add DICT2 to DICT1 with setup key=filename+"_AE"
