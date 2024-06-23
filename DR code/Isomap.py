from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, lil_matrix
import pandas as pd
import os
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Function to compute KNN preservation
def knn_preservation(X, X_embedded, n_neighbors):
    nn_orig = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    nn_embedded = NearestNeighbors(n_neighbors=n_neighbors).fit(X_embedded)
    
    orig_neighbors = nn_orig.kneighbors(X, return_distance=False)
    embedded_neighbors = nn_embedded.kneighbors(X_embedded, return_distance=False)
    
    preservation_count = 0
    for i in range(X.shape[0]):
        preservation_count += len(set(orig_neighbors[i]).intersection(set(embedded_neighbors[i])))
    
    return preservation_count / (X.shape[0] * n_neighbors)

# Function to find the best Isomap configuration
def find_best_isomap_configuration(X, min_neighbors=10, max_neighbors=40, min_components=2, max_components=100, threshold=0.1):
    best_configuration = None
    num_features = X.shape[1]

    # Adjust max_components if it exceeds the number of features
    max_components = min(max_components, num_features)

    # Try different number of neighbors
    for neighbors in range(min_neighbors, max_neighbors):
        for component in range(min_components, max_components):
            try:
                isomap = Isomap(n_neighbors=neighbors, n_components=component, eigen_solver='dense')
                X_isomap = isomap.fit_transform(X)

                # Calculate the KNN preservation score
                # score = knn_preservation(X, X_isomap, neighbors)
                score = isomap.reconstruction_error()
                # print(f"Neighbors: {neighbors}, Component: {component}, KNN Preservation: {knn_preservation_score}")

                # Update best configuration based on criteria: knn_preservation > knn_threshold, then smaller component, then smaller neighbors
                if score <= threshold:
                    if (best_configuration is None or
                        component < best_configuration[1] or
                        (component == best_configuration[1] and score < best_configuration[2]) or
                        (component == best_configuration[1] and score == best_configuration[2] and neighbors < best_configuration[0])):
                        best_configuration = (neighbors, component, score)
            except ValueError as e:
                print(f"Failed to fit Isomap with neighbors={neighbors}, components={component}: {e}")
                doNothing = True
    if best_configuration:
        neighbors, component, score = best_configuration
        print(f"Best Configuration:\nNeighbors: {neighbors}, Component: {component}, score: {score}")
    else:
        print("No suitable configuration found with score above the threshold.")

    return best_configuration

# Function to process and save data
def process_and_save_data(base_dir, folder, split, method):
    # Construct the file paths for train
    train_name = f"{folder}_{split}_{method}_train.csv"
    train_file = os.path.join(base_dir, folder, train_name)
    output_train_file = os.path.join(base_dir, folder, 'Isomap', f"{folder}_{split}_{method}_DR_train.csv")

    # file path for test
    test_name = f"{folder}_{split}_{method}_test.csv"
    test_file = os.path.join(base_dir, folder, test_name)
    output_test_file = os.path.join(base_dir, folder, 'Isomap', f"{folder}_{split}_{method}_DR_test.csv")
    
    # Load the data
    data_train = pd.read_csv(train_file)
    data_test = pd.read_csv(test_file)

    TOD_train = data_train.pop('TOD_pos')
    TOD_test = data_test.pop('TOD_pos')

    # Extract the relevant columns for dimensionality reduction
    X_train = data_train.values
    X_test = data_test.values

    # Find the best Isomap configuration
    best_isomap_config = find_best_isomap_configuration(X_train)

    
    if best_isomap_config:
        neighbors, components, preservation_ratio = best_isomap_config

        # Apply Isomap with the best configuration to the training data
        isomap = Isomap(n_neighbors=neighbors, n_components=components)
        X_train_isomap = isomap.fit_transform(X_train)
        X_test_isomap = isomap.transform(X_test)  # Transform the test data using the same Isomap model

        # Replace the original columns with the reduced-dimension data
        reduced_columns = [f'Component_{i+1}' for i in range(components)]
        reduced_train_df = pd.DataFrame(X_train_isomap, columns=reduced_columns)
        reduced_test_df = pd.DataFrame(X_test_isomap, columns=reduced_columns)

        # Concatenate with the 'TOD' column
        final_train_df = pd.concat([TOD_train.reset_index(drop=True), reduced_train_df], axis=1)
        final_test_df = pd.concat([TOD_test.reset_index(drop=True), reduced_test_df], axis=1)

        # Save the resulting DataFrames to new CSV files
        os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
        final_train_df.to_csv(output_train_file, index=False)
        final_test_df.to_csv(output_test_file, index=False)

        print(f"Transformed training data saved successfully to {output_train_file}")
        print(f"Transformed test data saved successfully to {output_test_file}")

# get the path to data
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '..', 'data', 'train test split data')
data_dir = os.path.normpath(data_dir)

# folder names
folder_BA11 = 'BA11'
folder_BA47 = 'BA47'
split_60 = '60'
split_70 = '70'
split_80 = '80'
method_log = 'log'
method_MM = 'MM'
method_None = 'nonnormalized'

folders = [folder_BA11, folder_BA47]
splits = [split_60, split_70, split_80]
methods = [method_log, method_MM, method_None]

for folder in folders:
    for split in splits:
        for method in methods:
            process_and_save_data(data_dir, folder, split, method)