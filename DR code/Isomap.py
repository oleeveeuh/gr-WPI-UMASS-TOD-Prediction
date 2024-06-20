from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os

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
def find_best_isomap_configuration(X, min_neighbors=5, max_neighbors=30, min_components=2, max_components=30, knn_threshold=0.9):
    best_configuration = None

    # Try different number of neighbors
    for neighbors in range(min_neighbors, max_neighbors):
        for component in range(min_components, max_components):
            isomap = Isomap(n_neighbors=neighbors, n_components=component)
            X_isomap = isomap.fit_transform(X)

            # Calculate the KNN preservation score
            knn_preservation_score = knn_preservation(X, X_isomap, neighbors)

            # print(f"Neighbors: {neighbors}, Component: {component}, KNN Preservation: {knn_preservation_score}")

            # Update best configuration based on criteria: knn_preservation > knn_threshold, then smaller component, then smaller neighbors
            if knn_preservation_score >= knn_threshold:
                if (best_configuration is None or
                    component < best_configuration[1] or
                    (component == best_configuration[1] and knn_preservation_score > best_configuration[2]) or
                    (component == best_configuration[1] and knn_preservation_score == best_configuration[2] and neighbors < best_configuration[0])):
                    best_configuration = (neighbors, component, knn_preservation_score)

    if best_configuration:
        neighbors, component, knn_preservation_score = best_configuration
        print(f"Best Configuration:\nNeighbors: {neighbors}, Component: {component}, KNN Preservation: {knn_preservation_score}")
    else:
        print("No suitable configuration found with KNN preservation score above the threshold.")

    return best_configuration

# Function to process and save data
def process_and_save_data(base_dir, folder, split, method, start_col='PER3'):
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

    # Define the columns to be reduced (from 'PER3' onwards)
    numerical_columns_train = data_train.columns[data_train.columns.get_loc(start_col):]
    numerical_columns_test = data_test.columns[data_test.columns.get_loc(start_col):]

    # Extract the relevant columns for dimensionality reduction
    X_train = data_train[numerical_columns_train].values
    X_test = data_test[numerical_columns_test].values

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

        # Concatenate with the original non-reduced columns
        final_train_df = pd.concat([data_train.iloc[:, :data_train.columns.get_loc(start_col)], reduced_train_df], axis=1)
        final_test_df = pd.concat([data_test.iloc[:, :data_test.columns.get_loc(start_col)], reduced_test_df], axis=1)

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