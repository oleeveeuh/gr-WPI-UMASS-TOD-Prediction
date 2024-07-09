from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
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
def find_best_isomap_configuration(X, min_neighbors=10, max_neighbors=50, min_components=10, max_components=56, threshold=0.2):
    best_configuration = None
    num_features = X.shape[1]

    # Adjust max_components if it exceeds the number of features
    max_components = min(max_components, num_features)
    min_score = 99.0
    # Try different number of neighbors
    for neighbors in range(min_neighbors, max_neighbors):
        for component in range(min_components, max_components):
            try:
                isomap = Isomap(n_neighbors=neighbors, n_components=component, tol=1e-6)

                X_isomap = isomap.fit_transform(X)
            
                # Calculate the KNN preservation score
                # score = knn_preservation(X, X_isomap, neighbors)
                score = isomap.reconstruction_error()
                # print(f"Neighbors: {neighbors}, Component: {component}, KNN Preservation: {knn_preservation_score}")

                # Update best configuration based on criteria: knn_preservation > knn_threshold, then smaller component, then smaller neighbors
                if score <= min_score:
                    if (best_configuration is None or
                        component > best_configuration[1] or
                        (component == best_configuration[1] and score < best_configuration[2]) or
                        (component == best_configuration[1] and score == best_configuration[2] and neighbors < best_configuration[0])):
                        best_configuration = (neighbors, component, score)
                        min_score = score
            except ValueError as e:
                print(f"Failed to fit Isomap with neighbors={neighbors}, components={component}: {e}")
                doNothing = True
    if best_configuration:
        neighbors, component, score = best_configuration
        print(f"Best Configuration:\nNeighbors: {neighbors}, Component: {component}, score: {score}")
    else:
        print("No suitable configuration found with score above the threshold.")

    return best_configuration

reduce_encoded_folder = 'reduced_encoded'
reduce_folder = 'reduced_data'
encoded_folder = 'encoded'
train_test_split = 'train_test_split_data'

# Function to process and save data
def process_and_save_data(base_dir, folder,split, method, output_folder, input_folder ,output_postfix = '', intput_postfix = ''):
    
    # Construct the file paths for train
    train_name = f"{folder}_{split}_{method}_{intput_postfix}train.csv"
    test_name = f"{folder}_{split}_{method}_{intput_postfix}test.csv"
    output_train_file = os.path.join(base_dir, '..',output_folder, f"{folder}_{split}_{method}_{output_postfix}Isomap_90_train.csv")
    output_test_file = os.path.join(base_dir, '..',output_folder, f"{folder}_{split}_{method}_{output_postfix}Isomap_90_test.csv")

    if folder == folder_full:
        train_name = f"full_{split}_{method}_{intput_postfix}train.csv"
        test_name = f"full_{split}_{method}_{intput_postfix}test.csv"
        output_train_file = os.path.join(base_dir, '..',output_folder, f"full_{split}_{method}_{output_postfix}Isomap_90_train.csv")
        output_test_file = os.path.join(base_dir, '..',output_folder, f"full_{split}_{method}_{output_postfix}Isomap_90_test.csv")

    train_file = os.path.join(base_dir, '..', input_folder, folder, train_name)
    

    # file path for test
    test_file = os.path.join(base_dir, '..', input_folder, folder, test_name)
    
    
    # Load the data
    data_train = pd.read_csv(train_file)
    data_test = pd.read_csv(test_file)

    TOD_train = data_train.pop('TOD')
    TOD_test = data_test.pop('TOD')

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
data_dir = os.path.join(script_dir, '..', 'data', 'window')
data_dir = os.path.normpath(data_dir)

# folder names
folder_BA11 = 'BA11'
folder_BA47 = 'BA47'
folder_full = 'full_data'
split_60 = '60'
split_70 = '70'
split_80 = '80'
method_log = 'log'
method_MM = 'MM'
method_None = 'nonnormalized'

folders = [folder_full]
splits = [split_60, split_70, split_80]
methods = [method_log, method_MM]

for folder in folders:
    for split in splits:
        for method in methods:
            process_and_save_data(data_dir, folder, split, method, output_folder = reduce_folder, input_folder= train_test_split)
# for split in splits:
#     for method in methods:
#         process_and_save_data(data_dir, folder_full, split, method)