from sklearn.manifold import Isomap
import pandas as pd
import os

# Function to find the best Isomap configuration
def find_best_isomap_configuration(X, min_neighbors=5, max_neighbors=50, min_components=2, max_components=10, max_residual_variance=0.1):
    best_configuration = None
    best_residual_variance = float('inf')
    best_component = max_components + 1

    # Try different number of neighbors
    for neighbors in range(min_neighbors, max_neighbors):
        for component in range(min_components, max_components):
            isomap = Isomap(n_neighbors=neighbors, n_components=component)
            isomap.fit_transform(X)
            residual_variance = isomap.reconstruction_error()

            if residual_variance < max_residual_variance:
                # print(f"Neighbors: {neighbors}, Component: {component}, \nResidual Variance: {residual_variance}")

                if component < best_component and residual_variance < best_residual_variance:
                    best_configuration = (neighbors, component, residual_variance)
                    best_residual_variance = residual_variance
                    best_component = component

    if best_configuration:
        neighbors, component, residual_variance = best_configuration
        print(f"Best Configuration:\nNeighbors: {neighbors}, Component: {component}, \nResidual Variance: {residual_variance}")
    else:
        print("No suitable configuration found with residual variance below the specified threshold.")

    return best_configuration



# get the path to data
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '..', 'data', 'train test split data')
data_dir = os.path.normpath(data_dir)

# folder names
folder_BA11 = 'BA11'
folder_BA47 = 'BA47'
split_60 = '60'
split_70 = '70'
split_80 = '08'
method_log = 'log'
method_MM = 'MM'
method_None = 'nonnormalized'

test_name = 'BA11_60_log_test.csv'
test_file = os.path.join(data_dir, folder_BA11, test_name)
output_dir = os.path.join(data_dir, folder_BA11, 'BA11_60_log_DR_test.csv')
data = pd.read_csv(test_file)

# Reduce the columns of genes
start_col = 'PER3'
numerical_columns = data.columns[data.columns.get_loc(start_col):]
X = data[numerical_columns].values

config = find_best_isomap_configuration(X)

if config:
    neighbor, component, variance = config

    # Apply Isomap with the best configuration to the training data
    isomap = Isomap(n_neighbors=neighbor, n_components=component)
    X_isomap = isomap.fit_transform(X)

    # Replace the original columns with the reduced-dimension data
    reduced_columns = [f'Component_{i+1}' for i in range(component)]
    reduced_df = pd.DataFrame(X_isomap, columns=reduced_columns)

    # Concatenate with the original non-reduced columns
    final_df = pd.concat([data.iloc[:, :data.columns.get_loc(start_col)], reduced_df], axis=1)

    # Save the resulting DataFrame to a new CSV file
    final_df.to_csv(output_dir, index=False)

    print(f"Transformed data saved successfully to {output_dir}")