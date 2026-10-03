import pandas as pd
import numpy as np
import tensorflow as tf
import random


import tensorflow as tf

# List all physical devices available
physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs:", physical_devices)

# Check if TensorFlow is currently using the GPU
print("Is TensorFlow using the GPU:", tf.test.is_gpu_available())



def create_windowed_dataframe(df, window_size):
    """
    Function to create a windowed dataframe.

    Args:
    df (pd.DataFrame): Input dataframe.
    window_size (int): Size of the window.

    Returns:
    pd.DataFrame: Windowed dataframe.
    """
    rows, cols = df.shape
    new_data = []
    df = df.sort_values(by='TOD').reset_index(drop=True)
    for i in range(window_size, rows - window_size):
        new_row = {}
        for col in df.columns:
            window = df.loc[i - window_size:i + window_size, col].tolist()
            if col in df.columns[:3]:
                new_row[col] = window[window_size]
            else:
                new_row[col] = window
        new_data.append(new_row)

    new_df = pd.DataFrame(new_data)
    return new_df

def set_random_seed(seed):
    """
    Function to set random seed for reproducibility.

    Args:
    seed (int): Random seed value.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def apply_convolution_and_pooling(df, kernel_size, pool_size, filters, batch_size, seed=42):
    """
    Apply convolution and pooling operations to each cell of the dataframe in batches.

    Args:
    df (pd.DataFrame): Input dataframe where each cell contains a list of 7 numbers.
    kernel_size (tuple): Kernel size for the Conv2D layer.
    pool_size (tuple): Pool size for the MaxPooling2D layer.
    filters (int): Number of filters for the Conv2D layer.
    batch_size (int): Batch size for processing.
    seed (int): Random seed for reproducibility.

    Returns:
    pd.DataFrame: Dataframe with each cell containing the result of the convolution and pooling operation.
    """
    set_random_seed(seed)

    rows, cols = df.shape
    data = np.array(df.values.tolist())
    data = data.reshape(rows * cols, 5, 1, 1)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(5, 1, 1), padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size, padding='same'),
        tf.keras.layers.Conv2D(filters=filters * 2, kernel_size=kernel_size, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu')
    ])

    # Split data into batches
    num_batches = (rows * cols + batch_size - 1) // batch_size
    results = []

    for i in range(num_batches):
        batch_data = data[i * batch_size:(i + 1) * batch_size]
        batch_results = model.predict(batch_data)
        batch_results = batch_results.mean(axis=1)
        results.append(batch_results)

    # Combine all batches
    results = np.concatenate(results)
    results = results.reshape(rows, cols)

    convolved_df = pd.DataFrame(results, index=df.index, columns=df.columns)
    return convolved_df

def process_and_save_data(input_path, output_path, window_size=2, kernel_size=(3, 1), pool_size=(3, 1), filters=3, batch_size=16, random_seed=42):
    """
    Function to process data, apply convolution and pooling, and save the final output.

    Args:
    input_path (str): Path to the input CSV file.
    output_path (str): Path to save the processed output CSV file.
    window_size (int): Size of the window for windowed dataframe creation.
    kernel_size (tuple): Kernel size for the Conv2D layer.
    pool_size (tuple): Pool size for the MaxPooling2D layer.
    filters (int): Number of filters for the Conv2D layer.
    batch_size (int): Batch size for processing.
    random_seed (int): Random seed for reproducibility.
    """
    # Read input data
    df = pd.read_csv(input_path)

    # Create windowed dataframe
    windowed_df = create_windowed_dataframe(df, window_size)

    # Separate columns to remove
    columns_to_remove = ['Age', 'Sex', 'TOD']
    data_set_conv = windowed_df.drop(columns=columns_to_remove)
    age_sex_tod = windowed_df[columns_to_remove]

    # Apply convolution and pooling
    convolved_df = apply_convolution_and_pooling(data_set_conv, kernel_size, pool_size, filters, batch_size, random_seed)

    # Concatenate with age_sex_tod dataframe
    final_df = pd.concat([convolved_df, age_sex_tod], axis=1)

    # Save final dataframe to CSV
    final_df.to_csv(output_path, index=False)


ratios = ['80']
dr_types = ['MM', 'log', 'nonnormalized']
datasets = ['BA11', 'BA47']
splits = ['test', 'train']
base_input_path = 'split_80/'
base_output_path = 'multiple_layers_w2_conv/'

# Generate file paths
input_files = [f'{base_input_path}{dataset}_{ratio}_{dr_type}_{split}.csv'
               for dataset in datasets
               for ratio in ratios
               for dr_type in dr_types
               for split in splits]

output_files = [f'{base_output_path}{dataset}_{ratio}_{dr_type}_{split}.csv'
                for dataset in datasets
                for ratio in ratios
                for dr_type in dr_types
                for split in splits]

# input_files = ['split_80/BA47_80_nonnormalized_test.csv','split_80/BA47_80_nonnormalized_train.csv']
# output_files = ['multiple_layers_w2_conv/BA47_80_nonnormalized_test.csv','multiple_layers_w2_conv/BA47_80_nonnormalized_train.csv']


# Process and save all files
for input_path, output_path in zip(input_files, output_files):
    process_and_save_data(input_path, output_path)

## if want to run this code 
# - make sure that you change the filepath above in the base_input_path and the base_output_path
# - for window sizes : 1, 2, 3 the input shape in the conv2d layer changes each time
# - for 1 the shape is (1,3,1,1) and for 2 the shape is (1,5,1,1) and for 3 the shape is (1,7,1,1)
# - understand the first 1 in the above shape is for the batch_size