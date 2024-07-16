import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Use the current directory
'''
directory = os.getcwd()
files = os.listdir(directory)

# Assuming the postfix are '_A' and '_B'
postfix_a = '_test.csv'
postfix_b = '_train.csv'

# Dictionary to store file pairs
file_pairs = {}

# Organize files into pairs
for file in files:
    if file.endswith(postfix_a):
        base_name = file[:-len(postfix_a)]
        pair_file = base_name + postfix_b
        if pair_file in files:
            file_pairs[file] = pair_file

# DataFrame to store comparison results
comparison_results = pd.DataFrame(columns=['File1', 'File2', 'Common_Values'])

# Read and compare files
for file_a, file_b in file_pairs.items():
    df_a = pd.read_csv(os.path.join(directory, file_a))
    df_b = pd.read_csv(os.path.join(directory, file_b))

    # Check if 'TOD' column exists in both DataFrames
    if 'TOD' not in df_a or 'TOD' not in df_b:
        print(f"Column 'TOD' not found in {file_a} or {file_b}. Skipping comparison.")
        continue

    # Find common values in the 'TOD' column
    common_values = set(df_a['TOD']).intersection(df_b['TOD'])

    # Store results
    comparison_results = pd.concat([comparison_results, pd.DataFrame([{
        'File1': file_a,
        'File2': file_b,
        'Common_Values': list(common_values)  # Convert set to list for better readability
    }])], axis=0, join='outer')

# Display or save the results
print(comparison_results)
# Optionally, save to a CSV file
# comparison_results.to_csv('intersection_values_comparison.csv', index=False)
'''
testfile = 'data/reduced_encoded/BA11_80_log_window1_ICA_95_test.csv'
trainfile = 'data/reduced_encoded/BA11_80_log_window1_ICA_95_train.csv'

train_df = pd.read_csv(trainfile)
test_df = pd.read_csv(testfile)

print(f'Reading files {trainfile} and {testfile}')

# Split features and target variable
y_train_log = train_df.pop('TOD')  # log-transformed target
X_train = train_df

y_test_log = test_df.pop('TOD')  # log-transformed target
X_test = test_df

model = LinearRegression()
model.fit(X_train, y_train_log)

# Predictions on both train and test sets (log scale)
y_pred_train_log = model.predict(X_train)
y_pred_test_log = model.predict(X_test)

# Transform predictions back to original scale
y_pred_train = np.exp(y_pred_train_log)
y_pred_test = np.exp(y_pred_test_log)
y_train = np.exp(y_train_log)
y_test = np.exp(y_test_log)

# Calculating and printing performance metrics for the training set on original scale
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
print("Training Set Performance (Original Scale):")
print("Mean Squared Error (MSE):", mse_train)
print("Coefficient of Determination (R²):", r2_train)

# Calculating and printing performance metrics for the testing set on original scale
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
print("Testing Set Performance (Original Scale):")
print("Mean Squared Error (MSE):", mse_test)
print("Coefficient of Determination (R²):", r2_test)


residuals = y_pred_test - y_test

# Calculate the standard deviation of the residuals
residuals_std = np.std(residuals)

print("Standard Deviation of Residuals:", residuals_std)