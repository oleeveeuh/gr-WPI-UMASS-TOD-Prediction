import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold
import matplotlib.pyplot as plt
# get the path to data
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '..', '..', 'data', 'train test split data')
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
DR_Isomap = "Isomap"

folders = [folder_BA11, folder_BA47]
splits = [split_60, split_70, split_80]
methods = [method_log, method_MM, method_None]



def process_folder(folder):
    folder_path = os.path.join(data_dir, folder)
    train_files = [f for f in os.listdir(folder_path) if 'train' in f]
    test_files = [f for f in os.listdir(folder_path) if 'test' in f]
    
    results = {'mse': [], 'r2': [], 'files': []}
    for train_file, test_file in zip(train_files, test_files):
        train_file_path = os.path.join(folder_path, train_file)
        test_file_path = os.path.join(folder_path, test_file)
        
        # Load data
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        
        # Preprocess data
        y_train = train_data.pop('TOD')
        X_train = train_data

        y_test = test_data.pop('TOD')
        X_test = test_data

        # Handle infinite and too large values
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

        X_train.fillna(X_train.mean(), inplace=True)
        X_test.fillna(X_test.mean(), inplace=True)

        # Define the SVM model
        svm = SVR()

        # Define the parameter grid for RandomizedSearchCV
        param_grid = {
            'svr__C': [0.1, 1, 10, 100, 1000],
            'svr__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'svr__kernel': ['linear', 'rbf']
        }

        # Create a pipeline
        pipeline = Pipeline([
            ('svr', svm)
        ])

        # Set up k-fold cross-validation with random search
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=kf, verbose=1, n_jobs=-1)

        # Fit the model
        random_search.fit(X_train, y_train)

        # Best parameters
        print(f"Best parameters found: {random_search.best_params_}")

        # Predict on the test data
        y_pred = random_search.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")


        # Store results
        results['mse'].append(mse)
        results['r2'].append(r2)
        results['files'].append(test_file)

    return results

folder = 'KPCA(90%)'
result = process_folder(folder)

# Plot the results
fig, axes = plt.subplots(2, 1, figsize=(14, 14))

# Plot Mean Squared Error
bar_width = 0.4
index = np.arange(len(result['mse']))

axes[0].bar(index, result['mse'], bar_width, label='SVR')


# Adding file names as labels
file_names = result['files']
axes[0].set_xticks(index + bar_width / 2)
axes[0].set_xticklabels(file_names, rotation=45, ha="right")

axes[0].set_title('Mean Squared Error (MSE) by File')
axes[0].set_xlabel('File')
axes[0].set_ylabel('MSE')
axes[0].legend()

# Plot R^2 Score
axes[1].bar(index, result['r2'], bar_width, label='SVR')

# Adding file names as labels
axes[1].set_xticks(index + bar_width / 2)
axes[1].set_xticklabels(file_names, rotation=45, ha="right")

axes[1].set_title('R^2 Score by File')
axes[1].set_xlabel('File')
axes[1].set_ylabel('R^2 Score')
axes[1].legend()

plt.tight_layout()
plt.show()