import pandas as pd
import os
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# get the path to data
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '..', '..', 'data', 'train_test_split_data')
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
DR_Isomap = 'Isomap'
DR_ICA = 'ICA'
DR_PCA = 'PCA'

folders = [folder_BA11, folder_BA47]
DR_folders = ['ICA_90', 'ICA_95', 'KPCA_95', 'KPCA_90', 'PCA_90', 'PCA_95']
splits = [split_60, split_70, split_80]
methods = [method_log, method_MM, method_None]
value = [90, 95]

use_reduce = True

if(use_reduce):
    train_file_name = f"{folder_BA11}_{split_60}_{method_log}_DR_train.csv"
    train_file = os.path.join(data_dir, folder_BA11, DR_Isomap, train_file_name)
    test_file_name = f"{folder_BA11}_{split_60}_{method_log}_DR_test.csv"
    test_file = os.path.join(data_dir, folder_BA11, DR_Isomap, test_file_name)
else:
    train_file_name = f"{folder_BA11}_{split_60}_{method_MM}_train.csv"
    train_file = os.path.join(data_dir, folder_BA11, train_file_name)
    test_file_name = f"{folder_BA11}_{split_60}_{method_MM}_test.csv"
    test_file = os.path.join(data_dir, folder_BA11, test_file_name)

# Function to load data and train models
def process_folder(folder):
    folder_path = os.path.join(data_dir, folder)
    train_files = [f for f in os.listdir(folder_path) if 'train' in f and 'nonnormalized' not in f]
    test_files = [f for f in os.listdir(folder_path) if 'test' in f and 'nonnormalized' not in f]
    
    results = {'RandomForest': {'mse': [], 'r2': [], 'files': []}, 'ExtraTrees': {'mse': [], 'r2': [], 'files': []}}
    for train_file, test_file in zip(train_files, test_files):
        train_file_path = os.path.join(folder_path, train_file)
        test_file_path = os.path.join(folder_path, test_file)
        
        # Load data
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        
        # Preprocess data
        y_train = train_data.pop('TOD_pos')
        X_train = train_data

        y_test = test_data.pop('TOD_pos')
        X_test = test_data

        # Define models
        models = {
            'RandomForest': RandomForestRegressor(random_state=42),
            'ExtraTrees': ExtraTreesRegressor(random_state=42)
        }

        # Define parameter grids for RandomizedSearchCV
        param_grids = {
            'RandomForest': {
                'randomforest__n_estimators': [10, 50, 100, 200],
                'randomforest__max_depth': [None, 10, 20, 30],
                'randomforest__min_samples_split': [2, 5, 10],
                'randomforest__min_samples_leaf': [1, 2, 4]
            },
            'ExtraTrees': {
                'extratrees__n_estimators': [10, 50, 100, 200],
                'extratrees__max_depth': [None, 10, 20, 30],
                'extratrees__min_samples_split': [2, 5, 10],
                'extratrees__min_samples_leaf': [1, 2, 4]
            }
        }

        for model_name, model in models.items():
            print(f"Training {model_name} on {train_file}")

            # Create pipeline
            pipeline = Pipeline([
                (model_name.lower(), model)
            ])

            # Set up k-fold cross-validation with random search
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            random_search = RandomizedSearchCV(pipeline, param_distributions=param_grids[model_name], n_iter=10, cv=kf, verbose=1, n_jobs=-1, random_state=42)

            # Fit the model
            random_search.fit(X_train, y_train)

            # Best parameters
            print(f"Best parameters found for {model_name} on {train_file}: {random_search.best_params_}")

            # Predict on the test data
            y_pred = random_search.predict(X_test)

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"{model_name} - Mean Squared Error: {mse}")
            print(f"{model_name} - R^2 Score: {r2}")

            # Store results
            results[model_name]['mse'].append(mse)
            results[model_name]['r2'].append(r2)
            results[model_name]['files'].append(test_file)

    return results
    

# Process each folder
# for folder in DR_folders:
#     process_folder(folder)
folder = 'BA11/PCA95'
result = process_folder(folder)

# Plot the results
fig, axes = plt.subplots(2, 1, figsize=(14, 14))

# Plot Mean Squared Error
bar_width = 0.4
index = np.arange(len(result['RandomForest']['mse']))

axes[0].bar(index, result['RandomForest']['mse'], bar_width, label='RandomForest')
axes[0].bar(index + bar_width, result['ExtraTrees']['mse'], bar_width, label='ExtraTrees')

# Adding file names as labels
file_names = result['RandomForest']['files']
axes[0].set_xticks(index + bar_width / 2)
axes[0].set_xticklabels(file_names, rotation=45, ha="right")

axes[0].set_title('Mean Squared Error (MSE) by File')
axes[0].set_xlabel('File')
axes[0].set_ylabel('MSE')
axes[0].legend()

# Plot R^2 Score
axes[1].bar(index, result['RandomForest']['r2'], bar_width, label='RandomForest')
axes[1].bar(index + bar_width, result['ExtraTrees']['r2'], bar_width, label='ExtraTrees')

# Adding file names as labels
axes[1].set_xticks(index + bar_width / 2)
axes[1].set_xticklabels(file_names, rotation=45, ha="right")

axes[1].set_title('R^2 Score by File')
axes[1].set_xlabel('File')
axes[1].set_ylabel('R^2 Score')
axes[1].legend()

plt.tight_layout()
plt.show()

# Output summary
summary = []
for model_name in ['RandomForest', 'ExtraTrees']:
    for file, mse, r2 in zip(result[model_name]['files'], result[model_name]['mse'], result[model_name]['r2']):
        summary.append(f"{model_name} - {file}: MSE = {mse}, R^2 = {r2}")

summary_text = "\n".join(summary)
print("Summary:\n", summary_text)