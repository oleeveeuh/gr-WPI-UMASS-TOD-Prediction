import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold

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
# Load data
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
# Preprocess data
y_train = train_data.pop('TOD_pos')
X_train = train_data

y_test = test_data.pop('TOD_pos')
X_test = test_data


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

