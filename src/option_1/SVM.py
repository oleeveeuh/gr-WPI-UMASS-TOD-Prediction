import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
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

# train_file_name = f"{folder_BA11}_{split_60}_{method_log}_DR_train.csv"
# train_file = os.path.join(data_dir, folder_BA11, DR_Isomap, train_file_name)
# test_file_name = f"{folder_BA11}_{split_60}_{method_log}_DR_test.csv"
# test_file = os.path.join(data_dir, folder_BA11, DR_Isomap, test_file_name)
train_file_name = f"{folder_BA11}_{split_60}_{method_log}_train.csv"
train_file = os.path.join(data_dir, folder_BA11, train_file_name)
test_file_name = f"{folder_BA11}_{split_60}_{method_log}_test.csv"
test_file = os.path.join(data_dir, folder_BA11, test_file_name)
# Load data
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
# Preprocess data
y_train = train_data.pop('TOD_pos')
X_train = train_data

y_test = test_data.pop('TOD_pos')
X_test = test_data

# Bin the target variable into 12 bins, each representing two hours
bins = range(0, 25, 2)
labels = range(12)
y_train_binned = pd.cut(y_train, bins=bins, labels=labels, right=False)
y_test_binned = pd.cut(y_test, bins=bins, labels=labels, right=False)

# Define the SVM model
svm = SVC()

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'svc__C': [0.1, 1, 10, 100, 1000],
    'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'svc__kernel': ['linear', 'rbf']
}

# Create a pipeline
pipeline = Pipeline([
    ('svc', svm)
])

# Set up k-fold cross-validation with random search
kf = KFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=kf, verbose=1, n_jobs=-1)

# Fit the model
random_search.fit(X_train, y_train_binned)

# Best parameters
print(f"Best parameters found: {random_search.best_params_}")

# Predict on the test data
y_pred = random_search.predict(X_test)

# Evaluate the model
print(classification_report(y_test_binned, y_pred))

