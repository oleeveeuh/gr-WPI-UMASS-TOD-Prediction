import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import os

# get the path to data
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '..', 'data', 'train_test_split_data')
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
DR_folders = ['ICA_90', 'ICA_95', 'KPCA_95', 'KPCA_90', 'PCA_90', 'PCA_95']
use_reduce = True

train_name = 'BA11_60_log_ICA_90_train.csv'
test_name = 'BA11_60_log_ICA_90_test.csv'

train_path = os.path.join(data_dir, DR_folders[0], train_name)
test_path = os.path.join(data_dir, DR_folders[0], test_name)

 # Load data
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Preprocess data
y_train = train_data.pop('TOD')
X_train = train_data

y_test = test_data.pop('TOD')
X_test = test_data

xgb_model = XGBRegressor(
    objective='reg:squarederror',
    eta=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    alpha=0.1,
    reg_lambda = 1,
    n_estimators=100,
)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on the test set
preds = xgb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, preds)
print(f"Mean Squared Error: {mse}")