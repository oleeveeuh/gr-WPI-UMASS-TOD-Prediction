import numpy as np
import pandas as pd
from read_train import *
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from skorch import NeuralNetRegressor
from model_definitions import LSTMRegressor, create_sequences, set_seeds

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

set_seeds(RANDOM_STATE)

brain_areas = ('BA11')#,'BA47')

best_combinations_BA11 = {
    'Option 1': ('LSTM', 'PCA_90','MM', 80),
    'Option 1 PCT': ('LSTM', 'PCA_90','log', 80),

    'Option 2': ('ExtraTrees', 'Isomap_90', 'window3','MM', 80),
    'Option 2 PCT': ('AdaBoost', 'PCA_90', 'window2','log', 80),

    'Option 3': ('Random Forest', 'KPCA_90', 'window1','MM', 60),
    'Option 3 PCT': ('AdaBoost', 'KPCA_90', 'window1','log', 80),

    'Option 3 Flatten': ('Random Forest', 'KPCA_90', 'window1','MM', 60),
    'Option 3 Flatten PCT': ('AdaBoost', 'KPCA_90', 'window1','log', 80),
}

best_models_BA11 = {
    'Option 1': 
        (NeuralNetRegressor(
            LSTMRegressor,
            module__hidden_size=75,
            module__num_layers=1,
            module__output_size=1,
            max_epochs=30,
            batch_size = 16,
            lr=0.1,
            iterator_train__shuffle=False,
            train_split=None,
            device='cpu'
            )),
    'Option 1 PCT': 
        (NeuralNetRegressor(
                LSTMRegressor,
                module__hidden_size=25,
                module__num_layers=1,
                module__output_size=1,
                max_epochs=20,
                batch_size = 16,
                lr=0.1,
                iterator_train__shuffle=False,
                train_split=None,
                device='cpu'
                )),

    'Option 2': ExtraTreesRegressor(n_estimators = 200, 
                                     min_samples_split = 5, 
                                     min_samples_leaf =2, 
                                     max_depth =None,
                                     random_state=42
                                     ),
    'Option 2 PCT': AdaBoostRegressor(n_estimators=50, 
                                  loss = 'exponential', 
                                  learning_rate = .01,
                                  random_state=42
                                  ),

    'Option 3': RandomForestRegressor(
        n_estimators = 10, 
        min_samples_split = 10, 
        min_samples_leaf = 2,
        max_depth = 10,
        random_state = 42
    ),
    'Option 3 PCT': AdaBoostRegressor(n_estimators=200, 
                                  loss = 'square', 
                                  learning_rate = .01,
                                  random_state=42
                                  ),

    'Option 3 Flatten': RandomForestRegressor(
        n_estimators =50, 
        min_samples_split = 2,
        min_samples_leaf = 4, 
        max_depth = 20,
        random_state=42
    ),
    'Option 3 Flatten PCT': AdaBoostRegressor(
        n_estimators=400, 
        loss = 'square', 
        learning_rate = 1.0,
        random_state=42
        ),
}


best_combinations_BA47 = {
    'Option 1': ('LSTM', 'PCA_90', 'MM', 80),
    'Option 1 PCT': ('LSTM', 'PCA_90','log', 80),

    'Option 2': ('AdaBoost', 'PCA_95', 'window3','MM', 70),
    'Option 2 PCT': ('AdaBoost', 'KPCA_95', 'window3','log', 80),

    'Option 3': ('LSTM', 'PCA_90', 'window2','MM', 70),
    'Option 3 PCT': ('LSTM', 'Isomap_90', 'window3','log', 80),

    'Option 3 Flatten': ('LSTM', 'PCA_90', 'window2','MM', 70),
    'Option 3 Flatten PCT': ('LSTM', 'Isomap_90', 'window3','log', 80),
}

best_models_BA47 = {
    'Option 1':
      (NeuralNetRegressor(
            LSTMRegressor,
            module__hidden_size=75,
            module__num_layers=1,
            module__output_size=1,
            max_epochs=30,
            batch_size = 16,
            lr=0.1,
            iterator_train__shuffle=False,
            train_split=None,
            device='cpu'
            )),
    'Option 1 PCT': 
        (NeuralNetRegressor(
                LSTMRegressor,
                module__hidden_size=25,
                module__num_layers=1,
                module__output_size=1,
                max_epochs=20,
                batch_size = 16,
                lr=0.1,
                iterator_train__shuffle=False,
                train_split=None,
                device='cpu'
                )),
    'Option 2': AdaBoostRegressor(n_estimators=400, 
                                  loss = 'square', 
                                  learning_rate = 1.0,
                                  random_state=42
                                  ),
    'Option 2 PCT': AdaBoostRegressor(n_estimators=400, 
                                  loss = 'square', 
                                  learning_rate = 1.0,
                                  random_state=42
                                  ),          
    'Option 3': 
        (NeuralNetRegressor(
                LSTMRegressor,
                module__hidden_size=25,
                module__num_layers=1,
                module__output_size=1,
                max_epochs=20,
                batch_size = 16,
                lr=0.1,
                iterator_train__shuffle=False,
                train_split=None,
                device='cpu'
                )),
    'Option 3 PCT': 
        (NeuralNetRegressor(
                LSTMRegressor,
                module__hidden_size=25,
                module__num_layers=1,
                module__output_size=1,
                max_epochs=20,
                batch_size = 16,
                lr=0.1,
                iterator_train__shuffle=False,
                train_split=None,
                device='cpu'
                )),

    'Option 3 Flatten': 
        (NeuralNetRegressor(
                LSTMRegressor,
                module__hidden_size=25,
                module__num_layers=1,
                module__output_size=1,
                max_epochs=20,
                batch_size = 16,
                lr=0.1,
                iterator_train__shuffle=False,
                train_split=None,
                device='cpu'
                )),
    'Option 3 Flatten PCT':
        (NeuralNetRegressor(
                LSTMRegressor,
                module__hidden_size=75,
                module__num_layers=1,
                module__output_size=1,
                max_epochs=30,
                batch_size = 16,
                lr=0.1,
                iterator_train__shuffle=False,
                train_split=None,
                device='cpu'
                )),
}

paths = {
    'Option 1': r"data/reduced_data/",
    'Option 2': r"data/reduced_encoded/",
    'Option 3': r"data/reduced_CNN/",
    'Option 3 Flatten': r"data/reduced_CNN_flatten/"
}

path = ("/Users/olivialiau/Downloads/gr-WPI-UMASS-TOD-Project/data/train_test_split_data/BA11")
original_data_60 = pd.read_csv(f"{path}/BA11_60_nonnormalized_train.csv")
original_data_70 = pd.read_csv(f"{path}/BA11_70_nonnormalized_train.csv")
original_data_80 = pd.read_csv(f"{path}/BA11_80_nonnormalized_train.csv")

MM_cols = {
    60: original_data_60['TOD'],
    70: original_data_70['TOD'],
    80: original_data_80['TOD'],
}

# normalization_equations = {
#     "log": {y *[max(TOD_BA11) - min(TOD_BA11)]+ min(TOD_BA11)},
#     "minmax": {np.exp(y)}
# }


def get_pred():
    train_path = os.path.join(paths[option[:8]], train_file)
    test_path = os.path.join(paths[option[:8]], test_file)
        
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    y_train = train_data.pop('TOD')
    X_train = train_data

    y_test = test_data.pop('TOD')
    X_test = test_data

    if model_name == 'LSTM':
        X_train = X_train.values.astype(np.float32)
        X_test = X_test.values.astype(np.float32)
        y_train = y_train.values.astype(np.float32)
        y_test = y_test.values.astype(np.float32)
        X_train, y_train = create_sequences(X_train, y_train)
        X_test, y_test = create_sequences(X_test, y_test)

    # print(f"Fitting {model}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred 


def transform_TOD():
    y_test, y_pred = get_pred()
    # y_test = y_test.tolist()
    # y_pred = y_pred.tolist()

    transformed_y_test = []
    transformed_y_pred = []
    
    if n_method == 'MM':
        col = MM_cols[split]
        # print(f"Transforming MinMax")
        for (test_value, pred_value) in zip(y_test, y_pred):
            transformed_test = (test_value * (max(col) - min(col)) + min(col))
            transformed_pred = (pred_value * (max(col) - min(col)) + min(col))

            transformed_y_test.append(transformed_test)
            transformed_y_pred.append(transformed_pred)

    else:
        # print(f"Transforming Log")

        for (test_value, pred_value) in zip(y_test, y_pred):
            transformed_test = (np.exp(test_value))
            transformed_pred = (np.exp(pred_value))

            transformed_y_test.append(transformed_test)
            transformed_y_pred.append(transformed_pred)

    transformed_y_test = np.array(transformed_y_test)
    transformed_y_pred = np.array(transformed_y_pred)

    # print(transformed_y_test)

    # differences = transformed_y_test - transformed_y_pred
    # std_dev = differences.std()
    # mse = mean_squared_error(transformed_y_test, transformed_y_pred)
    # mae = mean_absolute_error(transformed_y_test, transformed_y_pred)
    # mape = mean_absolute_percentage_error( transformed_y_test, transformed_y_pred)
    # rmse = mse ** 0.5
    # smape = 100 * (2 * np.abs(transformed_y_test - transformed_y_pred) / (np.abs(transformed_y_test) + np.abs(transformed_y_pred))).mean()
    
    # print(f"{brain_area} {option}: {model_name}")
    # print(f"STD: {std_dev}\nMSE: {mse}\nMAE: {mae}\nMAPE: {mape}\nRMSE: {rmse}\nSMAPE: {smape}")

    # add_graph(transformed_y_test.tolist(), transformed_y_pred.tolist())
    return transformed_y_test, transformed_y_pred


def add_graph():
    y_test, y_pred = transform_TOD()
    # i = np.arange(0,len(y_pred))
    ax[x, y].scatter(y_test, y_pred, s=10, c='red', marker='o', label='Predicted')
    ax[x, y].plot(y_test, y_test, c='green', label='Actual')
    ax[x, y].set_title(f'{brain_area} {option}: {model_name} w/ {dr_method} & {n_method}{split}')
    ax[x, y].set_xlabel('Actual TOD')
    ax[x, y].set_ylabel('TOD')
    ax[x, y].legend()

    differences = y_test - y_pred
    std_dev = differences.std()
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae}")
    ax[x, y].text(0.97, 0.10, f"STD: {std_dev:.3f}", ha='right',va = 'top', fontsize=13, transform=ax[x, y].transAxes, bbox=dict(facecolor='pink', alpha=.6))




for brain_area in brain_areas:
    fig, ax = plt.subplots(4, 2, figsize=(12, 16), layout="constrained")
    y = 0
    x = 0
    if brain_area == 'BA11':
        model_dir = best_models_BA11
        combinations_dir = best_combinations_BA11
    else: 
        model_dir = best_models_BA47
        combinations_dir = best_combinations_BA47

    
    for option, model in model_dir.items():

        if option == 'Option 1' or option == "Option 1 PCT":
            model_name, dr_method, n_method, split = (combinations_dir[option])
            train_file = (f"{brain_area}_{split}_{n_method}_{dr_method}_train.csv")
            test_file = (f"{brain_area}_{split}_{n_method}_{dr_method}_test.csv")
        # print(f"Opening {train_file} and {test_file}")
        else:
            model_name, dr_method, window, n_method, split = (combinations_dir[option])
            train_file = (f"{brain_area}_{split}_{n_method}_{window}_{dr_method}_train.csv")
            test_file = (f"{brain_area}_{split}_{n_method}_{window}_{dr_method}_test.csv")
            # print(f"Opening {train_file} and {test_file}")

        add_graph()
        # transform_TOD()
        if y == 0: y=1
        elif y == 1: 
            y=0
            x+=1
    # plt.subplots_adjust(right=0.85)
    # plt.tight_layout()
    plt.savefig(f"{brain_area}_plot.png")



