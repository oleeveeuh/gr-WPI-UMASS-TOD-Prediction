import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
from read_train import *
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import os
from skorch import NeuralNetRegressor
import torch.optim as optim
from model_definitions import *
import math

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seeds(RANDOM_STATE)

brain_areas = ('BA11','BA47')

best_combinations_BA11 = {
    'Option 1': ('LSTM', 'PCA_90', 'None', 'MM', 80),
    # 'Option 1 PCT': ('LSTM', 'PCA_90', 'None', 'log', 80),

    'Option 2': ('ExtraTrees', 'Isomap_90', 'window3','MM', 80),
    # 'Option 2 PCT': ('AdaBoost', 'PCA_90', 'window2','log', 80),

    # 'Option 3': ('Random Forest', 'KPCA_90', 'window1','MM', 60),
    # 'Option 3 PCT': ('AdaBoost', 'KPCA_90', 'window1','log', 80),

    'Option 3': ('Bagging', 'PCA_90', 'window3','MM', 70),
    # 'Option 4 Flatten PCT': ('SVR', 'PCA_90', 'window3','log', 80),
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
    # 'Option 1 PCT': 
    #     (NeuralNetRegressor(
    #             LSTMRegressor,
    #             module__hidden_size=25,
    #             module__num_layers=1,
    #             module__output_size=1,
    #             max_epochs=20,
    #             batch_size = 16,
    #             lr=0.1,
    #             iterator_train__shuffle=False,
    #             train_split=None,
    #             device='cpu'
    #             )),

    'Option 2': 
    
    ExtraTreesRegressor(n_estimators = 200, 
                                     min_samples_split = 5, 
                                     min_samples_leaf =2, 
                                     max_depth =None,
                                     random_state=42
                                     ),
    # 'Option 2 PCT': AdaBoostRegressor(n_estimators=50, 
    #                               loss = 'exponential', 
    #                               learning_rate = .01,
    #                               random_state=42
    #                               ),

    # 'Option 3': RandomForestRegressor(
    #     n_estimators = 10, 
    #     min_samples_split = 10, 
    #     min_samples_leaf = 2,
    #     max_depth = 10,
    #     random_state = 42
    # ),
    # 'Option 3 PCT': AdaBoostRegressor(n_estimators=200, 
    #                               loss = 'square', 
    #                               learning_rate = .01,
    #                               random_state=42
    #                               ),

    'Option 3': BaggingRegressor(random_state=42,
                                     n_estimators = np.int64(50), 
                                     max_samples = np.float64(0.7), 
                                     max_features = np.float64(0.9)
    ),
    # 'Option 4 Flatten PCT': SVR(max_iter=1000,
    #                             kernel = 'rbf', 
    #                             gamma = 0.0001, 
    #                             C= 100),
}


best_combinations_BA47 = {
    'Option 1': ('LSTM', 'PCA_90', 'None', 'MM', 80),
    # 'Option 1 PCT': ('LSTM', 'PCA_90','None','log', 80),

    'Option 2': ('AdaBoost', 'PCA_95', 'window3','MM', 70),
    # 'Option 2 PCT': ('AdaBoost', 'KPCA_95', 'window3','log', 80),

    # 'Option 3': ('MLP', 'KPCA_95', 'window3','MM', 80),
    # 'Option 3 PCT': ('Bagging', 'KPCA_95', 'window1','log', 80),

    # 'Option 3': ('MLP', 'PCA_90', 'window3','MM', 80),
    'Option 3': ('AdaBoost', 'KPCA_95', 'window3','log', 80),
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
    # 'Option 1 PCT': 
    #     (NeuralNetRegressor(
    #             LSTMRegressor,
    #             module__hidden_size=25,
    #             module__num_layers=1,
    #             module__output_size=1,
    #             max_epochs=20,
    #             batch_size = 16,
    #             lr=0.1,
    #             iterator_train__shuffle=False,
    #             train_split=None,
    #             device='cpu'
                # )),
    'Option 2': AdaBoostRegressor(n_estimators=400, 
                                  loss = 'square', 
                                  learning_rate = 1.0,
                                  random_state=42
                                  ),
    # 'Option 2 PCT': AdaBoostRegressor(n_estimators=400, 
    #                               loss = 'square', 
    #                               learning_rate = 1.0,
    #                               random_state=42
    #                               ),   

    # 'Option 3': NeuralNetRegressor(
    #         MLPRegressor,
    #         max_epochs=20,  # You can adjust this
    #         lr=0.1,
    #         module__num_units =10, 
    #         module__activation_func = nn.Tanh(), 
    #         iterator_train__shuffle=False,
    #         optimizer=optim.Adam,
    #         device = 'cpu',
    #         ),
    # 'Option 3 PCT': BaggingRegressor(random_state=42,
    #                                  n_estimators = np.int64(150), 
    #                                  max_samples = np.float64(0.1), 
    #                                  max_features = np.float64(1.0)),
    # 'Option 3': NeuralNetRegressor(
    #         MLPRegressor,
    #         # module__input_size=input_size,
    #         max_epochs=20,  # You can adjust this
    #         lr=0.05,
    #         module__num_units = 20, 
    #         module__activation_func =  nn.ReLU(),
    #         iterator_train__shuffle=False,
    #         optimizer=optim.Adam,
    #         device = 'cpu',
            # ),
    'Option 3': AdaBoostRegressor(n_estimators=250, 
                                  loss = 'exponential', 
                                  learning_rate = 0.01,
                                  random_state=42
                                  ),   
}

paths = {
    'Option 1': r"data/reduced_data/",
    # 'Option 1 PCT': r"data/reduced_data/",
    'Option 2': r"data/reduced_encoded/",
    # 'Option 2 PCT': r"data/reduced_encoded/",
    # 'Option 3': r"data/reduced_CNN/",
    # 'Option 3 PCT': r"data/reduced_CNN/",
    # 'Option 4 Flatten': r"data/reduced_CNN_flatten/",
    # 'Option 4 Flatten PCT': r"data/reduced_CNN_flatten/",
    'Option 3': r"data/reduced_CNN_flatten/",

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
best_std_multi = {
        'Option 1': 100,
        'Option 2': 100,
        'Option 3': 100,
        'Option 4': 100
    }
best_graphs_multi = {
        'Option 1': [],
        'Option 2': [],
        'Option 3': [],
        'Option 4': []
    }
std_deviations = {}
x_values = []

def get_dir(brain_area):
    if brain_area == 'BA11':
        model_dir = best_models_BA11
        combinations_dir = best_combinations_BA11
    else: 
        model_dir = best_models_BA47
        combinations_dir = best_combinations_BA47
    return combinations_dir, model_dir

def get_pred(brain_area, option, model, model_name, dr_method, window, n_method, split):
    if window == 'None':
        train_file = (f"{brain_area}_{split}_{n_method}_{dr_method}_train.csv")
        test_file = (f"{brain_area}_{split}_{n_method}_{dr_method}_test.csv")
    else:
        train_file = (f"{brain_area}_{split}_{n_method}_{window}_{dr_method}_train.csv")
        test_file = (f"{brain_area}_{split}_{n_method}_{window}_{dr_method}_test.csv")

    train_path = os.path.join(paths[option], train_file)
    test_path = os.path.join(paths[option], test_file)

        
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    y_train = train_data.pop('TOD')
    X_train = train_data

    y_test = test_data.pop('TOD')
    X_test = test_data

    if model_name == 'LSTM' or model_name == 'MLP':
        X_train = X_train.values.astype(np.float32)
        X_test = X_test.values.astype(np.float32)
        y_train = y_train.values.astype(np.float32)
        y_test = y_test.values.astype(np.float32)
        
        if model_name == 'LSTM':
            X_train, y_train = create_sequences(X_train, y_train)
            X_test, y_test = create_sequences(X_test, y_test)

    print(f"Fitting {model} {option}{window}{dr_method}{split}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred, n_method, split

def transform_TOD(brain_area, option, model, model_name, dr_method, window, n_method, split):
    y_test, y_pred, n_method, split = get_pred(brain_area, option, model, model_name, dr_method, window, n_method, split)
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


    # differences = transformed_y_test - transformed_y_pred
    # std_dev = differences.std()
    # mse = mean_squared_error(transformed_y_test, transformed_y_pred)
    # mae = mean_absolute_error(transformed_y_test, transformed_y_pred)
    # mape = mean_absolute_percentage_error( transformed_y_test, transformed_y_pred)
    # rmse = mse ** 0.5
    # smape = 100 * (2 * np.abs(transformed_y_test - transformed_y_pred) / (np.abs(transformed_y_test) + np.abs(transformed_y_pred))).mean()

    # add_graph(transformed_y_test.tolist(), transformed_y_pred.tolist())
    return transformed_y_test, transformed_y_pred

def regression_graph(brain_area, ax, y_test, y_pred, model_name, dr_method, window, n_method, split, option, std_dev, mae):
    y = 0
    x =  int(option[7]) - 1
    if option[-3:] == 'PCT':
        y = 1
    
    ax[x, y].scatter(y_test, y_pred, s=40, c='red', marker='o', label='Predicted')
    ax[x, y].plot(y_test, y_test, c='green', label='Actual')
    if option[:8] == "Option 1":
        ax[x, y].set_title(f'{brain_area} {option}: {model_name} ({dr_method}, {n_method}{split})')
    else:
        ax[x, y].set_title(f'{brain_area} {option}: {model_name} ({window}, {dr_method}, {n_method}{split})')

    ax[x, y].set_xlabel('Actual TOD', fontsize=20)
    ax[x, y].set_ylabel('TOD',fontsize=20)
    ax[x, y].legend()

    p = (f"STD: {std_dev:.3f} | MAE: {mae:.3f}")
    ax[x, y].text(0.97, 0.12, p, ha='right',va = 'top', fontsize=20, transform=ax[x, y].transAxes, bbox=dict(facecolor='pink', alpha=.6))

def reset_list():
    global best_std_multi
    best_std_multi = {
        'Option 1': 100,
        'Option 2': 100,
        'Option 3': 100,
        'Option 4': 100
    }
    print("reset")

def multi_axis_graph(brain_area, y_test, y_pred, model_name, dr_method, window, n_method, split, option, std_dev, done=False):
    global x_values
    if std_dev < best_std_multi[option[:8]]:
        best_std_multi[option[:8]] = std_dev
        best_graphs_multi[option[:8]] = [y_test, y_pred]
        if (len(y_test) < len(x_values)) or (len(x_values) == 0): 
            x_values = np.array(y_test)
            pyplot.xlim(min(x_values) - .5, max(x_values) +.5)

    if done:
        colors = {
            'Option 1': 'green',
            'Option 2': 'magenta',
            'Option 3': 'orange',
            'Option 4': 'red'
        }
        y_err = 2
        # y_err = differences.std() * np.sqrt(1/len(x_values) + (x_values - x_values.mean())**2 / np.sum((x_values - x_values.mean())**2))

        pyplot.fill_between(x_values, x_values-y_err, x_values+y_err, step=None, alpha = .4)
        # pyplot.plot(x_values, x_values, c = 'blue', label='Actual')

        for opt, values in best_graphs_multi.items():
            if opt[:8] == "Option 3":
                continue
            y_test, y_pred = values
            pyplot.scatter(y_test, y_pred, s=14, c= colors[opt], marker='o')
            pyplot.plot(y_test, y_pred, c = colors[opt], label=f'{opt} Predicted')

            # pyplot.plot(y_test, y_test, c='red', )
        pyplot.tight_layout()
        pyplot.legend()
        pyplot.savefig(f"{brain_area}_multi_plot.png")
        pyplot.clf()

def get_stats(y_test, y_pred):
    differences = y_test - y_pred
    std_dev = differences.std()
    mae = mean_absolute_error(y_test, y_pred)
    return std_dev, mae

def regression_graph2(brain_area, ax, y_test, y_pred, model_name, dr_method, window, n_method, split, option, std_dev, mae):
    x =  int(option[7]) - 1
    if int(option[7]) == 3:
        x = 1
    if int(option[7]) == 2:
        x=2

    colors = {
        'BA11': '#BB3f04',
        'BA47': '#089396',
    }
    shapes = {  
            'BA11': 'o',
            'BA47': '^',
        }
    
    ax[x].scatter(y_test, y_pred, s=100, c=colors[brain_area], marker=shapes[brain_area], label=f'{brain_area} Predicted')
    ax[x].plot(y_test, y_test, c='black', label='Actual')

    # if option[:8] == "Option 1":
    #     ax[x].set_title(f'{brain_area} {option}: {model_name} ({dr_method}, {n_method}{split})')
    # else:
    #     ax[x].set_title(f'{brain_area} {option}: {model_name} ({window}, {dr_method}, {n_method}{split})')

    ax[2].set_xlabel('Actual TOD', fontsize=20)
    ax[1].set_ylabel('Predicted TOD',fontsize=20)
    ax[x].set_facecolor('#FFFAEC')
    ax[x].tick_params(axis='both', which='major', labelsize=12)
    legend = ax[x].legend(ncol=1, fontsize='large', loc = 'upper left')
    legend.get_frame().set_alpha(0)


    # p = (f"STD: {std_dev:.4f} | MAE: {mae:.3f}")
    std_deviations[brain_area + option]=(mae)
    # =ax[x].text(0.97, 0.12, p, ha='right',va = 'top', fontsize=20, transform=ax[x].transAxes, bbox=dict(facecolor='pink', alpha=.6))

def main(regression2 = False, multi_axis=False): 

    if regression2: 
        fig, ax = plt.subplots(3, 1, figsize=(6, 10), layout="constrained", facecolor='#FFFAEC')

    for brain_area in brain_areas:

        # if regression: fig, ax = plt.subplots(4, 2, figsize=(12, 16), layout="constrained")
        if multi_axis:
            pyplot.figure(figsize=(12, 6))  # Width = 10 inches, Height = 6 inches
            global x_values
            x_values = []
            reset_list()

        combinations_dir, model_dir = get_dir(brain_area)
        for option, model in model_dir.items():
            model_name, dr_method, window, n_method, split = (combinations_dir[option])
            transformed_y_test, transformed_y_pred = transform_TOD(brain_area, option, model, model_name, dr_method, window, n_method, split)
            std_dev, mae = get_stats(transformed_y_test, transformed_y_pred)

            if regression2:
                regression_graph2(brain_area, ax, transformed_y_test, transformed_y_pred, model_name, dr_method, window, n_method, split, option, std_dev, mae)

            if multi_axis:
                if model == model_dir['Option 4 Flatten PCT']: multi_axis_graph(brain_area, transformed_y_test, transformed_y_pred, model_name, dr_method, window, n_method, split, option, std_dev, done=True)
                else: multi_axis_graph(brain_area, transformed_y_test, transformed_y_pred, model_name, dr_method, window, n_method, split, option, std_dev, done=False)

        # if regression: plt.savefig(f"{brain_area}_testplot.png")
    # plt.savefig(f"{brain_area}_testplot.png")
    print(std_deviations)


# main(regression2 = True, multi_axis=False)


