import numpy as np
import os
import fnmatch
import re
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

os.chdir(os.path.dirname(os.path.abspath(__file__)))

folder_list = ["/Users/tillieslosser/Downloads/Academic/Research/gr-WPI-UMASS-TOD-Project/data/train_test_split_data/" + f + "/" for f in ["ICA_90", "ICA_95", "KPCA_90", "KPCA_95", "PCA_90", "PCA_95"]]
data_dict = {}
file_names = []
TOD_dict = {}
regions = ["BA11", "BA47", "full"]
splits = ["80", "70", "60"]

for subfolder in folder_list:
    for file in os.listdir(subfolder):
        if fnmatch.fnmatch(file, '*.csv'):
            name = re.match(".+(?=_(train)|_(test)\.csv)", file)
            file_names.append(name.group())
            data_dict[file] = pd.DataFrame(np.genfromtxt(subfolder + "/" + file, delimiter=',', skip_header=1))
file_names = list(set(file_names))
print(file_names)

final_data_grouping = {}
for key in data_dict.keys():
    for name in file_names:
        if name not in final_data_grouping.keys():
            final_data_grouping[name] = {}
        if name in key:
            final_data_grouping[name][key] = data_dict[key]

# Print groupings
for data_group in final_data_grouping.keys():
    print("Data group:", data_group)
    for df in final_data_grouping[data_group].keys():
        print("        " , df)

# Assuming final_data_grouping is already populated
for data_group in list(final_data_grouping.keys()):

    for key in final_data_grouping[data_group].keys():
       if ("train" in key):
           train_data = final_data_grouping[data_group][key]
       elif ("test" in key):
           test_data = final_data_grouping[data_group][key]
       else:
           print("ERROR: ", key, "NOT FOUND")
    # Preprocess data
    X_train = train_data.iloc[:, :-1].values  # All columns except the last
    y_train = train_data.iloc[:, -1].values  # The last column

    y_test = test_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values

    # Create an SGDClassifier model
    sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)

    # Train the model
    print(data_group)
    print("----------X_train: ----------", X_train)
    print("----------y_train: ----------", y_train)
    sgd_regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = sgd_regressor.predict(X_test)

    # Evaluate the model
    rmse = mean_squared_error(y_test, y_pred, squared = False)

    print(f"{data_group} RMSE: {rmse}")
