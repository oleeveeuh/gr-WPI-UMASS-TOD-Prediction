import numpy as np
from sklearn.decomposition import FastICA
import os
import fnmatch
import re
import pandas as pd

BA11 = "data/train test split data/BA11"
BA47 = "data/train test split data/BA47"
full_data = "data/train test split data/full data"

folder_list = [BA11, BA47, full_data]
data_dict = {}
file_names = []
for subfolder in folder_list:
    for file in os.listdir(subfolder):
        if fnmatch.fnmatch(file, '*.csv'):
            name = re.match(".+(?=_(train)|_(test)\.csv)", file)
            file_names.append(name.group())
            data_dict[file] = np.genfromtxt(subfolder + "/" + file, delimiter=',', skip_header=1)
file_names = list(set(file_names))

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
    for dataset in final_data_grouping[data_group].keys():
        if "train" in dataset:
            to_DR_train = np.delete(final_data_grouping[data_group][dataset], 3, axis=1)
            train_name = dataset
        else:
            to_DR_test = np.delete(final_data_grouping[data_group][dataset], 3, axis=1)
            test_name = dataset
    ica_test = FastICA(n_components=100, algorithm='parallel', whiten=True)
    S_ica_test = ica_test.fit_transform(to_DR_train)  # Get the independent components from training data
    # Determine the number of components to use using the explained variance criterion
    explained_variance = np.var(S_ica_test, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    n_components_95 = np.argmax(np.cumsum(explained_variance_ratio) >= 0.95) + 1
    n_components_90 = np.argmax(np.cumsum(explained_variance_ratio) >= 0.90) + 1

    #take the number of components for explaining 95, 90 % reconstruction variance
    ica_95 = FastICA(n_components=n_components_95, algorithm='parallel', whiten=True)
    ica_90 = FastICA(n_components=n_components_90, algorithm='parallel', whiten=True)
    # Get the various independent components
    S_ica_95_train = pd.DataFrame(ica_95.fit_transform(to_DR_train))
    S_ica_95_test = pd.DataFrame(ica_95.transform(to_DR_test))
    S_ica_90_train = pd.DataFrame(ica_90.fit_transform(to_DR_train))
    S_ica_90_test = pd.DataFrame(ica_90.transform(to_DR_test))

    #write to csv files
    train_file_name_95 = "data/train test split data/ICA (95%)/" + re.match(".+(?=_(train)|_(test)\.csv)", train_name).group() + "_ICA_95_train.csv"
    test_file_name_95 = "data/train test split data/ICA (95%)/" +re.match(".+(?=_(train)|_(test)\.csv)", test_name).group() + "_ICA_95_test.csv"
    train_file_name_90 = "data/train test split data/ICA (90%)/" +re.match(".+(?=_(train)|_(test)\.csv)", train_name).group() + "_ICA_90_train.csv"
    test_file_name_90 = "data/train test split data/ICA (90%)/" +re.match(".+(?=_(train)|_(test)\.csv)", test_name).group() + "_ICA_90_test.csv"

    S_ica_95_train.to_csv(train_file_name_95, index=False)
    S_ica_95_test.to_csv(test_file_name_95, index=False)
    S_ica_90_train.to_csv(train_file_name_90, index=False)
    S_ica_90_test.to_csv(test_file_name_90, index=False)
