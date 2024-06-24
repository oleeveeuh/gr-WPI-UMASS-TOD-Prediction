import numpy as np
from sklearn.decomposition import FastICA
import os
import fnmatch
import re
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score

folder_list = ["data/train_test_split_data/" + f + "/" for f in ["ICA_90", "ICA_95", "KPCA_90", "KPCA_95", "PCA_90", "PCA_95"]]
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
for data_group in list(final_data_grouping.keys())[:1]:
    for dataset in final_data_grouping[data_group].keys():

# ________________________________________________________________________________________________________


# Create an SGDClassifier model
sgd_classifier = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# Train the model
sgd_classifier.fit(X_train, y_train)

# Make predictions
y_pred = sgd_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)