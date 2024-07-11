import numpy as np
from sklearn.decomposition import FastICA
import pandas as pd

# For Option 1:____________________________________________________________
regions = ["BA11", "BA47", "full"]
splits = ["80", "70", "60"]
normals = ["log", "MM"]


for region in regions:
    for split in splits:
        for normal in normals:
            folder = "full_data" if region == "full" else region
            print("Working on: " + "../data/train_test_split_data/"+folder + "/" + region + "_" + split + "_" + normal + "_train.csv")
            print("\tAnd: " + "../data/train_test_split_data/"+folder + "/" + region + "_" + split + "_" + normal + "_test.csv")

            out_train_name_90 = f"{region}_{split}_{normal}_ICA_90_train.csv"
            out_test_name_90 = f"{region}_{split}_{normal}_ICA_90_test.csv"
            out_train_name_95 = f"{region}_{split}_{normal}_ICA_95_train.csv"
            out_test_name_95 = f"{region}_{split}_{normal}_ICA_95_test.csv"

            train_df = pd.read_csv("../data/train_test_split_data/"+folder + "/" + region + "_" + split + "_" + normal + "_train.csv")
            test_df = pd.read_csv("../data/train_test_split_data/"+folder + "/" + region + "_" + split + "_" + normal + "_test.csv")
            # Preserve TOD columns
            train_TOD = train_df['TOD']
            test_TOD = test_df['TOD']
            # Drop TOD from dataframes pre reduction
            train_df = train_df.drop(['TOD'], axis = 1)
            test_df = test_df.drop(['TOD'], axis=1)

            #Fit ICA
            ica_test = FastICA(n_components=100, algorithm='parallel', whiten=True)
            S_ica_test = ica_test.fit_transform(train_df)  # Get the independent components from training data
            # Determine the number of components to use using the explained variance criterion
            explained_variance = np.var(S_ica_test, axis=0)
            explained_variance_ratio = explained_variance / np.sum(explained_variance)
            n_components_95 = np.argmax(np.cumsum(explained_variance_ratio) >= 0.95) + 1
            n_components_90 = np.argmax(np.cumsum(explained_variance_ratio) >= 0.90) + 1

            # take the number of components for explaining 95, 90 % reconstruction variance
            ica_95 = FastICA(n_components=n_components_95, algorithm='parallel', whiten=True)
            ica_90 = FastICA(n_components=n_components_90, algorithm='parallel', whiten=True)
            # Get the various independent components
            S_ica_95_train = pd.DataFrame(ica_95.fit_transform(train_df))
            S_ica_95_test = pd.DataFrame(ica_95.transform(test_df))
            S_ica_90_train = pd.DataFrame(ica_90.fit_transform(train_df))
            S_ica_90_test = pd.DataFrame(ica_90.transform(test_df))

            # Append TOD back on
            S_ica_95_train['TOD'] = train_TOD
            S_ica_95_test['TOD'] = test_TOD
            S_ica_90_train['TOD'] = train_TOD
            S_ica_90_test['TOD'] = test_TOD

            # write to csv files - MAKE SURE TO CHANGE DESTINATION IF NECESSARY
            S_ica_95_train.to_csv("../data/reduced_data/"+out_train_name_90, index=False)
            S_ica_95_test.to_csv("../data/reduced_data/"+out_test_name_90, index=False)
            S_ica_90_train.to_csv("../data/reduced_data/"+out_train_name_95, index=False)
            S_ica_90_test.to_csv("../data/reduced_data/"+out_test_name_95, index=False)



"""
# For Option 2:____________________________________________________________

regions = ["BA11", "BA47", "full"]
splits = ["80", "70", "60"]
normals = ["log", "MM"]

for region in regions:
    for split in splits:
        for normal in normals:
            out_train_name_90 = f"{region}_{split}_{normal}_ICA_90_train.csv"
            out_test_name_90 = f"{region}_{split}_{normal}_ICA_90_test.csv"
            out_train_name_95 = f"{region}_{split}_{normal}_ICA_95_train.csv"
            out_test_name_95 = f"{region}_{split}_{normal}_ICA_95_test.csv"

            train_df = pd.read_csv("../data/encoded/" + region + "_" + split + "_" + normal + "_train.csv")
            test_df = pd.read_csv("../data/encoded/" + region + "_" + split + "_" + normal + "_test.csv")
            # Preserve TOD columns
            train_TOD = train_df['TOD']
            test_TOD = test_df['TOD']
            # Drop TOD from dataframes pre reduction
            train_df = train_df.drop(['TOD'], axis = 1)
            test_df = test_df.drop(['TOD'], axis=1)

            #Fit ICA
            ica_test = FastICA(n_components=100, algorithm='parallel', whiten=True)
            S_ica_test = ica_test.fit_transform(train_df)  # Get the independent components from training data
            # Determine the number of components to use using the explained variance criterion
            explained_variance = np.var(S_ica_test, axis=0)
            explained_variance_ratio = explained_variance / np.sum(explained_variance)
            n_components_95 = np.argmax(np.cumsum(explained_variance_ratio) >= 0.95) + 1
            n_components_90 = np.argmax(np.cumsum(explained_variance_ratio) >= 0.90) + 1

            # take the number of components for explaining 95, 90 % reconstruction variance
            ica_95 = FastICA(n_components=n_components_95, algorithm='parallel', whiten=True)
            ica_90 = FastICA(n_components=n_components_90, algorithm='parallel', whiten=True)
            # Get the various independent components
            S_ica_95_train = pd.DataFrame(ica_95.fit_transform(train_df))
            S_ica_95_test = pd.DataFrame(ica_95.transform(test_df))
            S_ica_90_train = pd.DataFrame(ica_90.fit_transform(train_df))
            S_ica_90_test = pd.DataFrame(ica_90.transform(test_df))

            # Append TOD back on
            S_ica_95_train['TOD'] = train_TOD
            S_ica_95_test['TOD'] = test_TOD
            S_ica_90_train['TOD'] = train_TOD
            S_ica_90_test['TOD'] = test_TOD

            # write to csv files - MAKE SURE TO CHANGE DESTINATION IF NECESSARY
            S_ica_95_train.to_csv("../data/reduced_encoded/"+out_train_name_90, index=False)
            S_ica_95_test.to_csv("../data/reduced_encoded/"+out_test_name_90, index=False)
            S_ica_90_train.to_csv("../data/reduced_encoded/"+out_train_name_95, index=False)
            S_ica_90_test.to_csv("../data/reduced_encoded/"+out_test_name_95, index=False)


"""
"""
# For Option 3:____________________________________________________________
folders = ["w1_conv_dense/", "w2_conv_dense/", "w3_conv_dense/"]
regions = ["BA11", "BA47"]
splits = ["80", "70", "60"]
normals = ["log", "MM"]

for folder in folders:
    for region in regions:
        for split in splits:
            for normal in normals:
                window_string = {"w1" in folder: "window1", "w2" in folder: "window2", "w3" in folder: "window3"}.get(
                    True, "ERROR")
                out_train_name_90 = f"{region}_{split}_{normal}_{window_string}_ICA_90_train.csv"
                out_test_name_90 = f"{region}_{split}_{normal}_{window_string}_ICA_90_test.csv"
                out_train_name_95 = f"{region}_{split}_{normal}_{window_string}_ICA_95_train.csv"
                out_test_name_95 = f"{region}_{split}_{normal}_{window_string}_ICA_95_test.csv"

                print("Working on: " + "../data/"+folder + region + "_" + split + "_" + normal + "_train.csv")
                print("\tAnd: " + "../data/" + folder + region + "_" + split + "_" + normal + "_test.csv")
                train_df = pd.read_csv("../data/"+folder + region + "_" + split + "_" + normal + "_train.csv")
                test_df = pd.read_csv("../data/"+folder + region + "_" + split + "_" + normal + "_test.csv")
                # Preserve TOD columns
                train_TOD = train_df['TOD']
                test_TOD = test_df['TOD']
                # Drop TOD from dataframes pre reduction
                train_df = train_df.drop(['TOD'], axis = 1)
                test_df = test_df.drop(['TOD'], axis=1)

                #Fit ICA
                ica_test = FastICA(n_components=100, algorithm='parallel', whiten=True)
                S_ica_test = ica_test.fit_transform(train_df)  # Get the independent components from training data
                # Determine the number of components to use using the explained variance criterion
                explained_variance = np.var(S_ica_test, axis=0)
                explained_variance_ratio = explained_variance / np.sum(explained_variance)
                n_components_95 = np.argmax(np.cumsum(explained_variance_ratio) >= 0.95) + 1
                n_components_90 = np.argmax(np.cumsum(explained_variance_ratio) >= 0.90) + 1

                # take the number of components for explaining 95, 90 % reconstruction variance
                ica_95 = FastICA(n_components=n_components_95, algorithm='parallel', whiten=True)
                ica_90 = FastICA(n_components=n_components_90, algorithm='parallel', whiten=True)
                # Get the various independent components
                S_ica_95_train = pd.DataFrame(ica_95.fit_transform(train_df))
                S_ica_95_test = pd.DataFrame(ica_95.transform(test_df))
                S_ica_90_train = pd.DataFrame(ica_90.fit_transform(train_df))
                S_ica_90_test = pd.DataFrame(ica_90.transform(test_df))

                # Append TOD back on
                S_ica_95_train['TOD'] = train_TOD
                S_ica_95_test['TOD'] = test_TOD
                S_ica_90_train['TOD'] = train_TOD
                S_ica_90_test['TOD'] = test_TOD

                # write to csv files - MAKE SURE TO CHANGE DESTINATION IF NECESSARY
                S_ica_95_train.to_csv("../data/reduced_CNN/"+out_train_name_90, index=False)
                S_ica_95_test.to_csv("../data/reduced_CNN/"+out_test_name_90, index=False)
                S_ica_90_train.to_csv("../data/reduced_CNN/"+out_train_name_95, index=False)
                S_ica_90_test.to_csv("../data/reduced_CNN/"+out_test_name_95, index=False)
"""