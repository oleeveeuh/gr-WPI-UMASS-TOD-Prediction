
BA11 = "../data/train_test_split_data/BA11"
BA47 = "../data/train_test_split_data/BA47"
full_data = "../data/train_test_split_data/full_data"

folder_list = [BA11, BA47, full_data]
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
            for region in regions:
                for split in splits:
                    if ((region + "_train" not in TOD_dict.keys()) & (region in file) & (split in file) & ("train" in file)):
                        TOD_dict[region + "_" + split + "_train"] = data_dict[file][2]
                    elif ((region + "_test" not in TOD_dict.keys()) & (region in file) & (split in file) & ("test" in file)):
                        TOD_dict[region + "_" + split + "_test"] = data_dict[file][2]

file_names = list(set(file_names))

#check TOD split lengths
for tod in TOD_dict.keys():
    print("Length of ", tod, ": ", len(TOD_dict[tod]))

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


# Parameter grid for KPCA
kernels = ['poly']
gammas = [0.01, 0.1, 1.0, 10.0, 15.0]
degrees = np.arange(1, 5)
n_components = np.arange(2, 20)
# Manual Grid Search (scored by reconstruction error)
best_score = 1.0000
best_params = None
best_model = None
apply_model = None
reconstruct = None
# Assuming final_data_grouping is already populated
for data_group in list(final_data_grouping.keys()):
    for dataset in final_data_grouping[data_group].keys():
        if "train" in dataset:
            to_DR_train = np.delete(final_data_grouping[data_group][dataset], 2, axis=1)
            train_name = dataset
        else:
            to_DR_test = np.delete(final_data_grouping[data_group][dataset], 2, axis=1)
            test_name = dataset

