import openpyxl
import pandas as pd
import os
import re
import fnmatch
def find_best_model(path, is_window = False):
    total_df = pd.DataFrame(columns = ['full_df_name', 'model', 'best_parameters', 'MSE', 'MAE', 'MAPE', 'RMSE', 'SMAPE'])
    if is_window:
        for w in range(1,4):
            subfolder_path = f"{path}/window{w}/"
            for sheet in os.listdir(subfolder_path):
                if not fnmatch.fnmatch(sheet, '~$*'):
                    sheet_df = parse_sheet(sheet, sheet_path = f"{subfolder_path}/{sheet}", window = f"window{w}_")
                    total_df = pd.concat([total_df, sheet_df])
    else:
        for sheet in os.listdir(path):
            if not fnmatch.fnmatch(sheet, '~$*'):
                sheet_df = parse_sheet(sheet, sheet_path=f"{path}/{sheet}")
                total_df = pd.concat([total_df, sheet_df])
#Print the best model according to each category: not working yet (index issue somewhere, probably in idxmin, but figure that out later)
    """"# Print best model by each characteristic
    for measure in ['MSE', 'MAE', 'MAPE', 'RMSE', 'SMAPE']:
        col_index = list(total_df.columns.values).index(measure)
        col = total_df[measure]
        index_min = pd.Series(col).idxmin()
        print("For", measure, ":")
        print("\tThe lowest value is ", total_df.iloc[index_min, col_index])
        print("\tFrom the", total_df.iloc[index_min, 1])
        print("\tUsing the", total_df.iloc[index_min, 0], "dataset")"""
    return(total_df)

def parse_sheet(sheet, sheet_path, window =""):
    print(sheet_path)
    sheet_df = pd.DataFrame(columns = ['full_df_name', 'model', 'best_parameters', 'MSE', 'MAE', 'MAPE', 'RMSE', 'SMAPE'])
    df_name = re.match(".*(?= Overall Model Peformance Results)", sheet).group()
    workbook = openpyxl.load_workbook(sheet_path)
    for sheetname in workbook.sheetnames:
        sheet = workbook[sheetname]
        train_test_ratio = sheet['B1'].value
        DR_technique = sheet['B2'].value
        normal_method = sheet['B3'].value
        # Variance Level 90
        full_name_90 = df_name + "_" + str(train_test_ratio) + "_" + str(normal_method) + "_" + window + str(
            DR_technique) + '_90'
        full_name_95 = df_name + "_" + str(train_test_ratio) + "_" + str(normal_method) + "_" + window + str(
            DR_technique) + '_95'
        for row in ['7', '8', '9', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '22', '23', '24']:
            row_list_90 = [full_name_90, sheet['A' + row].value, sheet['C' + row].value, sheet['D' + row].value,
                           sheet['E' + row].value, sheet['F' + row].value, sheet['G' + row].value,
                           sheet['H' + row].value]
            row_list_95 = [full_name_95, sheet['I' + row].value, sheet['K' + row].value, sheet['L' + row].value,
                           sheet['M' + row].value,
                           sheet['N' + row].value, sheet['O' + row].value, sheet['P' + row].value]
            # check if row list is empty or contains all NA - skip if true
            """if all_nans(row_list_90[3:]):
                print(f"{full_name_90} is all NaNs for model {sheet['A' + row].value}")
            if all_nans(row_list_95[3:]):
                print(f"{full_name_95} is all NaNs for model {sheet['I' + row].value}")"""
            if not all_nans(row_list_90[3:]):
                sheet_df.loc[len(sheet_df) + 1] = row_list_90
            if not all_nans(row_list_95[3:]):
                sheet_df.loc[len(sheet_df) + 1] = row_list_95
    print("Sheet", df_name, ":\n", sheet_df)
    return sheet_df

def all_nans(lst):
    return all((x is None) for x in lst)

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    # Make DF for Option 1 Model Results
    opt1_dir = os.path.join(script_dir, '..', 'data', 'performance_sheets')
    opt1_dir = os.path.normpath(opt1_dir)

    opt1_df = find_best_model(opt1_dir)
    print("Option 1:\n", opt1_df)
    opt1_df.to_csv(script_dir+"/opt_1_all_models_performance.csv")

    # Make DF for Option 2 Model Results
    opt2_dir = os.path.join(script_dir, '..', 'data', 'performance_sheets_option2')
    opt2_dir = os.path.normpath(opt2_dir)

    opt2_df = find_best_model(opt2_dir, is_window=True)
    print("Option 2:\n", opt2_df)
    opt2_df.to_csv(script_dir + "/opt_2_all_models_performance.csv")

    # Make DF for Option 2 Model Results
    opt3_dir = os.path.join(script_dir, '..', 'data', 'performance_sheets_option3')
    opt3_dir = os.path.normpath(opt3_dir)

    opt3_df = find_best_model(opt3_dir, is_window=True)
    print("Option 2:\n", opt3_df)
    opt3_df.to_csv(script_dir + "/opt_3_all_models_performance.csv")




