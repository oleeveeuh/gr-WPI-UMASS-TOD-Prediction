import openpyxl
import pandas as pd
import os
import re
import fnmatch
def find_best_model(path):
    total_df = pd.DataFrame(columns = ['full_df_name', 'model', 'best_parameters', 'MSE', 'MAE', 'MAPE', 'RMSE', 'SMAPE'])
    for sheet in os.listdir(path):
        if not fnmatch.fnmatch(sheet, '~$*'):
            df_name = re.match(".+(?= Model Peformance Results)", sheet).group()
            workbook = openpyxl.load_workbook(path + "/" + sheet)
            for sheetname in workbook.sheetnames:
                sheet = workbook[sheetname]
                train_test_ratio = sheet['B1'].value
                DR_technique = sheet['B2'].value
                normal_method = sheet['B3'].value
                #Variance Level 90
                full_name_90 = df_name + "_" + str(train_test_ratio)+ "_" + str(normal_method)+ "_" + str(DR_technique) + '_90'
                full_name_95 = df_name + "_" + str(train_test_ratio) + "_" + str(normal_method) + "_" + str(DR_technique) + '_95'
                for row in ['7', '8', '9', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '22', '23', '24']:
                    row_list_90 = [full_name_90, sheet['A'+row].value, sheet['C'+row].value, sheet['D'+row].value, sheet['E'+row].value, sheet['F'+row].value, sheet['G'+row].value, sheet['H'+row].value]
                    row_list_95 = [full_name_95, sheet['I' + row].value, sheet['K' + row].value, sheet['L' + row].value,
                                sheet['M' + row].value,
                                sheet['N' + row].value, sheet['O' + row].value, sheet['P' + row].value]
                    # check if row list is empty or contains all NA - skip if true
                    if not all_nans(row_list_90[3:]):
                        total_df.loc[len(total_df) + 1] = row_list_90
                    if not all_nans(row_list_95[3:]):
                        total_df.loc[len(total_df) + 1] = row_list_95
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


def all_nans(lst):
    return all((x is None) for x in lst)

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    performance_dir = os.path.join(script_dir, '..', 'data', 'performance_sheets')
    performance_dir = os.path.normpath(performance_dir)

    total_df = find_best_model(performance_dir)
    print(total_df)
    total_df.to_csv(script_dir+"/all_models_by_performance.csv", )


