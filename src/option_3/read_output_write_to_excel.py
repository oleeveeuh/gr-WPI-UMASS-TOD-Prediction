import sys
import os
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to the sys.path
sys.path.append(parent_dir)
# Now you can import the module
from read_train import *
import pandas as pd
import glob

saved_output_dir = 'data\program_output\*.csv'

# List all CSV files in the folder
csv_files = glob.glob(saved_output_dir)


print(csv_files)
# Load each CSV file into a DataFrame and store them in a list
dataframes = [pd.read_csv(file) for file in csv_files]
windowSize = 0
is_this_flatten = True
for dataframe in dataframes:
    windowSize = windowSize % 3
    if windowSize == 0:
        # flip the is_this_flatten
        is_this_flatten = not is_this_flatten 
    windowSize = windowSize + 1
    if not is_this_flatten:
        write_results_to_excel(dataframe, target_folder=f'performance_sheets_option3/window{windowSize}',verbose=True)
    else:
        write_results_to_excel(dataframe, target_folder=f'performance_sheets_option3_flatten/window{windowSize}',verbose=True)