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

# Load each CSV file into a DataFrame and store them in a list
dataframes = [pd.read_csv(file) for file in csv_files]
windowSize = 0
for dataframe in dataframes:
    windowSize = (windowSize) % 3 + 1
    write_results_to_excel(dataframe, target_folder=f'performance_sheets_option2/window{windowSize}',verbose=True)