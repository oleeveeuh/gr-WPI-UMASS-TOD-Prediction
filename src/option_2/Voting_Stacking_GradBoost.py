import sys
import os
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to the sys.path
sys.path.append(parent_dir)
# Now you can import the module
from read_train import *
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor

if __name__ == "__main__":
    # read_file(Target.BA11, Split.S60, Normalize_Method.Log, DR_Method.ICA, Variance.V90)

    # Define models
    models = {
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
        'Stacking Regressor': StackingRegressor(estimators = [('lr', LinearRegression()),
                                                              ('svr', LinearSVR(random_state=42)),
                                                              ('rf', RandomForestRegressor(random_state=42)),
                                                              ('gb', GradientBoostingRegressor(random_state=42))]),
        'Voting Regressor': VotingRegressor([
            ('rf', RandomForestRegressor(random_state=42)),
            ('gb', GradientBoostingRegressor(random_state=42)),
            ('lr', LinearRegression())]
        )
    }

    # Define parameter grids for RandomizedSearchCV
    param_grids = {

        'Gradient Boosting Regressor': {
            'n_estimators': [10, 50, 100, 200],
            'learning_rate': [.05, .1, .5, 1],
            'max_depth': [None, 10, 15, 20, 30], 
            'min_samples_split':[2, 5, 10], 
            'min_samples_leaf': [2, 5, 10],
            'max_features': [None, 'sqrt', 'log2'],
            'min_impurity_decrease': [0, 1, 2],
            'max_leaf_nodes': [None, 5, 10, 15, 20]
        },
        'Voting Regressor': {
            'rf__n_estimators': [10, 50, 100, 200],
            'gb__n_estimators': [10, 50, 100, 200],
        },
        'Stacking Regressor':{
            'final_estimator': [RandomForestRegressor(random_state=42),
                                LinearRegression(),
                                LinearSVR(random_state=42),
                                RandomForestRegressor(random_state=42),
                                GradientBoostingRegressor(random_state=42)]
        }
    }

    # Specify which datasets to use
    combinations = filter_combinations(
        targets=[Target.BA11, Target.BA47],
        splits=[Split.S60, Split.S70, Split.S80],
        n_methods=[Normalize_Method.Log, Normalize_Method.MM],
        DR_methods=[DR_Method.KPCA],
        variances=[Variance.V90, Variance.V95]
    )
    
    results_df = train_test_model(models, param_grids, combinations, data_read_function=read_reduced_encoded_file,verbose=True, save_result=True)
    print(results_df)
    write_results_to_excel(results_df, target_folder='performance_sheets_option2',verbose=True)