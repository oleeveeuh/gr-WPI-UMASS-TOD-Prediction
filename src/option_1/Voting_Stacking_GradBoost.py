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
        # 'Support Vector Regressor': SVR(max_iter=1000),
        # 'Random Forest Regressor': RandomForestRegressor(random_state=42),
        # 'ExtraTreesRegressor': ExtraTreesRegressor(random_state=42),
        # 'Decision Tree Regressor': DecisionTreeRegressor(random_state = 42),
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
        # 'Support Vector Regressor': {
        #     'C': [0.1, 1, 10, 100, 1000],
        #     'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        #     'kernel': ['linear', 'rbf']
        # },
        # 'Random Forest Regressor': {
        #     'n_estimators': [10, 50, 100, 200],
        #     'max_depth': [None, 10, 20, 30],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4]
        # },
        # 'ExtraTreesRegressor': {
        #     'n_estimators': [10, 50, 100, 200],
        #     'max_depth': [None, 10, 20, 30],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4]
        # },
        # 'Decision Tree Regressor': {
        #     'max_depth': [None, 10, 15, 20, 30], 
        #     'min_samples_split':[2, 5, 10], 
        #     'min_samples_leaf': [2, 5, 10],
        #     'max_features': [None, 'sqrt', 'log2'],
        #     'min_impurity_decrease': [0, 1, 2],
        #     'max_leaf_nodes': [None, 5, 10, 15, 20]
        # },
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
        DR_methods=[DR_Method.ICA, DR_Method.KPCA, DR_Method.PCA, DR_Method.Isomap],
        variances=[Variance.V90, Variance.V95]
    )
    
    results_df = train_test_model(models, param_grids, combinations, verbose=True, save_result=True)
    # results_df = pd.read_csv('D:\WPI\DirectedResearch\gr-WPI-UMASS-TOD-Project\data\program_output\model_results_20240625_201447.csv')
    print(results_df)
    # Convert DataFrame to list of dictionaries
    # results_list = results_df.to_dict(orient='records')
    write_results_to_excel(results_df, verbose=True)