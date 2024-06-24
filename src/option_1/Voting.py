import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
random_forest = RandomForestRegressor(random_state=42)
gradient_boosting = GradientBoostingRegressor(random_state=42)
linear_regression = LinearRegression()
voting_regressor = VotingRegressor([
    ('rf', random_forest),
    ('gb', gradient_boosting),
    ('lr', linear_regression)
])
param_grid = {
    'rf__n_estimators': [50, 100],
    'gb__n_estimators': [50, 100],
    # Add other hyperparameters for individual estimators if needed
}
grid_search = GridSearchCV(estimator=voting_regressor, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=kfold, verbose=1, n_jobs=-1)
grid_search.fit(X, y)
print("Best parameters found: ", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Optionally, evaluate the best model on a hold-out test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on test set:", mse)
