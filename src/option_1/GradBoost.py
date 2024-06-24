import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

train_path = (r"/Users/olivialiau/Downloads/KPCADATA90/KPCA_BA11_60_MM_train.csv")
df = pd.read_csv(train_path)
X = df.drop(columns = ['TOD'])
y = df['TOD'].values

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

gb_regressor = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(estimator=gb_regressor, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=kfold, verbose=1, n_jobs=-1)
grid_search.fit(X, y)

print("Best parameters found: ", grid_search.best_params_)
best_model = grid_search.best_estimator_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on test set:", mse)


