import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

train_path = (r"/Users/olivialiau/Downloads/KPCADATA90/KPCA_BA11_60_MM_train.csv")
df = pd.read_csv(train_path)
X = df.drop(columns = ['TOD'])
y = df['TOD'].values

dt = DecisionTreeRegressor(random_state=42, criterion = 'squared_error')
kf = KFold(n_splits=5)

param_grid = {
    'max_depth': np.arange(1, 21), 
    'min_samples_split': np.arange(2, 11), 
    'min_samples_leaf': np.arange(1, 11),
    'ccp_alpha': np.arange(0, 2),
    'max_features': ['sqrt', 'log2'],
    'min_impurity_decrease': [0, 1, 2],
    'max_leaf_nodes': np.arange(5, 21)
}

best_params = None
best_mse = float('inf')

random_search = GridSearchCV(estimator=dt, param_grid= param_grid,
                                  scoring='neg_mean_squared_error',
                                   cv=kf, n_jobs=-1)

random_search.fit(X, y)

best_dt = random_search.best_estimator_
best_params = random_search.best_params_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_pred = best_dt.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Best Parameters:", best_params)
print("Mean Squared Error:", mse)

