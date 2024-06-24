import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import pca_script
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    objective='reg:squarederror',
    eta=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    alpha=0.1,
    reg_lambda = 1,
    n_estimators=100,
)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on the test set
preds = xgb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, preds)
print(f"Mean Squared Error: {mse}")