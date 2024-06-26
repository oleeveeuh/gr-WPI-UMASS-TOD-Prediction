#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:27:57 2024

@author: tillieslosser
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# load the dataset
dataframe = pd.read_csv('/Users/tillieslosser/Downloads/Academic/Research/WPI/data/final_data.csv', index_col=0, engine='python')
data = pd.concat([dataframe['TOD'], dataframe.loc[:, 'PER3':'SMARCA1']], axis = 1)
"""
dataset = data.values
print(dataset)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset[:, 1:] = scaler.fit_transform(dataset[:, 1:])

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

"""

#_____ start GPT

scaler = MinMaxScaler()
data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])


def create_sequences(df, sequence_length):
    X, y = [], []
    feature_columns = df.columns[1:]  # All columns except the first one
    target_column = df.columns[0]  # The first column is the target

    for i in range(len(df) - sequence_length):
        # Extract a sequence of features from index i to i + sequence_length
        X.append(df[feature_columns].iloc[i:i + sequence_length].values)

        # The target value is the time at the end of the sequence
        y.append(df[target_column].iloc[i + sequence_length])

    return np.array(X), np.array(y)

sequence_length = 10  # Example sequence length
X, y = create_sequences(data, sequence_length)

# Split data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
print("\n\n\nX_TRAIN:\n", X_train)
print("\n\n\nX_TEST:\n", X_test)
y_train, y_test = y[:split], y[split:]
print("\n\n\nY_TRAIN:\n", y_train)
print("\n\n\nY_TEST:\n", y_test)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(sequence_length, 235)))
model.add(Dense(1))  # Output layer for predicting time

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Predict
y_pred = model.predict(X_test)

plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Time', color='blue')
plt.plot(y_pred, label='Predicted Time', color='red', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Time')
plt.title('Actual vs Predicted Time')
plt.legend()
plt.show()

# Evaluate
mse = model.evaluate(X_test, y_test)
print(f'Mean Squared Error: {mse}')

print("done")