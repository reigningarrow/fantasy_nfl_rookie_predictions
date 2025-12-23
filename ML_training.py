# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:17:32 2024

@author: sambi
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import os

os.chdir('basic_game_data')

df = pd.DataFrame()
for file in os.listdir('games//seasons'):
    df_tmp = pd.read_csv('//games//'+file, index_col=False)
    df = pd.concat([df, df_tmp])


y = df['score']
x = df.drop(['score'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


model = XGBRegressor(objective='reg:squarederror', random_state=42)


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=3, verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Best Hyperparameters: {grid_search.best_params_}')
