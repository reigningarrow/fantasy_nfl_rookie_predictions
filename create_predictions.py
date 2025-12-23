# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:38:38 2024

@author: sambi
"""

from xgboost import XGBRegressor
import pandas as pd
import os
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
import keras.callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def model_3():
    model = Sequential()
    model.add(
        Dense(X_train.shape[1], input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    opt = Adam(learning_rate=0.01)
    model.compile(loss=root_mean_squared_error, optimizer=opt)
    return model


df = pd.read_csv('basic_game_data/games/full_data.csv')
df_valid = df[df['season'] == 2024]
df = df[df['season'] < 2024]


def pre_process(df, position=''):
    df = df.fillna(0)
    encoder = OrdinalEncoder()
    # Fit and transform the 'color' column
    df['against'] = encoder.fit_transform(df[['against']])
    if position in ['WR', 'QB', 'RB', 'TE', 'DEF']:
        df = df[df['position'] == position]
    df = df[df.columns.drop(list(df.filter(regex='date')))]
    df.drop(['tm', 'player',
            'position'], axis=1, inplace=True)
    return df


df = pre_process(df)
df = df[df.columns.drop(list(df.filter(regex='date')))]
df_dtypes = df.dtypes
# Split the data into features and target
X = df.drop('score', axis=1)
y = df['score']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model_xgb = XGBRegressor(objective='reg:squarederror',
                         device="cuda",
                         colsample_bytree=0.2, learning_rate=0.01, max_depth=3,
                         n_estimators=600, subsample=0.8, random_state=42)
model_xgb.fit(X_train, y_train)


model_nn = load_model('model_weights.h5')

pred_data = pd.read_csv('basic_game_data/games/to_predict.csv')
# save names
names = pred_data['player']
pos = pred_data['position']
pred_data = pre_process(pred_data)


predicted_nn = model_nn.predict(pred_data)
predicted_xgb = model_xgb.predict(pred_data)
# add names back in after preprocessing for readability
pred_data['player'] = names
pred_data['position'] = pos


df_pred = pred_data
df_pred["nn_predictions"] = predicted_nn
df_pred['nn_predictions'] = df_pred.nn_predictions.round(2)
df_pred["xgb_predictions"] = predicted_xgb
df_pred['xgb_predictions'] = df_pred.xgb_predictions.round(2)
df_pred['avg_predictions'] = df_pred[['xgb_predictions',
                                     'nn_predictions']].mean(axis=1).round(2)
df_pred = df_pred[['player', 'position', 'nn_predictions',
                   'xgb_predictions', 'avg_predictions']]
df_pred = df_pred.sort_values(by=['position', 'avg_predictions'],
                              ascending=False)
df_pred.reset_index(inplace=True)
df_pred.drop('index', axis=1, inplace=True)

df_pred.to_csv(
    f'predictions-2024-week-{df_valid["week"].max()+1}.csv', index=False)
