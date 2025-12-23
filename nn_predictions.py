# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 00:30:53 2024

@author: sambi
"""

import os

import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import root_mean_squared_error as RMSE
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
import keras.callbacks


np.random.seed(23)
warnings.filterwarnings("ignore")


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def model_1():
    model = Sequential()
    model.add(
        Dense(X_train.shape[1], input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(loss=root_mean_squared_error, optimizer="adam")
    model.summary()
    return model


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
    model.compile(loss="mean_squared_error", optimizer=opt)
    return model


def save_pred(predictor, y_pred, df):
    df_pred = df
    df_pred["pred"] = y_pred
    df_pred['pred'] = df_pred.pred.round(2)
    df_pred = df_pred[['player', 'position', 'pred']]
    df_pred = df_pred.sort_values(by=['position', 'pred'], ascending=False)
    df_pred.to_csv(
        os.path.join(
            "predictions/{}-{}.csv".format(
                predictor, pd.Timestamp.now().strftime("%Y%m%d-%Hh%Mm")
            ),
        ),
        index=False,
    )


df = pd.read_csv('basic_game_data/games/full_data.csv')
df_valid = df[df['season'] == 2024]
df = df[df['season'] < 2024]


def pre_process(df):
    df = df.fillna(0)
    encoder = OrdinalEncoder()
    # Fit and transform the 'color' column
    df['against'] = encoder.fit_transform(df[['against']])
    # drop anything containing dates
    df = df[df.columns.drop(list(df.filter(regex='date')))]
    # drop any other unnecessary columns
    df.drop(['tm', 'player',
            'position'], axis=1, inplace=True)
    return df


df = pre_process(df)
df = df[df.columns.drop(list(df.filter(regex='date')))]
df_valid = pre_process(df_valid)
# Split the data into features and target
X_train = df.drop('score', axis=1)
y_train = df['score']

X_test = df_valid.drop('score', axis=1)
y_test = df_valid['score']


# X_train = MinMaxScaler().fit_transform(X_train)
# X_test = MinMaxScaler().fit_transform(X_test)

callbacks = [keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True),
             keras.callbacks.ModelCheckpoint(
                 filepath='model_weights.h5', save_best_only=True),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                               patience=10, min_lr=0.0001)]
model = KerasRegressor(
    build_fn=model_3,
    epochs=1024,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    verbose=1, callbacks=callbacks
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("<--- Testing Error --->")
print(MAE(y_pred, y_test))
print(RMSE(y_pred, y_test))


pred_data = pd.read_csv('basic_game_data/games/to_predict.csv')
# save names
names = pred_data['player']
pos = pred_data['position']
pred_data = pre_process(pred_data)
predicted = model.predict(pred_data)
# add names back in after preprocessing for readability
pred_data['player'] = names
pred_data['position'] = pos
save_pred("nn", predicted, pred_data)
