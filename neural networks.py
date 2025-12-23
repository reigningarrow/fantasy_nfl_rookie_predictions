# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 00:11:02 2024

@author: sambi
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint
import keras.callbacks
import utils
from keras.optimizers import Adam

np.random.seed(23)


warnings.filterwarnings("ignore")


def root_mean_squared_error(y_true, y_pred):
    # Custom loss function for keras
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def get_modelcheckpoint_path(model_num):
    # Create a file path for a model and save models in hdf5 files with datetime, validation losses and epochs
    parent = "Modelling/NN/Model_{}/".format(model_num)
    child = (
        pd.Timestamp.now().strftime("%Y%m%d-%Hh%Mm")
        + "-model-epoch_{epoch:02d}-rmse_{val_loss:.5f}.hdf5"
    )
    return parent + child


def get_weights_path_and_epoch(model_num):
    filepaths = glob.glob(
        "Modelling/NN/Model_{}/*.hdf5".format(str(model_num))
    )
    losses = [float(filepath[-12:-5]) for filepath in filepaths]
    epochs = losses.index(min(losses))
    print(
        "Model {} | Lowest Valid Error: {} at Epoch {}".format(
            model_num, min(losses), epochs
        )
    )
    return (filepaths[losses.index(min(losses))], epochs)


def plot_learning_process(hist_list):
    for i, hist in enumerate(hist_list):
        plt.subplot(1, 1, 1)
        plt.plot(hist.history["loss"])
        plt.plot(hist.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper right")
        plt.show()
    return None


def run_models():
    for i in range(1, 5):
        # print(create_model(i)().summary())
        model = KerasRegressor(
            build_fn=create_model(i),
            epochs=1024,
            batch_size=32,
            validation_split=0.2,
            shuffle=True,
            verbose=1,
        )

        filepath = get_modelcheckpoint_path(i)
        callbacks = [keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True),
                     keras.callbacks.ModelCheckpoint(
                         filepath=filepath, save_best_only=True),
                     keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                       patience=10, min_lr=0.000001)]
        callbacks_list = [callbacks]

        if i == 1:
            h1 = model.fit(X, y, callbacks=callbacks_list)
        elif i == 2:
            h2 = model.fit(X, y, callbacks=callbacks_list)
        elif i == 3:
            h3 = model.fit(X, y, callbacks=callbacks_list)
        elif i == 4:
            h4 = model.fit(X, y, callbacks=callbacks_list)

    return [h1, h2, h3, h4]


def create_model(model_num):
    def model_1():
        model = Sequential()
        model.add(Dense(X.shape[1], input_dim=X.shape[1], activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))
        opt = Adam(learning_rate=0.01)
        model.compile(loss="mean_squared_error", optimizer=opt)
        return model

    def model_2():
        model = Sequential()
        model.add(Dense(X.shape[1], input_dim=X.shape[1], activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        opt = Adam(learning_rate=0.01)
        model.compile(loss="mean_squared_error", optimizer=opt)
        return model

    def model_3():
        model = Sequential()
        model.add(Dense(X.shape[1], input_dim=X.shape[1], activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        opt = Adam(learning_rate=0.01)
        model.compile(loss="mean_squared_error", optimizer=opt)
        return model

    def model_4():
        model = Sequential()
        model.add(Dense(X.shape[1], input_dim=X.shape[1], activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        opt = Adam(learning_rate=0.01)
        model.compile(loss="mean_squared_error", optimizer=opt)
        return model
    if model_num == 1:
        return model_1
    elif model_num == 2:
        return model_2
    elif model_num == 3:
        return model_3
    elif model_num == 4:
        return model_4
    else:
        return "invalid model_num"


df = pd.read_csv('basic_game_data/games/full_data.csv')
df_valid = df[df['season'] == 2024]
df = df[df['season'] < 2024]


def pre_process(df):
    df = df.fillna(0)
    encoder = OrdinalEncoder()
    # Fit and transform the 'color' column
    df['against'] = encoder.fit_transform(df[['against']])
    # df = df[df['position'] == position]
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


# Takes about ~ 30 mins for training 100 epochs for all models
hist_list = run_models()

# The models seem to overfit at an early stage for all models
plot_learning_process(hist_list)

for i in range(1, 5):
    weights_path, epochs = get_weights_path_and_epoch(i)
    model = KerasRegressor(
        build_fn=create_model(i),
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        shuffle=True,
        verbose=0,
    )

    errors = utils.cross_val(model, X, y, n_folds=5)
    utils.summarize_errors(errors, verbose=1)
