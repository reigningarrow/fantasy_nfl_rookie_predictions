# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 23:30:18 2025

@author: sambi
"""
import os
os.environ["KERAS_BACKEND"] = "torch"

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers, callbacks
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


import random

import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int = 42):
    """
    Set global random seed for reproducibility across Python, NumPy, and XGBoost.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    keras.utils.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # scikit-learn and XGBoost respect random_state arguments separately
    print(f"Global seed set to {seed}")


class NNRegressorKeras:
    """
    A regression model built with Keras (TensorFlow backend).
    """

    def __init__(self, input_dim, learning_rate=0.05, patience=100):
        self.model = self._build_model(input_dim)

        # Compile with optimizer, loss, and metrics
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",   # Mean Squared Error
            metrics=[
                keras.metrics.RootMeanSquaredError(name="rmse"),
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.R2Score(name="r2")  # TF >= 2.11
            ]
        )

        # Callbacks: EarlyStopping + ReduceLROnPlateau
        self.callbacks = [
            callbacks.EarlyStopping(
                patience=patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.25,
                patience=5,
                verbose=1
            )
        ]

    def _build_model(self, input_dim):
        """
        Build the neural network architecture in Keras.
        """
        inputs = keras.Input(shape=(input_dim, 1)
                             )  # [batch, seq_len, channels]

        x = layers.Conv1D(filters=52, kernel_size=5, activation=None)(inputs)
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

        x = layers.Conv1D(filters=16, kernel_size=5, activation=None)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

        x = layers.Flatten()(x)

        x = layers.Dense(128)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Dense(128)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Dense(64)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Dense(32)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.01)(x)

        outputs = layers.Dense(1)(x)  # Regression output

        return keras.Model(inputs, outputs)

    def train(self, X_train, y_train, X_test, y_test, epochs=5120, batch_size=32):
        """
        Train the model with early stopping and learning rate scheduling.
        """
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
            verbose=2
        )
        return history

    def predict(self, data):
        """
        Predict continuous values.
        """
        preds = self.model.predict(data)
        return preds.flatten()


def pre_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess fantasy football dataset and return the full DataFrame:
    - Keep identifier columns (player, position, season, etc.)
    - Scale percentage columns with MinMaxScaler
    - Log + Standard scale touchdown-related columns
    - Standard scale remaining continuous features
    - Return full DataFrame with identifiers + transformed features
    """
    # Keep identifiers separately
    id_cols = ['player', 'season', 'tm', 'against', 'date']
    keep_cols = [c for c in id_cols if c in df.columns]
    # df_id = df[keep_cols].copy()

    # Work on a copy of numeric features
    df_num = df.drop(columns=keep_cols, errors="ignore").fillna(0)

    # Identify feature groups
    pct_cols = [c for c in df_num.columns if "percentage" in c or "rate" in c]
    td_cols = [
        c for c in df_num.columns if "score" in c or "week_" in c or "home" in c]
    other_cols = [
        c for c in df_num.columns if c not in pct_cols + td_cols or "week" not in c]

    # Scale percentages
    if pct_cols:
        df_num[pct_cols] = MinMaxScaler().fit_transform(df_num[pct_cols])

    # Scale touchdowns (log + standardize)
    if td_cols:
        df_num[td_cols] = df_num[td_cols].clip(lower=0)
        df_num[td_cols] = np.log1p(df_num[td_cols])
        df_num[td_cols] = StandardScaler().fit_transform(df_num[td_cols])

    # Scale remaining continuous features
    if other_cols:
        df_num[other_cols] = StandardScaler().fit_transform(df_num[other_cols])

    # Concatenate identifiers + processed numeric features
    df_processed = df_num.reset_index(drop=True)

    return df_processed


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(model, X_test, y_test, X_valid, y_valid,
                   n_repeats=30, random_state=42, history=None):
    """
    Evaluate a Keras model on test and validation sets.
    Prints MAE, RMSE, R², and plots residuals and training curves.
    Optionally computes permutation importance if model is sklearn-wrapped.

    Args:
        model: Trained Keras model (or sklearn-wrapped KerasRegressor).
        X_test, y_test: Test data and labels.
        X_valid, y_valid: Validation data and labels.
        n_repeats: Number of repeats for permutation importance (sklearn models only).
        random_state: Random seed for permutation importance.
        history: Keras History object from model.fit() (optional, for training curves).
    """
    model_name = model.__class__.__name__

    # Predictions
    y_pred = model.predict(X_test).flatten()
    y_pred_valid = model.predict(X_valid).flatten()

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, r²: {r2:.4f}")

    mae_val = mean_absolute_error(y_valid, y_pred_valid)
    rmse_val = root_mean_squared_error(y_valid, y_pred_valid)
    r2_val = r2_score(y_valid, y_pred_valid)
    print(f"Validation MAE: {mae_val:.4f}, RMSE: {
          rmse_val:.4f}, r²: {r2_val:.4f}")

    # Permutation importance (only if sklearn-compatible)
    if hasattr(model, "fit") and hasattr(model, "score"):
        perm_importance = permutation_importance(
            model, X_valid, y_valid,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        importance_df = pd.DataFrame({
            "feature": X_valid.columns,
            "importance_mean": perm_importance.importances_mean,
            "importance_std": perm_importance.importances_std
        }).sort_values(by="importance_mean", ascending=False)

        importance_df.head(10).plot(
            x="feature", y="importance_mean",
            kind="barh", figsize=(12, 6),
            title=f"Top 10 Permutation Importances {model_name}",
            legend=False
        )
        plt.gca().invert_yaxis()
        plt.show()

    # Residual plots
    residuals = y_test - y_pred
    residuals_valid = y_valid - y_pred_valid
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, s=5, label="Test Set Residuals")
    plt.scatter(y_pred_valid, residuals_valid, alpha=0.7,
                s=5, label="Validation Set Residuals")
    plt.axhline(y=0, color="red", linestyle="--", linewidth=2)
    plt.title("Residual Plot: Residuals vs. Predicted Values")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Training curve visualization (if History object provided)
    if history is not None:
        plt.figure(figsize=(10, 6))
        if "rmse" in history.history:
            plt.plot(history.history["rmse"], label="Train RMSE", color="blue")
        if "val_rmse" in history.history:
            plt.plot(history.history["val_rmse"],
                     label="Validation RMSE", color="orange")
        plt.title("Training Curve: RMSE over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ----------------------------
# Main Execution (Keras version)
# ----------------------------
if __name__ == "__main__":
    # --- Global seed ---
    seed = 42
    set_global_seed(seed)

    position = ''

    # --- Load data ---
    df = pd.read_csv('basic_game_data/games/full_data.csv')

    if position != '':
        df = df[df['position'] == position]
    else:
        df = df[df['position'].isin(['QB', 'WR', 'RB', 'TE'])]
        df = pd.get_dummies(df, columns=['position'], drop_first=False)

    # --- Train/Validation split by season ---
    seasons = df['season'].sort_values().unique()
    val_season = seasons[-2]
    df_valid = df[df['season'] == val_season]
    df_train = df[df['season'] < val_season]

    # --- Preprocess ---
    X = df_train.drop('score', axis=1)
    y = df_train['score']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train = pre_process(X_train)
    X_test = pre_process(X_test)

    # Validation set
    X_valid = df_valid.drop('score', axis=1)
    X_valid = pre_process(X_valid)
    y_valid = df_valid['score']
    
    X_train = np.expand_dims(X_train, axis=-1)
    X_test  = np.expand_dims(X_test, axis=-1)
    X_valid = np.expand_dims(X_valid, axis=-1)


    # --- Initialize and train the neural network regressor ---
    nn_regressor = NNRegressorKeras(input_dim=X_train.shape[1])

    # Train the model (Keras fit)
    history = nn_regressor.train(
        X_train, y_train,
        X_valid, y_valid,
        epochs=5120,
       batch_size=32
    )

    # --- Final evaluation on the test set ---
    print("\n===== Final Model Evaluation =====")
    test_metrics = nn_regressor.model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_metrics[0]:.4f}")
    print(f"Test RMSE: {test_metrics[1]:.4f}")
    print(f"Test MAE : {test_metrics[2]:.4f}")
    print(f"Test R²  : {test_metrics[3]:.4f}")

    # --- Validation evaluation ---
    val_metrics = nn_regressor.model.evaluate(X_valid, y_valid, verbose=0)
    print(f"Validation Loss: {val_metrics[0]:.4f}")
    print(f"Validation RMSE: {val_metrics[1]:.4f}")
    print(f"Validation MAE : {val_metrics[2]:.4f}")
    print(f"Validation R²  : {val_metrics[3]:.4f}")

    # --- Evaluate with custom function ---
    evaluate_model(nn_regressor.model, X_test, y_test, X_valid, y_valid,
                   history=history)

    print('===================================')
    print(repr(nn_regressor))

    # --- Prediction example ---
    '''
    pred_data = pd.read_csv('basic_game_data/games/to_predict.csv')
    names = pred_data['player']
    pos = pred_data['position']
    pred_data_processed = pre_process(pred_data)

    predicted_nn = nn_regressor.predict(pred_data_processed)

    pred_data['player'] = names
    pred_data['position'] = pos
    pred_data["nn_predictions"] = np.round(predicted_nn, 2)

    df_pred = pred_data[['player', 'position', 'nn_predictions']]
    print(df_pred)
    '''
