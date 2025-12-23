# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 23:30:18 2025

@author: sambi

non conv-
Test MAE: 4.8440, RMSE: 6.6925, r²: 0.3253
Validation MAE: 4.7498, RMSE: 6.5541, r²: 0.3460
"""

import torch
import os
os.environ["KERAS_BACKEND"] = "torch"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import seaborn as sns
import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import layers, callbacks
from sklearn.model_selection import ParameterGrid
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance






def plot_model_diagnostics(model, X_test, y_test, title: str):
    """
    Plot diagnostic charts for a regression model:
    - True vs Predicted scatter plot with R² and RMSE annotation.
    - Distribution of prediction errors (residuals).

    Args:
        model: Trained regression model with a `.predict()` method.
        X_test (array-like): Test feature set.
        y_test (array-like): True target values.
        title (str): Title for the plot.

    Returns:
        None: Displays the diagnostic plots.
    """
    # Predictions and residuals
    y_pred = model.predict(X_test)
    # If it's (n_samples, 1), flatten it:
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()
    errors = y_pred - y_test

    # Compute metrics
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    # ---- Plot Layout ----
    fig = plt.figure(figsize=(10, 10))
    grid = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)

    # ---------------------------------------------------------
    # 1. TRUE vs PREDICTED SCATTER
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.scatter(y_test, y_pred, s=10, c='blue', alpha=0.7)
    ax1.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2)

    ax1.set_title(f"{title}\nTrue vs Predicted", fontsize=14)
    ax1.set_xlabel("True Values")
    ax1.set_ylabel("Predicted Values")
    ax1.grid(True)

    # Annotate R² and RMSE
    ax1.text(
        0.05, 0.95,
        f"R² = {r2:.4f}\nRMSE = {rmse:.4f}",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.7)
    )

    # ---------------------------------------------------------
    # 2. ERROR DISTRIBUTION (Residuals)
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(grid[1, 0])
    sns.histplot(errors, bins=40, kde=True, ax=ax2, color='purple')

    ax2.set_title("Distribution of Prediction Errors", fontsize=14)
    ax2.set_xlabel("Error (Predicted - True)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True)

    fig.tight_layout()
    plt.show()


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

    def __init__(self, input_dim, learning_rate=0.05, patience=100,
                 conv1_filters=52, conv2_filters=16, dense_units=128, dropout_rate=0.1):
        self.model = self._build_model(input_dim, conv1_filters, conv2_filters,
                                       dense_units, dropout_rate, conv=True)

        # Compile with optimizer, loss, and metrics
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=[
                keras.metrics.RootMeanSquaredError(name="rmse"),
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.R2Score(name="r2")  # TF >= 2.11
            ]
        )

        # Callbacks
        self.callbacks = [
            callbacks.EarlyStopping(
                patience=patience, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.25, patience=5, verbose=1)
        ]

    def _build_model(self, input_dim, conv1_filters, conv2_filters, dense_units, dropout_rate, conv: bool = True):
        if conv is True:
            inputs = keras.Input(shape=(input_dim, 1))

            x = layers.Conv1D(filters=conv1_filters,
                              kernel_size=5, activation=None)(inputs)
            x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

            x = layers.Conv1D(filters=conv2_filters,
                              kernel_size=5, activation=None)(x)
            x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

            x = layers.Flatten()(x)

            # Example: one tunable dense layer
            x = layers.Dense(dense_units)(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(dropout_rate)(x)

            outputs = layers.Dense(1)(x)
            return keras.Model(inputs, outputs)
        elif conv is False:
            # Define the Neural Network model
            model = keras.Sequential([
                layers.Dense(
                    64, input_dim=X_train.shape[1], activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(128, activation='leaky_relu'),
                layers.Dropout(0.2),
                layers.Dense(64, activation='leaky_relu'),
                layers.Dropout(0.2),
                layers.Dense(32, activation='leaky_relu'),
                # Output layer for regression
                layers.Dense(1, activation='linear')
            ])
            return model
        else:
            raise

    def train(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
            verbose=0
        )
        return history

    def predict(self, data):
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
    
    
    plot_model_diagnostics(model, X_test, y_test, f"Residual Plot {model_name}: Residuals vs. Predicted Values")


# ----------------------------
# Main Execution (Keras version)
# ----------------------------
if __name__ == "__main__":
    # --- Global seed ---
    # Set a fixed random seed for reproducibility across NumPy, TensorFlow, etc.
    seed = 42
    set_global_seed(seed)

    # Position filter (empty string means include multiple positions)
    position = ''

    # --- Load data ---
    # Read the full dataset containing game statistics
    df = pd.read_csv('basic_game_data/games/full_data.csv')

    # Filter dataset by position if specified, otherwise include QB/WR/RB/TE
    if position != '':
        df = df[df['position'] == position]
    else:
        df = df[df['position'].isin(['QB', 'WR', 'RB', 'TE'])]
        # One-hot encode the 'position' column
        df = pd.get_dummies(df, columns=['position'], drop_first=False)

    # --- Train/Validation split by season ---
    # Sort seasons and pick the second-to-last season as validation
    seasons = df['season'].sort_values().unique()
    val_season = seasons[-2]
    df_valid = df[df['season'] == val_season]   # validation set
    df_train = df[df['season'] < val_season]    # training set

    # --- Preprocess ---
    # Separate features (X) and target (y)
    X = df_train.drop('score', axis=1)
    y = df_train['score']

    # Split training data into train/test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Apply preprocessing (scaling, normalization, etc.)
    X_train = pre_process(X_train)
    X_test = pre_process(X_test)

    # Prepare validation set
    X_valid = df_valid.drop('score', axis=1)
    X_valid = pre_process(X_valid)
    y_valid = df_valid['score']

    # --- Reshape for Keras CNN input ---
    # Expand dimensions to match expected input shape (samples, features, channels)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    X_valid = np.expand_dims(X_valid, axis=-1)

    # Explicit reshape to (n_samples, n_features, 1)
    X_train = X_train.reshape((-1, X_train.shape[1], 1))
    X_test = X_test.reshape((-1, X_train.shape[1], 1))
    X_valid = X_valid.reshape((-1, X_valid.shape[1], 1))

    # --- Initialize and train the neural network regressor ---
    nn_regressor = NNRegressorKeras(input_dim=X_train.shape[1])

    # --- Random search for hyperparameters ---
    random_results = []
    for _ in range(10):
        # Randomly sample hyperparameters
        params = {
            'conv1_filters': random.choice([32, 52, 64]),
            'conv2_filters': random.choice([8, 16, 32]),
            'dense_units': random.choice([64, 128, 256]),
            'dropout_rate': random.choice([0.1, 0.2, 0.3]),
            'learning_rate': random.choice([1e-2, 1e-3, 1e-4])
        }
        # Train model with sampled parameters
        model = NNRegressorKeras(input_dim=X_train.shape[1], **params)
        history = model.train(X_train, y_train, X_valid, y_valid, epochs=256)
        # Track best validation MAE
        val_mae = min(history.history['val_mae'])
        random_results.append((params, val_mae))

    # Select best parameters from random search
    best_params = min(random_results, key=lambda x: x[1])[0]

    # --- Grid search around best parameters ---
    param_grid = {
        'conv1_filters': [best_params['conv1_filters']-16, best_params['conv1_filters'], best_params['conv1_filters']+16],
        'conv2_filters': [best_params['conv2_filters']],
        'dense_units': [best_params['dense_units']-64, best_params['dense_units'], best_params['dense_units']+64],
        'dropout_rate': [max(0, best_params['dropout_rate']-0.1), best_params['dropout_rate'], min(0.5, best_params['dropout_rate']+0.1)],
        'learning_rate': [best_params['learning_rate']]
    }

    grid_results = []
    for params in ParameterGrid(param_grid):
        # Train model with grid parameters
        model = NNRegressorKeras(input_dim=X_train.shape[1], **params)
        history = model.train(X_train, y_train, X_valid, y_valid, epochs=256)
        val_mae = min(history.history['val_mae'])
        grid_results.append((params, val_mae))

    # Select best parameters from grid search
    best_grid_params = min(grid_results, key=lambda x: x[1])[0]

    # --- Visualization of grid search results ---
    units = [r[0]['dense_units'] for r in grid_results]
    mae = [r[1] for r in grid_results]

    plt.scatter(units, mae, s=5)
    plt.xlabel("Dense Units")
    plt.ylabel("Validation MAE")
    plt.title("Validation Error vs Dense Units")
    plt.show()

    # --- Train final model with best grid parameters ---
    best_model = NNRegressorKeras(input_dim=20, **best_grid_params)
    history = best_model.train(X_train, y_train, X_valid, y_valid, epochs=256)

    # --- Evaluate on validation set ---
    final_results = best_model.model.evaluate(X_valid, y_valid, verbose=0)

    print("\nFinal Evaluation Metrics:")
    print(f"Loss (MSE): {final_results[0]:.4f}")
    print(f"RMSE: {final_results[1]:.4f}")
    print(f"MAE: {final_results[2]:.4f}")
    print(f"R²: {final_results[3]:.4f}")

    # Print model architecture summary
    print("\nBest Model Architecture:")
    best_model.model.summary()

    # --- Evaluate with custom function ---
    evaluate_model(best_model, X_test, y_test, X_valid, y_valid,
                   history=history)

    print('===================================')
    # print(repr(nn_regressor))

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
