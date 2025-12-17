# -*- coding: utf-8 -*-

"""
Machine Learning Pipeline for NFL Rookie Data Analysis

Steps:
1. Encode categorical variables (Ordinal and One-Hot Encoding).
2. Handle missing values and prepare features/target.
3. Scale features using MinMaxScaler.
4. Apply PCA for dimensionality reduction.
5. Train multiple ML models (XGBoost, RandomForest, SVR) using HalvingGridSearchCV.
6. Evaluate models using multiple metrics and plot diagnostics.

Author: SHB6
Created: 17 Dec 2025
"""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.experimental import enable_halving_search_cv  # noqa Required for HalvingGridSearchCV
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers, callbacks,optimizers

# -----------------------------
# Configuration
# -----------------------------
USE_LOGGING = True
LOG_FILE = 'ml_pipeline.log'

# -----------------------------
# Logging Setup
# -----------------------------
if USE_LOGGING:
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("ML pipeline started.")


def log_and_print(message: str, level: str = "info"):
    """
    Log and/or print messages based on configuration.

    Args:
        message (str): Message to display.
        level (str): Logging level ('info', 'debug', 'error').

    Returns:
        None
    """
    if USE_LOGGING:
        getattr(logging, level)(message)
    print(message)


# -----------------------------
# Data Preprocessing Functions
# -----------------------------


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using Ordinal and One-Hot Encoding.

    - Ordinal encode 'school' column.
    - One-hot encode 'position' column.

    Args:
        df (pd.DataFrame): Input DataFrame with 'school' and 'position' columns.

    Returns:
        pd.DataFrame: DataFrame with encoded features.
    """
    # Ordinal encode the players college
    encoder = OrdinalEncoder()
    df['school'] = encoder.fit_transform(df[['school']])

    # One-hot encode 'position'
    df = pd.get_dummies(df, columns=['position'], prefix='position', dummy_na=False)

    log_and_print("Encoded 'school' and 'position' columns.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values by filling NaNs with zeros.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with NaNs replaced by zeros.
    """
    df.fillna(0, inplace=True)
    log_and_print("Filled missing values with zeros.")
    return df


def split_and_scale(df: pd.DataFrame, target_col: str):
    """
    Split data into training and test sets and apply MinMax scaling.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Name of the target column.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_and_print("Split data and applied MinMax scaling.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def apply_pca(X_train, X_test, variance_threshold=0.95):
    """
    Apply PCA for dimensionality reduction.

    Args:
        X_train (array): Scaled training features.
        X_test (array): Scaled test features.
        variance_threshold (float): Variance to retain (default=0.95).

    Returns:
        tuple: (X_train_pca, X_test_pca, pca_model)
    """
    pca = PCA(variance_threshold)
    pca.fit(X_train)

    log_and_print(f"PCA applied. Number of components retained: {pca.n_components_}")
    return pca.transform(X_train), pca.transform(X_test), pca


# -----------------------------
# Model Training and Evaluation
# -----------------------------

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
    errors = y_pred - y_test

    # Compute metrics
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    log_and_print(f"Diagnostics for {title}: R²={r2:.4f}, RMSE={rmse:.4f}")

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

    plt.tight_layout()


def neural_net(X_train, y_train, X_test, y_test,
               epochs: int = 200, batch_size: int = 32, learning_rate: float = 0.001):
    """
    Build, train, and evaluate a feedforward neural network for regression tasks.

    Architecture:
    - Input layer: Dense with ReLU activation.
    - Hidden layers: Dense layers with ReLU activation and Dropout for regularisation.
    - Output layer: Dense with linear activation for regression.

    Callbacks:
    - EarlyStopping: Stops training when validation loss stops improving.
    - ReduceLROnPlateau: Reduces learning rate when validation loss plateaus.

    Args:
        X_train (array-like): Training feature set.
        y_train (array-like): Training target values.
        X_test (array-like): Test feature set.
        y_test (array-like): Test target values.
        epochs (int): Number of training epochs (default=200).
        batch_size (int): Batch size for training (default=32).
        learning_rate (float): Initial learning rate for Adam optimiser (default=0.001).

    Returns:
        model (Sequential): Trained Keras Sequential model.
    """
    # Define the Neural Network model
    model = keras.Sequential([
        layers.Dense(128, input_dim=X_train.shape[1], activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')  # Output layer for regression
    ])

    # Compile the model
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse', 'mae'])

    # Define callbacks
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                            min_lr=1e-6, verbose=1)

    log_and_print("Training Neural Network with EarlyStopping and ReduceLROnPlateau...")
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    # Evaluate the model
    loss, mse, mae = model.evaluate(X_test, y_test, verbose=0)
    rmse = np.sqrt(mse)
    y_pred = model.predict(X_test).flatten()
    r2 = r2_score(y_test, y_pred)

    # Log and print evaluation metrics
    log_and_print("\n--- Neural Network Evaluation ---")
    log_and_print(f"Test Loss: {loss:.4f}")
    log_and_print(f"Test MSE: {mse:.4f}")
    log_and_print(f"Test MAE: {mae:.4f}")
    log_and_print(f"Test RMSE: {rmse:.4f}")
    log_and_print(f"Test R²: {r2:.4f}")

    return model

def train_and_evaluate_models(X_train, y_train, X_test, y_test,pca=False):
    """
    Train multiple ML models using HalvingGridSearchCV and evaluate them.

    Models:
    - XGBoost
    - RandomForest
    - SVR

    Args:
        X_train (array): Training features.
        y_train (array): Training target.
        X_test (array): Test features.
        y_test (array): Test target.

    Returns:
        None
    """
    models = {
        'XGBoost': {
            'model': XGBRegressor(objective='reg:squarederror', random_state=42),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        },
        'SVR': {
            'model': SVR(),
            'param_grid': {
                'C': [0.1, 1, 10],
                'epsilon': [0.01, 0.1, 0.2],
                'kernel': ['linear', 'rbf']
            }
        }
    }

    scoring_metrics = {
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'rmse': lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),
        'r2': r2_score
    }

    for name, config in models.items():
        log_and_print(f"\n--- Training and evaluating {name} ---")

        search = HalvingGridSearchCV(
            config['model'],
            config['param_grid'],
            cv=5,
            factor=2,
            resource='n_samples',
            min_resources='exhaust',
            scoring='neg_root_mean_squared_error',
            refit=True
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        log_and_print(f"Best parameters for {name}: {search.best_params_}")
        log_and_print(f"Best RMSE score (CV): {search.best_score_}")

        # Evaluate on test set
        log_and_print(f"Test scores for {name}:")
        y_pred = best_model.predict(X_test)

        for metric_name, metric_func in scoring_metrics.items():
            score = metric_func(y_test, y_pred)
            log_and_print(f"  {metric_name}: {score:.4f}")
        if pca is True:
            title= f'Prediction graph for {name} with PCA'
        elif pca is False:
            title= f'Prediction graph for {name} without PCA'
        plot_model_diagnostics(best_model, X_test, y_test, title )


# -----------------------------
# Main Pipeline
# -----------------------------

def main(df_final: pd.DataFrame):
    """
    Main function to execute the ML pipeline.

    Args:
        df_final (pd.DataFrame): Final DataFrame with features and target.

    Returns:
        None
    """
    log_and_print("Starting ML pipeline...")

    # Encode categorical features
    df_final = encode_features(df_final)

    # Handle missing values
    df_final = handle_missing_values(df_final)

    # Save processed data
    df_final.to_csv('nfl_rookie_data.csv', index=False)
    log_and_print("Saved processed data to 'nfl_rookie_data.csv'.")

    # Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_final, target_col='nfl_score')

    # Apply PCA
    X_train_pca, X_test_pca, pca_model = apply_pca(X_train, X_test)

    # Train and evaluate models
    train_and_evaluate_models(X_train_pca, y_train, X_test_pca, y_test,pca=True)
    train_and_evaluate_models(X_train, y_train, X_test, y_test,pca=False)
    
    nn_model_pca = neural_net(X_train_pca, y_train, X_test_pca, y_test)
    nn_model = neural_net(X_train, y_train, X_test, y_test)

    plot_model_diagnostics(nn_model_pca, X_test_pca, y_test, 'Prediction graph for NN with PCA')
    plot_model_diagnostics(nn_model, X_test, y_test, 'Prediction graph for NN with PCA')

# Example usage:
if __name__=='__main__':
    df = pd.read_csv('college_and_nfl_dataset.csv')
    main(df)

