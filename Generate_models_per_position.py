# -*- coding: utf-8 -*-
"""
Fantasy Football Prediction with XGBoost
----------------------------------------
Pipeline:
1. Data preprocessing (scaling, encoding, log transforms).
2. Train/validation split by season.
3. Two-stage hyperparameter tuning:
   - Stage 1: RandomizedSearchCV (broad exploration).
   - Stage 2: GridSearchCV (refinement around best params).
4. Model evaluation (test + validation).
5. Feature importance visualization.

Author: sambi
"""

import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit, HalvingGridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (mean_absolute_error as MAE,
                             root_mean_squared_error, r2_score)
import joblib
from sklearn.inspection import permutation_importance


def set_global_seed(seed: int = 42):
    """
    Set global random seed for reproducibility across Python, NumPy,
    and XGBoost.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # scikit-learn and XGBoost respect random_state arguments separately
    print(f"Global seed set to {seed}")


# -----------------------------
# Preprocessing
# -----------------------------
def pre_process(df: pd.DataFrame, keep_position=True) -> pd.DataFrame:
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
    if keep_position is False:
        id_cols += ['position']
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


# -----------------------------
# Hyperparameter Search
# -----------------------------
def hyperparameter_search(model, X_train, y_train, param_dist,
                          n_iter=75, cv=3, random_state=42,
                          n_jobs=-1, verbose=2):
    """
    Two-stage hyperparameter tuning:
    1. RandomizedSearchCV for broad exploration
    2. GridSearchCV for fine-tuning around best params

    Parameters
    ----------
    model : estimator
        The base model (e.g., XGBRegressor()).
    X_train, y_train : array-like
        Training data.
    param_dist : dict
        Parameter distributions for RandomizedSearchCV.
    n_iter : int
        Number of random samples for RandomizedSearchCV.
    cv : int
        Number of cross-validation folds.
    random_state : int
        Random seed.
    n_jobs : int
        Number of parallel jobs.
    verbose : int
        Verbosity level.

    Returns
    -------
    dict
        Best parameters after grid search refinement.
    """
    # Example: 5 splits, expanding window
    tscv = TimeSeriesSplit(n_splits=cv)
    # Stage 1: Randomized Search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=tscv,
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=random_state
    )
    print("Running Randomized Search (Stage 1)...")
    random_search.fit(X_train, y_train)
    best_params_stage1 = random_search.best_params_
    print("Best params from Randomized Search:", best_params_stage1)

    # Stage 2: Build refined grid around best params
    param_grid = {}
    for param, value in best_params_stage1.items():
        if isinstance(value, int):
            step = max(1, value // 5)  # integer step
            param_grid[param] = list(
                {max(1, value - step), value, value + step})
        elif isinstance(value, float):
            step = value * 0.2 if value != 0 else 0.1
            param_grid[param] = list(
                {max(1e-6, value - step), value, value + step})
        else:
            # categorical/discrete params
            param_grid[param] = [value]

    grid_search = HalvingGridSearchCV(
        estimator=model.__class__(**best_params_stage1),  # re-init model
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=tscv,
        verbose=verbose,
        n_jobs=n_jobs,
        factor=3
    )
    print("Running Grid Search (Stage 2)...")
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best params from Grid Search:", best_params)

    return best_params


# -----------------------------
# Model Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test, X_valid, y_valid, n_repeats=30,
                   random_state=42,position=''):
    """
    Evaluate model on test and validation sets.
    Prints MAE and RMSE, and plots top permutation importances.
    Works with any sklearn-compatible model.
    """
    model_name = model.__class__.__name__
    # Test set
    y_pred = model.predict(X_test)
    mae = MAE(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)  # RMSE
    r2 = r2_score(y_test, y_pred)
    print(f'{model_name} results {position}')
    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, r2: {r2:.4f}")

    # Validation set
    y_pred_valid = model.predict(X_valid)
    mae_val = MAE(y_valid, y_pred_valid)
    rmse_val = root_mean_squared_error(y_valid, y_pred_valid)
    r2_val = r2_score(y_valid, y_pred_valid)
    print(f"Validation MAE: {mae_val:.4f}, RMSE: {
          rmse_val:.4f}, r2: {r2_val:.4f}")

    # Permutation importance (model-agnostic)
    perm_importance = permutation_importance(
        model, X_valid, y_valid,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    # Put into DataFrame
    importance_df = pd.DataFrame({
        "feature": X_valid.columns,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    # Plot top 10
    importance_df.head(10).plot(
        x="feature", y="importance_mean",
        kind="barh", figsize=(12, 6),
        title=f"Top 10 Permutation Importances {model_name} {position}",
        legend=False
    )
    plt.gca().invert_yaxis()
    plt.show()

    residuals = y_test - y_pred
    residuals_valid = y_valid - y_pred_valid
    # Create the residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, s=5, label='Test Set Residuals')
    plt.scatter(y_pred_valid, residuals_valid, alpha=0.7, s=5,
                label='Validation Set Residuals')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title(f"Residual Plot: Residuals vs. Predicted Values {model_name} {position}")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.legend()
    plt.show()


def save_model(model, position='', directory="models"):
    """
    Save a trained model with a dynamically generated filename.
    - If the model supports `.save_model()`, use it (e.g., XGBoost, LightGBM).
    - Otherwise, fall back to joblib serialization.

    Parameters
    ----------
    model : estimator
        The trained model (e.g., XGBRegressor, RandomForestRegressor).
    position: str,optional
        saves the name of the model if it is filtered for position
    directory : str, optional
        Directory where the model will be saved. Default is "models".

    Returns
    -------
    str
        Path to the saved model file.
    """
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)

    # Get model class name (e.g., "XGBRegressor")
    model_name = model.__class__.__name__

    # Create timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    '''
    # Decide file extension
    if hasattr(model, "save_model"):
        filename = f"{model_name}_{timestamp}.json"
        filepath = os.path.join(directory, filename)
        model.save_model(filepath)
    else:
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(directory, filename)
        joblib.dump(model, filepath)

    # or
    '''

    # Build filename
    filename = f"{model_name}_{position}_{timestamp}.pkl"
    filepath = os.path.join(directory, filename)

    # Save model as pickle
    joblib.dump(model, filepath)

    print(f"Model saved to: {filepath}")
    return filepath


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    # --- Global seed ---
    seed = 42
    set_global_seed(seed)

    # --- Load data ---
    df = pd.read_csv('basic_game_data/games/full_data.csv')
    seasons = df['season'].sort_values().unique()
    val_season = seasons[-2]

    for position in ['QB', 'WR', 'RB', 'TE']:

        if position != '':
            df_pos = df[df['position'] == position]
        else:
            df_pos = df[df['position'].isin(['QB', 'WR', 'RB', 'TE'])]
            df_pos = pd.get_dummies(df, columns=['position'], drop_first=False)

        df_valid = df_pos[df_pos['season'] == val_season]
        df_train = df_pos[df_pos['season'] < val_season]

        # --- Preprocess ---
        X = df_train.drop('score', axis=1)
        y = df_train['score']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=seed)
        X_train = pre_process(X_train, keep_position=False)
        X_test = pre_process(X_test, keep_position=False)
        # --- Time-aware CV ---
        tscv = 3

        # --- Base model ---
        base_model = XGBRegressor(objective='reg:squarederror',
                                  device="cuda", random_state=seed)

        # --- Param distribution ---
        param_dist = {
            'n_estimators': np.arange(100, 1000, 100),
            'learning_rate': np.logspace(-4, 0, 10),
            'max_depth': np.arange(2, 12, 2),
            'subsample': np.linspace(0.5, 1.0, 6),
            'colsample_bytree': np.linspace(0.5, 1.0, 6)
        }

        # --- Hyperparameter tuning ---
        best_params = hyperparameter_search(base_model, X_train, y_train,
                                            param_dist, cv=tscv,
                                            random_state=seed)

        # --- Final model ---
        final_model = XGBRegressor(**best_params, random_state=seed)
        final_model.fit(X_train, y_train)

        # --- Save + Evaluate ---
        save_model(final_model, position=position)
        X_valid = df_valid.drop('score', axis=1)

        X_valid = pre_process(X_valid, keep_position=False)
        y_valid = df_valid['score']

        evaluate_model(final_model, X_test, y_test, X_valid, y_valid,
                       position=position)

        # --- Base model ---
        base_model = XGBRegressor(objective='reg:squarederror',
                                  device="cuda", random_state=seed)

        base_model = RandomForestRegressor(criterion='squared_error',
                                           random_state=seed)

        # --- Param distribution ---
        param_dist = {
            'n_estimators': np.arange(100, 1000, 100),
            'max_depth': np.arange(2, 12, 2),
            'min_samples_split': np.arange(2, 12, 2),
            'min_samples_leaf': np.arange(1, 12, 2),
            'min_weight_fraction_leaf': np.arange(0, 1, 0.1),
            'max_features': np.arange(1, 10, 1),
        }

        # --- Hyperparameter tuning ---
        best_params = hyperparameter_search(base_model, X_train, y_train,
                                            param_dist, cv=tscv,
                                            random_state=seed)

        # --- Final model ---
        final_model = RandomForestRegressor(**best_params, random_state=seed)
        final_model.fit(X_train, y_train)

        # --- Save + Evaluate ---
        save_model(final_model, position=position)
        X_valid = df_valid.drop('score', axis=1)
        X_valid = pre_process(X_valid, keep_position=False)

        y_valid = df_valid['score']
        evaluate_model(final_model, X_test, y_test, X_valid, y_valid,
                       position=position)
