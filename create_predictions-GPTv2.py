# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:04:31 2025

@author: sambi
"""

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import random
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int = 42):
    """
    Set global random seed for reproducibility across Python, NumPy, and XGBoost.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # scikit-learn and XGBoost respect random_state arguments separately
    print(f"Global seed set to {seed}")


# ----------------------------
# Define the Neural Network and Training Logic
# ----------------------------
class NNRegressor:
    """
    A regression model built with PyTorch.
    It predicts continuous values (not categories).
    """

    def __init__(self, input_dim, learning_rate=0.05, patience=100):
        """
        Initialize the model and training setup.

        Args:
            input_dim (int): Number of input features (sequence length).
            learning_rate (float): How fast the optimizer updates weights.
            patience (int): How many epochs to wait before stopping
                            if validation loss does not improve.
        """
        # Build the actual neural network
        self.model = self._build_model(input_dim)

        # Loss function: Mean Squared Error (MSE)
        # We later take sqrt to get RMSE (Root Mean Squared Error)
        self.criterion = nn.MSELoss()

        # Optimizer: Adam (adaptive learning rate, usually better than plain SGD)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Scheduler: reduces learning rate if validation loss plateaus
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.25, patience=5
        )

        # Early stopping patience
        self.patience = patience
        self.train_rmse_history = []
        self.val_rmse_history = []

    def _build_model(self, input_dim):
        """
        Build the neural network architecture.

        - Convolutional layers extract local patterns from sequences.
        - Fully connected layers combine features for regression.
        - BatchNorm stabilizes training.
        - Dropout prevents overfitting.
        - LeakyReLU avoids "dying ReLU" problem.
        Build the neural network model.
        current best-
        model = nn.Sequential(nn.Linear(input_dim, 256),
                              nn.BatchNorm1d(256),
                              nn.LeakyReLU(),
                              nn.Dropout(0.1),
                              nn.Linear(256, 512),
                              nn.BatchNorm1d(512),
                              nn.LeakyReLU(),
                              nn.Dropout(0.1),
                              nn.Linear(512, 256),
                              nn.BatchNorm1d(256),
                              nn.LeakyReLU(),
                              nn.Dropout(0.1),

                              nn.Linear(256, 128),
                              nn.BatchNorm1d(128),
                              nn.LeakyReLU(), nn.Dropout(0.1),
                              nn.Linear(128, 64),
                              nn.BatchNorm1d(64),
                              nn.LeakyReLU(),
                              nn.Dropout(0.1),
                              nn.Linear(64, 32),
                              nn.LeakyReLU(),
                              nn.Linear(32, 1)) # Regression output layer """

        model = nn.Sequential(
            # First convolution: detect local patterns in input
            nn.Conv1d(in_channels=1, out_channels=52, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsample

            # Second convolution: refine features
            nn.Conv1d(in_channels=52, out_channels=16, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Flatten(),  # Flatten to feed into fully connected layers

            # Fully connected layers (dense layers)
            nn.Linear(16 * 39, 128),   # 39 = sequence length after conv+pool
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.01),

            nn.Linear(32, 1)  # Final regression output (single value)
        )
        
        '''
        model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)  # regression output
        )
        '''

        # Weight initialization: helps training converge faster
        for m in model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

        return model.to(device)

    def train(self, train_loader, val_loader, epochs=5120):
        """
        Train the model with early stopping and learning rate scheduling.

        Args:
            train_loader (DataLoader): Training dataset loader.
            val_loader (DataLoader): Validation dataset loader.
            epochs (int): Maximum number of training epochs.
        """
        best_val_loss = float("inf")  # Track best validation loss
        epochs_no_improve = 0         # Counter for early stopping
        early_stop = False
        best_model_state = None       # Save best weights

        for epoch in range(epochs):
            self.model.train()        # Put model in training mode
            epoch_loss = 0.0

            # Loop over mini-batches
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                self.optimizer.zero_grad()   # Reset gradients
                # Add channel dimension: [batch, 1, input_dim]
                xb = xb.unsqueeze(1)
                preds = self.model(xb)       # Forward pass

                # Loss = RMSE
                loss = torch.sqrt(self.criterion(preds, yb))
                loss.backward()              # Backpropagation
                self.optimizer.step()        # Update weights

                epoch_loss += loss.item()

            # Validation step
            val_loss, val_rmse, val_mae, val_r2 = self.validate(val_loader)

            # Average training RMSE for this epoch
            train_rmse = epoch_loss / len(train_loader)
            # Append to history lists
            self.train_rmse_history.append(train_rmse)
            self.val_rmse_history.append(val_rmse)

            # Adjust learning rate if validation RMSE plateaus
            self.scheduler.step(val_rmse)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:

                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train RMSE: {train_rmse:.4f} | "
                      f"Val RMSE: {val_rmse:.4f} | "
                      f"Val MAE: {val_mae:.4f} | "
                      f"Val R²: {val_r2:.4f}")

            # Early stopping check
            if train_rmse < best_val_loss:
                best_val_loss = train_rmse
                epochs_no_improve = 0
                best_model_state = self.model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    early_stop = True
                    break

        # Restore best weights if early stopping triggered
        if early_stop and best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def validate(self, val_loader):
        """
        Validate the model and compute metrics.

        Args:
            val_loader (DataLoader): Validation dataset loader.

        Returns:
            tuple: (val_loss, val_rmse, val_mae, val_r2)
        """
        self.model.eval()  # Evaluation mode (no dropout, batchnorm frozen)
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():  # Disable gradient calculation
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                xb = xb.unsqueeze(1)   # [batch, 1, input_dim]
                preds = self.model(xb)

                # RMSE loss
                v_loss = torch.sqrt(self.criterion(preds, yb))
                val_loss += v_loss.item()

                # Save predictions and targets for metrics
                val_preds.append(preds.cpu().numpy())
                val_targets.append(yb.cpu().numpy())

        val_loss /= len(val_loader)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)

        # Compute metrics
        val_rmse = root_mean_squared_error(val_targets, val_preds)
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)

        return val_loss, val_rmse, val_mae, val_r2

    def predict(self, data):
        """
        Make predictions on new input data.

        Args:
            data (pd.DataFrame or np.ndarray): Input features.
                   - If it's a pandas DataFrame, we use .values to get the raw numbers.
                   - If it's already a NumPy array, we use it directly.

        Returns:
            np.ndarray: Predicted values as a flat array.
        """
        self.model.eval()  # Put model in evaluation mode (no dropout, batchnorm frozen)

        with torch.no_grad():  # Disable gradient tracking (faster, uses less memory)
            # Convert input data into a PyTorch tensor
            data_tensor = torch.tensor(
                data.values if hasattr(data, "values") else data,
                dtype=torch.float32
            ).to(device)

            # If the input is 2D (batch_size, features), add a channel dimension
            # because Conv1d expects input of shape [batch, channels, sequence_length]
            if data_tensor.ndim == 2:
                data_tensor = data_tensor.unsqueeze(1)

            # Forward pass through the model
            predictions = self.model(data_tensor)

            # Convert predictions back to NumPy and flatten to 1D array
            predictions = predictions.cpu().numpy().flatten()

        return predictions

    def __repr__(self):
        """
        Return a string representation of the model architecture.
        This is useful if you just print the NNRegressor object.
        """
        return repr(self.model)

# ----------------------------
# Data Preprocessing
# ----------------------------


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


def evaluate_model(model, X_test, y_test, X_valid, y_valid, n_repeats=30, random_state=42):
    """
    Evaluate model on test and validation sets.
    Prints MAE and RMSE, and plots permutation importances, residuals, and training curves.
    Works with both PyTorch and sklearn/XGBoost models.
    """
    model_name = model.__class__.__name__

    # Check if model is a PyTorch model
    if hasattr(model, 'predict'):
        # For sklearn/XGBoost models
        y_pred = model.predict(X_test)
    else:
        # For PyTorch model
        y_pred = model.predict(torch.tensor(
            X_test.values, dtype=torch.float32).to(device))

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, r²: {r2:.4f}")

    # Validation set predictions
    if hasattr(model, 'predict'):
        y_pred_valid = model.predict(X_valid)
    else:
        y_pred_valid = model.predict(torch.tensor(
            X_valid.values, dtype=torch.float32).to(device))

    mae_val = mean_absolute_error(y_valid, y_pred_valid)
    rmse_val = root_mean_squared_error(y_valid, y_pred_valid)
    r2_val = r2_score(y_valid, y_pred_valid)

    print(f"Validation MAE: {mae_val:.4f}, RMSE: {
          rmse_val:.4f}, r²: {r2_val:.4f}")

    # Permutation importance (only for sklearn-compatible models)
    if hasattr(model, 'predict') and hasattr(model, 'fit'):
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
    plt.scatter(y_pred, residuals, alpha=0.7, s=5, label='Test Set Residuals')
    plt.scatter(y_pred_valid, residuals_valid, alpha=0.7,
                s=5, label='Validation Set Residuals')

    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title("Residual Plot: Residuals vs. Predicted Values")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Training curve visualisation (only if history attributes exist)
    if hasattr(model, 'train_rmse_history') and hasattr(model, 'val_rmse_history'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.train_rmse_history, label='Train RMSE', color='blue')
        plt.plot(model.val_rmse_history,
                 label='Validation RMSE', color='orange')
        plt.title("Training Curve: RMSE over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # --- Global seed ---
    seed = 42
    set_global_seed(seed)

    position = ''

    # --- Load data ---
    df = pd.read_csv('basic_game_data/games/full_data.csv')

    if position != '':
        df = df[['position'] == position]
    else:
        df = df[df['position'].isin(['QB', 'WR', 'RB', 'TE'])]
        df = pd.get_dummies(df, columns=['position'], drop_first=False)

    seasons = df['season'].sort_values().unique()
    val_season = seasons[-2]
    df_valid = df[df['season'] == val_season]
    df_train = df[df['season'] < val_season]

    # --- Preprocess ---
    # df_train = pre_process(df_train)

    X = df_train.drop('score', axis=1)
    y = df_train['score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=seed)
    X_train = pre_process(X_train)
    X_test = pre_process(X_test)

    # Convert data to tensors and create DataLoader for validation set
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(
        y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Convert data to tensors and create DataLoader for training set
    X_train_tensor = torch.tensor(
        X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(
        y_train.values, dtype=torch.float32).view(-1, 1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize and train the neural network regressor
    nn_regressor = NNRegressor(input_dim=X_train.shape[1])
    nn_regressor.train(train_loader, val_loader, epochs=5120)

    # Final evaluation on the validation set
    val_loss, val_rmse, val_mae, val_r2 = nn_regressor.validate(val_loader)

    print("\n===== Final Model Evaluation =====")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Validation MAE : {val_mae:.4f}")
    print(f"Validation R²  : {val_r2:.4f}")

    X_valid = df_valid.drop('score', axis=1)
    X_valid = pre_process(X_valid)
    y_valid = df_valid['score']
    evaluate_model(nn_regressor, X_test, y_test, X_valid, y_valid)
    print('===================================')
    print(repr(nn_regressor))
    '''
    # Load prediction data
    pred_data = pd.read_csv('basic_game_data/games/to_predict.csv')
    names = pred_data['player']
    pos = pred_data['position']
    pred_data_processed = pre_process(pred_data)

    # Make predictions with the trained model
    predicted_nn = nn_regressor.predict(pred_data_processed)

    # Combine results into final DataFrame
    pred_data['player'] = names
    pred_data['position'] = pos
    pred_data["nn_predictions"] = np.round(predicted_nn, 2)

    # Select relevant columns for final output
    df_pred = pred_data[['player', 'position', 'nn_predictions']]
    print(df_pred)
    '''
