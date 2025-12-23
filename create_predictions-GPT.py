# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:04:31 2025

@author: sambi
"""


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
    # scikit-learn and XGBoost respect random_state arguments separately
    print(f"Global seed set to {seed}")


# ----------------------------
# Define PyTorch Neural Network
# ----------------------------
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        
        self.model = nn.Sequential(
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
        
        # Optional: weight initialization
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)


# ----------------------------
# Data Preprocessing
# ----------------------------
def pre_process(df: pd.DataFrame, position: str = '') -> pd.DataFrame:
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

    if position != '':
        df = df[['position'] == position]
    else:
        df = df[df['position'].isin(['QB', 'WR', 'RB', 'TE'])]
        df = pd.get_dummies(df, columns=['position'], drop_first=False)

    # Work on a copy of numeric features
    df_num = df.drop(columns=keep_cols, errors="ignore").fillna(0)

    # Identify feature groups
    pct_cols = [c for c in df_num.columns if "percentage" in c or "rate" in c]
    td_cols = [c for c in df_num.columns if "score" in c or "week_" in c or "home" in c]
    other_cols = [c for c in df_num.columns if c not in pct_cols + td_cols or "week" not in c]

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


# ----------------------------
# Load and Split Data
# ----------------------------
df = pd.read_csv('basic_game_data/games/full_data.csv')

seasons_covered = df['season'].sort_values().unique()
validation_season = seasons_covered[-2]
df_valid = df[df['season'] == validation_season]
df = df[df['season'] < validation_season]


dtypes = df.dtypes
df = pre_process(df)
X = df.drop('score', axis=1)
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# ----------------------------
# Validation Split (from your test set or a dedicated validation set)
# ----------------------------
X_val_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# ----------------------------
# Early Stopping Parameters
# ----------------------------
patience = 100          # how many epochs to wait for improvement
best_val_loss = float("inf")
epochs_no_improve = 0
early_stop = False

# ----------------------------
# Train PyTorch NN
# ----------------------------


X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model_nn = Net(input_dim=X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_nn.parameters(), lr=0.01)

# Define scheduler (monitor validation RMSE)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',        # we want to minimize validation loss
    factor=0.5,        # reduce LR by half
    patience=5,        # wait 5 epochs with no improvement
)


epochs = 256*20
model_nn.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model_nn(xb)
        loss = torch.sqrt(criterion(preds, yb))  # RMSE
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # ---- Validation Step ----
    model_nn.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_preds, val_targets = [], []
        for xb, yb in val_loader:
            preds = model_nn(xb)
            v_loss = torch.sqrt(criterion(preds, yb))
            val_loss += v_loss.item()
            val_preds.append(preds.cpu().numpy())
            val_targets.append(yb.cpu().numpy())
        val_loss /= len(val_loader)
        # Compute metrics
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_rmse = root_mean_squared_error(val_targets, val_preds)
        val_mae  = mean_absolute_error(val_targets, val_preds)
        val_r2   = r2_score(val_targets, val_preds)
    model_nn.train()
    
    # ---- Step the scheduler ----
    scheduler.step(val_rmse)

    # ---- Print Progress ----
    if (epoch + 1) % 10 == 0:
        train_rmse = epoch_loss / len(train_loader)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Val R²: {val_r2:.4f}"
        )

    # ---- Early Stopping Check ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model_nn.state_dict()  # save best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            early_stop = True
            break


# Load best model weights (from early stopping)
model_nn.load_state_dict(best_model_state)
model_nn.eval()

with torch.no_grad():
    all_preds, all_targets = [], []
    for xb, yb in val_loader:
        preds = model_nn(xb)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)


# Compute final metrics
final_rmse = root_mean_squared_error(all_targets, all_preds)
final_mae  = mean_absolute_error(all_targets, all_preds)
final_r2   = r2_score(all_targets, all_preds)

print("\n===== Final Model Evaluation =====")
print(f"Validation RMSE: {final_rmse:.4f}")
print(f"Validation MAE : {final_mae:.4f}")
print(f"Validation R²  : {final_r2:.4f}")

# Option 1: Save only weights
torch.save(model_nn.state_dict(), "models/model_weights.pth")


# Load best model weights
if early_stop:
    model_nn.load_state_dict(best_model_state)

# ----------------------------
# Save Model (weights + full model)
# ----------------------------


# Option 2: Save the entire model (architecture + weights)
# torch.save(model_nn, "full_model.pth")


# ----------------------------
# Load Model Later
# ----------------------------
# If you only saved weights:
loaded_model = Net(input_dim=X_train.shape[1]).to(device)
loaded_model.load_state_dict(torch.load("models/model_weights.pth", map_location=device))
loaded_model.eval()

# If you saved the full model:
# loaded_model = torch.load("full_model.pth", map_location=device)
# loaded_model.eval()


# ----------------------------
# Predictions
# ----------------------------
pred_data = pd.read_csv('basic_game_data/games/to_predict.csv')
names = pred_data['player']
pos = pred_data['position']
pred_data = pre_process(pred_data)

# XGBoost predictions
#predicted_xgb = model_xgb.predict(pred_data)

# PyTorch predictions
with torch.no_grad():
    pred_tensor = torch.tensor(pred_data.values, dtype=torch.float32).to(device)
    predicted_nn = loaded_model(pred_tensor).cpu().numpy().flatten()

# ----------------------------
# Combine Results
# ----------------------------
pred_data['player'] = names
pred_data['position'] = pos
pred_data["nn_predictions"] = np.round(predicted_nn, 2)
pred_data["xgb_predictions"] = np.round(predicted_xgb, 2)
pred_data['avg_predictions'] = pred_data[['xgb_predictions', 'nn_predictions']].mean(axis=1).round(2)

df_pred = pred_data[['player', 'position', 'nn_predictions', 'xgb_predictions', 'avg_predictions']]
df_pred = df_pred