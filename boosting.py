# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:31:14 2024

@author: sambi
"""

# %load_ext cudf.pandas
import pandas as pd

# explicitly require this experimental feature

from sklearn.experimental import enable_halving_search_cv  # noqa

# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor
import numpy as np
import os

df = pd.read_csv('basic_game_data/games/full_data.csv')
seasons_covered = df['season'].sort_values().unique()
validation_season = seasons_covered[-2]
df_valid = df[df['season'] == validation_season]
df = df[df['season'] < validation_season]


def pre_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess fantasy football dataset:
    - Drop non-numeric/categorical columns
    - Scale percentage columns with MinMaxScaler
    - Log + Standard scale touchdown-related columns
    - Standard scale remaining continuous features
    """
    # --- Drop irrelevant columns ---
    df = df.drop(columns=['tm', 'player', 'against', 'date', 'season'], errors="ignore")
    df = df.fillna(0)
    df = pd.get_dummies(df, columns=['position'], drop_first=True)
    # --- Identify column groups ---
    pct_cols = [col for col in df.columns if "percentage" in col or "rate" in col]
    td_cols  = [col for col in df.columns if "score" in col or "week_" in col or "home" in col]
    other_cols = [col for col in df.columns if col not in pct_cols + td_cols or "week" not in col]

    # --- Scale percentages (0–1) ---
    if pct_cols:
        mm_scaler = MinMaxScaler()
        df[pct_cols] = mm_scaler.fit_transform(df[pct_cols])

    # --- Scale touchdowns (log + standardize) ---
    if td_cols:
        df[td_cols] = df[td_cols].clip(lower=0) 
        for col in td_cols:
            if (df[col] < 0).any():
                print(df[td_cols].describe())
                print((df[td_cols] < 0).sum())   # count negatives
                raise ValueError(f"Negative values found in {col}, cannot apply log1p safely.")
        df[td_cols] = np.log1p(df[td_cols])  # log1p handles zeros
        td_scaler = StandardScaler()
        df[td_cols] = td_scaler.fit_transform(df[td_cols])

    # --- Scale remaining continuous features ---
    if other_cols:
        cont_scaler = StandardScaler()
        df[other_cols] = cont_scaler.fit_transform(df[other_cols])

    return df



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


df = pre_process(df)
df = df[df.columns.drop(list(df.filter(regex='date')))]
df_dtypes = df.dtypes

# Identify boolean columns
bool_cols = df.select_dtypes(include='bool').columns

# Convert boolean columns to integers (True → 1, False → 0)
df[bool_cols] = df[bool_cols].astype(int)

# Split the data into features and target
X = df.drop('score', axis=1)
y = df['score']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


#X_train_gpu = cp.asarray(X_train.to_numpy())
#y_train_gpu = cp.asarray(y_train.to_numpy())

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900],
    'learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4],
    'max_depth': [1, 2, 3, 5, 7, 9],
    'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
    'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0]
}

# Create a DMatrix for training data on GPU
#dtrain = DMatrix(X_train, label=y_train)

# Create the XGBoost regressor
model = XGBRegressor(objective='reg:squarederror',
                     device="cuda", random_state=42)

# Set up Grid Search
grid_search = HalvingGridSearchCV(estimator=model, param_grid=param_grid,
                                  scoring='neg_mean_squared_error',
                                  cv=3, verbose=3, n_jobs=-1)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Make predictions
y_pred = grid_search.predict(X_test)

print(best_params)
# Calculate Mean Squared Error
mse = MAE(y_test, y_pred)
rmse = RMSE(y_test, y_pred)
print(f'Mean Squared Error: {mse:.6f}')
print(f'Root Mean Squared Error: {rmse:.6f}')
model = XGBRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)


model.save_model('models/xgb_model.json')

print(model.get_booster().get_score(importance_type="gain"))
print(model.get_booster().get_score(importance_type="weight"))


df_valid = pre_process(df_valid)

X_valid = df_valid.drop('score', axis=1)
y_valid = df_valid['score']

y_pred2 = model.predict(X_valid)

mse = MAE(y_valid, y_pred2)
rmse = RMSE(y_valid, y_pred2)
print(f'Mean Squared Error: {mse:.6f}')
print(f'Root Mean Squared Error: {rmse:.6f}')

# pred = df_test['']

feature_important = model.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=[
                    "score"]).sort_values(by="score", ascending=False)
data.nlargest(10, columns="score").plot(
    kind='barh', figsize=(20, 10))  # plot top 40 features


pred_data = pd.read_csv('basic_game_data/games/to_predict.csv')
# save names
names = pred_data['player']
pos = pred_data['position']
pred_data = pre_process(pred_data)
predicted = model.predict(pred_data)
# add names back in after preprocessing for readability
pred_data['player'] = names
pred_data['position'] = pos
#save_pred("xgb", predicted, pred_data)

