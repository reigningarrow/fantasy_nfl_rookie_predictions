# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 21:15:10 2025

@author: sambi
"""


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor, XGBClassifier
import scipy.stats as st
import cupy as cp

df = pd.read_csv(
    "C:\\Users\\sambi\\Documents\\Python Scripts\\nfl_rookie_data.csv", index_col=0)

X = df.drop('nfl_score', axis=1)
y = df['nfl_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Normalize the data
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_no_pca = X_train
X_test_no_pca = X_test

print('running classifier')
# Define high scorer threshold (e.g., top 25%)
cutoff = np.quantile(y_train, 0.75)
high_scorer = (y_train >= cutoff).astype(int)

# XGBoost base model
xgb_classifier = XGBClassifier(colsample_bytree=1.0,
                               learning_rate=0.07, max_depth=8,
                               n_estimators=100, subsample=0.6, random_state=42)


# Best XGBoost parameters: {'colsample_bytree': 1.0, 'learning_rate': 0.07, 'max_depth': 8, 'n_estimators': 100, 'subsample': 0.6}
xgb_classifier.fit(X_train_no_pca, high_scorer)

# Get predictions to feed into neural net
xgb_train_preds = xgb_classifier.predict(X_train_no_pca).reshape(-1, 1)
xgb_test_preds = xgb_classifier.predict(X_test_no_pca).reshape(-1, 1)

print('---------------------------------------')
print('Now estimating the score using NN:\n\n')
# Feature stacking
X_nn_train = np.hstack([X_train_no_pca, xgb_train_preds])
X_nn_test = np.hstack([X_test_no_pca, xgb_test_preds])

X_nn_train = cp.asarray(X_nn_train)

# Create HalvingGridSearchCV object
xgb_reg = XGBRegressor(tree_method="hist", device="cuda", random_state=42)
'''
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 9, 11, 15],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.5, 0.7, 0.9, 1.0],
    'colsample_bytree': [0.3, 0.5, 0.7, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5, 1],
    'reg_alpha': [0, 0.1, 0.5, 1, 10],
    'reg_lambda': [0, 0.1, 0.5, 1, 10],
    'scale_pos_weight': [1, 5, 10, 20],
    'objective': ['reg:squarederror'],
    'eval_metric': ['rmse', 'mae']
}

search = HalvingGridSearchCV(xgb_reg,
                             param_grid=param_grid,
                             cv=5,
                             factor=3,
                             resource='n_samples',
                             min_resources='exhaust',
                             scoring='neg_root_mean_squared_error',
                             random_state=42,
                             verbose=2
                             )

# Fit the model
search.fit(X_nn_train, y_train)


# ------------------------------------------------------------------
# 2. Parameter *distributions*  (lists OR scipy.stats objects)
#    – you can keep the same discrete choices …
#    – … or get fancier and replace some lists with probability
#      distributions to explore a wider space.
# ------------------------------------------------------------------
param_distributions = {
    # uniform integers 50-599
    "n_estimators":      st.randint(50, 600),
    # 0.001–0.4 on log scale
    "learning_rate":     st.loguniform(1e-3, 4e-1),
    "max_depth":         st.randint(3, 16),                # 3-15 inclusive
    "min_child_weight":  st.randint(1, 8),                 # 1-7
    "subsample":         st.uniform(0.5, 0.5),             # 0.5-1.0
    "colsample_bytree":  st.uniform(0.3, 0.7),             # 0.3-1.0
    "gamma":             st.uniform(0, 1),                 # 0-1
    "reg_alpha":         st.loguniform(1e-3, 1e1),         # 0.001-10
    "reg_lambda":        st.loguniform(1e-3, 1e1),         # 0.001-10
    "scale_pos_weight":  st.randint(1, 21),                # 1-20
    "objective":        ["reg:squarederror"],              # fixed
    "eval_metric":      ["rmse", "mae"]                    # categorical
}

# ------------------------------------------------------------------
# 3. HalvingRandomSearchCV (successive-halving + randomness)
#    – n_candidates: initial number of random configs to draw
#    – factor: shrinkage factor per round (same as before)
#    – resource: what we halve (here: number of training samples)
# ------------------------------------------------------------------
search = HalvingRandomSearchCV(
    estimator=xgb_reg,
    param_distributions=param_distributions,
    n_candidates=2056,              # starting pool (tweak to taste)
    factor=3,                       # same as your Grid variant
    resource="n_samples",           # successive halving on sample size
    max_resources="auto",           # let it find the full dataset size
    min_resources="exhaust",        # use as much data as possible per round
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    verbose=2,
    n_jobs=50,                      # parallel evaluation if you have the cores
    error_score='raise'             # ← critical for debugging!
)

# ------------------------------------------------------------------
# 4. Fit
# ------------------------------------------------------------------
'''

# Grid around the best parameters
param_grid = {
    'colsample_bytree': [0.73, 0.75, 0.76, 0.77, 0.78],
    'gamma': [0.4, 0.42, 0.43, 0.45],
    'learning_rate': [0.0065, 0.007, 0.0075],
    'max_depth': [2, 3, 4],
    'min_child_weight': [6, 7, 8],
    'n_estimators': [500, 550, 600],
    'reg_alpha': [2.5, 2.7, 2.9],
    'reg_lambda': [0.01, 0.015, 0.02],
    'scale_pos_weight': [5, 6, 7],
    'subsample': [0.94, 0.95, 0.96, 0.97]
}

search = HalvingGridSearchCV(xgb_reg,
                             param_grid=param_grid,
                             cv=2,
                             factor=4,
                             resource='n_samples',
                             min_resources='exhaust',
                             scoring='neg_root_mean_squared_error',
                             random_state=42,
                             verbose=1,
                             n_jobs=40,
                             aggressive_elimination=True)

search.fit(X_nn_train, y_train)


# Get the best parameters
best_params = search.best_params_
print("Best parameters found: ", best_params)

# Get the best estimator
best_estimator = search.best_estimator_
print("Best estimator: ", best_estimator)


# Output the best parameters to a text file
with open('best_params_XGB.txt', 'w') as f:
    f.write(str(best_params))


# Predict on the test set
y_pred = best_estimator.predict(X_nn_test)

print("Predictions for y_test using XGBoost with HalvingGridSearchCV:")


# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Create a DataFrame to display the results
results = pd.DataFrame({
    'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'R-squared (R²)'],
    'Value': [mae, mse, rmse, r2]
})

# Print the results
print(results)

# 2.2 Create a DataFrame to hold true and predicted values
df = pd.DataFrame({
    'y_true':  y_test,
    'y_pred':  y_pred
})

# 2.3 Determine the 75th percentile threshold of the true targets
threshold = np.percentile(df['y_true'], 75)

# 2.4 Filter to only those rows where y_true is in the top 25%
top25 = df[df['y_true'] >= threshold]

# Compute metrics
rmse_top25 = mean_squared_error(
    top25['y_true'], top25['y_pred'], squared=False)
mae_top25 = mean_absolute_error(top25['y_true'], top25['y_pred'])
r2_top25 = r2_score(top25['y_true'], top25['y_pred'])

print(f"Top 25% subset size: {len(top25)} samples")
print(f"RMSE (top25): {rmse_top25:.4f}")
print(f"MAE  (top25): {mae_top25:.4f}")
print(f"R²   (top25): {r2_top25:.4f}")

plt.figure(figsize=(8, 8))
sns.scatterplot(
    x='true_score',
    y='pred_score',
    data=df,
    alpha=0.6,
    edgecolor=None
)

# Draw the 45° reference line (perfect prediction)
min_val = min(df[['true_score', 'pred_score']].min()) - 1
max_val = max(df[['true_score', 'pred_score']].max()) + 1
plt.plot([min_val, max_val],
         [min_val, max_val],
         ls='--',
         c='gray',
         linewidth=1)

# Formatting
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xlabel('True Fantasy Football Score')
plt.ylabel('Predicted Fantasy Football Score')
plt.title('True vs. Predicted Fantasy Football Scores')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
