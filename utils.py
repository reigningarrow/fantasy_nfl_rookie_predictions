# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:42:37 2024

@author: sambi
"""

import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def csv_concatenate(folder_path, nested=False, _type=''):
    # Concatenate all csv files under a directory
    if nested is True:
        files = glob.glob(folder_path + "/*/*.csv")
    else:
        files = glob.glob(folder_path + "/*.csv")

    df_list = []
    print('concat')
    print(os.path.abspath(folder_path))
    print(files)
    for file in files:
        if _type == '':
            df_list.append(pd.read_csv(file, parse_dates=True,
                           infer_datetime_format=True))
        elif _type in file:
            df_list.append(pd.read_csv(file, parse_dates=True,
                           infer_datetime_format=True))

    # Fill nan with 0s as some values are empty for percentage points
    df = pd.concat(df_list).fillna(0).reset_index(drop=True)

    return df


def create_defence(df):
    # Define aggregation functions
    agg_funcs = {}
    for col in df.columns:
        if 'average' in col or 'per' in col or 'avg ' in col or 'rate ' in col:
            # Average for columns with 'average' in their name
            agg_funcs[col] = 'mean'
        elif 'long' in col or 'home' in col or 'week' in col or 'date' in col or 'points' in col or col == 'yards allowed':
            agg_funcs[col] = 'max'
        else:
            agg_funcs[col] = 'sum'    # Sum for all other columns
    agg_funcs.pop('tm')
    # Group by 'team' and aggregate
    result = df.groupby('tm').agg(agg_funcs).reset_index()
    result['Position'] = 'DEF'
    result['player'] = result['tm']
    df = pd.concat([df, result])
    return df


def calc_score(file_name):
    df = pd.read_csv(file_name, index_col=False)
    try:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
    except:
        pass
    if 'offence' in file_name:
        scores = []
        for i, row in df.iterrows():
            pass_score = row['Pass Yds']/25 + \
                row['Pass TD']*6+row['Int']*-2+row['Sack']*-1

            if row['Pass Yds'] >= 300:
                pass_score += 2
            if row['Pass Yds'] >= 400:
                pass_score += 2

            rush_score = row['Rush Yds']/10+row['Rush TD']*6
            if row['Rush Yds'] >= 100:
                rush_score += 2
            if row['Rush Yds'] >= 200:
                rush_score += 2

            rec_score = row['Rec']*0.5+row['Rec TD']*6+row['Rec Yds']/10
            if row['Rec Yds'] >= 100:
                rush_score += 2
            if row['Rec Yds'] >= 200:
                rush_score += 2
            try:
                ret_score = row['punt yards']/10+row['punt td']*6+row['kick return yards']/10 + \
                    row['kickoff td']*6
            except KeyError:
                ret_score = 0
            score = pass_score+rush_score+rec_score + \
                ret_score-2*row['Fumble Lost']

            scores.append(score)

        df['score'] = scores
    elif 'defence' in file_name:
        if 'DEF' not in df.columns:
            df = create_defence(df)
        scores = []
        for i, row in df.iterrows():
            try:
                score = row['kickoff td']*6+row['punt td']*6
            except KeyError:
                score = 0
            if row['Position'] == 'DEF':
                score += row['Fumble Return TD']*6+row['Sacks']+row['Interceptions']*2 + \
                    row['Intercept. Ret. TD']*6+row['Fumbles Recovered']*2
                try:
                    score += row['kickoff td']*6+row['punt td']*6
                except KeyError:
                    pass
                if row['points allowed'] == 0:
                    score += 10
                elif row['points allowed'] > 1 and row['points allowed'] <= 6:
                    score += 7
                elif row['points allowed'] > 6 and row['points allowed'] <= 13:
                    score += 4
                elif row['points allowed'] > 13 and row['points allowed'] <= 20:
                    score += 1
                elif row['points allowed'] > 20 and row['points allowed'] <= 27:
                    score += 0
                elif row['points allowed'] > 27 and row['points allowed'] <= 34:
                    score += -1
                elif row['points allowed'] > 34:
                    score += -4

                if row['points allowed'] < 100:
                    score += 10
                elif row['points allowed'] >= 100 and row['points allowed'] < 200:
                    score += 7
                elif row['points allowed'] >= 200 and row['points allowed'] < 300:
                    score += 4
                elif row['points allowed'] >= 300 and row['points allowed'] < 400:
                    score += 1
            else:

                score += row['Tackles Solo']+row['Assists']+row['Sacks']*2+row['Interceptions']*2 + \
                    row['Fumbles Forced']*2+row['Fumbles Recovered'] + \
                    row['Fumble Return TD']*6 + \
                    row['Intercept. Ret. TD']*6+row['Passes Defended']

                if row['Tackles Combined'] >= 10:
                    score += 2

                if row['Sacks'] >= 2:
                    score += 2

            scores.append(score)

        df['score'] = scores

    return df


def calculate_MAE(pred, true):
    n = len(pred)
    # true = true.reset_index()
    true = true.reset_index()['score']
    abs_error = 0
    for i in range(n):
        abs_error += abs(pred[i] - true[i])
    mae = abs_error / n
    return mae


def calculate_RMSE(pred, true):
    return np.sqrt(mean_squared_error(pred, true))


def load_full_dataset(weighting="quad", folder='', role='', nest=False):
    if folder == '':
        print(os.path.abspath(os.path.join("Modelling", "Features", weighting)))
        df_features = csv_concatenate(
            os.path.join("Modelling", "Features", weighting), _type=role, nested=nest)
    else:
        print(os.path.abspath(os.path.join(
            folder, "Modelling", "Features", weighting)))
        df_features = csv_concatenate(
            os.path.join(folder, "Modelling", "Features", weighting), _type=role, nested=nest)
    df_features.drop(['position', 'tm'], axis=1, inplace=True)
    X = df_features.loc[:, df_features.columns[5:]]
    X = MinMaxScaler().fit_transform(X)
    y = df_features["score"].values.reshape(-1, 1).flatten()
    return X, y


def cross_val(reg_base, X, y, n_folds=5, isLightGBM=False, params=None, verbose=0):
    errors = {"MAE": {"train": [], "valid": []},
              "RMSE": {"train": [], "valid": []}}

    for i in range(n_folds):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=1 / n_folds, stratify=None, random_state=i
        )
        # print('train-\n', X_train)
        # print('\n\n\ntest-\n', X_valid)
        if isLightGBM == True:
            if params == None:
                return "Error: Specify parameters for LightGBM"
            else:
                d_train = lgb.Dataset(X_train, label=y_train)
                d_valid = lgb.Dataset(X_valid, label=y_valid)
                watchlist = [d_valid]
                print('train-\n', d_train)
                print('\n\n\nparams-\n', params)
                reg = lgb.train(params, d_train, watchlist)

        else:
            reg = reg_base
            reg.fit(X_train, y_train)
        y_pred_train = reg.predict(X_train)

        errors["MAE"]["train"].append(calculate_MAE(y_pred_train, y_train))
        errors["RMSE"]["train"].append(calculate_RMSE(y_pred_train, y_train))

        y_pred_valid = reg.predict(X_valid)

        errors["MAE"]["valid"].append(calculate_MAE(y_pred_valid, y_valid))
        errors["RMSE"]["valid"].append(calculate_RMSE(y_pred_valid, y_valid))

        if verbose == 1:
            print("<--- Training Error ({}/{})--->".format(i + 1, n_folds))
            print(" MAE: {}".format(round(errors["MAE"]["train"][i], 5)))
            print("RMSE: {}\n".format(round(errors["RMSE"]["train"][i], 5)))

            print("<--- Validation Error ({}/{}) --->".format(i + 1, n_folds))
            print(" MAE: {}".format(round(errors["MAE"]["valid"][i], 5)))
            print("RMSE: {}\n\n".format(round(errors["RMSE"]["valid"][i], 5)))

    return errors


def summarize_errors(errors, verbose=0):
    df_mae = pd.DataFrame(errors["MAE"]).T
    df_rmse = pd.DataFrame(errors["RMSE"]).T

    df_mae.columns = ["" for i in df_mae.columns]
    df_rmse.columns = ["" for i in df_rmse.columns]

    df_mae.index.name = "MAE"
    df_rmse.index.name = "RMSE"

    if verbose == 1:
        print(df_mae, df_rmse)

    print("\n   <--- Validation Errors --->")
    print(
        "MAE  | Mean: {}, SD: {}".format(
            str(round(np.mean(errors["MAE"]["valid"]), 5)),
            str(round(np.std(errors["MAE"]["valid"]), 5)),
        )
    )
    print(
        "RMSE | Mean: {}, SD: {}\n".format(
            str(round(np.mean(errors["RMSE"]["valid"]), 5)),
            str(round(np.std(errors["RMSE"]["valid"]), 5)),
        )
    )
    return None
