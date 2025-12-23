# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:47:17 2024

@author: sambi
"""
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import os


df = pd.read_csv('games/seasons/2018_offence.csv')


df = df[df['position'] == 'QB']
# df.set_index('week', inplace=True)


plt.errorbar(df['week'], df['score'], yerr=df['risk'], color="r",
             elinewidth=1, capsize=1, alpha=0.45)
plt.plot(df["week"], df["score"], alpha=1, c='b')
plt.show()


def draw_weights():
    plt.figure()
    weights_dic = {}
    weighting = ["sqrt_0.5", "linear_1.0", "quad_2.0"]

    for key in weighting:
        weights = np.array([np.power(i, float(key[-3:]))
                           for i in range(1, 11)])
        weights = weights / weights.sum()
        weights_dic[key[:-4]] = weights

    sns.set_style("darkgrid")

    for key in weights_dic.keys():
        plt.plot(weights_dic[key])

    plt.xlabel("n-th game", fontsize=12)
    plt.ylabel("Weight", fontsize=12)
    plt.xticks([i for i in range(0, 10)], [i for i in range(1, 11)])
    plt.show()


draw_weights()


def calculate_weighted_mean(weights, values):
    n = len(weights)
    # weighted_sum = [weights[i] * values[i] for i in range(n)]
    weights = np.array(weights)  # Convert to a numpy array if not already
    values = np.array(values)   # Convert to a numpy array if not already
    try:
        weighted_sum = (weights * values)
        weighted_mean = sum(weighted_sum) / sum(weights)
    except Exception as e:
        print('\n\n\n\n\n')
        print('weights-', weights)
        print('values-', values)
        raise e

    return weighted_mean


def generate_features(df, weighting):
    # df.drop_duplicates(subset=['player'], inplace=True)
    df.reset_index(inplace=True)
    df2 = df.copy()[0:0].to_dict()
    df2['rest'] = {}
    rest = {}
    for i in range(len(df)):

        date = df.loc[i, "date"]
        name = df.loc[i, "player"]

        df_name = df.loc[df["player"] == name].reset_index(drop=True)
        index = df_name.loc[df_name["date"] == date].index[0]

        # r = df_name.loc[index]
        # print(r.to_dict())
        # Generate features from the past 10 games
        prev_games = 3
        if index >= prev_games:
            print(name)
            df_past = df_name[index - prev_games: index].reset_index(drop=True)

            # Consider the number of days between the current game and the previous game
            current = datetime.strptime(
                str(df_name.loc[index, "date"]), "%Y%m%d")
            previous = datetime.strptime(
                str(df_past.loc[df_past.shape[0] - 1, "date"]), "%Y%m%d"
            )
            rest_period = current - previous

            rest[index] = rest_period.days

            # Weights higehr towards the most recent game
            if weighting == "linear":
                weights = [i for i in range(1, prev_games+1)]

            elif weighting == "quad":
                weights = [i**2 for i in range(1, prev_games+1)]

            elif weighting == "sqrt":
                weights = [i ** (1 / 2) for i in range(1, prev_games+1)]

            for key in df2.keys():
                if key in ["date", "player", "score", 'tm', 'position']:
                    df2[key][i] = df_name.loc[index, key]
                elif key == "risk":
                    df2[key][i] = df_past["score"].std()
                elif key != "rest":
                    # print('KEEYYYYYY-', key)
                    weighted_mean = calculate_weighted_mean(
                        weights, df_past[key])
                    df2[key][i] = weighted_mean

    df2['rest'] = rest
    return pd.DataFrame(df2)


os.chdir('basic_game_data')
weighting_types = ["sqrt", "linear", "quad"]
seasons = ['2018', '2019', '2020', '2021', '2022', '2023', '2024']
# Takes ~ 2 hrs in total
# TODO: Optimize
for weighting in weighting_types:
    for season in seasons:
        for role in ['offence', 'defence']:
            df = pd.read_csv(
                os.path.join("games", "seasons",
                             "{}_{}.csv".format(season, role)), index_col=False
            )
            z = df['player'].value_counts()

            name = df.loc[0, "player"]
            df_name = df.loc[df["player"] == name].reset_index(drop=True)
            # a = df_name.loc[df_name["date"] == date]
            df_features = generate_features(df, weighting)
            # Add Starter, Listed Position and Team
            df_features = pd.merge(
                df.loc[
                    :,
                    [
                        "date",
                        "player"
                    ],
                ],
                df_features,
                on=["date", "player"],
                how="inner",
            )
            # Add roster information
            # df_features = df_features.loc[:, DF_FEATURES]
            if os.path.exists(os.path.join("Modelling", "Features", weighting)) is False:
                os.makedirs(os.path.join("Modelling", "Features", weighting))
            df_features.to_csv(
                os.path.join(
                    "Modelling",
                    "Features",
                    weighting,
                    "{}_{}.csv".format(season, role)
                ),
                index=False,
            )
