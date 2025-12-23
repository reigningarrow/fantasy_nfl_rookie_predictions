# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:35:32 2024

@author: sambi
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import utils
warnings.filterwarnings("ignore")


_type = 'offence'

for _type in ['offence', 'defence']:
    weighting = "quad"
    df_features = utils.csv_concatenate(
        os.path.join("Modelling", "Features", weighting), _type=_type)

    cols = list(df_features.columns)[6:]
    cols.remove('position')
    cols.remove('score')
    cols.remove('avg_score')
    X = df_features.loc[:, cols]
    X = MinMaxScaler().fit_transform(X)
    y = df_features["score"].values.reshape(-1, 1).flatten()

    # Takes 2 minutes
    model = GradientBoostingRegressor()
    model.fit(X, y)

    top_features = pd.Series(model.feature_importances_,
                             index=cols).sort_values()
    top_features.plot(kind="barh", figsize=(15, 10),
                      title=f"Top Features-{_type}")
    plt.show()

    omit_lowest = 20
    _selected = list(top_features[omit_lowest:].index)

    for feature_type in [cols, _selected]:
        for weighting in ["sqrt", "linear", "quad"]:
            df_features = utils.csv_concatenate(
                os.path.join("Modelling", "Features", weighting), _type=_type
            )
            print("\nweighting scheme: {}".format(weighting))

            X = df_features.loc[:, feature_type]
            X = MinMaxScaler().fit_transform(X)
            y = df_features["score"].values.reshape(-1, 1).flatten()

            reg = LinearRegression()

            errors = utils.cross_val(reg, X, y, verbose=0)
            utils.summarize_errors(errors, verbose=0)

        print("========================")
