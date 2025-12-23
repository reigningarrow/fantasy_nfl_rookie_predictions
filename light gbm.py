# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:03:55 2024

@author: sambi
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

from datetime import datetime


import lightgbm as lgb
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

import utils
warnings.filterwarnings("ignore")


def bayes_parameter_opt_lgb(X, y, init_round, opt_round, n_folds, random_seed):
    train_data = lgb.Dataset(data=X, label=y)

    def lgb_eval(
        feature_fraction,
        bagging_fraction,
        lambda_l1,
        lambda_l2,
        max_depth,
        num_leaves,
        min_split_gain,
        min_child_weight,
        learning_rate,
        n_estimators,
    ):
        params = {
            "objective": "regression",
            "max_bin": 255,
            "bagging_freq": 1,
            "min_child_samples": 20,
            "boosting": "gbdt",
            "verbosity": 1,
            "early_stopping_round": 200,
            "metric": "rmse",
        }

        params["feature_fraction"] = max(min(feature_fraction, 1), 0)
        params["bagging_fraction"] = max(min(bagging_fraction, 1), 0)
        params["lambda_l1"] = max(lambda_l1, 0)
        params["lambda_l2"] = max(lambda_l2, 0)
        params["max_depth"] = int(round(max_depth))
        params["num_leaves"] = int(round(num_leaves))
        params["min_split_gain"] = min_split_gain
        params["min_child_weight"] = min_child_weight
        params["learning_rate"] = learning_rate
        params["n_estimators"] = int(round(n_estimators))

        cv_result = lgb.cv(
            params,
            train_data,
            nfold=n_folds,
            seed=random_seed,
            verbose_eval=None,
            stratified=False,
        )

        # Print RMSE for each round of lgbBO for rough tracking of the optimization process
        min_index = cv_result["rmse-mean"].index(min(cv_result["rmse-mean"]))
        print(
            "RMSE: {} +- {}".format(
                round(cv_result["rmse-mean"][min_index], 5),
                round(cv_result["rmse-stdv"][min_index], 5),
            )
        )

        return (-1.0 * np.array(cv_result["rmse-mean"])).max()

    lgbBO = BayesianOptimization(
        lgb_eval,
        {
            "feature_fraction": (0.6, 0.9),
            "bagging_fraction": (0.8, 1),
            "lambda_l1": (0, 3),
            "lambda_l2": (0, 3),
            "max_depth": (5, 100),
            "num_leaves": (10, 300),
            "min_split_gain": (0.001, 0.1),
            "min_child_weight": (0, 1),
            "learning_rate": (0.001, 0.1),
            "n_estimators": (50, 5000),
        },
        random_state=random_seed,
    )

    # Save progress for each round into a JSON file which can be monitored on a editor (i.e. VSCode)
    # This somehow suppresses the terminal output (https://github.com/fmfn/BayesianOptimization/issues/167)
    logger = JSONLogger(
        path="/Models/LightGBM/Params/{}.json".format(
            pd.Timestamp.now().strftime("%Y%m%d-%Hh%Mm")
        )
    )
    lgbBO.subscribe(Events.OPTMIZATION_STEP, logger)

    lgbBO.maximize(init_points=init_round, n_iter=opt_round, acq="ei")

    return lgbBO.max["params"]


X, y = utils.load_full_dataset("quad", role='offence')
opt_params = None
if os.path.exists("Modelling//LightGBM/Params") is False:
    print('folder doesnt exiiissstttt')
    os.makedirs("Modelling/LightGBM/Params/")

# Load the newest file in the params directory, otherwise set manually
path_params = sorted(glob.glob("Modelling//LightGBM//Params//*.json"))[-1]
z = pd.read_json(path_params, lines=True)
df_params = pd.read_json(path_params, lines=True)
df_params = (
    df_params.loc[:, ["target", "params"]]
    .sort_values(by="target", ascending=False)
    .reset_index()
)

if opt_params == None:
    opt_params = df_params.loc[0, "params"]
else:
    print("opt_params is already defined")

for key in opt_params.keys():
    if key in ["max_depth", "num_leaves", "n_estimators"]:
        opt_params[key] = int(round(opt_params[key]))

model = lgb.LGBMModel()
errors = utils.cross_val(model, X, y, isLightGBM=True,
                         params=opt_params, verbose=0)
utils.summarize_errors(errors)
