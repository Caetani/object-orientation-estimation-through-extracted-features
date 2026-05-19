import sys
sys.path.insert(0, ".")

import emlearn
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate, GridSearchCV

from src.models.dataset_definitions import X_cols, y_cols
from src.utils.model_evaluation_utils import (
    evaluate,
    plot_euler_hist,
    plot_accuracy_threshold,
    plot_hist_geodesic,
    quaternion_to_euler,
    geodesic_rmse_scorer
)
from src.utils.model_conversion_utils import *

K_FOLDS = 10
OBJECT_ID   = 4
MODELS_DIR  = f'test/models/object_{OBJECT_ID}/decision_tree'
DATA_PATH   = 'processed/splitted_train.xlsx'
OUTPUT_DIR = f'{MODELS_DIR}/performance'


PARAMS = {}

criterion_arr = ["squared_error", "friedman_mse", "absolute_error", "poisson"]


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_excel('processed/splitted_train_70_30.xlsx')
    df = df[df['object_id'] == OBJECT_ID]
    #df = df[(df['frame_id'] != 1277) & (df['frame_id'] != 1295)] # Frames with gimbal lock (yaw == roll, pitch at +/- 90 deg)
    
    df_train = df[df['set'] == 'train']
    #df_test = df[df['set'] == 'test']

    X_train = df_train[X_cols]
    y_train = df_train[y_cols].values

    """ model = DecisionTreeRegressor(**PARAMS)
    cv_results = pd.DataFrame(cross_validate(model, X_train, y_train, cv=K_FOLDS, scoring='neg_root_mean_squared_error'))
    mean_rmse = -cv_results['test_score'].mean()
    std_rmse = cv_results['test_score'].std()
    print(f"Results before pruning:\n\tE(RMSE) = {mean_rmse}\n\ts(RMSE) = {std_rmse}") """

    for criterion in criterion_arr:
        PARAMS = {
            'criterion': criterion,
        }
        print(f"Criterion: {criterion}")
        model = DecisionTreeRegressor(**PARAMS)
        model.fit(X_train, y_train)

        path = model.cost_complexity_pruning_path(X_train, y_train)
        alphas, impurities = path.ccp_alphas, path.impurities

        PARAM_GRID = {
            'ccp_alpha': alphas,
        }

        grid_search = GridSearchCV(
            estimator=DecisionTreeRegressor(**PARAMS),
            param_grid=PARAM_GRID,
            cv=K_FOLDS,
            scoring=geodesic_rmse_scorer, #'neg_root_mean_squared_error',
            n_jobs=4,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        cv_results  = pd.DataFrame(grid_search.cv_results_)
        split_cols  = [f'split{i}_test_score' for i in range(K_FOLDS)]
        best_idx    = grid_search.best_index_
        mu_best     = cv_results.loc[best_idx, 'mean_test_score']
        sigma_best  = cv_results.loc[best_idx, 'std_test_score']
        best_scores = cv_results.loc[best_idx, split_cols].values.astype(float)

        print(cv_results.sort_values(by=['rank_test_score'], ascending=True))