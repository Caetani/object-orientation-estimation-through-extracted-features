import sys
sys.path.insert(0, ".")

import joblib
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate, GridSearchCV
import gc
import matplotlib.pyplot as plt
from scipy import stats

from src.models.dataset_definitions import X_cols, y_cols
from src.utils.model_evaluation_utils import (
    evaluate,
    plot_euler_hist,
    plot_accuracy_threshold,
    plot_hist_geodesic,
    quaternion_to_euler,
    geodesic_rmse_scorer
)
from src.utils.model_conversion_utils import convert_model


if __name__ == '__main__':
    K_FOLDS = 10
    OBJECT_IDS = [4]#list(np.arange(1, 16, 1))
    SPLIT = '70_30'
    
    PARAMS = {}
    
    original_df = pd.read_excel(f'processed/splitted_train_{SPLIT}.xlsx')
    original_df = original_df[(original_df['frame_id'] != 1277) & (original_df['frame_id'] != 1295)] # Gimbal lock (Pitch = 90% - Yaw == Roll)

    for OBJECT_ID in OBJECT_IDS:
        print(f"\n\nSeaching model configuration for object {OBJECT_ID}...")

        MODELS_DIR  = f'models/object_{OBJECT_ID}/decision_tree_{SPLIT}/quantization'
        OUTPUT_DIR = f'{MODELS_DIR}'

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        df_train = original_df[(original_df['set'] == 'train') & (original_df['object_id'] == OBJECT_ID)]
        df_test = original_df[(original_df['set'] == 'test') & (original_df['object_id'] == OBJECT_ID)]

        num_bits_arr = []
        rmse_geodesic_val_arr = []

        for num_bits in range(2, 16+1, 1):
            num_bits_arr.append(num_bits)
            n_levels = 2**num_bits - 1

            X_train = df_train[X_cols]
            X_train = X_train*n_levels
            X_train = np.round(X_train)
            X_train = X_train.astype('int')
            
            y_train = df_train[y_cols]
            y_train = y_train*n_levels
            y_train = np.round(y_train)
            y_train = y_train.astype('int')

            model = DecisionTreeRegressor(**PARAMS)
            model.fit(X_train, y_train)

            path = model.cost_complexity_pruning_path(X_train, y_train)
            alphas, impurities = path.ccp_alphas, path.impurities

            PARAM_GRID = {
                'ccp_alpha': alphas
            }

            grid_search = GridSearchCV(
                estimator=DecisionTreeRegressor(**PARAMS),
                param_grid=PARAM_GRID,
                cv=K_FOLDS,
                scoring=geodesic_rmse_scorer,
                n_jobs=4,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            cv_results  = pd.DataFrame(grid_search.cv_results_)
            best_model = grid_search.best_estimator_

            

            joblib.dump(best_model, f'{MODELS_DIR}/{num_bits}_model.pkl')
            convert_model(best_model, f"{MODELS_DIR}/{num_bits}_bits")


    print(f"End of Execution.")