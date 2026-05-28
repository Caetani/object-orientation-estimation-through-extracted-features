import sys
sys.path.insert(0, ".")

import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import gc
import matplotlib.pyplot as plt
from time import time

from src.models.dataset_definitions import X_cols, y_cols
from src.utils.model_evaluation_utils import (
    evaluate,
    plot_euler_hist,
    plot_accuracy_threshold,
    plot_hist_geodesic,
    quaternion_to_euler,
    geodesic_rmse_scorer,
    geodesic_rmse_oob,
)


if __name__ == '__main__':
    K_FOLDS = 10
    OBJECT_IDS = [4] #list(np.arange(1, 16, 1))
    SPLIT = '70_30'
    
    PARAMS = {
        'oob_score': geodesic_rmse_oob,
    }
    
    original_df = pd.read_excel(f'processed/splitted_train_{SPLIT}.xlsx')
    original_df = original_df[(original_df['frame_id'] != 1277) & (original_df['frame_id'] != 1295)] # Gimbal lock (Pitch = 90% - Yaw == Roll)

    for OBJECT_ID in OBJECT_IDS:
        print(f"\n\nSeaching model configuration for object {OBJECT_ID}...")

        MODELS_DIR  = f'test_rf/models/object_{OBJECT_ID}/random_forest_{SPLIT}'
        OUTPUT_DIR = f'{MODELS_DIR}/performance'

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        df_train = original_df[(original_df['set'] == 'train') & (original_df['object_id'] == OBJECT_ID)]
        df_test = original_df[(original_df['set'] == 'test') & (original_df['object_id'] == OBJECT_ID)]

        X_train = df_train[X_cols]
        y_train = df_train[y_cols].values

        X_test = df_test[X_cols]
        y_test = df_test[y_cols].values

        PARAM_GRID = {
            'n_estimators': [100, 300, 500, 1_000],
            'max_features': [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 24, 28],
            'max_depth':    [5, 10, 20, None],
            'min_samples_leaf': [1, 5, 10, 20],
        }

        """ model = RandomForestRegressor(**PARAMS)
        model.fit(X_train, y_train)
        print(f"Model Geodesic Error = {-model.oob_score_} degrees.") 

        y_train_norm, y_pred_train, errors_train = evaluate(model, X_train, y_train, "Training set")
        y_test_norm,  y_pred_test,  errors_test  = evaluate(model, X_test, y_test, "Testing set")
        """
        initial_time = time()
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(**PARAMS),
            param_grid=PARAM_GRID,
            cv=K_FOLDS,
            scoring=geodesic_rmse_scorer,
            n_jobs=4,
            verbose=3
        )
        grid_search.fit(X_train, y_train)
        print(f"Final time = {time() - initial_time}")

        cv_results  = pd.DataFrame(grid_search.cv_results_)
        cv_results.sort_values(by=['rank_test_score'], ascending=True, inplace=True)
        cv_results.reset_index(inplace=True)
        print(cv_results)
        cv_results.to_excel(f'{MODELS_DIR}/grid_search_results.xlsx', index=False)

        best_model = grid_search.best_estimator_
        joblib.dump(best_model, f'{MODELS_DIR}/model.pkl', compress=3)

        y_train_norm, y_pred_train, errors_train = evaluate(best_model, X_train, y_train, "Training set")
        y_test_norm,  y_pred_test,  errors_test  = evaluate(best_model, X_test, y_test, "Testing set")

        euler_train_true = np.array([quaternion_to_euler(_q) for _q in y_train_norm])
        euler_train_pred = np.array([quaternion_to_euler(_q) for _q in y_pred_train])
        euler_test_true  = np.array([quaternion_to_euler(_q) for _q in y_test_norm])
        euler_test_pred  = np.array([quaternion_to_euler(_q) for _q in y_pred_test])

        plot_euler_hist(euler_train_true, euler_train_pred, 'Train', 'train', OUTPUT_DIR)
        plot_euler_hist(euler_test_true, euler_test_pred, 'Test', 'test', OUTPUT_DIR)
        plot_accuracy_threshold(errors_train, errors_test, OUTPUT_DIR)
        plot_hist_geodesic(errors_train, OUTPUT_DIR, "train", "Treinamento")
        plot_hist_geodesic(errors_test, OUTPUT_DIR, "test", "Teste")

        plt.close('all')
        gc.collect()

    print(f"End of Execution.")