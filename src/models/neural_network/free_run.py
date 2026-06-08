import sys
sys.path.insert(0, ".")

import joblib
import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
import gc
import matplotlib.pyplot as plt
from itertools import product

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
        'loss': 'squared_error',
        'activation': 'relu',
        'solver': 'adam',
        'batch_size': 32,
        'max_iter': 10_000,
        'hidden_layer_sizes': (350, 350),
        'alpha': 0.01
    }

    #{'alpha': 0.01, 'hidden_layer_sizes': (250, 250)}
    
    original_df = pd.read_excel(f'processed/splitted_train_{SPLIT}.xlsx')
    original_df = original_df[(original_df['frame_id'] != 1277) & (original_df['frame_id'] != 1295)] # Gimbal lock (Pitch = 90% - Yaw == Roll)

    for OBJECT_ID in OBJECT_IDS:
        print(f"\n\nSeaching model configuration for object {OBJECT_ID}...")

        MODELS_DIR  = f'free_run_nn_2/models/object_{OBJECT_ID}/neural_network_{SPLIT}'
        OUTPUT_DIR = f'{MODELS_DIR}/performance'

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        df_train = original_df[(original_df['set'] == 'train') & (original_df['object_id'] == OBJECT_ID)]
        df_test = original_df[(original_df['set'] == 'test') & (original_df['object_id'] == OBJECT_ID)]

        X_train = df_train[X_cols]
        y_train = df_train[y_cols].values

        X_test = df_test[X_cols]
        y_test = df_test[y_cols].values

        power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        X_train = power_transformer.fit_transform(X_train)
        X_test = power_transformer.transform(X_test)


        best_model = MLPRegressor(**PARAMS)
        best_model.fit(X_train, y_train)
        X_train_pred = best_model.predict(X_train)
        X_test_pred = best_model.predict(X_test)
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