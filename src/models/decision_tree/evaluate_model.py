import sys
sys.path.insert(0, ".")

import joblib
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
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
    OBJECT_IDS = [4] #list(np.arange(1, 16, 1))
    NUM_BITS = 4
    N_LEVELS = 2**NUM_BITS - 1
    SPLIT = '70_30'
    
    PARAMS = {}
    
    original_df = pd.read_excel(f'processed/splitted_train_{SPLIT}.xlsx')
    original_df = original_df[(original_df['frame_id'] != 1277) & (original_df['frame_id'] != 1295)] # Gimbal lock (Pitch = 90% - Yaw == Roll)

    for OBJECT_ID in OBJECT_IDS:
        print(f"\n\nSeaching model configuration for object {OBJECT_ID}...")

        MODELS_DIR  = f'models/object_{OBJECT_ID}/decision_tree_{SPLIT}/quantization/{NUM_BITS}_bits'
        OUTPUT_DIR = f'{MODELS_DIR}/performance'

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        df_train = original_df[(original_df['set'] == 'train') & (original_df['object_id'] == OBJECT_ID)]
        df_test = original_df[(original_df['set'] == 'test') & (original_df['object_id'] == OBJECT_ID)]

        y_train = df_train[y_cols].values
        y_test = df_test[y_cols].values

        X_train = df_train[X_cols]
        train_min, train_max = X_train.min(), X_train.max()
        X_train = (X_train - train_min) / (train_max - train_min)
        X_train = X_train*N_LEVELS
        X_train = np.round(X_train)
        X_train = X_train.astype(np.uint8)

        X_test = df_test[X_cols]
        X_test = (X_test - train_min) / (train_max - train_min)
        X_test = X_test*N_LEVELS
        X_test = np.round(X_test)
        X_test = X_test.astype(np.uint8)

        best_model = joblib.load(f"{MODELS_DIR}/{NUM_BITS}_model.pkl")

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