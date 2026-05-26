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

        MODELS_DIR  = f'models/object_{OBJECT_ID}/random_forest_{SPLIT}'
        OUTPUT_DIR = f'{MODELS_DIR}/performance'

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        df_train = original_df[(original_df['set'] == 'train') & (original_df['object_id'] == OBJECT_ID)]
        df_test = original_df[(original_df['set'] == 'test') & (original_df['object_id'] == OBJECT_ID)]

        X_train = df_train[X_cols]
        y_train = df_train[y_cols].values

        X_test = df_test[X_cols]
        y_test = df_test[y_cols].values
        
        best_model = joblib.load(f'{MODELS_DIR}/model.pkl')
        joblib.dump(best_model, f'{MODELS_DIR}/model_compressed.pkl', compress=3)

    print(f"End of Execution.")