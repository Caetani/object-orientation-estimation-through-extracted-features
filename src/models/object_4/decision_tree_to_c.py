import sys
sys.path.insert(0, ".")

import emlearn
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.models.dataset_definitions import X_cols, y_cols
from src.utils.model_evaluation_utils import (
    evaluate,
    plot_euler_hist,
    plot_accuracy_threshold,
    plot_hist_geodesic,
    quaternion_to_euler,
)
from src.utils.model_conversion_utils import *

OBJECT_ID   = 4
MODELS_DIR  = f'models/object_{OBJECT_ID}/decision_tree'
DATA_PATH   = 'processed/splitted_train.xlsx'
OUTPUT_DIR = f'{MODELS_DIR}/performance'

BEST_PARAMS = {
    'ccp_alpha':         0,
    'max_depth':         11,
    'max_features':      1.0,
    'min_samples_leaf':  2,
    'min_samples_split': 10,
}

if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_excel('processed/splitted_train_70_30.xlsx')
    df = df[df['object_id'] == OBJECT_ID]
    #df = df[(df['frame_id'] != 1277) & (df['frame_id'] != 1295)] # Frames with gimbal lock (yaw == roll, pitch at +/- 90 deg)
    
    df_train = df[df['set'] == 'train']
    df_test = df[df['set'] == 'test']

    X_train = df_train[X_cols]
    y_train = df_train[y_cols].values

    X_test = df_test[X_cols]
    y_test = df_test[y_cols].values

    model = DecisionTreeRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    y_train_norm, y_pred_train, errors_train = evaluate(model, X_train, y_train, "Training set")
    y_test_norm,  y_pred_test,  errors_test  = evaluate(model, X_test, y_test, "Testing set")

    euler_train_true = np.array([quaternion_to_euler(_q) for _q in y_train_norm])
    euler_train_pred = np.array([quaternion_to_euler(_q) for _q in y_pred_train])
    euler_test_true  = np.array([quaternion_to_euler(_q) for _q in y_test_norm])
    euler_test_pred  = np.array([quaternion_to_euler(_q) for _q in y_pred_test])

    plot_euler_hist(euler_train_true, euler_train_pred, 'Train', 'train', OUTPUT_DIR)
    plot_euler_hist(euler_test_true, euler_test_pred, 'Test', 'test', OUTPUT_DIR)
    plot_accuracy_threshold(errors_train, errors_test, OUTPUT_DIR)
    plot_hist_geodesic(errors_train, errors_test, OUTPUT_DIR)

    joblib.dump(model, f"{MODELS_DIR}/model.pkl")

    for i, col in enumerate(['qw', 'qx', 'qy', 'qz']):
        proxy = SingleOutputRegressorProxy(model, i)
        cmodel = emlearn.convert(proxy, kind='DecisionTreeRegressor', method='inline')
        cmodel.save(name=f'model_{col}', file=f'{MODELS_DIR}/model_{col}.h')
