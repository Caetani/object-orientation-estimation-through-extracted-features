import sys
sys.path.insert(0, ".")

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.compose import TransformedTargetRegressor

from src.utils.orientation_utils import geodesic_error, quaternion_to_euler
from src.utils.model_evaluation_utils import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gc

# ── Colunas ───────────────────────────────────────────────────────────────────
hu_cols = ['hu_1', 'hu_2', 'hu_3', 'hu_4']
passthrough_cols = [
    'compactness_ext',
    'dist_centroid',
    'angle_centroid',
    'sin_centroid',
    'cos_centroid',
    'aspect_ratio',
    'eccentricity',
    'sin_major',
    'cos_major',
    'hog_0_30',
    'hog_30_60',
    'hog_60_90',
    'hog_90_120',
    'hog_120_150',
    'hog_150_180',
    'hog_180_210',
    'hog_210_240',
    'hog_240_270',
    'hog_270_300',
    'hog_300_330',
    'hog_330_360',
    'hu_5',
    'hu_6',
    'hu_7',
]
X_cols = passthrough_cols + hu_cols
y_cols = ['qw', 'qx', 'qy', 'qz']

OBJECT_ID  = 4
SPLIT = '70_30'
MODELS_DIR = f'models/object_{OBJECT_ID}/linear_regression_{SPLIT}'
DATA_PATH  = f'processed/splitted_train_{SPLIT}.xlsx'
OUTPUT_DIR = MODELS_DIR + "/performance"


def build_pipeline() -> TransformedTargetRegressor:
    col_transformer = ColumnTransformer(
        transformers=[
            ('pca_hu',      PCA(n_components=2), hu_cols),
            ('passthrough', 'passthrough',        passthrough_cols),
        ]
    )

    pipeline_X = Pipeline([
        ('col_transformer', col_transformer),
        ('power_transform',  PowerTransformer(method='yeo-johnson', standardize=True)),
        ('regressor',        MultiOutputRegressor(LinearRegression())),
    ])

    pipeline = TransformedTargetRegressor(
        regressor=pipeline_X,
        transformer=StandardScaler(),
    )

    return pipeline




if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Carrega e prepara dados ───────────────────────────────────────────────
    df = pd.read_excel(DATA_PATH)
    df = df[df['object_id'] == OBJECT_ID]
    df = df[(df['frame_id'] != 1277) & (df['frame_id'] != 1295)]

    train_df = df[df['set'] == 'train'].reset_index(drop=True)
    test_df  = df[df['set'] == 'test'].reset_index(drop=True)

    X_train = train_df[X_cols]
    y_train = train_df[y_cols].values
    X_test  = test_df[X_cols]
    y_test  = test_df[y_cols].values

    print(f'Amostras de treino: {len(X_train)} | Amostras de teste: {len(X_test)}')

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, f'{MODELS_DIR}/model.pkl')
    print(f'\nPipeline salva em: {MODELS_DIR}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    y_train_norm, y_pred_train, errors_train = evaluate(pipeline, X_train, y_train, 'Conjunto de Treinamento')
    y_test_norm,  y_pred_test,  errors_test  = evaluate(pipeline, X_test,  y_test,  'Conjunto de Teste')

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