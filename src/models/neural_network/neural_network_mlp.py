import sys
sys.path.insert(0, ".")

import joblib
import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import gc
import matplotlib.pyplot as plt

from src.models.dataset_definitions import X_cols, y_cols
from src.utils.model_evaluation_utils import (
    evaluate,
    plot_euler_hist,
    plot_accuracy_threshold,
    plot_hist_geodesic,
    quaternion_to_euler,
    geodesic_rmse_scorer,
)


if __name__ == '__main__':
    K_FOLDS = 10
    OBJECT_IDS = [4] #list(np.arange(1, 16, 1))
    SPLIT = '70_30'
    
    PARAMS = {
        'loss': 'squared_error',
        'solver': 'adam',
        'max_iter': 10_000,
        'learning_rate': 'constant',
        'activation': 'relu',
        'batch_size': 32,
        'validation_fraction': 0.1,
        'hidden_layer_sizes': (128, 128),
        'learning_rate_init': 0.01,
        'batch_size': 32,
        'alpha': 0.0001,
        'early_stopping': True,
        'n_iter_no_change': 20,
        'tol': 1e-5, 
    }
    
    original_df = pd.read_excel(f'processed/splitted_train_{SPLIT}.xlsx')
    original_df = original_df[(original_df['frame_id'] != 1277) & (original_df['frame_id'] != 1295)] # Gimbal lock (Pitch = 90% - Yaw == Roll)

    for OBJECT_ID in OBJECT_IDS:
        print(f"\n\nSeaching model configuration for object {OBJECT_ID}...")

        MODELS_DIR  = f'models/object_{OBJECT_ID}/MLPRegressor_Best_Model_2_{SPLIT}'
        OUTPUT_DIR = f'{MODELS_DIR}/performance'

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        df_train = original_df[(original_df['set'] == 'train') & (original_df['object_id'] == OBJECT_ID)]
        df_test = original_df[(original_df['set'] == 'test') & (original_df['object_id'] == OBJECT_ID)]

        X_train = df_train[X_cols]
        y_train = df_train[y_cols].values

        X_test = df_test[X_cols]
        y_test = df_test[y_cols].values

        hu_pca_cols = [f'hu_{i}' for i in range(1, 4+1, 1)]
        hu_pca_transformer = Pipeline(
            steps=[
                ("power_transformer", PowerTransformer(
                    method='yeo-johnson',
                    standardize=True
                )),
                ('PCA', PCA(n_components=3)),
                ('scaler', StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ('hu_moments_pca', hu_pca_transformer, hu_pca_cols),
            ], remainder=PowerTransformer(method='yeo-johnson', standardize=True)
        )

        best_model = Pipeline([
            ("preprocessor", preprocessor),
            ("neural_network", MLPRegressor(**PARAMS)),
        ])

        best_model.fit(X_train, y_train)

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

        mlp = best_model.named_steps["neural_network"]
        plt.figure(figsize=(8, 5))
        plt.plot(mlp.loss_curve_, color="tab:blue", label="Treinamento")

        if getattr(mlp, "validation_scores_", None) is not None:
            plt.plot(mlp.validation_scores_, color="tab:red", label="Validação")

        plt.xlabel("Época")
        plt.ylabel("Loss (erro quadrático médio)")
        plt.title("Curva de perda durante o treinamento")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.close('all')
        gc.collect()

    print(f"End of Execution.")