import sys
sys.path.insert(0, ".")

import joblib
import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
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
    OBJECT_IDS = [4]  # list(np.arange(1, 16, 1))
    SPLIT = '70_30'

    PARAMS = {
        'loss': 'squared_error',
        'solver': 'adam',
        'max_iter': 10_000,
        'learning_rate': 'constant',
        'activation': 'relu',
        'batch_size': 16,
        #'validation_fraction': 0.1,
        'hidden_layer_sizes': (512, 512),
        'learning_rate_init': 0.001,
        'alpha': 0.0001,
        #'early_stopping': True,
        #'n_iter_no_change': 20,
        #'tol': 1e-5,
    }

    # Parâmetros usados apenas no loop manual de treino/validação (curva de loss)
    MANUAL_TRAIN_EPOCHS = 2000
    MANUAL_PATIENCE = 100
    MANUAL_TOL = 1e-5
    MANUAL_VAL_FRACTION = 0.2

    original_df = pd.read_excel(f'processed/splitted_train_{SPLIT}.xlsx')
    original_df = original_df[(original_df['frame_id'] != 1277) & (original_df['frame_id'] != 1295)]  # Gimbal lock (Pitch = 90% - Yaw == Roll)

    for OBJECT_ID in OBJECT_IDS:
        print(f"\n\nSeaching model configuration for object {OBJECT_ID}...")

        MODELS_DIR = f'models/object_{OBJECT_ID}/MLPRegressor_Best_Model_2_{SPLIT}'
        OUTPUT_DIR = f'{MODELS_DIR}/performance'

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        df_train = original_df[(original_df['set'] == 'train') & (original_df['object_id'] == OBJECT_ID)]
        df_test = original_df[(original_df['set'] == 'test') & (original_df['object_id'] == OBJECT_ID)]

        X_train = df_train[X_cols]
        y_train = df_train[y_cols].values

        X_test = df_test[X_cols]
        y_test = df_test[y_cols].values

        hu_pca_cols = [f'hu_{i}' for i in range(1, 4 + 1, 1)]
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

        # ---- Treinamento "oficial" do modelo final (usado em todas as métricas/gráficos) ----
        best_model = Pipeline([
            ("preprocessor", preprocessor),
            ("neural_network", MLPRegressor(**PARAMS)),
        ])

        best_model.fit(X_train, y_train)

        y_train_norm, y_pred_train, errors_train = evaluate(best_model, X_train, y_train, "Training set")
        y_test_norm, y_pred_test, errors_test = evaluate(best_model, X_test, y_test, "Testing set")

        euler_train_true = np.array([quaternion_to_euler(_q) for _q in y_train_norm])
        euler_train_pred = np.array([quaternion_to_euler(_q) for _q in y_pred_train])
        euler_test_true = np.array([quaternion_to_euler(_q) for _q in y_test_norm])
        euler_test_pred = np.array([quaternion_to_euler(_q) for _q in y_pred_test])

        plot_euler_hist(euler_train_true, euler_train_pred, 'Train', 'train', OUTPUT_DIR)
        plot_euler_hist(euler_test_true, euler_test_pred, 'Test', 'test', OUTPUT_DIR)
        plot_accuracy_threshold(errors_train, errors_test, OUTPUT_DIR)
        plot_hist_geodesic(errors_train, OUTPUT_DIR, "train", "Treinamento")
        plot_hist_geodesic(errors_test, OUTPUT_DIR, "test", "Teste")

        joblib.dump(best_model, f'{MODELS_DIR}/model.pkl', compress=3)

        # ---- Loop manual de treino/validação, apenas para a curva de loss (mesma métrica: MSE) ----
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=MANUAL_VAL_FRACTION, random_state=42
        )

        # Preprocessador ajustado apenas no sub-treino (X_tr), evitando vazamento com X_val
        manual_preprocessor = ColumnTransformer(
            transformers=[
                ('hu_moments_pca', hu_pca_transformer, hu_pca_cols),
            ], remainder=PowerTransformer(method='yeo-johnson', standardize=True)
        )
        X_tr_transformed = manual_preprocessor.fit_transform(X_tr)
        X_val_transformed = manual_preprocessor.transform(X_val)

        mlp_params = {
            k: v for k, v in PARAMS.items()
            if k not in ("early_stopping", "n_iter_no_change", "tol", "validation_fraction")
        }
        manual_mlp = MLPRegressor(**{**mlp_params, "warm_start": True, "max_iter": 1})

        train_losses = []
        val_losses = []
        best_val_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(MANUAL_TRAIN_EPOCHS):
            manual_mlp.partial_fit(X_tr_transformed, y_tr)
            train_losses.append(manual_mlp.loss_)

            y_val_pred = manual_mlp.predict(X_val_transformed)
            val_loss = np.mean((y_val - y_val_pred) ** 2)
            val_losses.append(val_loss)

            if val_loss < best_val_loss - MANUAL_TOL:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= MANUAL_PATIENCE:
                print(f"Early stopping manual na época {epoch}")
                break

        best_epoch = int(np.argmin(val_losses))

        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, color="tab:blue", label="Treinamento")
        plt.plot(val_losses, color="tab:red", label="Validação")
        plt.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label="Melhor época")
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