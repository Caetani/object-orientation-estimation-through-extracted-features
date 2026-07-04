import sys
sys.path.insert(0, ".")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import ctypes

def _preload_cuda_libs():
    """
    Force-loads CUDA .so files via ctypes before TensorFlow is imported.
    os.environ["LD_LIBRARY_PATH"] has no effect on the current process —
    the dynamic linker is already initialized. ctypes.CDLL triggers dlopen now.
    """
    venv = os.environ.get("VIRTUAL_ENV", "")
    if not venv:
        print("WARNING: VIRTUAL_ENV not set, skipping CUDA preload.")
        return

    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    tf_dir = os.path.join(venv, "lib", py_ver, "site-packages", "tensorflow")
    wsl_dir = "/usr/lib/wsl/lib"

    libs = [
        (wsl_dir, "libcuda.so.1"),
        (tf_dir,  "libcudart.so.12"),
        (tf_dir,  "libnvJitLink.so.12"),
        (tf_dir,  "libcublas.so.12"),
        (tf_dir,  "libcublasLt.so.12"),
        (tf_dir,  "libcufft.so.11"),
        (tf_dir,  "libcurand.so.10"),
        (tf_dir,  "libcusolver.so.11"),
        (tf_dir,  "libcusparse.so.12"),
        (tf_dir,  "libcudnn.so.9"),
        (tf_dir,  "libnccl.so.2"),
    ]

    for directory, lib in libs:
        path = os.path.join(directory, lib)
        try:
            ctypes.CDLL(path)
        except OSError as e:
            print(f"WARNING: could not preload {lib}: {e}")

_preload_cuda_libs()

import joblib
import numpy as np
import pandas as pd
from itertools import product
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scikeras.wrappers import KerasRegressor
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

print("GPU available:", tf.config.list_physical_devices('GPU'))

from src.models.dataset_definitions import X_cols, y_cols
from src.utils.model_evaluation_utils import (
    evaluate,
    plot_euler_hist,
    plot_accuracy_threshold,
    plot_hist_geodesic,
    quaternion_to_euler,
    geodesic_rmse_scorer,
    geodesic_rmse_score_func,
)

def geodesic_rmse_keras(y_true, y_pred):
    result = tf.py_function(
        lambda yt, yp: geodesic_rmse_score_func(yt.numpy(), yp.numpy()),
        [y_true, y_pred],
        tf.float32
    )
    result.set_shape([])  # informa ao Keras que o resultado é um escalar
    return result

geodesic_rmse_keras.__name__ = "geodesic_rmse"


def build_model(hidden_layer_sizes=(64, 32), activation="relu", lr=1e-3, n_features_in=27, **kwargs):
    n_outputs = 4

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in,)))
    for units in hidden_layer_sizes:
        model.add(keras.layers.Dense(units, activation=activation))
    model.add(keras.layers.Dense(n_outputs))  # quaternion: 4 outputs, no activation

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[geodesic_rmse_keras],
        run_eagerly=True,
    )
    return model


if __name__ == '__main__':
    K_FOLDS = 10
    OBJECT_IDS = [4]  # list(np.arange(1, 16, 1))
    SPLIT = '70_30'

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        min_delta=1e-3,
        restore_best_weights=True,
        verbose=0,
    )

    # ---- Grid search space ----
    layers_range = [i for i in range(5, 50+5, 5)]
    LAYERS_1 = [(j,) for j in layers_range]
    LAYERS_2 = list(product(layers_range, repeat=2))
    LAYERS = LAYERS_1 + LAYERS_2

    LEARNING_RATES = [10**i for i in range(-3, -2 + 1, 1)]
    ACTIVATIONS = ['relu']
    BATCH_SIZES = [16, 32]

    PARAM_GRID = {
        'neural_network__model__hidden_layer_sizes': LAYERS,
        'neural_network__model__lr': LEARNING_RATES,
        'neural_network__model__activation': ACTIVATIONS,
        'neural_network__model__batch_size': BATCH_SIZES,
    }

    original_df = pd.read_excel(f'processed/splitted_train_{SPLIT}.xlsx')
    original_df = original_df[(original_df['frame_id'] != 1277) & (original_df['frame_id'] != 1295)]  # Gimbal lock (Pitch = 90% - Yaw == Roll)

    for OBJECT_ID in OBJECT_IDS:
        print(f"\n\nSeaching model configuration for object {OBJECT_ID}...")

        MODELS_DIR = f'models/object_{OBJECT_ID}/Keras_Complete_Search_neural_network_{SPLIT}'
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

        neural_network = KerasRegressor(
            model=build_model,
            n_features_in=27,
            loss="mse",
            metrics=[geodesic_rmse_keras],
            epochs=10_000,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0,
        )

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("neural_network", neural_network),
        ])

        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=PARAM_GRID,
            cv=K_FOLDS,
            scoring=geodesic_rmse_scorer,
            n_jobs=4,
            verbose=3,
        )
        grid_search.fit(X_train, y_train)

        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.sort_values(by=['rank_test_score'], ascending=True, inplace=True)
        cv_results.reset_index(inplace=True)
        print(cv_results)
        cv_results.to_excel(f'{MODELS_DIR}/grid_search_results.xlsx', index=False)

        best_model = grid_search.best_estimator_
        joblib.dump(best_model, f'{MODELS_DIR}/model.pkl', compress=3)

        history = best_model.named_steps["neural_network"].history_
        best_epoch = np.argmin(history["val_loss"])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history["loss"], color="tab:blue", label="Treinamento")
        ax.plot(history["val_loss"], color="tab:red", label="Validação")
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label="Melhor época")
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title(f"Objeto {OBJECT_ID} — Curva de perda")
        ax.legend()
        ax.grid(True)
        fig.savefig(f'{OUTPUT_DIR}/loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        metric_key = [k for k in history.keys() if k not in ("loss", "val_loss") and not k.startswith("val_")][0]
        val_metric_key = f"val_{metric_key}"

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history[metric_key], color="tab:blue", label="Treinamento")
        ax.plot(history[val_metric_key], color="tab:red", label="Validação")
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label="Melhor época")
        ax.set_xlabel("Época")
        ax.set_ylabel("Erro geodésico (RMSE)")
        ax.set_title(f"Objeto {OBJECT_ID} — Curva da métrica")
        ax.legend()
        ax.grid(True)
        fig.savefig(f'{OUTPUT_DIR}/metric_curve.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

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

        plt.close('all')
        gc.collect()

    print(f"End of Execution.")