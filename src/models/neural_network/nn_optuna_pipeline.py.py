import sys
sys.path.insert(0, ".")
import os
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

    # Order matters: load dependencies before the libs that depend on them
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

import pandas as pd
import numpy as np
import joblib
import gc
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scikeras.wrappers import KerasRegressor
from tensorflow import keras
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU configurada: {gpus[0].name}")
else:
    print("AVISO: nenhuma GPU detectada, rodando em CPU.")

from src.models.dataset_definitions import X_cols, y_cols
from src.utils.model_evaluation_utils import (
    evaluate,
    plot_euler_hist,
    plot_accuracy_threshold,
    plot_hist_geodesic,
    quaternion_to_euler,
    geodesic_rmse_scorer,
    geodesic_rmse_oob,
    geodesic_rmse_score_func,
)


def geodesic_rmse_keras(y_true, y_pred):
    # Normalize quaternions to unit length
    y_true = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred = tf.nn.l2_normalize(y_pred, axis=-1)
    # Dot product — take abs to handle antipodal equivalence (q == -q)
    dot = tf.abs(tf.reduce_sum(y_true * y_pred, axis=-1))
    # Clamp to [-1, 1] for numerical safety before acos
    dot = tf.clip_by_value(dot, 0.0, 1.0)
    # Geodesic error in radians: 2 * arccos(|q1 · q2|)
    error = 2.0 * tf.acos(dot)
    return tf.sqrt(tf.reduce_mean(error ** 2))
geodesic_rmse_keras.__name__ = "geodesic_rmse"

def build_model(meta, hidden_layer_sizes=(64, 32), activation="relu", lr=1e-3, dropout_rate=0.0):
    n_features_in = meta["n_features_in_"]
    n_outputs = meta["n_outputs_"]

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in,)))
    for units in hidden_layer_sizes:
        model.add(keras.layers.Dense(units, activation=activation))
        if dropout_rate > 0.0:
            model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(n_outputs))  # quaternion: 4 outputs, no activation

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[geodesic_rmse_keras],
    )
    return model

if __name__ == '__main__':
    print("GPU available:", tf.config.list_physical_devices('GPU'))

    N_TRIALS   = 50
    OBJECT_IDS = [4] #list(np.arange(1, 16, 1))
    SPLIT      = '70_30'

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial noise

    original_df = pd.read_excel(f'processed/splitted_train_{SPLIT}.xlsx')
    original_df = original_df[(original_df['frame_id'] != 1277) & (original_df['frame_id'] != 1295)] # Gimbal lock (Pitch = 90% - Yaw == Roll)

    for OBJECT_ID in OBJECT_IDS:
        print(f"\n\nSearching model configuration for object {OBJECT_ID}...")

        MODELS_DIR = f'Search_3_model_optimization/models/object_{OBJECT_ID}/neural_network_{SPLIT}'
        OUTPUT_DIR = f'{MODELS_DIR}/performance'

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        df_train = original_df[(original_df['set'] == 'train') & (original_df['object_id'] == OBJECT_ID)]
        df_test  = original_df[(original_df['set'] == 'test')  & (original_df['object_id'] == OBJECT_ID)]

        X_train = df_train[X_cols]
        y_train = df_train[y_cols].values

        X_test = df_test[X_cols]
        y_test = df_test[y_cols].values

        # ── Optuna objective ──────────────────────────────────────────────────────────
        def objective(trial):
            lr           = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            n_layers     = trial.suggest_int(  "n_layers", 1, 2)
            units        = trial.suggest_int("units",  128, 512)
            activation   = trial.suggest_categorical("activation", ["relu"])
            batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])
            dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

            hidden_layer_sizes = tuple([units] * n_layers)

            pca_hu_moments_cols = [f"hu_{i}" for i in range(1, 4+1, 1)]
            pca_hu_moments_pipe = Pipeline(
                steps=[
                    ('power_transformer', PowerTransformer(
                        method='yeo-johnson',
                        standardize=True
                    )),
                    ('pca', PCA(n_components=2)),
                    ('scaling', StandardScaler()),
                ]
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ('pca_hu_moments', pca_hu_moments_pipe, pca_hu_moments_cols)
                ], remainder=PowerTransformer(
                    method='yeo-johnson',
                    standardize=True
                )
            )

            pipe = Pipeline([
                ('preprocessing', preprocessor),
                ("neural_network", KerasRegressor(
                    model=build_model,
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    lr=lr,
                    dropout_rate=dropout_rate,
                    epochs=10_000,
                    batch_size=batch_size,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=500,
                            restore_best_weights=True,
                        )
                    ],
                )),
            ])

            try:
                pipe.fit(X_train, y_train)
                history = pipe.named_steps["neural_network"].history_
                return min(history["val_loss"])  # best epoch's val_loss (EarlyStopping)
            except Exception as e:
                print(f"  WARNING: trial {trial.number} failed — {e}")
                return float("inf")

        # ── Run search ────────────────────────────────────────────────────────────────
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            study_name=f"object_{OBJECT_ID}",
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

        best = study.best_params
        print(f"\n  Best trial   : {study.best_trial.number}")
        print(f"  Best val loss: {study.best_value:.6f}")
        print(f"  Best params  : {best}")

        # ── Optuna plots ──────────────────────────────────────────────────────────────
        try:
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
            )
            ax = plot_optimization_history(study)
            ax.get_figure().savefig(f'{OUTPUT_DIR}/optuna_history.png', dpi=150, bbox_inches='tight')
            plt.close('all')

            ax = plot_param_importances(study)
            ax.get_figure().savefig(f'{OUTPUT_DIR}/optuna_importances.png', dpi=150, bbox_inches='tight')
            plt.close('all')

            ax = plot_parallel_coordinate(study)
            ax.get_figure().savefig(f'{OUTPUT_DIR}/optuna_parallel.png', dpi=150, bbox_inches='tight')
            plt.close('all')
        except Exception as e:
            print(f"  WARNING: Optuna plots failed — {e}")

        # ── Final model with best params, trained on full X_train ─────────────────────
        best_hidden_layer_sizes = tuple([best["units"]] * best["n_layers"])

        print(f"\n  Retraining final model on full training set...")
        pca_hu_moments_cols = [f"hu_{i}" for i in range(1, 4+1, 1)]
        pca_hu_moments_pipe = Pipeline(
            steps=[
                ('power_transformer', PowerTransformer(
                    method='yeo-johnson',
                    standardize=True
                )),
                ('pca', PCA(n_components=2)),
                ('scaling', StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ('pca_hu_moments', pca_hu_moments_pipe, pca_hu_moments_cols)
            ], remainder=PowerTransformer(
                method='yeo-johnson',
                standardize=True
            )
        )
        pipe = Pipeline([
                ('preprocessing', preprocessor),
                ("neural_network", KerasRegressor(
                model=build_model,
                hidden_layer_sizes=best_hidden_layer_sizes,
                activation=best["activation"],
                lr=best["lr"],
                dropout_rate=best["dropout_rate"],
                epochs=10_000,
                batch_size=best["batch_size"],
                verbose=1,
                validation_split=0.1,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=500,
                        restore_best_weights=True,
                    )
                ],
            )),
        ])

        pipe.fit(X_train, y_train)
        print("History keys:", list(pipe.named_steps["neural_network"].history_.keys()))

        y_train_pred = pipe.predict(X_train)
        y_test_pred  = pipe.predict(X_test)

        history = pipe.named_steps["neural_network"].history_

        # Loss curve
        fig, ax = plt.subplots()
        ax.plot(history["loss"],     label="Train loss")
        ax.plot(history["val_loss"], label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title(f"Object {OBJECT_ID} — Loss")
        ax.legend()
        fig.savefig(f'{OUTPUT_DIR}/loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Metric curve
        metric_keys = [k for k in history.keys() if k not in ("loss", "val_loss") and not k.startswith("val_")]
        if metric_keys:
            metric_key     = metric_keys[0]
            val_metric_key = f"val_{metric_key}"
            fig, ax = plt.subplots()
            ax.plot(history[metric_key],     label="Train metric")
            ax.plot(history[val_metric_key], label="Val metric")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Geodesic RMSE")
            ax.set_title(f"Object {OBJECT_ID} — Metric")
            ax.legend()
            fig.savefig(f'{OUTPUT_DIR}/metric_curve.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            print("  WARNING: No metric found in history — only loss was tracked.")

        joblib.dump(pipe, f'{MODELS_DIR}/model.pkl', compress=3)
        joblib.dump(study, f'{MODELS_DIR}/optuna_study.pkl')

        y_train_norm, y_pred_train, errors_train = evaluate(pipe, X_train, y_train, "Training set")
        y_test_norm,  y_pred_test,  errors_test  = evaluate(pipe, X_test, y_test, "Testing set")

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