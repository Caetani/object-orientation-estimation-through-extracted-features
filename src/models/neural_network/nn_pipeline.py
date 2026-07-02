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
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
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
)

def build_model(meta, hidden_layer_sizes=(64, 32), activation="relu", lr=1e-3):
    n_features_in = meta["n_features_in_"]
    n_outputs = meta["n_outputs_"]

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in,)))
    for units in hidden_layer_sizes:
        model.add(keras.layers.Dense(units, activation=activation))
    model.add(keras.layers.Dense(n_outputs))  # quaternion: 4 outputs, no activation

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
    )
    return model

if __name__ == '__main__':
    print("GPU available:", tf.config.list_physical_devices('GPU'))
    K_FOLDS = 10
    OBJECT_IDS = [4] #list(np.arange(1, 16, 1))
    SPLIT = '70_30'

    original_df = pd.read_excel(f'processed/splitted_train_{SPLIT}.xlsx')
    original_df = original_df[(original_df['frame_id'] != 1277) & (original_df['frame_id'] != 1295)] # Gimbal lock (Pitch = 90% - Yaw == Roll)

    for OBJECT_ID in OBJECT_IDS:
        print(f"\n\nSeaching model configuration for object {OBJECT_ID}...")

        MODELS_DIR  = f'models/object_{OBJECT_ID}/neural_network_{SPLIT}'
        OUTPUT_DIR = f'{MODELS_DIR}/performance'

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        df_train = original_df[(original_df['set'] == 'train') & (original_df['object_id'] == OBJECT_ID)]
        df_test = original_df[(original_df['set'] == 'test') & (original_df['object_id'] == OBJECT_ID)]

        X_train = df_train[X_cols]
        y_train = df_train[y_cols].values

        X_test = df_test[X_cols]
        y_test = df_test[y_cols].values

        pipe = Pipeline([
            ("power_transformer", PowerTransformer(
                method='yeo-johnson',
                standardize=True
            )),
            ("pca", PCA(n_components=21)),
            ("scaler", StandardScaler()),
            ("neural_network", KerasRegressor(
                model=build_model,
                hidden_layer_sizes=(1024, 1024),  # e.g. (4,), (4, 4), (4, 4, 4)
                activation="relu",
                lr=1e-4,
                epochs=5000,
                batch_size=16,
                verbose=1,
            )),
        ])

        pipe.fit(X_train, y_train)

        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)

        joblib.dump(pipe, f'{MODELS_DIR}/model.pkl', compress=3)

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