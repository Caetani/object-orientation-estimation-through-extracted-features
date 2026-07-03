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
        'max_iter': 1_000,
        'learning_rate': 'constant'
        #'activation': 'relu',
        #'batch_size': 16,
        #'early_stopping': True,
        #'validation_fraction': 0.1,
        #'n_iter_no_change': 10,
    }

    
    """ #Param grid 1
    layers_range = [2**i for i in range(3, 8+1, 1)]
    LAYERS = [(j,) for j in layers_range] + [(i, i) for i in layers_range]
    ALPHAS = [10**i for i in range(-5, -3, 1)]
    LEARNING_RATES = [10**i for i in range(-5, -2+1, 1)]
    ACTIVATIONS = ['relu']#['relu', 'logistic', 'tanh']
    BATCH_SIZES = [2**i for i in range(3, 5+1, 1)]


    PARAM_GRID = {
        'neural_network__hidden_layer_sizes': LAYERS,
        'neural_network__alpha': ALPHAS,
        'neural_network__learning_rate_init': LEARNING_RATES,
        'neural_network__activation': ACTIVATIONS,
        'neural_network__batch_size': BATCH_SIZES,
    } """

    layers_range = [2**i for i in range(6, 9+1, 1)]
    LAYERS_1 = [(j,) for j in layers_range]
    LAYERS_2 = list(product(layers_range, repeat=2))
    LAYERS = LAYERS_1 + LAYERS_2
    ALPHAS = [10**i for i in range(-5, -3, 1)]
    LEARNING_RATES = [10**i for i in range(-3, -1+1, 1)]
    ACTIVATIONS = ['relu']
    BATCH_SIZES = [2**i for i in range(4, 5+1, 1)]


    PARAM_GRID = {
        'neural_network__hidden_layer_sizes': LAYERS,
        'neural_network__alpha': ALPHAS,
        'neural_network__learning_rate_init': LEARNING_RATES,
        'neural_network__activation': ACTIVATIONS,
        'neural_network__batch_size': BATCH_SIZES,
    }

    """ PARAM_GRID = {
        'hidden_layer_sizes': LAYERS,
        'alpha': ALPHAS,
        'learning_rate_init': LEARNING_RATES,
        'activation': ACTIVATIONS,
        'batch_size': BATCH_SIZES,
    } """
    
    original_df = pd.read_excel(f'processed/splitted_train_{SPLIT}.xlsx')
    original_df = original_df[(original_df['frame_id'] != 1277) & (original_df['frame_id'] != 1295)] # Gimbal lock (Pitch = 90% - Yaw == Roll)

    for OBJECT_ID in OBJECT_IDS:
        print(f"\n\nSeaching model configuration for object {OBJECT_ID}...")

        MODELS_DIR  = f'models/object_{OBJECT_ID}/MLPRegressor_Search_2_neural_network_{SPLIT}'
        OUTPUT_DIR = f'{MODELS_DIR}/performance'

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        df_train = original_df[(original_df['set'] == 'train') & (original_df['object_id'] == OBJECT_ID)]
        df_test = original_df[(original_df['set'] == 'test') & (original_df['object_id'] == OBJECT_ID)]

        X_train = df_train[X_cols]
        y_train = df_train[y_cols].values

        X_test = df_test[X_cols]
        y_test = df_test[y_cols].values

        """ hu_pca_cols = [f'hu_{i}' for i in range(1, 4+1, 1)]
        hu_pca_transformer = Pipeline(
            steps=[
                ('power_transformer', PowerTransformer(
                    method='yeo-johnson',
                    standardize=True
                ))
            ]
        ) """

        pipe = Pipeline([
            ("power_transformer", PowerTransformer(
                method='yeo-johnson',
                standardize=True
            )),
            ("pca", PCA(n_components=21)),
            ("scaler", StandardScaler()),
            ("neural_network", MLPRegressor(**PARAMS)),
        ])

        """ pt = PowerTransformer(method='yeo-johnson', standardize=True)
        sc = StandardScaler()

        X_train = pd.DataFrame(pt.fit_transform(X_train), columns=X_cols)
        X_test = pd.DataFrame(pt.transform(X_test), columns=X_cols) """

        """ X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X_cols)
        X_test = pd.DataFrame(sc.transform(X_test), columns=X_cols) """

        grid_search = GridSearchCV(
            estimator=pipe,
            #estimator=MLPRegressor(**PARAMS),
            param_grid=PARAM_GRID,
            cv=K_FOLDS,
            scoring=geodesic_rmse_scorer,
            n_jobs=4,
            verbose=3
        )
        grid_search.fit(X_train, y_train)

        cv_results  = pd.DataFrame(grid_search.cv_results_)
        cv_results.sort_values(by=['rank_test_score'], ascending=True, inplace=True)
        cv_results.reset_index(inplace=True)
        print(cv_results)
        cv_results.to_excel(f'{MODELS_DIR}/grid_search_results.xlsx', index=False)

        best_model = grid_search.best_estimator_
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