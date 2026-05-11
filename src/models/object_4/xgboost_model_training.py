import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from src.utils.orientation_utils import geodesic_error
from src.models.dataset_definitions import X_cols, y_cols

OBJECT_ID = 4
SET       = 'train'
K_FOLDS   = 10
SEED      = 42

# Parâmetros da pesquisa em grade
# n_estimators e max_depth controlam capacidade do modelo
# learning_rate e subsample ajudam a evitar overfitting
# min_child_weight e gamma adicionam regularização
PARAM_GRID = {
    'estimator__n_estimators':    [100, 300, 500],
    'estimator__max_depth':       [3, 5, 7],
    'estimator__learning_rate':   [0.01, 0.05, 0.1],
    'estimator__subsample':       [0.7, 0.9],
    'estimator__min_child_weight':[3, 5],
    'estimator__gamma':           [0, 0.1],
}

if __name__ == '__main__':
    df = pd.read_excel('processed/train_extracted_features.xlsx')
    df = df[df['object_id'] == OBJECT_ID].reset_index(drop=True)

    # Aleatoriza a ordem das amostras
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    X = df[X_cols].values
    y = df[y_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    print(f'Amostras de treino: {len(X_train)} | Amostras de teste: {len(X_test)}')

    base_model = XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=SEED,
        verbosity=0,
    )
    model = MultiOutputRegressor(base_model)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        scoring='neg_root_mean_squared_error',
        cv=K_FOLDS,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    print('Iniciando pesquisa em grade...')
    grid_search.fit(X_train, y_train)

    print(f'\nMelhores hiperparâmetros: {grid_search.best_params_}')
    print(f'Melhor RMSE (CV): {-grid_search.best_score_:.6f}')

    # Avaliação no conjunto de teste
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Normaliza os quaternions preditos para norma unitária
    norms  = np.linalg.norm(y_pred, axis=1, keepdims=True)
    y_pred = y_pred / norms

    # Erro geodésico por amostra
    errors = np.array([geodesic_error(y_test[i], y_pred[i]) for i in range(len(y_test))])

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f'\n── Resultados no conjunto de teste (objeto {OBJECT_ID}) ──')
    print(f'RMSE quaternion:        {rmse:.6f}')
    print(f'Erro geodésico médio:   {errors.mean():.2f} graus')
    print(f'Erro geodésico mediano: {np.median(errors):.2f} graus')
    print(f'Erro geodésico mínimo:  {errors.min():.2f} graus')
    print(f'Erro geodésico máximo:  {errors.max():.2f} graus')
    print(f'Erro geodésico std:     {errors.std():.2f} graus')