import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from src.utils.orientation_utils import geodesic_error
from sklearn.model_selection import train_test_split
from src.models.dataset_definitions import X_cols, y_cols


OBJECT_ID = 4
SEED      = 12

BEST_PARAMS = {
    'gamma':           0,
    'learning_rate':   0.01,
    'max_depth':       7,
    'min_child_weight':3,
    'n_estimators':    500,
    'subsample':       0.9,
}


def print_metricas(errors, y_test, y_pred, label=''):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    if label:
        print(f'\n── {label} ──')
    print(f'RMSE quaternion:        {rmse:.6f}')
    print(f'Erro geodésico médio:   {errors.mean():.2f} graus')
    print(f'Erro geodésico mediano: {np.median(errors):.2f} graus')
    print(f'Erro geodésico mínimo:  {errors.min():.2f} graus')
    print(f'Erro geodésico máximo:  {errors.max():.2f} graus')
    print(f'Erro geodésico std:     {errors.std():.2f} graus')

    print(f'\n── Erro por componente do quaternion ──')
    for i, col in enumerate(y_cols):
        erros_comp = np.abs(y_test[:, i] - y_pred[:, i])
        print(f'  {col}: MAE={erros_comp.mean():.6f}  std={erros_comp.std():.6f}  max={erros_comp.max():.6f}')


def plot_resultados(errors, y_test, y_pred, object_id):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'XGBoost — Objeto {object_id} — Avaliação no Test Set', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Histograma do erro geodésico
    ax0 = fig.add_subplot(gs[0, :])
    ax0.hist(errors, bins=40, color='steelblue', edgecolor='white', linewidth=0.5)
    ax0.axvline(errors.mean(),   color='red',    linestyle='--', linewidth=1.5, label=f'Média: {errors.mean():.2f}°')
    ax0.axvline(np.median(errors), color='orange', linestyle='--', linewidth=1.5, label=f'Mediana: {np.median(errors):.2f}°')
    ax0.set_xlabel('Erro geodésico (graus)')
    ax0.set_ylabel('Frequência')
    ax0.set_title('Distribuição do Erro Geodésico Angular')
    ax0.legend()

    # Erro por componente do quaternion
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    for i, (col, color) in enumerate(zip(y_cols, colors)):
        ax = fig.add_subplot(gs[1, i] if i < 3 else gs[1, 2])
        erros_comp = y_test[:, i] - y_pred[:, i]
        ax.hist(erros_comp, bins=30, color=color, edgecolor='white', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=1.0)
        ax.axvline(erros_comp.mean(), color='red', linestyle='--', linewidth=1.2,
                   label=f'Média: {erros_comp.mean():.4f}')
        ax.set_xlabel(f'Erro {col}')
        ax.set_ylabel('Frequência')
        ax.set_title(f'Componente {col}')
        ax.legend(fontsize=8)

    plt.savefig(f'processed/xgboost_obj{object_id}_resultados.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'\nGráfico salvo em: processed/xgboost_obj{object_id}_resultados.png')


if __name__ == '__main__':

    # ── Carrega dados ─────────────────────────────────────────────────────────
    df_train = pd.read_excel('processed/train_extracted_features.xlsx')
    df_train = df_train[df_train['object_id'] == OBJECT_ID].reset_index(drop=True)
    df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)

    #df_test = pd.read_excel('processed/test_extracted_features.xlsx')
    df_train, df_test = train_test_split(df_train, test_size=0.2) 
    df_test = df_test[df_test['object_id'] == OBJECT_ID].reset_index(drop=True)

    X_train = df_train[X_cols].values
    y_train = df_train[y_cols].values
    X_test  = df_test[X_cols].values
    y_test  = df_test[y_cols].values

    print(f'Amostras de treino: {len(X_train)} | Amostras de teste: {len(X_test)}')

    # ── Treina com todos os dados de treino ───────────────────────────────────
    base_model = XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=SEED,
        verbosity=0,
        **BEST_PARAMS,
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    # ── Avalia no test set ────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    norms  = np.linalg.norm(y_pred, axis=1, keepdims=True)
    y_pred = y_pred / norms

    errors = np.array([geodesic_error(y_test[i], y_pred[i]) for i in range(len(y_test))])

    print_metricas(errors, y_test, y_pred, label=f'Resultados no Test Set — Objeto {OBJECT_ID}')
    plot_resultados(errors, y_test, y_pred, OBJECT_ID)