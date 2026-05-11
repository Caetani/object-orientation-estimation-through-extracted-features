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
MODELS_DIR = 'models/linear_regression'
DATA_PATH  = 'processed/splitted_train.xlsx'


def build_pipeline() -> TransformedTargetRegressor:
    # Etapa 1 — PCA nas colunas hu, passthrough nas demais
    col_transformer = ColumnTransformer(
        transformers=[
            ('pca_hu',      PCA(n_components=2), hu_cols),
            ('passthrough', 'passthrough',        passthrough_cols),
        ]
    )

    # Etapa 2 — PowerTransformer em todo X (PCs + passthrough)
    # Etapa 3 — Regressão linear múltipla
    pipeline_X = Pipeline([
        ('col_transformer', col_transformer),
        ('power_transform',  PowerTransformer(method='yeo-johnson', standardize=True)),
        ('regressor',        MultiOutputRegressor(LinearRegression())),
    ])

    # Encapsula com TransformedTargetRegressor para padronizar y com StandardScaler
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

    train_df = df[df['set'] == 'train'].reset_index(drop=True)
    test_df  = df[df['set'] == 'test'].reset_index(drop=True)

    X_train = train_df[X_cols]
    y_train = train_df[y_cols].values
    X_test  = test_df[X_cols]
    y_test  = test_df[y_cols].values

    print(f'Amostras de treino: {len(X_train)} | Amostras de teste: {len(X_test)}')

    # ── Treina pipeline ───────────────────────────────────────────────────────
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # ── Avalia ────────────────────────────────────────────────────────────────
    from src.utils.orientation_utils import geodesic_error, quaternion_to_euler
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    RESULTS_DIR = 'results/linear_regression'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def avaliar(X, y_true, label):
        y_pred  = pipeline.predict(X)
        y_pred  = y_pred / np.linalg.norm(y_pred, axis=1, keepdims=True)
        y_norm  = y_true / np.linalg.norm(y_true, axis=1, keepdims=True)
        errors  = np.array([geodesic_error(y_norm[i], y_pred[i]) for i in range(len(y_true))])
        rmse_geo = np.sqrt(np.mean(errors**2))
        print(f'\n── {label} ──')
        print(f'Erro geodésico médio:   {errors.mean():.2f} graus')
        print(f'Erro geodésico mediano: {np.median(errors):.2f} graus')
        print(f'Erro geodésico std:     {errors.std():.2f} graus')
        print(f'Erro geodésico RMSE:    {rmse_geo:.2f} graus')
        print(f'Erro geodésico mínimo:  {errors.min():.2f} graus')
        print(f'Erro geodésico máximo:  {errors.max():.2f} graus')
        return y_norm, y_pred, errors

    y_train_norm, y_pred_train, errors_train = avaliar(X_train, y_train, 'Conjunto de Treinamento')
    y_test_norm,  y_pred_test,  errors_test  = avaliar(X_test,  y_test,  'Conjunto de Teste')

    q_labels      = [r'$q_w$', r'$q_x$', r'$q_y$', r'$q_z$']
    colors_train  = '#2196F3'
    colors_test   = '#E53935'
    FONT_SIZE     = 11

    plt.rcParams.update({'font.size': FONT_SIZE})

    def plot_scatter(y_true, y_pred, label, suffix):
        """Dispersão q_pred vs q_true por componente."""
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        color = colors_train if suffix == 'train' else colors_test
        for i, (ax, ql) in enumerate(zip(axes, q_labels)):
            ax.scatter(y_true[:, i], y_pred[:, i], s=8, alpha=0.5, color=color)
            lim = [-1.1, 1.1]
            ax.plot(lim, lim, 'k--', linewidth=0.8)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel(f'Valor real {ql}')
            ax.set_ylabel(f'Valor predito {ql}')
            ax.set_aspect('equal')
        fig.suptitle(f'Predito vs. Real por Componente do Quaternion — {label}', fontsize=FONT_SIZE)
        plt.tight_layout()
        path = f'{RESULTS_DIR}/scatter_components_{suffix}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Salvo: {path}')

    def plot_hist_components(y_true, y_pred, label, suffix):
        """Histograma do erro por componente do quaternion — layout 2x2."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        color = colors_train if suffix == 'train' else colors_test
        for i, (ax, ql) in enumerate(zip(axes, q_labels)):
            err  = y_pred[:, i] - y_true[:, i]
            rmse = np.sqrt(np.mean(err**2))
            ax.hist(err, bins=30, color=color, edgecolor='white', linewidth=0.4)
            ax.axvline(0,          color='black', linewidth=0.8, linestyle='--')
            ax.axvline(err.mean(), color='red',   linewidth=1.0, linestyle='-',
                       label=f'Média: {err.mean():.3f}\nDesvio padrão: {err.std():.3f}\nRMSE: {rmse:.3f}')
            ax.set_xlabel(f'Erro {ql}')
            ax.set_ylabel('Frequência')
            ax.legend(fontsize=9)
        fig.suptitle(f'Distribuição do Erro por Componente do Quaternion — {label}', fontsize=FONT_SIZE)
        plt.tight_layout()
        path = f'{RESULTS_DIR}/hist_components_{suffix}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Salvo: {path}')

    def plot_hist_geodesic(errors_tr, errors_te):
        """Histograma do erro geodésico — treino e teste sobrepostos."""
        rmse_tr = np.sqrt(np.mean(errors_tr**2))
        rmse_te = np.sqrt(np.mean(errors_te**2))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(errors_tr, bins=30, color=colors_train, alpha=0.6,
                edgecolor='white', linewidth=0.4, label='Treinamento')
        ax.hist(errors_te, bins=30, color=colors_test,  alpha=0.6,
                edgecolor='white', linewidth=0.4, label='Teste')
        ax.axvline(errors_tr.mean(), color=colors_train, linewidth=1.2, linestyle='--',
                   label=f'Média treino: {errors_tr.mean():.1f}°  STD: {errors_tr.std():.1f}°  RMSE: {rmse_tr:.1f}°')
        ax.axvline(errors_te.mean(), color=colors_test,  linewidth=1.2, linestyle='--',
                   label=f'Média teste: {errors_te.mean():.1f}°  STD: {errors_te.std():.1f}°  RMSE: {rmse_te:.1f}°')
        ax.set_xlabel('Erro geodésico angular (graus)')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição do Erro Geodésico Angular')
        ax.legend(fontsize=9)
        plt.tight_layout()
        path = f'{RESULTS_DIR}/hist_geodesic.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Salvo: {path}')

    plot_scatter(y_train_norm, y_pred_train, 'Treinamento', 'train')
    plot_scatter(y_test_norm,  y_pred_test,  'Teste',       'test')
    plot_hist_components(y_train_norm, y_pred_train, 'Treinamento', 'train')
    plot_hist_components(y_test_norm,  y_pred_test,  'Teste',       'test')
    plot_hist_geodesic(errors_train, errors_test)

    # ── Converte quaternions para ângulos de Euler (graus) ────────────────────
    def to_euler_deg(y_q):
        """Converte array de quaternions para array de ângulos de Euler em graus."""
        return np.array([np.rad2deg(quaternion_to_euler(q)) for q in y_q])

    euler_labels = ['Roll (graus)', 'Pitch (graus)', 'Yaw (graus)']
    euler_short  = ['roll', 'pitch', 'yaw']

    euler_train_true = to_euler_deg(y_train_norm)
    euler_train_pred = to_euler_deg(y_pred_train)
    euler_test_true  = to_euler_deg(y_test_norm)
    euler_test_pred  = to_euler_deg(y_pred_test)

    def plot_euler_scatter(euler_true, euler_pred, label, suffix):
        """Dispersão euler_pred vs euler_true para roll, pitch e yaw."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        color = colors_train if suffix == 'train' else colors_test
        for i, (ax, el) in enumerate(zip(axes, euler_labels)):
            ax.scatter(euler_true[:, i], euler_pred[:, i], s=8, alpha=0.5, color=color)
            lim = [-185, 185]
            ax.plot(lim, lim, 'k--', linewidth=0.8)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel(f'Real — {el}')
            ax.set_ylabel(f'Predito — {el}')
            ax.set_aspect('equal')
        fig.suptitle(f'Predito vs. Real — Ângulos de Euler — {label}', fontsize=FONT_SIZE)
        plt.tight_layout()
        path = f'{RESULTS_DIR}/scatter_euler_{suffix}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Salvo: {path}')

    def plot_euler_hist(euler_true, euler_pred, label, suffix):
        """Histograma do erro em roll, pitch e yaw — layout 1x3."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        color = colors_train if suffix == 'train' else colors_test
        for i, (ax, el) in enumerate(zip(axes, euler_labels)):
            err  = euler_pred[:, i] - euler_true[:, i]
            rmse = np.sqrt(np.mean(err**2))
            ax.hist(err, bins=30, color=color, edgecolor='white', linewidth=0.4)
            ax.axvline(0,          color='black', linewidth=0.8, linestyle='--')
            ax.axvline(err.mean(), color='red',   linewidth=1.0, linestyle='-',
                       label=f'Média: {err.mean():.2f}°\nDesvio padrão: {err.std():.2f}°\nRMSE: {rmse:.2f}°')
            ax.set_xlabel(f'Erro — {el}')
            ax.set_ylabel('Frequência')
            ax.legend(fontsize=9)
        fig.suptitle(f'Distribuição do Erro — Ângulos de Euler — {label}', fontsize=FONT_SIZE)
        plt.tight_layout()
        path = f'{RESULTS_DIR}/hist_euler_{suffix}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Salvo: {path}')

    plot_euler_scatter(euler_train_true, euler_train_pred, 'Treinamento', 'train')
    plot_euler_scatter(euler_test_true,  euler_test_pred,  'Teste',       'test')
    plot_euler_hist(euler_train_true, euler_train_pred, 'Treinamento', 'train')
    plot_euler_hist(euler_test_true,  euler_test_pred,  'Teste',       'test')

    # ── Taxa de acerto por threshold geodésico ────────────────────────────────
    def plot_accuracy_threshold(errors_tr, errors_te):
        """Taxa de acerto em função do threshold do erro geodésico."""
        thresholds   = np.arange(1, 31)
        accuracy_tr  = [np.mean(errors_tr <= t) * 100 for t in thresholds]
        accuracy_te  = [np.mean(errors_te <= t) * 100 for t in thresholds]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(thresholds, accuracy_tr, color=colors_train, linewidth=2,
                marker='o', markersize=4, label='Treinamento')
        ax.plot(thresholds, accuracy_te, color=colors_test,  linewidth=2,
                marker='o', markersize=4, label='Teste')
        ax.set_xlabel('Threshold do erro geodésico (graus)')
        ax.set_ylabel('Taxa de acerto (%)')
        ax.set_title('Taxa de Acerto em Função do Threshold Geodésico')
        ax.set_xlim(1, 30)
        ax.set_ylim(0, 100)
        ax.set_xticks(thresholds)
        ax.grid(True, linewidth=0.4, alpha=0.6)
        ax.legend(fontsize=9)
        plt.tight_layout()
        path = f'{RESULTS_DIR}/accuracy_threshold.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Salvo: {path}')

    plot_accuracy_threshold(errors_train, errors_test)

    # ── Salva pipeline ────────────────────────────────────────────────────────
    joblib.dump(pipeline, f'{MODELS_DIR}/pipeline_obj{OBJECT_ID}.pkl')
    print(f'\nPipeline salva em: {MODELS_DIR}/pipeline_obj{OBJECT_ID}.pkl')