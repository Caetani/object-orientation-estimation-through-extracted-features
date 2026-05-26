import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
import pandas as pd

import sys
sys.path.insert(0, ".")

from src.utils.orientation_utils import *


q_labels     = [r'$q_w$', r'$q_x$', r'$q_y$', r'$q_z$']
set_colors = {
    'train': '#2196F3',
    'validation': "#00911D",
    'test': '#E53935'
}
FONT_SIZE    = 11
euler_labels = ['Roll (graus)', 'Pitch (graus)', 'Yaw (graus)']


def calculate_rmse(y_pred, y_true):
    errors = y_pred - y_true
    return np.sqrt(np.mean(errors**2))


def evaluate(model, X, y_true, label):
    y_pred = model.predict(X)
    y_pred = normalize_quarternions(y_pred)
    y_norm = normalize_quarternions(y_true)
    geodesic_errors = geodesic_error(y_norm, y_pred)
    rmse_geodesic = np.sqrt(np.mean(geodesic_errors**2))

    print(f'\n── {label} ──')
    print(f'Erro geodésico RMSE:    {rmse_geodesic:.2f} graus')
    print(f'Erro geodésico médio:   {geodesic_errors.mean():.2f} graus')
    print(f'Erro geodésico std:     {geodesic_errors.std():.2f} graus')
    print(f'Erro geodésico mínimo:  {geodesic_errors.min():.2f} graus')
    print(f'Erro geodésico máximo:  {geodesic_errors.max():.2f} graus')
    
    return y_norm, y_pred, geodesic_errors


def plot_euler_scatter(euler_true, euler_pred, label, suffix, results_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    color = set_colors[suffix]
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
    path = f'{results_dir}/scatter_euler_{suffix}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {path}')


def plot_euler_hist(euler_true, euler_pred, label, suffix, results_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    color = set_colors[suffix]
    for i, (ax, el) in enumerate(zip(axes, euler_labels)):
        # Erro circular — limitado ao intervalo [-180°, 180°]
        err  = np.rad2deg(angular_difference(euler_pred[:, i], euler_true[:, i])) #((euler_pred[:, i] - euler_true[:, i]) + 180) % 360 - 180
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
    path = f'{results_dir}/hist_euler_{suffix}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {path}')


def plot_accuracy_threshold(errors_tr, errors_te, results_dir):
    """Taxa de acerto em função do threshold do erro geodésico."""
    thresholds  = np.arange(1, 31)
    accuracy_tr = [np.mean(errors_tr <= t) * 100 for t in thresholds]
    accuracy_te = [np.mean(errors_te <= t) * 100 for t in thresholds]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, accuracy_tr, color=set_colors['train'], linewidth=2,
            marker='o', markersize=4, label='Treinamento')
    ax.plot(thresholds, accuracy_te, color=set_colors['test'],  linewidth=2,
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
    path = f'{results_dir}/accuracy_threshold.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {path}')


def plot_hist_components(y_true, y_pred, label, suffix, results_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    color = set_colors[suffix]
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
    path = f'{results_dir}/hist_components_{suffix}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {path}')


def plot_hist_geodesic(geodesic_errors, results_dir, suffix, label):
    rmse = np.sqrt(np.mean(geodesic_errors**2))
    fig, ax = plt.subplots(figsize=(7, 4))
    color = set_colors[suffix]
    ax.hist(geodesic_errors, bins=30, color=color, alpha=0.6,
            edgecolor='white', linewidth=0.4, label=label)
    ax.axvline(geodesic_errors.mean(), color=color, linewidth=1.2, linestyle='--',
               label=f'Média: {geodesic_errors.mean():.1f}°  STD: {geodesic_errors.std():.1f}°  RMSE: {rmse:.1f}°')
    ax.set_xlabel('Erro geodésico angular (graus)')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição do Erro Geodésico Angular')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = f'{results_dir}/{suffix}_hist_geodesic.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {path}')


def geodesic_rmse_score_func(y, y_pred, **kwargs):
    y = normalize_quarternions(y)
    y_pred = normalize_quarternions(y_pred)
    return np.sqrt(np.mean(geodesic_error(q_true=y, q_pred=y_pred)**2))


geodesic_rmse_scorer = make_scorer(geodesic_rmse_score_func, greater_is_better=False)

def geodesic_rmse_oob(y, y_pred):
    y = normalize_quarternions(y)
    y_pred = normalize_quarternions(y_pred)
    return -np.sqrt(np.mean(geodesic_error(q_true=y, q_pred=y_pred)**2))


def feature_importance(model, columns, results_dir):
    importances = pd.Series(model.feature_importances_, index=columns).sort_values()

    ax = importances.plot(kind='barh', figsize=(7, 5))
    plt.title("Importância da Característica (MDI)")

    for container in ax.containers:
        ax.bar_label(container, fmt=lambda x: f'{x * 100:.1f}%', padding=3)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/feature_importance.png")
