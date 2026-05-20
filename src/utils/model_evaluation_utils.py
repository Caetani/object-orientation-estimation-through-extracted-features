import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from scipy import stats

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
    y_pred   = model.predict(X)
    print(f"Shape do y_pred: {np.shape(y_pred), type(y_pred)}")
    y_pred   = normalize_quarternions(y_pred)
    y_norm   = normalize_quarternions(y_true)
    geodesic_errors   = geodesic_error(y_norm, y_pred)
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


def old_plot_hist_geodesic(errors_tr, errors_te, results_dir):
    rmse_tr = np.sqrt(np.mean(errors_tr**2))
    rmse_te = np.sqrt(np.mean(errors_te**2))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors_tr, bins=30, color=set_colors['train'], alpha=0.6,
            edgecolor='white', linewidth=0.4, label='Treinamento')
    ax.hist(errors_te, bins=30, color=set_colors['test'],  alpha=0.6,
            edgecolor='white', linewidth=0.4, label='Teste')
    ax.axvline(errors_tr.mean(), color=set_colors['train'], linewidth=1.2, linestyle='--',
               label=f'Média treino: {errors_tr.mean():.1f}°  STD: {errors_tr.std():.1f}°  RMSE: {rmse_tr:.1f}°')
    ax.axvline(errors_te.mean(), color=set_colors['test'],  linewidth=1.2, linestyle='--',
               label=f'Média teste: {errors_te.mean():.1f}°  STD: {errors_te.std():.1f}°  RMSE: {rmse_te:.1f}°')
    ax.set_xlabel('Erro geodésico angular (graus)')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição do Erro Geodésico Angular')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = f'{results_dir}/hist_geodesic.png'
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
    return np.sqrt(np.mean(geodesic_error(q_true=y, q_pred=y_pred)**2))


geodesic_rmse_scorer = make_scorer(geodesic_rmse_score_func, greater_is_better=False)
 
def ttest_two_sample(
    mean1: float,
    mean2: float,
    std1: float,
    std2: float,
    n1: int,
    n2: int,
    alpha: float = 0.05,
    test_type: str = "bilateral",
) -> dict:
    """
    Teste t de Welch para duas amostras independentes com variâncias distintas.
 
    Retorna
    -------
    dict com:
        t_stat   : estatística t amostral
        df       : graus de liberdade (Welch-Satterthwaite)
        p_value  : valor p do teste
        t_crit   : valor(es) crítico(s) de t
        power    : potência do teste (1 - β)
        beta     : probabilidade de erro tipo II
        reject   : bool — rejeita H₀ ao nível alpha?
    """
    test_type = test_type.lower()
    if test_type not in ("lower", "greater", "bilateral"):
        raise ValueError("test_type deve ser 'lower', 'greater' ou 'bilateral'.")
 
    # ------------------------------------------------------------------ #
    # 1. Estatística t e graus de liberdade (aproximação de Welch)        #
    # ------------------------------------------------------------------ #
    se = np.sqrt(std1**2 / n1 + std2**2 / n2)   # erro padrão da diferença
    t_stat = (mean1 - mean2) / se
 
    # Graus de liberdade de Welch-Satterthwaite
    num   = (std1**2 / n1 + std2**2 / n2) ** 2
    denom = (std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1)
    df    = num / denom
 
    t_dist = stats.t(df)
 
    # ------------------------------------------------------------------ #
    # 2. Valor P                                                          #
    # ------------------------------------------------------------------ #
    if test_type == "lower":
        # H₁: µ₁ < µ₂  →  rejeita quando t_stat << 0
        p_value = t_dist.cdf(t_stat)
    elif test_type == "greater":
        # H₁: µ₁ > µ₂  →  rejeita quando t_stat >> 0
        p_value = t_dist.sf(t_stat)          # sf = 1 − cdf
    else:
        # bilateral  →  rejeita nas duas caudas
        p_value = 2 * t_dist.sf(abs(t_stat))
 
    # ------------------------------------------------------------------ #
    # 3. Valor(es) crítico(s)                                             #
    # ------------------------------------------------------------------ #
    if test_type == "lower":
        t_crit = t_dist.ppf(alpha)           # negativo
    elif test_type == "greater":
        t_crit = t_dist.ppf(1 - alpha)      # positivo
    else:
        t_crit = t_dist.ppf(1 - alpha / 2)  # valor absoluto; rejeita em ±t_crit
 
    # ------------------------------------------------------------------ #
    # 4. Potência — Abordagem 1 (t central recentrada)                    #
    #                                                                     #
    # O deslocamento observado em unidades de erro padrão é t_stat.       #
    # Para cada tipo de teste, c* = limite crítico deslocado pela         #
    # diferença de médias observada.                                      #
    #                                                                     #
    # Sob H₁ verdadeiro (diferença = mean1 − mean2):                      #
    #   T' = (X̄₁ − X̄₂ − (mean1 − mean2)) / se  ~  t(df)                 #
    # Rejeição ocorre quando T cai na região crítica original, o que      #
    # equivale a T' cruzar o limite deslocado por t_stat.                 #
    # ------------------------------------------------------------------ #
    if test_type == "lower":
        # região de rejeição: T < t_crit
        # sob H₁: P(T' < t_crit − t_stat)
        c_star  = t_crit - t_stat
        power   = t_dist.cdf(c_star)
        beta    = 1 - power
 
    elif test_type == "greater":
        # região de rejeição: T > t_crit
        # sob H₁: P(T' > t_crit − t_stat)
        c_star  = t_crit - t_stat
        power   = t_dist.sf(c_star)
        beta    = 1 - power
 
    else:
        # bilateral: região de rejeição T < −t_crit  ou  T > +t_crit
        # sob H₁:
        #   poder = P(T' < −t_crit − t_stat) + P(T' > +t_crit − t_stat)
        c_left  = -t_crit - t_stat
        c_right =  t_crit - t_stat
        power   = t_dist.cdf(c_left) + t_dist.sf(c_right)
        beta    = 1 - power
 
    reject = p_value <= alpha
 
    return {
        "t_stat" : round(t_stat,  4),
        "df"     : round(df,      2),
        "p_value": round(p_value, 6),
        "t_crit" : round(t_crit,  4),
        "power"  : round(power,   6),
        "beta"   : round(beta,    6),
        "reject" : reject,
    }
