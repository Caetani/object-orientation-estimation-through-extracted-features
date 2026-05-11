import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, t
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from src.utils.orientation_utils import geodesic_error
from src.models.dataset_definitions import X_cols, y_cols

OBJECT_ID  = 4
SEED       = 42
K_FOLDS    = 20
ALPHA      = 0.05   # nível de significância
MIN_POWER  = 0.75   # potência mínima desejada

PARAM_GRID = {
    'estimator__max_depth':        [3, 5, 7, 10, 12, 15, 17, 20],
    'estimator__min_samples_split':[10, 20, 50],
    'estimator__min_samples_leaf': [5, 10, 20],
    'estimator__max_features':     [0.5, 0.7, 1.0],
    'estimator__ccp_alpha':        [0.0, 0.001, 0.005, 0.01, 0.05],
}


def complexidade(params):
    """Métrica de complexidade da árvore — menor é mais simples."""
    return (
        params['estimator__max_depth']
        - params['estimator__min_samples_split'] / 10
        - params['estimator__min_samples_leaf']  / 5
    )


def calcular_potencia(scores_a, scores_b, alpha=0.05):
    """
    Calcula a potência do teste t pareado entre dois vetores de scores.
    Usa o tamanho de efeito de Cohen's d pareado.

    Parâmetros
    ----------
    scores_a, scores_b : arrays de scores por fold
    alpha              : nível de significância

    Retorna
    -------
    power   : potência estimada
    p_value : p-value do teste t pareado
    """
    n      = len(scores_a)
    diff   = scores_a - scores_b
    mean_d = diff.mean()
    std_d  = diff.std(ddof=1)

    if std_d == 0:
        return 1.0 if mean_d == 0 else 0.0, 0.0

    # Estatística t observada
    t_stat = mean_d / (std_d / np.sqrt(n))
    df     = n - 1

    # p-value bilateral
    p_value = 2 * t.sf(np.abs(t_stat), df)

    # Valor crítico para alpha bilateral
    t_crit = t.ppf(1 - alpha / 2, df)

    # Cohen's d pareado
    cohens_d = mean_d / std_d

    # Parâmetro de não-centralidade
    ncp = cohens_d * np.sqrt(n)

    # Potência: P(|T| > t_crit | ncp)
    power = t.sf(t_crit, df, loc=ncp) + t.cdf(-t_crit, df, loc=ncp)

    return power, p_value


if __name__ == '__main__':
    df = pd.read_excel('processed/splitted_train.xlsx')
    df = df[df['object_id'] == OBJECT_ID]

    train_df = df[df['set'] == 'train'].reset_index(drop=True)
    test_df  = df[df['set'] == 'test'].reset_index(drop=True)

    X_train = train_df[X_cols].values
    y_train = train_df[y_cols].values
    X_test  = test_df[X_cols].values
    y_test  = test_df[y_cols].values

    print(f'Amostras de treino: {len(X_train)} | Amostras de teste: {len(X_test)}')

    base_model = DecisionTreeRegressor(random_state=SEED)
    model      = MultiOutputRegressor(base_model)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        scoring='neg_root_mean_squared_error',
        cv=K_FOLDS,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    print('\nIniciando pesquisa em grade...')
    grid_search.fit(X_train, y_train)

    # ── Seleção do modelo mais simples com potência >= MIN_POWER ─────────────
    cv_results  = pd.DataFrame(grid_search.cv_results_)
    split_cols  = [f'split{i}_test_score' for i in range(K_FOLDS)]
    best_idx    = grid_search.best_index_
    mu_best     = cv_results.loc[best_idx, 'mean_test_score']
    sigma_best  = cv_results.loc[best_idx, 'std_test_score']
    best_scores = cv_results.loc[best_idx, split_cols].values.astype(float)

    # Filtra candidatos estatisticamente equivalentes (p > alpha) com potência >= MIN_POWER
    cv_results['complexidade'] = cv_results['params'].apply(complexidade)
    cv_results_sorted = cv_results.sort_values('complexidade').reset_index(drop=False)

    idx_selecionado = None
    for _, row in cv_results_sorted.iterrows():
        orig_idx      = row['index']
        fold_scores   = row[split_cols].values.astype(float)
        power, p_val  = calcular_potencia(best_scores, fold_scores, alpha=ALPHA)

        # Equivalente ao melhor (p > alpha) com potência suficiente
        if p_val > ALPHA and power >= MIN_POWER:
            idx_selecionado = orig_idx
            power_sel       = power
            p_val_sel       = p_val
            break

    # Fallback: usa o melhor modelo se nenhum candidato satisfizer os critérios
    if idx_selecionado is None:
        print('Nenhum candidato satisfez os critérios — usando o melhor modelo diretamente.')
        idx_selecionado = best_idx
        power_sel, p_val_sel = calcular_potencia(best_scores, best_scores)

    params_simples = cv_results.loc[idx_selecionado, 'params']
    score_simples  = cv_results.loc[idx_selecionado, 'mean_test_score']

    print(f'\nMelhor configuração (menor RMSE CV):')
    print(f'  Parâmetros: {grid_search.best_params_}')
    print(f'  RMSE CV:    {-mu_best:.6f} +/- {sigma_best:.6f}')

    print(f'\nConfiguração mais simples (p > {ALPHA}, potência >= {MIN_POWER}):')
    print(f'  Parâmetros: {params_simples}')
    print(f'  RMSE CV:    {-score_simples:.6f}')
    print(f'  p-value:    {p_val_sel:.4f}')
    print(f'  Potência:   {power_sel:.4f}')

    # ── Retreina o modelo selecionado em todo o conjunto de treino ────────────
    simple_params = {k.replace('estimator__', ''): v for k, v in params_simples.items()}
    simple_model  = MultiOutputRegressor(
        DecisionTreeRegressor(random_state=SEED, **simple_params)
    )
    simple_model.fit(X_train, y_train)

    # ── Avaliação no conjunto de teste ────────────────────────────────────────
    y_pred      = simple_model.predict(X_test)
    y_pred      = y_pred / np.linalg.norm(y_pred, axis=1, keepdims=True)
    y_test_norm = y_test / np.linalg.norm(y_test, axis=1, keepdims=True)

    errors = np.array([geodesic_error(y_test_norm[i], y_pred[i]) for i in range(len(y_test))])

    print(f'\n── Resultados no conjunto de teste (objeto {OBJECT_ID}) ──')
    print(f'Erro geodésico médio:   {errors.mean():.2f} graus')
    print(f'Erro geodésico mediano: {np.median(errors):.2f} graus')
    print(f'Erro geodésico std:     {errors.std():.2f} graus')
    print(f'Erro geodésico mínimo:  {errors.min():.2f} graus')
    print(f'Erro geodésico máximo:  {errors.max():.2f} graus')