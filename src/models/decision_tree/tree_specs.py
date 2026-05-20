import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    OBJECT_ID = 4
    SPLIT = '70_30'
    MODEL_DIR = f'models/object_{OBJECT_ID}/decision_tree_{SPLIT}'
    
    cv_results = pd.read_excel(f"{MODEL_DIR}/grid_search_results.xlsx")
    cv_results.sort_values(by=['rank_test_score'], ascending=True, inplace=True)
    cv_results.reset_index(inplace=True)
    best_mean = cv_results.loc[0, 'mean_test_score']
    best_std = cv_results.loc[0, 'std_test_score']
    threshold = best_mean - (best_std / np.sqrt(10))
    print(f"Best mean = {best_mean}")
    #print(cv_results)

    
    
    filtered_results = cv_results[cv_results['mean_test_score'] > threshold]
    filtered_results.sort_values

    #plt.plot(np.log10(filtered_results['param_ccp_alpha'].values), 'bx')
    plt.plot(cv_results['param_ccp_alpha'].values, -cv_results['mean_test_score'].values, 'b.-')
    plt.ylabel("RMSE Médio do\nErro Geodésico (graus)")
    plt.xlabel(f"Parâmetro de Complexidade")
    plt.title("RMSE médio do erro geodésio\npara o conjunto treinamento (K=10)")
    plt.savefig(f"{MODEL_DIR}/ccp_performance_effect.png")
    #plt.show()