import sys
sys.path.insert(0, ".")

import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import gc
import matplotlib.pyplot as plt
from time import time

import seaborn as sns

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


if __name__ == '__main__':
    K_FOLDS = 10
    OBJECT_IDS = [4] #list(np.arange(1, 16, 1))
    SPLIT = '70_30'

    for OBJECT_ID in OBJECT_IDS:
        print(f"\n\nSeaching model configuration for object {OBJECT_ID}...")

        MODELS_DIR  = f'models/object_{OBJECT_ID}/random_forest_{SPLIT}'
        OUTPUT_DIR = f'{MODELS_DIR}/performance'

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        model = joblib.load(f"{MODELS_DIR}/model.pkl")
        print(model)

        all_depths = np.array([tree.get_depth() for tree in model.estimators_])
        print(f"Average depth: {np.mean(all_depths)}")

        cv_results = pd.read_excel(f'{MODELS_DIR}/grid_search_results.xlsx')
        cv_results['mean_test_score'] *= -1
        sns.pointplot(data=cv_results, hue=cv_results['param_n_estimators'], 
                        y='mean_test_score', x='param_max_features')
        plt.savefig(f'{OUTPUT_DIR}/cv_results.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(cv_results)

    print(f"End of Execution.")