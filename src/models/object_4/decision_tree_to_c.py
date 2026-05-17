import sys
sys.path.insert(0, ".")

import emlearn
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.models.dataset_definitions import X_cols, y_cols
from src.utils.model_evaluation_utils import evaluate

OBJECT_ID   = 4
SEED        = 42
MODELS_DIR  = 'models/decision_tree'
DATA_PATH   = 'processed/splitted_train.xlsx'
OUTPUT_DIR = 'results/decision_tree'

BEST_PARAMS = {
    'ccp_alpha':         0,
    'max_depth':         15,
    'max_features':      1.0,
    'min_samples_leaf':  5,
    'min_samples_split': 10,
}

if __name__ == '__main__':
    df = pd.read_excel('processed/splitted_train.xlsx')
    df = df[df['object_id'] == OBJECT_ID]
    df_train = df[df['set'] == 'train']
    df_test = df[df['set'] == 'test']

    X_train = df_train[X_cols]
    y_train = df_train[y_cols].values

    X_test = df_test[X_cols]
    y_test = df_test[y_cols].values

    model = DecisionTreeRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    y_train_norm, y_pred_train, errors_train = evaluate(model, X_train, y_train, "Training set")
    y_test_norm,  y_pred_test,  errors_test  = evaluate(model, X_test, y_test, "Testing set")

