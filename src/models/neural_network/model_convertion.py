import sys
sys.path.insert(0, ".")

import emlearn
import joblib
import os

from src.utils.model_conversion_utils import convert_model

if __name__ == '__main__':
    OBJECT_ID = 4
    SPLIT = '70_30'

    MODELS_DIR  = f'free_run_nn_2/models/object_{OBJECT_ID}/neural_network_{SPLIT}'

    os.makedirs(MODELS_DIR, exist_ok=True)

    model = joblib.load(f'{MODELS_DIR}/model.pkl')

    cmodel = emlearn.convert(estimator=model, kind='MLPRegressor', method='inline', dtype='float')
    cmodel.save(name=f'model', file=f'{MODELS_DIR}/model.h')