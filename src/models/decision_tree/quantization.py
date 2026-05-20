import pandas as pd
import numpy as np
import joblib

if __name__ == '__main__':
    OBJECT_ID = 4
    SPLIT = '70_30'
    MODEL_DIR = f'models/object_{OBJECT_ID}/decision_tree_{SPLIT}'
    
    model = joblib.load(f"{MODEL_DIR}/model.pkl")

    print(model)