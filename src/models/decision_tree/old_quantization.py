import pandas as pd
import numpy as np
import joblib

if __name__ == '__main__':
    OBJECT_ID = 4
    SPLIT = '70_30'
    MODEL_DIR = f'models/object_{OBJECT_ID}/decision_tree_{SPLIT}'
    MODELS_DIR  = f'models/object_{OBJECT_ID}/decision_tree_{SPLIT}'
    OUTPUT_DIR = f'{MODELS_DIR}/quantization'

    original_df = pd.read_excel(f'processed/splitted_train_{SPLIT}.xlsx')
    original_df = original_df[(original_df['frame_id'] != 1277) & (original_df['frame_id'] != 1295)] # Gimbal lock (Pitch = 90% - Yaw == Roll)

    model = joblib.load(f"{MODEL_DIR}/model.pkl")

    print(model)


    num_bits_arr = []
    train_accuracy_arr = []
    test_accuracy_arr = []

    for num_bits in range(2, 16+1, 1):
        num_bits_arr.append(num_bits)
        n_levels = 2**num_bits - 1

        X_train_qt = X_train*n_levels
        X_test_qt = X_test*n_levels

        X_train_qt = np.round(X_train_qt)
        X_train_qt = X_train_qt.astype('int')


        X_test_qt = np.round(X_test_qt)
        X_test_qt = X_test_qt.astype('int')

        model.fit(X_train_qt, y_train)

        y_train_pred = model.predict(X_train_qt)
        y_test_pred = model.predict(X_test_qt)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print(f"Train accuracy = {train_accuracy}")
        print(f"Test accuracy = {test_accuracy}")
        train_accuracy_arr.append(train_accuracy)
        test_accuracy_arr.append(test_accuracy)