import sys
sys.path.insert(0, ".")

import os
import cv2
import pandas as pd
from src.utils.general_utils import load_config, object_id_to_str
from src.utils.geometry_utils import mask_ellipse_axes

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np

if __name__ == '__main__':
    config = load_config()
    DATA_DIR = config['dataset']['train_data_dir']
    MASK_FOLDER = config['dataset']['mask_folder']
    RBG_FOLDER = config['dataset']['images_folder']

    json_data = pd.read_excel('processed/train_json_data.xlsx')

    major_sin_arr = []
    counter = 0
    for i, row in json_data.iterrows():
        counter += 1
        if counter >= 30: break
        frame_id = row['frame_id']
        img_name = row['image_file_name']
        object_id = row['object_id']

        mask_addrs = f'{DATA_DIR}/{object_id_to_str(object_id)}/{MASK_FOLDER}/{object_id_to_str(frame_id)}_{object_id_to_str(0)}.png'

        mask = cv2.imread(mask_addrs, cv2.IMREAD_GRAYSCALE)
        major_axis, major_angle_deg, minor_axis, minor_angle_deg, vis = mask_ellipse_axes(mask, show=True)

        output_dir = 'processed/ellipsis_train'
        output_folder    = os.path.join(output_dir, object_id_to_str(object_id))
        output_file_name = os.path.join(output_folder, f'{object_id_to_str(frame_id)}.png')

        os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(output_file_name, vis)