import sys
sys.path.insert(0, ".")

import cv2
import pandas as pd
from src.utils.general_utils import load_config, object_id_to_str
from src.utils.feature_extraction_utils import hog_gradient_bins, texture_hu_moments

if __name__ == '__main__':
    SET = 'test'
    config = load_config()
    DATA_DIR = config['dataset'][f'{SET}_data_dir']
    MASK_FOLDER = config['dataset']['mask_folder']
    RBG_FOLDER = config['dataset']['images_folder']

    json_data = pd.read_excel(f'processed/{SET}_json_data.xlsx')

    for i, row in json_data.iterrows():
        frame_id = row['frame_id']
        img_name = row['image_file_name']
        object_id = row['object_id']
        
        print(row)

        mask_addrs = f'{DATA_DIR}/{object_id_to_str(object_id)}/{MASK_FOLDER}/{object_id_to_str(frame_id)}_{object_id_to_str(0)}.png'
        rgb_addrs = f'{DATA_DIR}/{object_id_to_str(object_id)}/{RBG_FOLDER}/{img_name}'

        mask = cv2.imread(mask_addrs, cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(rgb_addrs)

        bins = hog_gradient_bins(mask)
        hu = texture_hu_moments(rgb, mask)
        #print(hu)