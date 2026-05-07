import sys
sys.path.insert(0, ".")

import pandas as pd
from src.utils.general_utils import load_config, object_id_to_str
from src.utils.image_processing_utils import remove_background
import cv2

if __name__ == '__main__':
    config = load_config()
    DATA_DIR = config['dataset']['test_data_dir']
    MASK_FOLDER = config['dataset']['mask_folder']
    RBG_FOLDER = config['dataset']['images_folder']

    json_data = pd.read_excel('processed/test_json_data.xlsx')

    object_name = 'cam'
    frame_id = 20
    row = json_data[(json_data['object_name'] == object_name) & (json_data['frame_id'] == frame_id)].iloc[0]

    img_name = row['image_file_name']
    object_id = row['object_id']
    
    img_addrs = f'{DATA_DIR}/{object_id_to_str(object_id)}/{RBG_FOLDER}/{img_name}'
    mask_addrs = f'{DATA_DIR}/{object_id_to_str(object_id)}/{MASK_FOLDER}/{object_id_to_str(frame_id)}_{object_id_to_str(0)}.png'
    img = cv2.imread(img_addrs)
    mask = cv2.imread(mask_addrs)
    masked_img = remove_background(img, mask)

    cv2.imshow('img', img)
    cv2.imshow('mask', mask)
    cv2.imshow('masked img', masked_img)
    cv2.waitKey()
    cv2.destroyAllWindows()