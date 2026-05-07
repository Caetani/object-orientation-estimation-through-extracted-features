import sys
sys.path.insert(0, ".")
import os
import pandas as pd
from src.utils.general_utils import load_config, object_id_to_str
from src.utils.image_processing_utils import remove_background
import cv2

if __name__ == '__main__':
    config = load_config()
    DATA_DIR    = config['dataset']['test_data_dir']
    MASK_FOLDER = config['dataset']['mask_folder']
    RBG_FOLDER  = config['dataset']['images_folder']

    json_data  = pd.read_excel('processed/test_json_data.xlsx')
    output_dir = 'processed/masked_test_images'

    for i, row in json_data.iterrows():
        frame_id  = row['frame_id']
        img_name  = row['image_file_name']
        object_id = row['object_id']

        img_addrs  = f'{DATA_DIR}/{object_id_to_str(object_id)}/{RBG_FOLDER}/{img_name}'
        mask_addrs = f'{DATA_DIR}/{object_id_to_str(object_id)}/{MASK_FOLDER}/{object_id_to_str(frame_id)}_{object_id_to_str(0)}.png'

        img        = cv2.imread(img_addrs)
        mask       = cv2.imread(mask_addrs)
        masked_img = remove_background(img, mask)

        output_folder    = os.path.join(output_dir, object_id_to_str(object_id))
        output_file_name = os.path.join(output_folder, f'{object_id_to_str(frame_id)}.png')

        os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(output_file_name, masked_img)

        """ cv2.imshow('img', img)
        cv2.imshow('mask', mask)
        cv2.imshow('masked img', masked_img)
        cv2.waitKey()
        cv2.destroyAllWindows() """