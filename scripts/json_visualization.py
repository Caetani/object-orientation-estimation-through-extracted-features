import sys
sys.path.insert(0, ".")

import pandas as pd
import json
import numpy as np
import os

from utils.orientation_utils import *

if __name__ == '__main__':
    object_id = '000004' # Camera
    images_folder = 'rgb'
    mask_folder = 'mask'

    scene_gt = 'scene_gt.json'
    scene_camera = 'scene_camera_json'
    scene_gt_info = 'scene_gt_info.json'

    DATA_DIR = f'/mnt/d/Datasets/linemod/lm_train/train/{object_id}/'

    image_files = os.listdir(f"{DATA_DIR}/{images_folder}/")
    

    with open(f'{DATA_DIR}/scene_gt.json', 'r') as f:
        data = json.load(f)

        for frame_id, objects in data.items():
            if int(frame_id) >= 1278: exit()
            for obj in objects:
                R_flat = obj['cam_R_m2c']           # flat list of 9 values
                R = np.array(R_flat).reshape(3, 3)  # reshape to 3x3 matrix
                t = np.array(obj['cam_t_m2c'])      # translation vector
                obj_id = obj['obj_id']
                
                q = rotation_matrix_to_quaternion(R)
                roll, pitch, yaw = np.rad2deg(quaternion_to_euler(q))

                print(f"Frame {frame_id}, Object {obj_id}: Pitch = {pitch} - Yaw = {yaw} - Roll = {roll} ({q})")
                #print(R)
                #print(rotation_matrix_to_quaternion(R))