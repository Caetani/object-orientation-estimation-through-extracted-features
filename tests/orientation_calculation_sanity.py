import pandas as pd
import json
import numpy as np
import os
import sys

# Adiciona src/ ao path para importar utils (script em src/preprocessing)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.utils.orientation_utils import *
from src.utils.general_utils import load_config, load_json, object_id_to_str
from src.models.dataset_definitions import OBJECT_NAMES

config = load_config()

DATA_DIR   = config['dataset']['train_data_dir']#'/mnt/d/Datasets/linemod/lm_train/train/'
OUTPUT_DIR = 'tests/results'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'train_verification_json_data.xlsx')

FLOAT_TOLERANCE = 1e-6


def verify_orientation_convertions(object_id_int):
    print(f"Object id = {object_id_int, type(object_id_int)}")
    object_id_str = f'{object_id_int:06d}'
    folder = os.path.join(DATA_DIR, object_id_str)
    gt = load_json(os.path.join(folder, 'scene_gt.json'))

    for frame_id_str in gt.keys():
        objects_gt   = gt[frame_id_str]
        for obj_gt in objects_gt:
            R = np.array(obj_gt['cam_R_m2c']).reshape(3, 3)
            
            
            q = rotation_matrix_to_quaternion(R)
            

            qw, qx, qy, qz = q

            t_roll, t_pitch, t_yaw = rotation_matrix_to_euler(R)
            _roll, _pitch, _yaw = quaternion_to_euler(q)

            assert abs(angular_difference(t_roll, _roll)) < FLOAT_TOLERANCE, \
                f"Roll mismatch: {t_roll} != {_roll}"

            assert abs(angular_difference(t_pitch, _pitch)) < FLOAT_TOLERANCE, \
                f"Pitch mismatch: {t_pitch} != {_pitch}"

            assert abs(angular_difference(t_yaw, _yaw)) < FLOAT_TOLERANCE, \
                f"Yaw mismatch: {t_yaw} != {_yaw}"

            """ assert abs(t_roll - _roll) < FLOAT_TOLERANCE, f"Roll mismatch: {t_roll} != {_roll}"
            assert abs(t_pitch - _pitch) < FLOAT_TOLERANCE, f"Pitch mismatch: {t_pitch} != {_pitch}"
            assert abs(t_yaw - _yaw) < FLOAT_TOLERANCE, f"Yaw mismatch: {t_yaw} != {_yaw}" """

            R_q = euler_to_rotation_matrix(_roll, _pitch, _yaw).reshape(1, 9)

            print(f"JSON: {np.array(obj_gt['cam_R_m2c'])}")
            print(f"Calculated: {R_q}")
            print(f"Diff = {(np.abs(np.array(obj_gt['cam_R_m2c']) - R_q)) > FLOAT_TOLERANCE}")
            #assert ((np.abs(R.reshape(1, 9) - R_q)) < 10**-3).all(), "Rotation matrices not equal"

            mod_q = np.linalg.norm(q)
            assert abs(mod_q - 1.0) < FLOAT_TOLERANCE, (
                f"Objeto {object_id_str}, frame {frame_id_str}: "
                f"módulo do quaternion != 1 (valor: {mod_q})"
            )

            roll_rad, pitch_rad, yaw_rad = quaternion_to_euler(q)



if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for obj_id in range(1, 16):
        print(f'Checking object {obj_id:02d} ({OBJECT_NAMES[obj_id]})...')
        try:
            verify_orientation_convertions(obj_id)
        except AssertionError as e:
            print(f'  ERRO DE INTEGRIDADE: {e}')
            raise
        except FileNotFoundError as e:
            print(f'  File not found: {e}')

    print("End of execution.")