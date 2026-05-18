import pandas as pd
import json
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.utils.orientation_utils import *
from src.utils.general_utils import load_config, load_json
from src.models.dataset_definitions import OBJECT_NAMES

config = load_config()

DATA_DIR   = config['dataset']['train_data_dir']
OUTPUT_DIR = 'tests/results'

FLOAT_TOLERANCE = 1e-3


def verify_orientation_convertions(object_id_int):
    object_id_str = f'{object_id_int:06d}'
    folder = os.path.join(DATA_DIR, object_id_str)
    gt = load_json(os.path.join(folder, 'scene_gt.json'))

    for frame_id_str in gt.keys():
        objects_gt   = gt[frame_id_str]
        for obj_gt in objects_gt:
            R = np.array(obj_gt['cam_R_m2c']).reshape(3, 3)
            
            q = rotation_matrix_to_quaternion(R)
            q_roll, q_pitch, q_yaw = quaternion_to_euler(q)

            roll, pitch, yaw = rotation_matrix_to_euler(R)
            q_euler = euler_to_quaternion(roll, pitch, yaw)

            q_error = geodesic_error([q], [q_euler])[0]
            if q_error > FLOAT_TOLERANCE: print(f"[Object: {object_id_str} - Frame: {frame_id_str}] Quarternion convertion error. Geodesic error = {q_error}")

            R_from_euler = euler_to_rotation_matrix(roll, pitch, yaw)
            R_from_q = euler_to_rotation_matrix(q_roll, q_pitch, q_yaw)

            assert abs(angular_difference(roll, q_roll)) < FLOAT_TOLERANCE, \
                f"Roll mismatch: {roll} != {q_roll}"

            assert abs(angular_difference(pitch, q_pitch)) < FLOAT_TOLERANCE, \
                f"Pitch mismatch: {pitch} != {q_pitch}"

            assert abs(angular_difference(yaw, q_yaw)) < FLOAT_TOLERANCE, \
                f"Yaw mismatch: {yaw} != {q_yaw}"

            if not np.allclose(R, R_from_q): print(f"[Object: {object_id_str} - Frame: {frame_id_str}] Quarternion to Rotation Matrix convertion error.")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for obj_id in OBJECT_NAMES.keys():
        print(f'Checking object {obj_id:02d} ({OBJECT_NAMES[obj_id]})...')
        try:
            verify_orientation_convertions(obj_id)
        except AssertionError as e:
            print(f'  INTEGRITY ERROR: {e}')
            raise
        except FileNotFoundError as e:
            print(f'  File not found: {e}')

    print("End of execution.")