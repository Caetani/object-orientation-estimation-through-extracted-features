import pandas as pd
import json
import numpy as np
import os

from utils.orientation_utils import *

if __name__ == '__main__':
    object_id = '000004' # Camera
    subfolder = 'rgb'
    scene_id = '000000'

    DATA_DIR = f'/mnt/d/Datasets/linemod/lm_train/train/{object_id}/'



    files = os.listdir(f"{DATA_DIR}/{subfolder}/")
    for f in files:
        print(f)