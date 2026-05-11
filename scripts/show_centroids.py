import sys
sys.path.insert(0, ".")
import pandas as pd
import json
import numpy as np
import os
import cv2
from src.utils.orientation_utils import *
from src.utils.geometry_utils import *

CONFIG_PATH = "config.json"  # preencha com o caminho para o config.json

if __name__ == '__main__':

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    images_folder = config['dataset']['images_folder']
    mask_folder   = config['dataset']['mask_folder']
    scene_gt      = config['dataset']['scene_gt']
    scene_camera  = config['dataset']['scene_camera']
    scene_gt_info = config['dataset']['scene_gt_info']

    object_id = '000004'  # Camera
    DATA_DIR  = os.path.join(config['dataset']['train_data_dir'], object_id)

    mask_files = os.listdir(os.path.join(DATA_DIR, mask_folder))

    for f in mask_files:
        print(f)
        mask = cv2.imread(os.path.join(DATA_DIR, mask_folder, f), cv2.IMREAD_GRAYSCALE)
        count = cv2.countNonZero(mask)
        print(count)
        sobelx = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=3)  # Bordas horizontais
        sobely = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=3)  # Bordas verticais

        # Magnitude do gradiente
        gradient_magnitude = cv2.magnitude(sobelx, sobely)

        # Converte para uint8
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
        threshold_gradient = gradient_magnitude.copy()
        threshold_gradient[threshold_gradient > 250]  = 255
        threshold_gradient[threshold_gradient <= 250] = 0
        canny    = cv2.Canny(mask, threshold1=250, threshold2=251)
        internal = mask_internal_perimeter(mask)
        external = mask_external_perimeter(mask)
        total    = mask_total_perimeter(mask)
        num_holes = mask_num_holes(mask)
        diameter  = mask_major_axis(mask)
        print(internal, external, total, num_holes)
        print(diameter)
        cx, cy = mask_centroid(mask, show=True)
        print(f"Centroid: {cx} - {cy}")
        continue

        contour_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_img, contours, -1, color=(0, 255, 0), thickness=2)

        cv2.imshow('contour', contour_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        exit()