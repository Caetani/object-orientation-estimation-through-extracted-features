import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
import pandas as pd
import os
from src.utils.general_utils import load_config, object_id_to_str
from src.utils.geometry_utils import (
    mask_external_perimeter, mask_internal_perimeter, mask_total_perimeter,
    mask_num_holes, mask_area, mask_centroid, mask_ellipse_axes
)
from src.utils.feature_extraction_utils import hog_gradient_bins, texture_hu_moments


def extrair_features(row, mask, rgb):
    """Extrai todas as features de uma linha do json_data."""

    # ── Descritores brutos ────────────────────────────────────────────────────
    p_ext   = mask_external_perimeter(mask)
    p_int   = mask_internal_perimeter(mask)
    p_total = mask_total_perimeter(mask)
    n_holes = mask_num_holes(mask)
    area    = mask_area(mask)
    cx, cy  = mask_centroid(mask)

    bbox_xmin = row['bbox_xmin']
    bbox_xmax = row['bbox_xmax']
    bbox_ymin = row['bbox_ymin']
    bbox_ymax = row['bbox_ymax']
    bbox_w    = row['bbox_w']
    bbox_h    = row['bbox_h']
    bbox_xc = (bbox_xmax + bbox_xmin) / 2
    bbox_yc = (bbox_ymax + bbox_ymin) / 2

    major_axis, major_angle_deg, minor_axis, _, _ = mask_ellipse_axes(mask)

    # ── Features para o modelo ────────────────────────────────────────────────
    compactness_ext   = (p_ext   ** 2) / area if area > 0 else 0.0
    compactness_int   = (p_int   ** 2) / area if area > 0 else 0.0
    compactness_total = (p_total ** 2) / area if area > 0 else 0.0

    # Deslocamento normalizado do centróide em relação ao centro da bbox
    delta_x = (cx - bbox_xc) / bbox_w if bbox_w > 0 else 0.0
    delta_y = (cy - bbox_yc) / bbox_h if bbox_h > 0 else 0.0

    dist_centroid  = np.sqrt(delta_x**2 + delta_y**2)
    angle_centroid = np.rad2deg(np.arctan2(delta_x, delta_y))
    sin_centroid   = np.sin(np.deg2rad(angle_centroid))
    cos_centroid   = np.cos(np.deg2rad(angle_centroid))

    aspect_ratio  = bbox_h / bbox_w if bbox_w > 0 else 0.0
    eccentricity  = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0.0

    sin_major = np.sin(np.deg2rad(major_angle_deg))
    cos_major = np.cos(np.deg2rad(major_angle_deg))

    # HOG — 12 bins fixos de 30 graus (0° a 360°)
    hog_bins = hog_gradient_bins(mask, resolution_deg=30.0)

    # Momentos invariantes de Hu sobre a imagem RGB (grayscale)
    hu = texture_hu_moments(rgb, mask)

    # ── Monta dicionário de features ──────────────────────────────────────────
    features = {
        'compactness_ext':   compactness_ext,
        'compactness_int':   compactness_int,
        'compactness_total': compactness_total,
        'dist_centroid':     dist_centroid,
        'angle_centroid':    angle_centroid,
        'sin_centroid':      sin_centroid,
        'cos_centroid':      cos_centroid,
        'aspect_ratio':      aspect_ratio,
        'eccentricity':      eccentricity,
        'sin_major':         sin_major,
        'cos_major':         cos_major,
    }

    # HoG bins — nomeados pelo intervalo angular
    bin_edges = np.arange(0, 361, 30)
    for i, val in enumerate(hog_bins):
        features[f'hog_{bin_edges[i]}_{bin_edges[i+1]}'] = val

    # Momentos de Hu
    for i, val in enumerate(hu):
        features[f'hu_{i+1}'] = val

    return features


if __name__ == '__main__':
    SET = 'train'
    config   = load_config()
    DATA_DIR    = config['dataset'][f'{SET}_data_dir']
    MASK_FOLDER = config['dataset']['mask_folder']
    RGB_FOLDER  = config['dataset']['images_folder']
    OUTPUT_DIR  = 'processed'

    json_data = pd.read_excel(f'{OUTPUT_DIR}/{SET}_json_data.xlsx')
    
    registros = []

    for i, row in json_data.iterrows():
        frame_id  = row['frame_id']
        img_name  = row['image_file_name']
        object_id = row['object_id']

        print(f'Processando objeto {object_id_to_str(object_id)}, frame {frame_id}...')

        mask_addrs = f'{DATA_DIR}/{object_id_to_str(object_id)}/{MASK_FOLDER}/{object_id_to_str(frame_id)}_{object_id_to_str(0)}.png'
        rgb_addrs  = f'{DATA_DIR}/{object_id_to_str(object_id)}/{RGB_FOLDER}/{img_name}'

        mask = cv2.imread(mask_addrs, cv2.IMREAD_GRAYSCALE)
        rgb  = cv2.imread(rgb_addrs)

        if mask is None or rgb is None:
            print(f'  AVISO: imagem não encontrada — objeto {object_id_to_str(object_id)}, frame {frame_id}')
            continue

        try:
            features = extrair_features(row, mask, rgb)
        except Exception as e:
            print(f'  ERRO: objeto {object_id_to_str(object_id)}, frame {frame_id}: {e}')
            continue

        # Une as colunas do json_data com as features extraídas
        registro = row.to_dict()
        registro.update(features)
        registros.append(registro)

    df = pd.DataFrame(registros)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_excel(f'{OUTPUT_DIR}/{SET}_extracted_features.xlsx', index=False)
    print(f'\nPlanilha salva em: {OUTPUT_DIR}/{SET}_extracted_features.xlsx')
    print(f'Total de registros: {len(df)}')
    print(f'Total de colunas:   {len(df.columns)}')