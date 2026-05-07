import pandas as pd
import json
import numpy as np
import os
import sys

# Adiciona src/ ao path para importar utils (script em src/preprocessing)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.orientation_utils import rotation_matrix_to_quaternion, quaternion_to_euler

#DATA_DIR   = '/mnt/d/Datasets/linemod/lm_train/train'
DATA_DIR   = '/mnt/d/Datasets/linemod/lm_test_bop19/test/'
OUTPUT_DIR = 'processed'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'test_json_data.xlsx')

# Tolerância para comparações numéricas de módulo do quaternion
QUAT_NORM_TOL = 1e-6

# Mapeamento de object_id para nome do objeto (Linemod)
OBJECT_NAMES = {
    1:  'ape',
    2:  'benchvise',
    3:  'bowl',
    4:  'cam',
    5:  'cat',
    6:  'cup',
    7:  'can',
    8:  'driller',
    9:  'duck',
    10: 'eggbox',
    11: 'glue',
    12: 'holepuncher',
    13: 'iron',
    14: 'lamp',
    15: 'phone',
}


def carregar_json(caminho):
    with open(caminho, 'r') as f:
        return json.load(f)


def nome_imagem(frame_id):
    """Retorna o nome do arquivo de imagem no formato 000000.png."""
    return f'{int(frame_id):06d}.png'


def processar_objeto(object_id_int):
    """
    Processa todos os frames de um objeto e retorna uma lista de dicionários.
    Lança AssertionError se alguma verificação de integridade falhar.
    """
    object_id_str = f'{object_id_int:06d}'
    pasta = os.path.join(DATA_DIR, object_id_str)

    gt      = carregar_json(os.path.join(pasta, 'scene_gt.json'))
    gt_info = carregar_json(os.path.join(pasta, 'scene_gt_info.json'))

    registros = []

    for frame_id_str in gt.keys():

        objetos_gt   = gt[frame_id_str]
        objetos_info = gt_info[frame_id_str]

        for obj_gt, obj_info in zip(objetos_gt, objetos_info):

            # ── Verificações de integridade ────────────────────────────────
            if not obj_info['bbox_obj'] == obj_info['bbox_visib']: print(
                f"Objeto {object_id_str}, frame {frame_id_str}: "
                f"bbox_obj != bbox_visib"
            )
            px_all   = obj_info['px_count_all']
            px_valid = obj_info['px_count_valid']
            px_visib = obj_info['px_count_visib']
            if not (px_all == px_valid == px_visib):
                print(
                    f"[DIVERGÊNCIA px_count] objeto {object_id_str}, frame {frame_id_str}: "
                    f"px_count_all={px_all}, px_count_valid={px_valid}, px_count_visib={px_visib} "
                    f"| diff all-valid={px_all - px_valid}, all-visib={px_all - px_visib}, valid-visib={px_valid - px_visib}"
                )
            visib_fract = obj_info['visib_fract']
            if visib_fract != 1.0:
                print(
                    f"[DIVERGÊNCIA visib_fract] objeto {object_id_str}, frame {frame_id_str}: "
                    f"visib_fract={visib_fract} | diff={1.0 - visib_fract:.6f}"
                )

            # ── Rotação e quaternion ───────────────────────────────────────
            R = np.array(obj_gt['cam_R_m2c']).reshape(3, 3)
            q = rotation_matrix_to_quaternion(R)
            qw, qx, qy, qz = q

            mod_q = np.linalg.norm(q)
            assert abs(mod_q - 1.0) < QUAT_NORM_TOL, (
                f"Objeto {object_id_str}, frame {frame_id_str}: "
                f"módulo do quaternion != 1 (valor: {mod_q})"
            )

            roll_rad, pitch_rad, yaw_rad = quaternion_to_euler(q)

            # ── Bounding box ───────────────────────────────────────────────
            x, y, w, h = obj_info['bbox_obj']
            bbox_xmin = x
            bbox_xmax = x + w
            bbox_ymin = y
            bbox_ymax = y + h

            registros.append({
                'object_id':       obj_gt['obj_id'],
                'object_name':     OBJECT_NAMES.get(obj_gt['obj_id'], 'unknown'),
                'frame_id':        int(frame_id_str),
                'image_file_name': nome_imagem(frame_id_str),
                'yaw_rad':         yaw_rad,
                'pitch_rad':       pitch_rad,
                'roll_rad':        roll_rad,
                'yaw_deg':         np.rad2deg(yaw_rad),
                'pitch_deg':       np.rad2deg(pitch_rad),
                'roll_deg':        np.rad2deg(roll_rad),
                'bbox_xc':         x,
                'bbox_yc':         y,
                'bbox_w':          w,
                'bbox_h':          h,
                'bbox_xmin':       bbox_xmin,
                'bbox_xmax':       bbox_xmax,
                'bbox_ymin':       bbox_ymin,
                'bbox_ymax':       bbox_ymax,
                'qw':              qw,
                'qx':              qx,
                'qy':              qy,
                'qz':              qz,
                'mod_q':           mod_q,
            })

    return registros


if __name__ == '__main__':

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    todos_registros = []

    for obj_id in range(1, 16):
        print(f'Processando objeto {obj_id:02d} ({OBJECT_NAMES[obj_id]})...')
        try:
            registros = processar_objeto(obj_id)
            todos_registros.extend(registros)
            print(f'  {len(registros)} frames adicionados.')
        except AssertionError as e:
            print(f'  ERRO DE INTEGRIDADE: {e}')
            raise
        except FileNotFoundError as e:
            print(f'  AVISO: pasta/arquivo não encontrado: {e}')

    df = pd.DataFrame(todos_registros)

    # Garante ordenação por objeto e frame
    df = df.sort_values(['object_id', 'frame_id']).reset_index(drop=True)

    df.to_excel(OUTPUT_FILE, index=False)
    print(f'\nPlanilha salva em: {OUTPUT_FILE}')
    print(f'Total de registros: {len(df)}')