import cv2
import numpy as np

def remove_background(
        img: cv2.Mat,
        mask: cv2.Mat
) -> cv2.Mat:
    """
    Remove o fundo da imagem RGB utilizando a máscara do objeto.
    Pixels onde a máscara é 0 são zerados nos 3 canais RGB.
    """
    mask_bool = mask != 0
    result = np.zeros_like(img)
    result[mask_bool] = img[mask_bool]
    return result