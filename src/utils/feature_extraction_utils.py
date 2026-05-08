import numpy as np
import cv2
from scipy.stats import circmean, circstd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import sys
sys.path.insert(0, ".")
from src.utils.image_processing_utils import remove_background

def _contour_angles(mask: cv2.Mat) -> np.ndarray:
    """Calcula os ângulos das tangentes entre pontos consecutivos do contorno externo."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    pts = contours[0].reshape(-1, 2).astype(np.float32)
    dx = np.diff(pts[:, 0], append=pts[0, 0])
    dy = np.diff(pts[:, 1], append=pts[0, 1])
    return np.rad2deg(np.arctan2(dy, dx)) % 360


def _contour_histogram(angle_vals: np.ndarray, resolution_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Calcula o histograma de ângulos do contorno com a resolução fornecida."""
    n_bins = round(360 / resolution_deg)
    hist, bin_edges = np.histogram(angle_vals, bins=n_bins, range=(0.0, 360.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return hist, bin_centers


def hog_gradient_metrics(
    mask: cv2.Mat,
    resolution_deg: float = 10.0,
) -> tuple[float, float, float, float, float]:
    """
    Calcula métricas dos ângulos do contorno da máscara.

    Retorna
    -------
    mean_angle, std_angle, mode_1, mode_2, mode_3
    """
    angle_vals = _contour_angles(mask)

    mean_angle = np.rad2deg(circmean(np.deg2rad(angle_vals), high=2*np.pi, low=0))
    std_angle  = np.rad2deg(circstd(np.deg2rad(angle_vals),  high=2*np.pi, low=0))

    hist, bin_centers = _contour_histogram(angle_vals, resolution_deg)
    top3_idx          = np.argsort(hist)[::-1][:3]
    mode_1, mode_2, mode_3 = bin_centers[top3_idx]

    return mean_angle, std_angle, mode_1, mode_2, mode_3


def hog_gradient_bins(
    mask: cv2.Mat,
    resolution_deg: float = 30.0,
) -> np.ndarray:
    """
    Retorna o histograma de ângulos do contorno como vetor de bins fixos,
    normalizado para uso direto como features em modelos de machine learning.

    Parâmetros
    ----------
    mask           : imagem binária da máscara (0 ou 255)
    resolution_deg : resolução angular em graus (default 30 → 12 bins)

    Retorna
    -------
    hist : array normalizado de tamanho (360 / resolution_deg,)
    """
    angle_vals        = _contour_angles(mask)
    hist, _           = _contour_histogram(angle_vals, resolution_deg)
    hist              = hist.astype(np.float32)
    norm              = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm
    return hist


def texture_hu_moments(
    img: cv2.Mat,
    mask: cv2.Mat,
) -> np.ndarray:
    """
    Calcula os momentos de Hu sobre a intensidade da imagem dentro da máscara.

    Retorna
    -------
    hu : array de 7 valores
    """
    img_shape = img.shape
    if len(img_shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_masked = remove_background(img, mask)
    M  = cv2.moments(img_masked)
    hu = cv2.HuMoments(M).flatten()
    
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu