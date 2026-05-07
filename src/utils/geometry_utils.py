import cv2
import numpy as np
from scipy.spatial.distance import cdist

def mask_total_perimeter(
    mask: cv2.Mat,
) -> float:
    contours, hierarchy = cv2.findContours(
        mask, 
        mode=cv2.RETR_CCOMP,
        method=cv2.CHAIN_APPROX_TC89_KCOS
    )
    total_perimeter_px = sum(cv2.arcLength(c, closed=True) for c in contours)
    return total_perimeter_px


def mask_external_perimeter(
    mask: cv2.Mat,      
) -> float:
    contours, hierarchy = cv2.findContours(
        mask, 
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_TC89_KCOS
    )
    external_perimeter_px = sum(cv2.arcLength(c, closed=True) for c in contours)
    return external_perimeter_px

def mask_internal_perimeter(
    mask: cv2.Mat,
) -> float:
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_TC89_KCOS
    )
    internal_perimeter_px = sum(
        cv2.arcLength(c, closed=True)
        for c, h in zip(contours, hierarchy[0]) if h[3] != -1
    )
    return internal_perimeter_px


def mask_area(
    mask: cv2.Mat,
) -> float:
    area_px = cv2.countNonZero(mask)
    return area_px


def mask_num_holes(
    mask: cv2.Mat,
    min_hole_area: int = 10,
) -> int:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    num_holes = sum(
        1 for c, h in zip(contours, hierarchy[0])
        if h[3] != -1 and cv2.contourArea(c) >= min_hole_area
    )
    return num_holes


def mask_centroid(
    mask: cv2.Mat,
    show: bool = False,
) -> tuple[float, float]:
    M = cv2.moments(mask)
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    if show:
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(vis, (int(cx), int(cy)), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.imshow('Centroide', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cx, cy


def mask_ellipse_axes(
    mask: cv2.Mat,
    show: bool = False,
) -> tuple[float, float, float, float, cv2.Mat]:
    """
    Ajusta uma elipse ao contorno externo da máscara e retorna os comprimentos
    e ângulos do major e minor axis.
    Retorna
    -------
    major_axis, major_angle_deg, minor_axis, minor_angle_deg, vis
    """
    contours, _ = cv2.findContours(
        mask,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_TC89_KCOS
    )
    if len(contours[0]) < 5:
        raise ValueError("Contorno com menos de 5 pontos: impossível ajustar elipse.")
    ellipse = cv2.fitEllipse(contours[0])
    (cx, cy), (minor_axis, major_axis), angle_deg = ellipse

    major_axis /= 2
    minor_axis /= 2

    major_angle_deg = angle_deg
    minor_angle_deg = major_angle_deg - 90

    vis = None
    if show:
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # Ângulos corrigidos apenas para visualização — fitEllipse mede a partir do eixo vertical
        vis_major_angle_deg = angle_deg - 90
        vis_minor_angle_deg = vis_major_angle_deg - 90
        # Desenha a elipse
        cv2.ellipse(vis, ellipse, color=(0, 255, 0), thickness=2)
        # Major axis (vermelho)
        major_dx = int(major_axis * np.cos(np.deg2rad(vis_major_angle_deg)))
        major_dy = int(major_axis * np.sin(np.deg2rad(vis_major_angle_deg)))
        cv2.line(vis,
                 (int(cx) - major_dx, int(cy) - major_dy),
                 (int(cx) + major_dx, int(cy) + major_dy),
                 color=(0, 0, 255), thickness=2)
        # Minor axis (azul)
        minor_dx = int(minor_axis * np.cos(np.deg2rad(vis_minor_angle_deg)))
        minor_dy = int(minor_axis * np.sin(np.deg2rad(vis_minor_angle_deg)))
        cv2.line(vis,
                 (int(cx) - minor_dx, int(cy) - minor_dy),
                 (int(cx) + minor_dx, int(cy) + minor_dy),
                 color=(255, 0, 0), thickness=2)
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color_text = (255, 255, 255)
        cv2.putText(vis, f'Major: {major_axis*2:.1f}px  {major_angle_deg:.1f}deg', (10, 20), font, font_scale, color_text, 1)
        cv2.putText(vis, f'Minor: {minor_axis*2:.1f}px  {minor_angle_deg:.1f}deg', (10, 40), font, font_scale, color_text, 1)

    return major_axis, major_angle_deg, minor_axis, minor_angle_deg, vis


def mask_major_axis(
    mask: cv2.Mat,
) -> tuple[float, float]:
    contours, hierarchy = cv2.findContours(
        mask, 
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_TC89_KCOS
    )
    hull = cv2.convexHull(contours[0])
    pts = hull.reshape(-1, 2).astype(np.float32)

    dist_matrix = cdist(pts, pts, metric='euclidean')

    idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    p1, p2 = pts[idx[0]], pts[idx[1]]

    diameter = dist_matrix[idx]

    delta_x = abs(p1[0] - p2[0])
    delta_y = abs(p1[1] - p2[1])
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.rad2deg(angle_rad)

    return diameter, angle_deg