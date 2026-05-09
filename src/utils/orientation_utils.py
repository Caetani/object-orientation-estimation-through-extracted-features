import numpy as np


def euler_to_quaternion(roll, pitch, yaw, canonicalize: bool = False):
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    q = np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ])
    if canonicalize: return q if q[0] >= 0 else -q
    else: return q


def rotation_matrix_to_quaternion(R, canonicalize: bool = False):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    if canonicalize: return q if q[0] >= 0 else -q
    else: return q


def euler_to_rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,             cp*cr            ]
    ])


def rotation_matrix_to_euler(R):
    pitch = np.arcsin(-R[2, 0])
    roll  = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
    yaw   = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))

    return np.array([roll, pitch, yaw])


def quaternion_to_euler(q):
    w, x, y, z = q
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(2*(w*y - z*x))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return np.array([roll, pitch, yaw])


def geodesic_error(q_true: np.ndarray, q_pred: np.ndarray) -> float:
    """
    Calcula o erro geodésico angular entre dois quaternions em graus.
    Lida corretamente com a ambiguidade de sinal (q e -q).
    """
    q_pred_norm = q_pred / np.linalg.norm(q_pred)
    dot = np.clip(np.abs(np.dot(q_true, q_pred_norm)), 0.0, 1.0)
    return np.rad2deg(2 * np.arccos(dot))