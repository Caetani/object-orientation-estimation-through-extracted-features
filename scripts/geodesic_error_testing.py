import sys
sys.path.insert(0, ".")

import numpy as np
from src.utils.orientation_utils import geodesic_error, euler_to_quaternion

if __name__ == '__main__':
    yaw1, pitch1, roll1 = 50, 110, 0
    yaw2, pitch2, roll2 = 60, 120, 0

    yaw1 = np.deg2rad(yaw1)
    pitch1 = np.deg2rad(pitch1)
    roll1 = np.deg2rad(roll1)
    yaw2 = np.deg2rad(yaw2)
    pitch2 = np.deg2rad(pitch2)
    roll2 = np.deg2rad(roll2)

    q1 = euler_to_quaternion(roll=roll1, pitch=pitch1, yaw=yaw1)#, canonicalize=True)
    q2 = euler_to_quaternion(roll=roll2, pitch=pitch2, yaw=yaw2)#, canonicalize=True)

    error = geodesic_error(q1, q2)

    print(error)