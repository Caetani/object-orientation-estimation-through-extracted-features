import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.utils.orientation_utils import *

minor_sin_arr = []
minor_cos_arr = []
processed_sin_arr = []

"Hardware implementation"

angles = np.deg2rad([30, 20, 10])

R = euler_to_rotation_matrix(*angles)

q = rotation_matrix_to_quaternion(R)

angles2 = quaternion_to_euler(q)

print(np.rad2deg(angles))
print(np.rad2deg(angles2))