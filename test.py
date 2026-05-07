import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

minor_sin_arr = []
minor_cos_arr = []

for angle in range(0, 2*360+5, 5):
    major_angle = np.deg2rad(angle)

    major_cos = np.cos(major_angle)
    major_sin = np.sin(major_angle)

    C_major_angle = np.array([major_cos, major_sin])

    R = np.array([
        [0, 1],
        [-1, 0]
    ])
    C_minor_angle = R @ C_major_angle

    minor_cos, minor_sin = C_minor_angle
    minor_sin_arr.append(minor_sin)
    minor_cos_arr.append(minor_cos)
    minor_angle = np.rad2deg(np.arctan2(minor_cos, minor_sin))


plt.plot(minor_cos_arr, 'r.-', label='cos')
plt.plot(minor_sin_arr, 'b.-', label='sin')
plt.grid()
plt.show()