import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.utils.orientation_utils import *


qw = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
qx = 10*qw
qy = 100*qw
qz = 5*qw

y_arr = np.vstack((qw, qx, qy, qz)).T

y = y_arr[0, :]

print(y_arr, y_arr.shape, len(y_arr.shape))
print(y, y.shape, len(y.shape))