import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.utils.orientation_utils import *


qw = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
qx = 10*qw
qy = 100*qw

y = np.vstack((qw, qx, qy)).T
print(y, y.shape)