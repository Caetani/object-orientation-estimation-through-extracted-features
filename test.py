import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x = np.arange(1, 9+1, 1)
y = 10**(np.arange(-6, -2).astype("float"))
z = np.outer(x, y).flatten()
print(z)