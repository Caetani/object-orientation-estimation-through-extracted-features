import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy import stats

alpha = 0.05
dof = 19

mean_1 = 0
mean_2 = - 1
mean_diff = mean_2 - mean_1

t_crit = stats.t.ppf(alpha, dof)
print(f"t_crit = {t_crit}")

t_stat = (mean_2 - mean_1)

p_value = stats.t.cdf(mean_diff, dof)
print(f"P-Value = {p_value}")