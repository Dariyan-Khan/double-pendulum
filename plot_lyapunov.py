import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from lyapunov_time_estimate import lyp_exp

# small perturbation
delta = np.pi/20
g = 9.81

th1_range = np.arange(-np.pi, np.pi, delta)
th2_range = np.arange(-np.pi, np.pi, delta)

initial_cond = np.array([np.pi/10, np.pi/10, 0, 0])
eps = 0.01  # the initial difference
tmax = 10


points_th1, points_th2 = [], []
vals = []
for th1 in tqdm(th1_range):
    for th2 in th2_range:
        points_th1.append(th1)
        points_th2.append(th2)
        initial_cond = np.array([th1, th2, 0, 0])
        vals.append(lyp_exp(initial_cond, eps, tmax, T=1))


points_th1, points_th2 = np.array(points_th1), np.array(points_th2)

fig, ax = plt.subplots()
scatter = ax.scatter(points_th1, points_th2, c=vals, cmap='hot', s=30,
                     linewidth=0)
ax.set_xlabel('Theta 1')
ax.set_ylabel('Theta 2')
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_title("The different lyapunov times")
cbar = fig.colorbar(scatter)
cbar.ax.tick_params(labelsize=5)
cbar.set_label('number of time units', fontsize=6, rotation=270)
plt.show()
