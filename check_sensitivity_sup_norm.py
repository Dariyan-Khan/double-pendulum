from dp_tidy import Pendulum
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# small perturbation
delta = np.pi/60
eps = 2 * delta
g = 9.81

# Assuming both pendulums have length 1
time_unit = np.sqrt(1/g)
max_time_unit = 20
# p1 = Pendulum(np.pi/2, np.pi/2, tmax=1e4 * time_unit)
# print('hi')
# p2 = Pendulum(np.pi/2 + delta, np.pi/2 + delta, tmax=1e4*time_unit)
# print('goodbye')

th1_range = np.arange(-np.pi, np.pi, delta)
th2_range = np.arange(-np.pi, np.pi, delta)


def step_to_time_unit(s, dt=0.04):
    # print(s)
    return (s * dt) / time_unit


t_conv = np.vectorize(step_to_time_unit)


def categorise(x):
    if x < 0:
        return 'w'

    elif x <= 0.25 * max_time_unit:
        return '#a6c8ff' # lightsteelblue'

    elif x <= 0.5 * max_time_unit:
        return '#0f62fe' #'cornflowerblue'

    elif x <= 0.75 * max_time_unit:
        return 'mediumblue'

    else:
        return 'b'


cat = np.vectorize(categorise)


# print(1000*np.sqrt(1/p1.g))
# Use the supremum metric to decide if error is big enough


points_th1, points_th2 = [], []
vals = []
for th1 in tqdm(th1_range):
    # print(th1)
    for th2 in th2_range:
        points_th1.append(th1)
        points_th2.append(th2)
        p1 = Pendulum(th1, th2, tmax=100 * time_unit)
        p2 = Pendulum(th1 + delta, th2 + delta, tmax=100 * time_unit)
        # print(p1.sol.shape[0])
        for s in range(p1.sol.shape[0]):
            if max(abs(p1.theta1[s] - p2.theta1[s]),
                   abs(p1.theta2[s] - p2.theta2[s])) >= eps:

                vals.append(s)
                break

            if s == p1.sol.shape[0] - 1:
                vals.append(float('inf'))

points_th1, points_th2 = np.array(points_th1), np.array(points_th2)
vals = t_conv(np.array(vals))
# print(vals)


# print(points)

# col = np.where(vals<1,'k',np.where(vals<5,'b','r')) # noqa
# col = cat(vals)

# print(col)

fig, ax = plt.subplots()
scatter = ax.scatter(points_th1, points_th2, c=vals, cmap='hot', s=10,
                     linewidth=0)
ax.set_xlabel('Theta 1')
ax.set_ylabel('Theta 2')
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_title("Number of time units for pendulums to diverge by eps")
cbar = fig.colorbar(scatter)
cbar.ax.tick_params(labelsize=5) 
cbar.set_label('number of time units', fontsize=6, rotation=270)
# produce a legend with the unique colors from the scatter
# legend1 = ax.legend(*scatter.legend_elements(),
#                      title="Classes")
# ax.add_artist(legend1)

plt.show()



# show_anim(p1, p2)