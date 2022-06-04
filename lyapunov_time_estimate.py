import numpy as np
from dp_tidy import Pendulum
from tqdm import tqdm
import time

initial_cond = np.array([np.pi/10, np.pi/10, 0, 0])
eps = 0.01
tmax = 100
err_0 = np.array([np.sqrt(eps / 4.0) for _ in range(4)])


def calc_E(y):
    """Return the total energy of the system."""
    m1 = m2 = L1 = L2 = 1
    g = 9.81

    th1, th2, th1d, th2d = y
    V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
    T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
            2*L1*L2*th1d*th2d*np.cos(th1-th2))
    return T + V


def principal(x):
    a = np.mod(x, 2 * np.pi)
    if a > np.pi:
        a = a - 2 * np.pi
    return a


def pendulum_vec(v, t):  # v = (th1, th2, th1_dot, th2_dot)
    return Pendulum(th1=principal(v[0]), th2=principal(v[1]),
                    th1_dot=v[2], th2_dot=v[3], tmax=t, to_trace=False,
                    trace_delete=False)

T = 1

p1 = pendulum_vec(initial_cond, T)

p2 = pendulum_vec(initial_cond + err_0, T)

assert p1.dt == p2.dt and p1.tmax == p2.tmax
step_rate = 1 / p1.dt

lyp_ests = []

for i in tqdm(range(tmax)):
    p1_state = p1.sol[-1]
    p1_state[[1, 2]] = p1_state[[2, 1]]  # swap second and third elements
    assert len(p1_state) == 4
    p2_state = p2.sol[-1]
    p2_state[[1, 2]] = p2_state[[2, 1]]  # swap second and third elements
    err_1 = p1_state - p2_state
    err1_norm = np.linalg.norm(err_1)
    err0_norm = np.linalg.norm(err_0)
    lyp_ests.append((1/T) * (np.log(err1_norm) - np.log(err0_norm)))
    err_0 = (err_1 / err1_norm) * eps  # scaled err_1
    # print(calc_E(p1_state), "energy")
    p1 = pendulum_vec(p1_state, T)
    p2 = pendulum_vec(p1_state + err_0, T)


print(lyp_ests)

print(sum(lyp_ests) / len(lyp_ests))
