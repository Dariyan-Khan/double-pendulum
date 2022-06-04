import numpy as np
from dp_tidy import Pendulum
from tqdm import tqdm

initial_cond = np.array([np.pi/2, np.pi/4, 0, 0])
eps = 0.01
tmax = 40
err_0 = np.array([np.sqrt(eps / 4.0) for i in range(4)])


def pendulum_vec(v, t):  # v = (th1, th2, th1_dot, th2_dot)
    return Pendulum(th1=v[0], th2=v[1], th1_dot=v[2], th2_dot=v[3], tmax=t,
                    to_trace=False, trace_delete=False)


T = 1

p1 = pendulum_vec(initial_cond, T)

p2 = pendulum_vec(initial_cond + err_0, T)

assert p1.dt == p2.dt and p1.tmax == p2.tmax
step_rate = 1 / p1.dt

lyp_ests = []

for i in tqdm(range(tmax)):
    p1_state = p1.sol[-1]
    assert len(p1_state) == 4
    p2_state = p2.sol[-1]
    err_1 = p1_state - p2_state
    err1_norm = np.linalg.norm(err_1)
    err0_norm = np.linalg.norm(err_0)
    lyp_ests.append((1/T) * (np.log(err1_norm) / np.log(err0_norm)))

    err_0 = (err_1 / err1_norm) * eps  # scaled err_1
    p1 = pendulum_vec(p1_state, T)
    p2 = pendulum_vec(p1_state + err_0, T)


print(lyp_ests)
