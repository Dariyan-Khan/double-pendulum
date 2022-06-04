import numpy as np
from dp_tidy import Pendulum

initial_cond = np.array(np.pi/2, np.pi/4, 0, 0)
eps = 0.01
err = np.array([np.sqrt(eps / 4.0) for i in range(4)])

def pendulum_vec(v, tmax): # v = (th1, th2, th1_dot, th2_dot)
    return Pendulum(th1=v[0], th2=v[1], th1_dot=v[2], th2_dot=v[3], tmax=tmax,
                    to_trace=False, trace_delete=False)

T = 1

p1 = pendulum_vec(initial_cond, 1000)

p2 = pendulum_vec(initial_cond + err, 1000)

assert p1.dt == p2.dt and p1.tmax == p2.tmax
step_rate = 1 / p1.dt

lyp_ests = []

for i in range(1, p1.tmax + 1):
    step_num = step_rate * i
    err = 

