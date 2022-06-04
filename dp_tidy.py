import numpy as np
from scipy.integrate import solve_ivp
from copy import deepcopy

# Pendulum rod lengths (m), bob masses (kg).
L1, L2 = 1, 1
m1, m2 = 1, 1
# The gravitational acceleration (m.s-2).
g = 9.81
to_trace = True
to_delete = True

# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 15, 0.04
num_frames = 250
t = np.arange(0, tmax+dt, dt)


class Pendulum():

    def __init__(self, th1, th2, th1_dot=0, th2_dot=0, L1=1, L2=1, m1=1, m2=1,
                 g=9.81, to_trace=True, trace_delete=True, tmax=15, dt=0.04,
                 restart=None):
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.to_trace = to_trace
        self.trace_delete = trace_delete
        self.tmax = tmax
        self.dt = dt
        self.restart = restart
        self.t = np.arange(0, tmax+dt, dt)
        self.num_frames = (50/3) * tmax
        self.y0 = [th1, th1_dot, th2, th2_dot]

        if restart is None:
            y_full = solve_ivp(self.deriv, np.array((0, tmax + dt)), self.y0,
                                t_eval=self.t,
                               args=(self.L1, self.L2, self.m1, self.m2))
            self.sol = (y_full.y).T
        else:
            self.sol = self.iterative_solve()
        # self.sol = (self.y.y).T
        self.theta1 = self.sol[:, 0]
        self.theta2 = self.sol[:, 2]
        self.x1 = self.L1 * np.sin(self.theta1)
        self.y1 = -self.L1 * np.cos(self.theta1)
        self.x2 = self.x1 + self.L2 * np.sin(self.theta2)
        self.y2 = self.y1 - self.L2 * np.cos(self.theta2)

    def deriv(self, t, y, L1, L2, m1, m2):
        """Return the first derivatives of y = theta1, z1, theta2, z2."""
        theta1, z1, theta2, z2 = y

        c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

        theta1dot = z1
        z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
                 (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
        theta2dot = z2
        z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) +
                 m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
        return theta1dot, z1dot, theta2dot, z2dot

    def iterative_solve(self):
        T = self.tmax + self.dt
        q, r = np.divmod(T, self.restart)
        if r!=0:
            dt = np.arange(0, r, self.dt)
            sol = solve_ivp(self.deriv, np.array((0, r)),
                            self.y0, method='RK23', t_eval=dt,
                            args=(self.L1, self.L2, self.m1, self.m2))
            sol = (sol.y).T

        dt_1 = np.arange(0, 1, self.dt)
        for _ in range(int(q)):
            try:
                d_sol = solve_ivp(self.deriv, np.array((0, 1)),
                                  sol[-1], method='RK23', t_eval=dt_1,
                                  args=(self.L1, self.L2, self.m1, self.m2))
                sol = np.concatenate((sol, (d_sol.y).T))

            except Exception:
                d_sol = solve_ivp(self.deriv, np.array((0, 1)),
                                  self.y0, method='RK23', t_eval=dt_1,
                                  args=(self.L1, self.L2, self.m1, self.m2))
                
                sol = deepcopy((d_sol.y).T)

        return sol


# p1 = Pendulum(0, 0, to_trace=False, trace_delete=False, tmax=2.04)