import numpy as np
from scipy.integrate import solve_ivp

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
                 g=9.81, to_trace=True, trace_delete=True, tmax=15, dt=0.04):
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.to_trace = to_trace
        self.trace_delete = trace_delete
        self.tmax = tmax
        self.dt = dt
        self.t = np.arange(0, tmax+dt, dt)
        self.num_frames = (50/3) * tmax
        self.y0 = [th1, th1_dot, th2, th2_dot]
        self.y = solve_ivp(self.deriv, np.array((0, tmax + dt)), self.y0, max_step=0.1,
                           t_eval=self.t, args=(self.L1, self.L2, self.m1,
                           self.m2))
        self.sol = (self.y.y).T
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