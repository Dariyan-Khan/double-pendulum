import numpy as np
from dp_tidy import Pendulum
import matplotlib.pyplot as plt
from matplotlib import animation

del_rate = 50  # int(p1.num_frames / 10)


def animate(i, p1, p2, ln1, ln2, trace1, trace2, time_text, time_template,
            tr1_x, tr1_y, tr2_x, tr2_y):
    ln1.set_data([0, p1.x1[i], p1.x2[i]], [0, p1.y1[i], p1.y2[i]])
    ln2.set_data([0, p2.x1[i], p2.x2[i]], [0, p2.y1[i], p2.y2[i]])
    if p1.to_trace:
        if i % del_rate == 0 and i > 0 and p1.trace_delete:
            del tr1_x[del_rate:]
            del tr1_y[del_rate:]

        tr1_x.append(p1.x2[i])
        tr1_y.append(p1.y2[i])
        trace1.set_data(tr1_x, tr1_y)

    if p2.to_trace:
        if i % del_rate == 0 and i > 0 and p2.trace_delete:
            del tr2_x[del_rate:]
            del tr2_y[del_rate:]
        tr2_x.append(p2.x2[i])
        tr2_y.append(p2.y2[i])
        trace2.set_data(tr2_x, tr2_y)

    time_text.set_text(time_template % (((p1.tmax + p1.dt) /
                                        p1.num_frames) * i))  # (i*dt))


def show_anim(p1, p2):
    assert p1.to_trace == p2.to_trace and p1.trace_delete == p2.trace_delete
    assert p1.tmax == p2.tmax and p1.dt == p2.dt
    tr1_x, tr1_y = [], []
    tr2_x, tr2_y = [], []
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_facecolor('w')
    ax.get_xaxis().set_ticks([])    # enable this to hide x axis ticks
    ax.get_yaxis().set_ticks([])    # enable this to hide y axis ticks
    ax.set_ylim(-4, 4)
    ax.set_xlim(-4, 4)

    time_template = 'time : %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    ln1, = plt.plot([], [], 'bo-', lw=1, markersize=8)
    ln2, = plt.plot([], [], 'ro-', lw=1, markersize=8)

    trace1, = plt.plot([], [], 'y-', alpha=0.75, lw=0.5, markersize=1)
    trace2, = plt.plot([], [], 'g-', alpha=0.75, lw=0.5, markersize=1)

    animate_wrapper = lambda i: animate(i, p1, p2, ln1, ln2, trace1, trace2, # noqa
                                        time_text, time_template, tr1_x,
                                        tr1_y, tr2_x, tr2_y)


    ani = animation.FuncAnimation(fig, animate_wrapper,
                                  frames=int(p1.num_frames),
                                  interval=50, repeat=False)  # 30 -> 500

    # Uncomment function to save animation as a gif
    # ani.save('pendulums.gif', writer='pillow', fps=len(t[t < 1]))

    plt.show()


if __name__ == "__main__":
    eps = np.pi / 10

    # y0 = np.array([np.pi, 0, np.pi/2, 0])
    # y1 = np.array([np.pi/7 + eps, 0, np.pi/7 + eps, 0])

    th1 = np.pi
    th2 = np.pi / 2

    p1 = Pendulum(th1, th2, to_trace=False, trace_delete=False)
    p2 = Pendulum(th1 + eps, th2 + eps, to_trace=False, trace_delete=False)

    print(p1.num_frames)

    show_anim(p1, p2)
