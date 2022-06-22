import numpy as np

from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * np.exp(-b * x) + (1-a) * np.exp(-c * x)



def membrane_time_constant(xp, yp, dt=0.001):
    imin = np.argmin(yp)
    xp = xp[imin:]
    xp -= xp[0]
    yp = yp[imin:]
    yp -= yp[-1]
    yp /= yp[0]

    x = np.arange(0, np.max(xp), dt)
    y = np.interp(x, xp, yp)

    return np.max(
        1 / curve_fit(func, x, y, bounds=([0,0,0], [1,1,1]))[0][1:]
    )



if __name__ == '__main__':
    membrane_time_constant(np.arange(0, 1000, 0.1), np.arange(1,0,-0.1))
