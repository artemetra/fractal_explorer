print("importing...")
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
from datetime import datetime

N = 5000
ITER_COUNT = 50
T = 0.1
D_MULT = 0.2  # proportion of the viewport to pan
C = 1
F = lambda z: z**z**z
r1, r2, i1, i2 = -3, 3, -3, 3


def ttt():
    return f"[{datetime.now().strftime('%H:%M:%S.%f')}]"


def supercos(x, n):
    for _ in range(n):
        x = np.sin(x)
    return x


def super_uhh(x, n):
    for _ in range(n):
        x = x**2 + C
    return x


def generate_points(r1, r2, i1, i2, n):
    """Generates n uniformly distributed complex numbers."""
    points = np.random.uniform(r1, r2, n) + 1j * np.random.uniform(i1, i2, n)
    return points


def test_convergence(arr, f, iter_count=ITER_COUNT, tolerance=1e-12):
    """Tests the convergence for arr of complex numbers."""

    def superf(z, n):
        for i in range(n):
            if i % (iter_count // 10) == 0:
                print(ttt(), f"iter: {i}")
            z = f(z)
        return z

    res = superf(arr, iter_count)
    res1 = f(res)
    # if the next application of cos is approx the same as
    # the previous one, then the original number converges.
    err = np.abs(res1 - res)
    return arr[err < tolerance]  # filter those that converged.


def zoom(x1, x2, y1, y2, zoom_in=True, t=T):
    """Given the current viewbox (x1,y1), (x2,y2), zoom in by t"""
    # equation of a line through (x1,y1), (x2,y2)
    f = lambda x: (((y2 - y1) / (x2 - x1)) * (x - x1) + y1)  # x2 is never x1
    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2
    # _p is for 'prime', these are the new points
    # we only really need to 'zoom in' the x coordinates,
    # since the equation of the line stays the same and
    # we can get new y's by applying f.
    if not zoom_in:
        t *= -1
    x1p = x1 + t * abs(xm - x1)
    x2p = x2 - t * abs(xm - x2)
    y1p = f(x1p)
    y2p = f(x2p)
    return (x1p, x2p, y1p, y2p)


def pan_x(x1, x2, y1, y2, is_pos=True, d_mult=D_MULT):
    d = ((abs(x1 - x2)) / 2) * d_mult
    if not is_pos:
        d *= -1
    return (x1 + d, x2 + d, y1, y2)


def pan_y(x1, x2, y1, y2, is_pos=True, d_mult=D_MULT):
    d = ((abs(x1 - x2)) / 2) * d_mult
    if not is_pos:
        d *= -1
    return (x1, x2, y1 + d, y2 + d)


def on_press(event):
    # TODO: fix this!!!
    global r1, r2, i1, i2  # Use global variables to update the bounds
    global N
    sys.stdout.flush()
    match event.key:
        case "up":
            r1, r2, i1, i2 = pan_y(r1, r2, i1, i2, True)
        case "down":
            r1, r2, i1, i2 = pan_y(r1, r2, i1, i2, False)
        case "right":
            r1, r2, i1, i2 = pan_x(r1, r2, i1, i2, True)
        case "left":
            r1, r2, i1, i2 = pan_x(r1, r2, i1, i2, False)
        case "i":
            r1, r2, i1, i2 = zoom(r1, r2, i1, i2, True)
        case "k":
            r1, r2, i1, i2 = zoom(r1, r2, i1, i2, False)
        case "+":
            N *= 2
        case "-":
            N //= 2
            if N == 0:
                N = 1
        case _ as key:
            print(f"key: {key}")
            # Explicitly don't replot whenever any other key is pressed
            return
    plot_the_thing(ax, r1, r2, i1, i2, N)


def plot_the_thing(ax, r1, r2, i1, i2, n=N):
    ax.clear()
    print(ttt(), "generating points...")
    points = generate_points(r1, r2, i1, i2, n)
    print(ttt(), f"testing convergence: {N} points, {ITER_COUNT} iterations...")
    convergent = test_convergence(points, F, tolerance=1e-12)
    min_conv = round(abs(np.min(convergent, where=~np.isnan(convergent), initial=999)),4)
    print(
        ttt(),
        f"done, convergence ratio: {convergent.size/N * 100}%, min value: {min_conv}",
    )
    print(ttt(), "plotting...")
    ax.scatter(convergent.real, convergent.imag, s=0.15)
    ax.set_xlim([r1, r2])
    ax.set_ylim([i1, i2])
    plt.draw()
    print(ttt(), "complete!")


fig, ax = plt.subplots()
plot_the_thing(ax, r1, r2, i1, i2)

fig.canvas.mpl_connect("key_press_event", on_press)

plt.show()
