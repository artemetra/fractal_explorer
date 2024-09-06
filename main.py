from typing import Callable, Any
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime
from enum import Enum

INIT_N = 2**12
point_count = INIT_N
# 10 is typically enough
ITER_COUNT = 1000
T = 0.1
D_MULT = 0.2  # proportion of the viewport to pan
C = 1
angle = 0
BOUND = 2


def F(z: np.ndarray, f_args: list[Any], f_kwargs: dict[Any, Any]) -> np.ndarray:
    res: np.ndarray = z**2 + f_args[0]
    return res


# initial r1, initial r2, ...
IR1, IR2, II1, II2 = -3, 3, -3, 3
r1, r2, i1, i2 = IR1, IR2, II1, II2


def ttt():
    return f"[{datetime.now().strftime('%H:%M:%S.%f')}]"


class ConvergenceStyle(Enum):
    DIFFERENCE = (1,)
    BOUND = (2,)


def test_convergence(
    arr: np.ndarray,
    f: Callable[[np.ndarray, list, dict], np.ndarray],
    iter_count=ITER_COUNT,
    tolerance=1e-12,
    convergence_style=ConvergenceStyle.BOUND,
    f_args=[],
    f_kwargs={},
):
    """Tests the convergence for arr of complex numbers."""

    def superf(z: np.ndarray, n: int) -> np.ndarray:
        for i in range(n):
            if iter_count >= 10:
                if i % (iter_count // 10) == 0:
                    # Prints on every 10%
                    print(ttt(), f"iter: {i+1}", end="\r")
            z = f(z, f_args, f_kwargs)
        print()
        return z

    match convergence_style:
        case ConvergenceStyle.BOUND:
            res = superf(arr, iter_count)
            res = np.abs(res)
            return arr[res < BOUND]
        case ConvergenceStyle.DIFFERENCE:
            res = superf(arr, iter_count)
            res1 = f(res, f_args, f_kwargs)
            # if the next application of f is approx the same as
            # the previous one, then the original number converges.
            err = np.abs(res1 - res)
            return arr[err < tolerance]  # filter those that converged.
        case _:
            raise NotImplementedError


def zoom(x1: float, x2: float, y1: float, y2: float, zoom_in=True, t=T):
    """Given the current viewbox (x1,y1), (x2,y2), zoom in by t"""
    # equation of a line through (x1,y1), (x2,y2)
    # x2 is never x1
    assert (x2 - x1) != 0

    def f(x):
        return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1

    xm = (x1 + x2) / 2

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


def generate_points(r1, r2, i1, i2, n) -> np.ndarray:
    """Generates n uniformly distributed complex numbers."""
    # We need to expand the limits by a factor of sqrt(2) in order to support rotation
    # TODO? not do this maybe? the problem is that only ≈63% of points are actually being
    # displayed at most, with the rest being out of the viewport but still tested
    r1p, r2p, i1p, i2p = zoom(r1, r2, i1, i2, zoom_in=False, t=1 / (2 * np.sqrt(2)))
    points = np.random.uniform(r1p, r2p, n) + 1j * np.random.uniform(i1p, i2p, n)
    return points


def pan_x(x1: float, x2: float, y1: float, y2: float, is_pos=True, d_mult=D_MULT):
    global angle
    d = ((abs(x1 - x2)) / 2) * d_mult
    if not is_pos:
        d *= -1
    return (
        x1 + d * np.cos(angle),
        x2 + d * np.cos(angle),
        y1 - d * np.sin(angle),
        y2 - d * np.sin(angle),
    )


def pan_y(x1: float, x2: float, y1: float, y2: float, is_pos=True, d_mult=D_MULT):
    global angle
    d = ((abs(x1 - x2)) / 2) * d_mult
    if not is_pos:
        d *= -1
    return (
        x1 + d * np.sin(angle),
        x2 + d * np.sin(angle),
        y1 + d * np.cos(angle),
        y2 + d * np.cos(angle),
    )


def on_press(event):
    # TODO: fix this!!!
    global r1, r2, i1, i2  # Use global variables to update the bounds
    global point_count
    global angle
    sys.stdout.flush()
    match event.key:
        case "up":
            r1, r2, i1, i2 = pan_y(r1, r2, i1, i2, is_pos=True)
        case "down":
            r1, r2, i1, i2 = pan_y(r1, r2, i1, i2, is_pos=False)
        case "right":
            r1, r2, i1, i2 = pan_x(r1, r2, i1, i2, is_pos=True)
        case "left":
            r1, r2, i1, i2 = pan_x(r1, r2, i1, i2, is_pos=False)
        case "i":
            r1, r2, i1, i2 = zoom(r1, r2, i1, i2, zoom_in=True)
        case "k":
            r1, r2, i1, i2 = zoom(r1, r2, i1, i2, zoom_in=False)
        case "j":
            angle -= (2 * np.pi) / 32
            angle %= 2 * np.pi
        case "l":
            angle += (2 * np.pi) / 32
            angle %= 2 * np.pi
        case "+":
            point_count *= 2
        case "-":
            point_count //= 2
            if point_count == 0:
                point_count = 1
        case "r":
            # re render
            pass
        case "ctrl+r":  # TODO make OS agnostic?
            # reset viewport
            r1, r2, i1, i2 = IR1, IR2, II1, II2
            point_count = INIT_N
            angle = 0
        case _ as key:
            print(f"key: {key}")
            # Explicitly don't replot whenever any other key is pressed
            return
    plot_the_thing(ax, r1, r2, i1, i2, point_count)


def plot_the_thing(ax: Axes, r1: float, r2: float, i1: float, i2: float, n=point_count):
    ax.clear()

    print(ttt(), "generating points...")
    points = generate_points(r1, r2, i1, i2, n)

    print(
        ttt(), f"testing convergence: {point_count} points, {ITER_COUNT} iterations..."
    )
    convergent = test_convergence(
        points, F, tolerance=sys.float_info.epsilon, f_args=[points]
    )
    min_conv = round(
        abs(np.min(convergent, where=~np.isnan(convergent), initial=99999)), 4
    )

    # https://math.stackexchange.com/questions/2119527/how-do-you-rotate-a-point-in-the-complex-plane-by-theta-radians
    viewport_center = (r1 + r2) / 2 + 1j * (i1 + i2) / 2
    w = viewport_center * np.ones(convergent.shape)
    convergent = w + np.exp(1j * angle) * (convergent - w)

    ax.scatter(convergent.real, convergent.imag, s=0.15)
    ax.set_xlim((r1, r2))
    ax.set_ylim((i1, i2))
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.draw()

    conv_percentage = round(convergent.size / point_count * 100, 3)
    print(
        ttt(),
        f"done, convergence ratio: {conv_percentage}%, min value: {min_conv}",
    )
    fmt_viewport_center = (
        float(round(viewport_center.real, 5)),
        float(round(viewport_center.imag, 5)),
    )
    ax.set_title(f"viewport_center={fmt_viewport_center}; angle={round(angle, 5)}")


fig, ax = plt.subplots()
plot_the_thing(ax, r1, r2, i1, i2)

fig.canvas.mpl_connect("key_press_event", on_press)
plt.show()
