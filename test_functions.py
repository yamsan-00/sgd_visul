"""
Test (objective) functions and their analytical gradients.

Each entry in the FUNCTIONS dictionary contains:
    func           : callable  -  the scalar loss  f(x)
    grad           : callable  -  analytical gradient  nabla f(x)
    min            : np.ndarray -  location of the global minimum
    range          : tuple(xmin, xmax, ymin, ymax)
    default_start  : tuple(x0, y0)  -  sensible starting point
    formula_tex    : str  -  LaTeX-style formula for display
"""

import numpy as np


# =====================================================================
#  Individual functions
# =====================================================================

def bowl_func(x):
    """f(x,y) = x^2 + y^2"""
    return x[0] ** 2 + x[1] ** 2


def bowl_grad(x):
    return np.array([2.0 * x[0], 2.0 * x[1]])


def rosenbrock_func(x, a=1.0, b=100.0):
    """f(x,y) = (a-x)^2 + b*(y - x^2)^2"""
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x, a=1.0, b=100.0):
    dx = -2.0 * (a - x[0]) - 4.0 * b * x[0] * (x[1] - x[0] ** 2)
    dy = 2.0 * b * (x[1] - x[0] ** 2)
    return np.array([dx, dy])


def rastrigin_func(x, A=10.0):
    """f(x,y) = 2A + (x^2 - A cos 2pi x) + (y^2 - A cos 2pi y)"""
    return (A * 2
            + (x[0] ** 2 - A * np.cos(2.0 * np.pi * x[0]))
            + (x[1] ** 2 - A * np.cos(2.0 * np.pi * x[1])))


def rastrigin_grad(x, A=10.0):
    dx = 2.0 * x[0] + 2.0 * np.pi * A * np.sin(2.0 * np.pi * x[0])
    dy = 2.0 * x[1] + 2.0 * np.pi * A * np.sin(2.0 * np.pi * x[1])
    return np.array([dx, dy])


def saddle_func(x):
    """f(x,y) = x^2 - y^2"""
    return x[0] ** 2 - x[1] ** 2


def saddle_grad(x):
    return np.array([2.0 * x[0], -2.0 * x[1]])


def elongated_func(x):
    """f(x,y) = 0.1 x^2 + y^2"""
    return 0.1 * x[0] ** 2 + x[1] ** 2


def elongated_grad(x):
    return np.array([0.2 * x[0], 2.0 * x[1]])


# =====================================================================
#  Function registry
# =====================================================================

FUNCTIONS = {
    "Quadratic Bowl": {
        "func": bowl_func,
        "grad": bowl_grad,
        "min": np.array([0.0, 0.0]),
        "range": (-5, 5, -5, 5),
        "default_start": (-4.0, -4.0),
        "formula_tex": r"$f(\mathbf{x}) = x_1^2 + x_2^2$",
        "grad_tex": r"$\nabla f = \left[2x_1,\; 2x_2\right]$",
    },
    "Rosenbrock": {
        "func": rosenbrock_func,
        "grad": rosenbrock_grad,
        "min": np.array([1.0, 1.0]),
        "range": (-2, 2, -1, 3),
        "default_start": (-1.0, 2.0),
        "formula_tex": r"$f(\mathbf{x}) = (1-x_1)^2 + 100\,(x_2 - x_1^2)^2$",
        "grad_tex": (
            r"$\frac{\partial f}{\partial x_1} = -2(1-x_1) - 400\,x_1(x_2 - x_1^2)$"
            "\n"
            r"$\frac{\partial f}{\partial x_2} = 200\,(x_2 - x_1^2)$"
        ),
    },
    "Rastrigin": {
        "func": rastrigin_func,
        "grad": rastrigin_grad,
        "min": np.array([0.0, 0.0]),
        "range": (-5.12, 5.12, -5.12, 5.12),
        "default_start": (4.0, 4.0),
        "formula_tex": (
            r"$f(\mathbf{x}) = 20 + \left(x_1^2 - 10\cos 2\pi x_1\right)"
            r" + \left(x_2^2 - 10\cos 2\pi x_2\right)$"
        ),
        "grad_tex": (
            r"$\nabla f = \left[2x_1 + 20\pi\sin 2\pi x_1,\;"
            r" 2x_2 + 20\pi\sin 2\pi x_2\right]$"
        ),
    },
    "Saddle Point": {
        "func": saddle_func,
        "grad": saddle_grad,
        "min": np.array([0.0, 0.0]),
        "range": (-5, 5, -5, 5),
        "default_start": (-4.0, -4.0),
        "formula_tex": r"$f(\mathbf{x}) = x_1^2 - x_2^2$",
        "grad_tex": r"$\nabla f = \left[2x_1,\; -2x_2\right]$",
    },
    "Elongated Bowl": {
        "func": elongated_func,
        "grad": elongated_grad,
        "min": np.array([0.0, 0.0]),
        "range": (-5, 5, -5, 5),
        "default_start": (-4.0, -4.0),
        "formula_tex": r"$f(\mathbf{x}) = 0.1\,x_1^2 + x_2^2$",
        "grad_tex": r"$\nabla f = \left[0.2\,x_1,\; 2\,x_2\right]$",
    },
}
