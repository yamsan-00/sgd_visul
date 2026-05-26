"""
Optimization algorithms for SGD convergence comparison.

Provides:
    - vanilla_sgd       : Standard gradient descent
    - momentum_sgd      : SGD with classical momentum
"""

import numpy as np


def vanilla_sgd(grad_func, x0, lr=0.01, n_steps=100):
    """
    Vanilla (standard) Stochastic Gradient Descent.

    Update rule:
        theta_{t+1} = theta_t - eta * nabla J(theta_t)

    Parameters
    ----------
    grad_func : callable
        Function returning the gradient at a given point.
    x0 : np.ndarray
        Initial parameter vector.
    lr : float
        Learning rate (eta).
    n_steps : int
        Number of optimisation steps.

    Returns
    -------
    np.ndarray, shape (n_steps+1, d)
        Full trajectory including the initial point.
    """
    trajectory = [x0.copy()]
    x = x0.copy()
    for _ in range(n_steps):
        grad = grad_func(x)
        x = x - lr * grad
        trajectory.append(x.copy())
    return np.array(trajectory)


def momentum_sgd(grad_func, x0, lr=0.01, momentum=0.9, n_steps=100):
    """
    SGD with (classical) Momentum.

    Update rules:
        v_t       = gamma * v_{t-1} + eta * nabla J(theta_t)
        theta_{t+1} = theta_t - v_t

    Parameters
    ----------
    grad_func : callable
        Function returning the gradient at a given point.
    x0 : np.ndarray
        Initial parameter vector.
    lr : float
        Learning rate (eta).
    momentum : float
        Momentum coefficient (gamma), in [0, 1).
    n_steps : int
        Number of optimisation steps.

    Returns
    -------
    np.ndarray, shape (n_steps+1, d)
        Full trajectory including the initial point.
    """
    trajectory = [x0.copy()]
    x = x0.copy()
    v = np.zeros_like(x)
    for _ in range(n_steps):
        grad = grad_func(x)
        v = momentum * v + lr * grad
        x = x - v
        trajectory.append(x.copy())
    return np.array(trajectory)
