"""Unit tests for sgd_visul.algorithms."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms import vanilla_sgd, momentum_sgd


def quadratic_grad(x):
    # grad of f(x) = 0.5 * (x0^2 + 10 x1^2)  -> minimum at origin
    return np.array([x[0], 10.0 * x[1]])


def test_vanilla_sgd_shape_contains_initial():
    x0 = np.array([2.0, 2.0])
    traj = vanilla_sgd(quadratic_grad, x0, lr=0.01, n_steps=50)
    assert traj.shape == (51, 2)
    np.testing.assert_allclose(traj[0], x0)


def test_vanilla_sgd_descends():
    x0 = np.array([2.0, 2.0])
    traj = vanilla_sgd(quadratic_grad, x0, lr=0.01, n_steps=200)
    # final point should have a much smaller magnitude than initial
    assert np.linalg.norm(traj[-1]) < 0.5 * np.linalg.norm(x0)


def test_momentum_sgd_descends_faster_initially():
    x0 = np.array([2.0, 2.0])
    sg = vanilla_sgd(quadratic_grad, x0.copy(), lr=0.005, n_steps=20)
    mom = momentum_sgd(quadratic_grad, x0.copy(), lr=0.005, momentum=0.8, n_steps=20)
    assert np.linalg.norm(mom[-1]) < np.linalg.norm(sg[-1])


def test_constant_grad_vanilla_sgd_step_matches_update_rule():
    grad = lambda x: np.array([1.0, 0.0])
    x0 = np.array([0.0, 0.0])
    traj = vanilla_sgd(grad, x0, lr=0.1, n_steps=3)
    # after 3 steps x0 = -0.3
    np.testing.assert_allclose(traj[-1], np.array([-0.3, 0.0]))


def test_momentum_sgd_shape():
    x0 = np.array([1.0, 1.0])
    traj = momentum_sgd(quadratic_grad, x0, lr=0.01, momentum=0.9, n_steps=15)
    assert traj.shape == (16, 2)
