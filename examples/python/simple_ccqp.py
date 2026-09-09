"""Minimal Python example for solving a CCQP with Gapese."""

import numpy as np

import gapese


problem = {
    "n_x": 2,
    "n_lambda": 1,
    "n_b": 1,
    "n_d": 2,
}

options = gapese.default_options()
options.update(
    print_level=0,
    dgap_a=0.5,
    dgap_b=2.0,
    tol_kkt=1e-5,
    tol_dgap=1e-5,
    mu_min=1e1,
    mu_max=1e6,
)

ccqp = {
    "Q_x": np.diag([1.0, 1.0]),
    "Q_lambda": np.array([[1.0]]),
    "Q_eta": np.array([[1.0]]),
    "r_x": np.array([-1.0, -1.0]),
    "r_lambda": np.array([-1.0]),
    "r_eta": np.array([-1.0]),
    "A_x": np.array([[1.0, 1.0]]),
    "A_lambda": np.zeros((1, 1)),
    "A_eta": np.zeros((1, 1)),
    "b": np.array([1.0]),
    "C_x": np.eye(2),
    "C_lambda": np.zeros((2, 1)),
    "C_eta": np.zeros((2, 1)),
    "d": np.zeros(2),
}

initial = {
    "x": np.array([0.25, -0.5]),
    "lambda": np.array([0.3]),
    "eta": np.array([0.7]),
}

solver = gapese.Solver(problem, options)
solution, info = solver.solve_ccqp(ccqp, initial)

print("status:", info["status"])
print("message:", info["msg"])
print("x:", solution["x"])
print("lambda:", solution["lambda"])
print("eta:", solution["eta"])
print("KKT residual:", info["KKT_res"])
print("D-gap residual:", info["D_gap_res"])
