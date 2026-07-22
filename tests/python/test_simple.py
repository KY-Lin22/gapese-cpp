import numpy as np

import gapese


def make_problem(use_sparse=False):
    problem = {"n_x": 2, "n_lambda": 1, "n_b": 1, "n_d": 2}
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
    if use_sparse:
        import scipy.sparse as sp

        for key in [
            "Q_x",
            "Q_lambda",
            "Q_eta",
            "A_x",
            "A_lambda",
            "A_eta",
            "C_x",
            "C_lambda",
            "C_eta",
        ]:
            ccqp[key] = sp.csc_matrix(ccqp[key])

    initial = {
        "x": np.array([0.25, -0.5]),
        "lambda": np.array([0.3]),
        "eta": np.array([0.7]),
    }
    return problem, options, ccqp, initial


def test_dense_and_stateful_warm_start():
    problem, options, ccqp, initial = make_problem(False)
    solver = gapese.Solver(problem, options)
    solution, info = solver.solve_ccqp(ccqp, initial)

    assert info["status"] == 1
    assert info["KKT_res"] <= options["tol_kkt"] * 1.01
    assert info["D_gap_res"] <= options["tol_dgap"] * 1.01
    assert abs(np.sum(solution["x"]) - 1.0) < 1e-7
    assert np.min(solution["x"]) >= -1e-7

    warm = dict(solution)
    warm_solution, warm_info = solver.solve_ccqp(ccqp, warm)
    assert warm_info["status"] == 1
    assert np.linalg.norm(warm_solution["x"] - solution["x"], ord=np.inf) < 1e-7


def test_scipy_sparse_input():
    try:
        problem, options, ccqp, initial = make_problem(True)
    except ImportError:
        return
    solver = gapese.Solver(problem, options)
    _, info = solver.solve_ccqp(ccqp, initial)
    assert info["status"] == 1


if __name__ == "__main__":
    test_dense_and_stateful_warm_start()
    test_scipy_sparse_input()
    print("All Python simple tests passed.")
