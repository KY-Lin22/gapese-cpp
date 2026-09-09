import numpy as np
import pytest

import gapese


CANONICAL_OPTION_NAMES = {
    "print_level",
    "dgap_a",
    "dgap_b",
    "fb_sigma",
    "max_iter",
    "tol_kkt",
    "tol_dgap",
    "newton_step_max",
    "mu_min",
    "mu_max",
    "mu_update",
    "nu_x",
    "nu_xi",
    "nu_b",
    "nu_d",
    "beta_init",
    "rho",
    "alpha_min",
    "nu_alpha",
    "nu_D",
}

LEGACY_OPTION_NAMES = {
    "D_gap_param_a",
    "D_gap_param_b",
    "FB_sigma",
    "tol_KKT_res",
    "tol_D_gap_res",
    "Newton_step_norm",
}

FLOAT_OPTION_NAMES = CANONICAL_OPTION_NAMES - {"print_level", "max_iter"}


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


def test_default_options_are_canonical():
    options = gapese.default_options()
    assert set(options) == CANONICAL_OPTION_NAMES
    assert not (set(options) & LEGACY_OPTION_NAMES)


def test_partial_options_are_allowed():
    problem, _, _, _ = make_problem(False)
    solver = gapese.Solver(problem, {"print_level": 0})
    assert solver is not None


def test_unknown_option_is_rejected():
    problem, _, _, _ = make_problem(False)
    with pytest.raises(ValueError, match="unknown Gapese option: tol_kktt"):
        gapese.Solver(problem, {"tol_kktt": 1e-5})


def test_non_string_option_name_is_rejected():
    problem, _, _, _ = make_problem(False)
    with pytest.raises(ValueError, match="Gapese option names must be strings"):
        gapese.Solver(problem, {1: 0})


@pytest.mark.parametrize("legacy_name", sorted(LEGACY_OPTION_NAMES))
def test_legacy_option_aliases_are_rejected(legacy_name):
    problem, _, _, _ = make_problem(False)
    with pytest.raises(ValueError, match="unknown Gapese option"):
        gapese.Solver(problem, {legacy_name: 1.0})


@pytest.mark.parametrize("option_name", sorted(FLOAT_OPTION_NAMES))
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_options_are_rejected(option_name, value):
    problem, options, _, _ = make_problem(False)
    options[option_name] = value
    with pytest.raises(ValueError):
        gapese.Solver(problem, options)


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
    test_default_options_are_canonical()
    test_partial_options_are_allowed()
    test_unknown_option_is_rejected()
    test_non_string_option_name_is_rejected()
    for legacy_name in sorted(LEGACY_OPTION_NAMES):
        test_legacy_option_aliases_are_rejected(legacy_name)
    for option_name in sorted(FLOAT_OPTION_NAMES):
        for value in (np.nan, np.inf, -np.inf):
            test_nonfinite_options_are_rejected(option_name, value)
    test_dense_and_stateful_warm_start()
    test_scipy_sparse_input()
    print("All Python simple tests passed.")
