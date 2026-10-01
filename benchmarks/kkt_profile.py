"""Small KKT timing benchmark for gapese-cpp.

Run from the repository root:
    python benchmarks/kkt_profile.py
"""

from pathlib import Path
from statistics import median

import numpy as np
import scipy.sparse as sp
import gapese


PROBLEMS = ("liswet1_100", "flp4_4", "liswet1_200")
WARMUP = 1
REPEATS = 5

# Same solver options as the frozen gapese-benchmark project.
OPTIONS = {
    "tol_kkt": 1e-5,
    "tol_dgap": 1e-6,
    "max_iter": 1000,
    "nu_D": 1e-4,
    "nu_x": 1e-5,
    "nu_xi": 1e-5,
    "nu_b": 1e-5,
    "nu_d": 1e-5,
    "mu_min": 1.0,
    "mu_update": 2.0,
    "nu_alpha": 0.7,
    "rho": 0.1,
    "beta_init": 1.0,
}

TIMING_FIELDS = (
    "total",
    "search_direction",
    "kkt_assembly",
    "lu_analyze",
    "lu_factorize",
    "lu_solve",
    "search_other",
)

DATA_DIR = Path(__file__).resolve().parent / "data"


def _csr(data, prefix):
    shape = tuple(int(v) for v in data[f"{prefix}_shape"])
    return sp.csr_matrix(
        (data[f"{prefix}_data"], data[f"{prefix}_indices"], data[f"{prefix}_indptr"]),
        shape=shape,
    )


def _load_problem(name):
    """Load a frozen MacMPEC problem and convert it to Gapese's public API."""
    with np.load(DATA_DIR / f"{name}.npz", allow_pickle=False) as data:
        Q = _csr(data, "Q")
        A = _csr(data, "A")
        L = _csr(data, "L")
        R = _csr(data, "R")
        g = np.asarray(data["g"], dtype=float)
        lb_A = np.asarray(data["lb_A"], dtype=float)
        ub_A = np.asarray(data["ub_A"], dtype=float)
        lb_L = np.asarray(data["lb_L"], dtype=float)
        lb_R = np.asarray(data["lb_R"], dtype=float)
        x0 = np.asarray(data["x0"], dtype=float)

    # Same two-sided constraint split used by gapese-benchmark.
    finite_lb = np.isfinite(lb_A)
    finite_ub = np.isfinite(ub_A)
    equality = finite_lb & finite_ub & (lb_A == ub_A)
    lower = finite_lb & ~equality
    upper = finite_ub & ~equality

    A_eq = A[equality, :].tocsr()
    b_eq = lb_A[equality].copy()

    C_blocks = []
    d_blocks = []
    if np.any(lower):
        C_blocks.append(A[lower, :].tocsr())
        d_blocks.append(lb_A[lower].copy())
    if np.any(upper):
        C_blocks.append((-A[upper, :]).tocsr())
        d_blocks.append((-ub_A[upper]).copy())

    if C_blocks:
        C = sp.vstack(C_blocks, format="csr")
        d = np.concatenate(d_blocks)
    else:
        C = sp.csr_matrix((0, A.shape[1]), dtype=float)
        d = np.zeros(0, dtype=float)

    n = int(Q.shape[0])
    m = int(L.shape[0])
    n_eq = int(A_eq.shape[0])
    n_d = int(C.shape[0])

    Z_eq_m = sp.csr_matrix((n_eq, m), dtype=float)
    Z_m_m = sp.csr_matrix((m, m), dtype=float)
    I_m = sp.eye(m, format="csr", dtype=float)

    A_x = sp.vstack([A_eq, L, R], format="csc")
    A_lambda = sp.vstack([Z_eq_m, -I_m, Z_m_m], format="csc")
    A_eta = sp.vstack([Z_eq_m, Z_m_m, -I_m], format="csc")

    problem_size = {
        "n_x": n,
        "n_lambda": m,
        "n_b": int(A_x.shape[0]),
        "n_d": n_d,
    }

    zero_hessian = sp.csc_matrix((m, m), dtype=float)
    ccqp = {
        "Q_x": Q.tocsc(),
        "Q_lambda": zero_hessian,
        "Q_eta": zero_hessian.copy(),
        "r_x": g.copy(),
        "r_lambda": np.zeros(m),
        "r_eta": np.zeros(m),
        "A_x": A_x,
        "A_lambda": A_lambda,
        "A_eta": A_eta,
        "b": np.concatenate([b_eq, lb_L, lb_R]),
        "C_x": C.tocsc(),
        "C_lambda": sp.csc_matrix((n_d, m), dtype=float),
        "C_eta": sp.csc_matrix((n_d, m), dtype=float),
        "d": d,
    }

    initial = {
        "x": x0.copy(),
        "lambda": np.asarray(L @ x0).reshape(-1) - lb_L,
        "eta": np.asarray(R @ x0).reshape(-1) - lb_R,
    }
    return problem_size, ccqp, initial


def _solve_once(problem_size, ccqp, initial, options):
    # Use a fresh Solver for each solve, matching the full benchmark.
    solver = gapese.Solver(problem_size, options)
    _, info = solver.solve_ccqp(ccqp, initial)

    timing = info.get("Time", {})
    missing = [name for name in TIMING_FIELDS if name not in timing]
    if missing:
        raise RuntimeError(
            "Missing timing fields: " + ", ".join(missing) +
            ". Rebuild gapese-cpp after applying the KKT profiling patch."
        )

    return {
        "status": int(info.get("status", 0)),
        "iter": int(info.get("iter_num", 0)),
        "KKT_res": float(info.get("KKT_res", np.nan)),
        "D_gap_res": float(info.get("D_gap_res", np.nan)),
        "cost": float(info.get("cost", np.nan)),
        **{name: float(timing[name]) for name in TIMING_FIELDS},
    }


def _ms(seconds):
    return 1000.0 * seconds


def _run_problem(name, options):
    problem_size, ccqp, initial = _load_problem(name)
    full_kkt_dim = (
        problem_size["n_x"]
        + 2 * problem_size["n_lambda"]
        + problem_size["n_b"]
        + 2 * problem_size["n_d"]
    )
    reduced_kkt_dim = full_kkt_dim - problem_size["n_d"]
    print(
        f"\n[{name}] reduced KKT dimension = {reduced_kkt_dim} "
        f"(full = {full_kkt_dim})"
    )

    for _ in range(WARMUP):
        _solve_once(problem_size, ccqp, initial, options)

    rows = []
    for i in range(REPEATS):
        row = _solve_once(problem_size, ccqp, initial, options)
        rows.append(row)
        print(
            f"  run {i + 1}: status={row['status']} iter={row['iter']}  "
            f"total={_ms(row['total']):8.3f}  "
            f"search={_ms(row['search_direction']):8.3f}  "
            f"assembly={_ms(row['kkt_assembly']):8.3f}  "
            f"analyze={_ms(row['lu_analyze']):8.3f}  "
            f"factorize={_ms(row['lu_factorize']):8.3f}  "
            f"solve={_ms(row['lu_solve']):8.3f}  "
            f"other={_ms(row['search_other']):8.3f}"
        )

    def med(field):
        return median(row[field] for row in rows)

    total = med("total")
    profiled_kkt = (
        med("kkt_assembly")
        + med("lu_analyze")
        + med("lu_factorize")
        + med("lu_solve")
    )
    print(
        "  median : "
        f"total={_ms(total):8.3f}  "
        f"search={_ms(med('search_direction')):8.3f}  "
        f"assembly={_ms(med('kkt_assembly')):8.3f}  "
        f"analyze={_ms(med('lu_analyze')):8.3f}  "
        f"factorize={_ms(med('lu_factorize')):8.3f}  "
        f"solve={_ms(med('lu_solve')):8.3f}  "
        f"other={_ms(med('search_other')):8.3f}  "
        f"KKT/total={100.0 * profiled_kkt / total:5.1f}%"
    )
    last = rows[-1]
    print(
        f"           status={last['status']}  iter={last['iter']}  "
        f"KKT_res={last['KKT_res']:.3e}  D_gap_res={last['D_gap_res']:.3e}  "
        f"cost={last['cost']:.12e}"
    )


def main():
    options = dict(gapese.default_options())
    options["print_level"] = 0
    options.update(OPTIONS)

    print(f"Gapese KKT profile: warmup={WARMUP}, repeats={REPEATS}")
    for name in PROBLEMS:
        _run_problem(name, options)


if __name__ == "__main__":
    main()
