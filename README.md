# Gapese C++ v0

A first C++17/Eigen implementation of the Gap Penalty Semismooth (Gapese) solver for the general CCQP

\[
\min_{x,\lambda,\eta}
\frac12[x;\lambda;\eta]^T\operatorname{diag}(Q_x,Q_\lambda,Q_\eta)[x;\lambda;\eta]
+[r_x;r_\lambda;r_\eta]^T[x;\lambda;\eta]
\]

subject to linear equalities, linear inequalities, and
`0 <= lambda ⟂ eta >= 0`.

This package contains:

- a stateful C++ core using Eigen sparse matrices;
- a stateful MATLAB MEX interface (`GapeseMex`);
- a stateful Python/pybind11 interface (`gapese.Solver`);
- a small C++ test, MATLAB regression test, and Python test.

## Algorithm parity with the MATLAB reference

The v0 core follows the supplied MATLAB implementation:

- interleaved `xi = [lambda_1, eta_1, ..., lambda_n, eta_n]`;
- the same D-gap value and gradient;
- regularization only for the strict indefinite region
  `b*lambda_i > eta_i > a*lambda_i`; boundaries use the adjacent PSD branch;
- the FB generalized derivative `(-1,-1)` at `(v_d,gamma_d)=(0,0)` when `sigma=0`;
- the same KKT residual and regularized Newton matrix;
- the same penalty continuation, including reevaluation after `mu` changes;
- the same merit-function backtracking line search;
- optional warm starts for `v_d`, `gamma_b`, and `gamma_d`.

The first version assembles the KKT sparse matrix on every Newton iteration. Pattern caching and symbolic-factorization reuse are intentionally deferred until numerical parity is established.

## Dependencies

- C++17 compiler
- CMake 3.18+
- Eigen 3.4
- MATLAB with a configured C++ MEX compiler, for the MATLAB interface
- Python 3.9+, NumPy, pybind11, and scikit-build-core, for the Python interface
- SciPy is optional and only needed to pass SciPy sparse matrices directly

CMake first searches for installed Eigen/pybind11. If they are unavailable and `GAPESE_FETCH_DEPENDENCIES=ON`, it downloads the pinned releases. For offline builds, install the dependencies locally and set `EIGEN3_INCLUDE_DIR`.

## Build and test the C++ core

```bash
cmake -S . -B build \
  -DGAPESE_BUILD_PYTHON=OFF \
  -DGAPESE_BUILD_TESTS=ON \
  -DEIGEN3_INCLUDE_DIR=/path/to/eigen-3.4.0
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

When Eigen is installed as a CMake package, omit `EIGEN3_INCLUDE_DIR`.

## MATLAB stateful interface

1. Put Eigen 3.4 somewhere on disk. The supplied directory must contain `Eigen/`.
2. In MATLAB:

```matlab
cd <gapese_cpp>/matlab
build_gapese_mex('C:\path\to\eigen-3.4.0')
addpath(pwd)
```

3. Construct and reuse a solver:

```matlab
Option = GapeseMex.create_Option();
solver = GapeseMex(Prob, Option);
[Sol, Info] = solver.solve_CCQP(CCQP, Init);
```

4. With the reference MATLAB `@Gapese` on the path, run:

```matlab
Result = test_gapese_mex_simple();
```

The MEX gateway accepts MATLAB dense or sparse matrices. Internally they are converted to Eigen column-major sparse matrices. `GapeseMex` owns an opaque MEX handle and destroys it automatically.

## Python stateful interface

From the project root:

```bash
python -m pip install .
```

For an offline Eigen installation:

```bash
CMAKE_ARGS="-DEIGEN3_INCLUDE_DIR=/path/to/eigen-3.4.0 -DGAPESE_FETCH_DEPENDENCIES=OFF" \
python -m pip install .
```

Usage:

```python
import gapese

options = gapese.default_options()
solver = gapese.Solver(problem, options)
solution, info = solver.solve_ccqp(ccqp, initial)
```

Both dense NumPy matrices and SciPy sparse matrices are accepted. See `python/example_simple.py` and `tests/python/test_simple.py`.

## Inputs and outputs

### Problem dimensions

```text
n_x, n_lambda, n_b, n_d
```

### CCQP data

```text
Q_x, Q_lambda, Q_eta
r_x, r_lambda, r_eta
A_x, A_lambda, A_eta, b
C_x, C_lambda, C_eta, d
```

### Initial guess

Required:

```text
x, lambda, eta
```

Optional primal-dual warm start:

```text
v_d, gamma_b, gamma_d
```

### Solution

```text
x, lambda, eta, v_d, gamma_b, gamma_d
```

### Information

```text
cost, D_gap_res, KKT_res, mu, iter_num, status, msg,
Time, line_search_status
```

## v0 limitations

- Eigen `SparseLU` is the primary linear solver; `SparseQR` is used as a robustness fallback.
- The KKT sparsity pattern and symbolic factorization are not yet cached.
- The MATLAB MEX source has to be compiled locally against the user's MATLAB and compiler.
- The Python wheel is built locally; prebuilt wheels are not included.
- Exact floating-point iterates can differ slightly from MATLAB `\` because the sparse linear solver and pivoting strategy differ.

## Suggested next milestone

After the simple regression tests pass:

1. compare the MEX and MATLAB solvers on a generated medium-size sparse CCQP;
2. compare repeated stateful solves with primal-dual warm start;
3. profile KKT assembly versus factorization;
4. add fixed-pattern value updates and symbolic-analysis reuse.
