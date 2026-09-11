# Gapese

Gapese is a C++17/Eigen solver for **complementarity-constrained quadratic programming (CCQP)**, with Python and MATLAB MEX interfaces.

The solver uses a D-gap penalty reformulation, a structure-exploiting semismooth Newton step, and a merit-function line search. The same CCQP data model and solver options are exposed across the C++, Python, and MATLAB interfaces.

## Problem formulation

Gapese considers CCQPs of the form

$$
\begin{aligned}
\min_{x,\lambda,\eta}\quad
& \frac{1}{2}x^\top Q_x x + r_x^\top x
+ \frac{1}{2}\lambda^\top Q_\lambda \lambda + r_\lambda^\top \lambda
+ \frac{1}{2}\eta^\top Q_\eta \eta + r_\eta^\top \eta \\
\text{s.t.}\quad
& A_x x + A_\lambda \lambda + A_\eta \eta - b = 0, \\
& C_x x + C_\lambda \lambda + C_\eta \eta - d \ge 0, \\
& 0 \le \lambda \perp \eta \ge 0.
\end{aligned}
$$

with

- $x \in \mathbb{R}^{n_x}$,
- $\lambda,\eta \in \mathbb{R}^{n_\lambda}$,
- $n_b$ linear equality constraints, and
- $n_d$ linear inequality constraints.

## Features

- C++17 core based on Eigen sparse matrices.
- Stateful `gapese::Solver`, `gapese.Solver`, and `GapeseMex` interfaces.
- D-gap penalty continuation for complementarity constraints.
- Semismooth Newton search direction with a modified D-gap Hessian.
- Merit-function backtracking line search.
- Optional primal-dual warm starts through `v_d`, `gamma_b`, and `gamma_d`.
- NumPy dense and SciPy sparse matrix input in Python.
- MATLAB dense and sparse matrix input through the MEX interface.
- Strict validation of problem dimensions and the canonical solver options.

## Dependencies

### C++

- C++17 compiler
- CMake 3.18+
- Eigen 3.4

The CMake build first searches for an installed Eigen package. If Eigen is not found and `GAPESE_FETCH_DEPENDENCIES=ON`, Eigen 3.4.0 is fetched automatically. For offline builds, provide `EIGEN3_INCLUDE_DIR` and disable dependency fetching.

### Python

- Python 3.9+
- NumPy
- pybind11 and scikit-build-core as build dependencies
- SciPy only when SciPy sparse matrices are passed directly

### MATLAB

- MATLAB with a configured C++ MEX compiler
- Eigen 3.4 headers

## Installation

### Python

From the repository root:

```bash
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e ".[test]"
```

For an offline build with a local Eigen installation, pass the corresponding CMake arguments through your build environment, including `EIGEN3_INCLUDE_DIR` and `GAPESE_FETCH_DEPENDENCIES=OFF`.

### C++

A direct CMake build creates only the C++ core by default. The Python module, tests, and examples are optional targets controlled by `GAPESE_BUILD_PYTHON`, `GAPESE_BUILD_TESTS`, and `GAPESE_BUILD_EXAMPLES`. Install/package rules are enabled by default for a top-level Gapese build and disabled when Gapese is included as a subdirectory; `GAPESE_INSTALL` can override that behavior.

Configure a release build with the C++ tests and example enabled:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGAPESE_BUILD_PYTHON=OFF -DGAPESE_BUILD_TESTS=ON -DGAPESE_BUILD_EXAMPLES=ON
cmake --build build --config Release
cmake -E chdir build ctest -C Release --output-on-failure
```

If Eigen is not installed as a CMake package and dependency fetching is disabled, also pass:

```bash
-DEIGEN3_INCLUDE_DIR=/path/to/eigen-3.4.0 -DGAPESE_FETCH_DEPENDENCIES=OFF
```

Install the C++ library to a prefix of your choice:

```bash
cmake --install build --config Release --prefix /path/to/gapese-install
```

A downstream CMake project can then use the installed package as follows:

```cmake
find_package(gapese CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE gapese::gapese)
```

Make the Gapese install prefix discoverable through `CMAKE_PREFIX_PATH` (or `gapese_DIR`). If Eigen is available only as a header tree rather than an installed CMake package, also provide `EIGEN3_INCLUDE_DIR` when configuring the downstream project.

### MATLAB MEX

From MATLAB, compile the MEX interface by providing the Eigen include directory:

```matlab
cd("path/to/gapese-cpp/matlab")
build_gapese_mex("path/to/eigen-3.4.0")
```

`build_gapese_mex` also reads the `EIGEN3_INCLUDE_DIR` environment variable when no path argument is supplied. After compilation, add only the repository `matlab/` directory to the MATLAB path when needed:

```matlab
addpath("path/to/gapese-cpp/matlab")
```

## Quick start: Python

The following example solves a two-variable CCQP with one complementarity pair:

```python
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

print(info["status"], info["msg"])
print("x =", solution["x"])
print("KKT residual =", info["KKT_res"])
print("D-gap residual =", info["D_gap_res"])
```

The full example is available at `examples/python/simple_ccqp.py`.

## C++ interface

The public C++ API is centered on:

```cpp
#include <gapese/solver.hpp>

// Define problem dimensions and solver options.
gapese::ProblemSize problem{n_x, n_lambda, n_b, n_d};
gapese::Options options;

gapese::Solver solver(problem, options);
gapese::SolveResult result = solver.solve(ccqp, initial);
```

The main public types are:

- `gapese::Options`
- `gapese::ProblemSize`
- `gapese::CcqpData`
- `gapese::InitialGuess`
- `gapese::Solution`
- `gapese::Timing`
- `gapese::Info`
- `gapese::SolveResult`
- `gapese::SparseMatrix`, `gapese::Vector`, and `gapese::Triplet`

See `examples/cpp/simple_ccqp.cpp` for a complete example.

## MATLAB MEX interface

The MATLAB interface follows the same workflow:

```matlab
Option = GapeseMex.default_options();
solver = GapeseMex(Prob, Option);
[Sol, Info] = solver.solve_ccqp(CCQP, Init);
```

A partial option structure is allowed; unspecified canonical options use their defaults. See `examples/matlab/simple_ccqp.m` for a complete example.

## Problem data

### Problem dimensions

| Field | Meaning |
| --- | --- |
| `n_x` | Dimension of $x$; must be positive |
| `n_lambda` | Number of complementarity pairs; must be positive |
| `n_b` | Number of equality constraints; must be nonnegative |
| `n_d` | Number of inequality constraints; must be nonnegative |

### CCQP data

| Field | Size |
| --- | --- |
| `Q_x` | $n_x \times n_x$ |
| `Q_lambda`, `Q_eta` | $n_\lambda \times n_\lambda$ |
| `r_x` | $n_x$ |
| `r_lambda`, `r_eta` | $n_\lambda$ |
| `A_x` | $n_b \times n_x$ |
| `A_lambda`, `A_eta` | $n_b \times n_\lambda$ |
| `b` | $n_b$ |
| `C_x` | $n_d \times n_x$ |
| `C_lambda`, `C_eta` | $n_d \times n_\lambda$ |
| `d` | $n_d$ |

### Initial guess

Required fields:

```text
x, lambda, eta
```

Optional primal-dual warm-start fields:

```text
v_d, gamma_b, gamma_d
```

If the optional fields are omitted, the solver initializes them to vectors of ones with the corresponding dimensions. A returned `Solution` can therefore be reused directly as a full warm start for a later solve with the same problem dimensions.

## Solver options

`gapese.default_options()` and `GapeseMex.default_options()` return the same 20 canonical options. The C++ interface exposes the same values through `gapese::Options`.

| Option | Default | Requirement | Meaning |
| --- | ---: | --- | --- |
| `print_level` | `2` | `0`, `1`, or `2` | `0`: silent; `1`: solver summary; `2`: iteration log and summary |
| `dgap_a` | `0.1` | finite, `> 0` | Lower D-gap parameter $a$ |
| `dgap_b` | `10.0` | finite, `> dgap_a` | Upper D-gap parameter $b$ |
| `fb_sigma` | `0.0` | finite, `>= 0` | Smoothing term in the Fisher-Burmeister mapping |
| `max_iter` | `500` | positive integer | Maximum solver iterations |
| `tol_kkt` | `1e-4` | finite, `> 0` | KKT residual tolerance |
| `tol_dgap` | `1e-4` | finite, `> 0` | D-gap residual tolerance |
| `newton_step_max` | `1e10` | finite, `> 0` | Failure threshold for the infinity norm of the Newton step |
| `mu_min` | `1.0` | finite, `> 0` | Initial D-gap penalty parameter |
| `mu_max` | `1e6` | finite, `>= mu_min` | Maximum D-gap penalty parameter |
| `mu_update` | `2.0` | finite, `> 1` | Multiplicative penalty update factor |
| `nu_x` | `1e-8` | finite, `>= 0` | Newton regularization for the $x$ block |
| `nu_xi` | `1e-8` | finite, `>= 0` | Newton regularization for the transformed complementarity block |
| `nu_b` | `1e-8` | finite, `>= 0` | Regularization associated with equality multipliers |
| `nu_d` | `1e-8` | finite, `>= 0` | Regularization associated with the inequality/FB block |
| `beta_init` | `1.0` | finite, `> 0` | Initial merit-function constraint weight |
| `rho` | `0.1` | finite, in `(0,1)` | Descent parameter used when selecting the merit weight |
| `alpha_min` | `1e-16` | finite, in `(0,1)` | Minimum line-search stepsize |
| `nu_alpha` | `0.5` | finite, in `(0,1)` | Backtracking contraction factor |
| `nu_D` | `1e-4` | finite, in `(0,1)` | Armijo reduction parameter |

Unknown option names are rejected. Python and MATLAB allow a partial canonical option dictionary/structure and fill unspecified values from the defaults.

## Solution and solver information

The returned solution contains:

```text
x, lambda, eta, v_d, gamma_b, gamma_d
```

Python and MATLAB return an information object/structure with:

```text
cost
D_gap_res
KKT_res
mu
iter_num
status
msg
Time
line_search_status
```

`status == 1` means that both the KKT and D-gap residuals satisfy their requested tolerances. `status == 0` indicates solver failure or termination without satisfying both criteria; `msg` contains the corresponding human-readable reason.

`Time` contains:

```text
check_termination
search_direction
step_size
else
total
```

The C++ `Info` and `Timing` structures expose the same information using the C++ field names `dgap_res`, `kkt_res`, `message`, `timing`, and `other`.

## Examples

- `examples/cpp/simple_ccqp.cpp`
- `examples/python/simple_ccqp.py`
- `examples/matlab/simple_ccqp.m`

The Python example can be run from the repository root after installing the package:

```bash
python examples/python/simple_ccqp.py
```

After compiling the MATLAB MEX interface, the MATLAB example can be run from the repository root with:

```matlab
run("examples/matlab/simple_ccqp.m")
```

## Tests

### C++

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGAPESE_BUILD_PYTHON=OFF -DGAPESE_BUILD_TESTS=ON -DGAPESE_BUILD_EXAMPLES=ON
cmake --build build --config Release
cmake -E chdir build ctest -C Release --output-on-failure
```

With `GAPESE_BUILD_EXAMPLES=ON`, the same build also compiles `gapese_example_simple` from `examples/cpp/simple_ccqp.cpp`.

### Python

```bash
python -m pip install -e ".[test]"
python -m pytest tests/python -q
```

### MATLAB

After compiling `gapese_mex`, run:

```matlab
run("tests/matlab/test_gapese_mex_simple.m")
```

## Algorithm

The numerical method is documented in [`docs/algorithm.md`](docs/algorithm.md). In brief, Gapese:

1. replaces complementarity with a D-gap penalty;
2. interleaves each pair $(\lambda_i,\eta_i)$ into a transformed variable $\xi_i$;
3. preserves the block structure of the D-gap generalized Hessian and regularizes only its strict indefinite region;
4. solves the penalized KKT system with a semismooth Newton step;
5. globalizes the iteration with an $\ell_1$ merit-function line search; and
6. declares success only when both the KKT and D-gap residual tolerances are satisfied.

## License

Gapese is released under the BSD 3-Clause License. See [`LICENSE`](LICENSE) for details.
