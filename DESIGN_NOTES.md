# Design notes

## Variable order

The Newton variable and KKT residual order is

```text
[x, xi, v_d, gamma_b, gamma_d]
```

where

```text
xi = [lambda_1, eta_1, ..., lambda_n, eta_n].
```

## KKT blocks

The assembled matrix is the direct C++ analogue of the MATLAB matrix:

```text
Q_x + nu_x I                 A_x'       -C_x'
                Q_xi+mu B    A_xi'      -C_xi'
A_x             A_xi         -nu_b I
C_x             C_xi  -I
                       D_v-nu_d I        D_gamma-nu_d I
```

with zero blocks omitted from sparse assembly.

## Stateful interfaces

The C++ `Solver` stores problem dimensions, options, and evaluators. The current v0 still rebuilds transformed data and the KKT matrix for each call/iteration. This preserves a clean baseline before pattern caching.

The MATLAB wrapper stores an opaque `uint64` handle in `GapeseMex`. The Python wrapper stores the C++ solver directly in a pybind11 object.

## Sparse input

MATLAB dense and sparse matrices, NumPy dense matrices, and SciPy CSC-compatible sparse matrices are converted to Eigen `SparseMatrix<double, ColMajor, int>`.
