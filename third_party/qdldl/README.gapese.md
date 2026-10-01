# Embedded QDLDL

Gapese vendors QDLDL internally for the quasi-definite reduced KKT solve.
The embedded build uses double precision and 32-bit integer indices so the
upper-triangular CSC matrix can be passed directly from Eigen without copying
its index or value arrays.

Upstream: https://github.com/osqp/qdldl
Version/API: 0.1.9
License: Apache-2.0 (see `LICENSE`).
