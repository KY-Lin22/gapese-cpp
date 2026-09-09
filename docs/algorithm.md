# Gapese algorithm

This document summarizes the numerical method implemented by Gapese for complementarity-constrained quadratic programming (CCQP). The presentation follows the current solver formulation and uses the same notation as the public problem data.

## 1. CCQP formulation

Gapese considers

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

Here $x\in\mathbb{R}^{n_x}$ and $\lambda,\eta\in\mathbb{R}^{n_\lambda}$. The complementarity condition means

$$
\lambda\ge0,\qquad \eta\ge0,\qquad \lambda^\top\eta=0.
$$

Direct treatment of complementarity constraints is difficult because standard nonlinear-programming constraint qualifications fail at feasible complementarity points. Gapese instead uses a D-gap penalty reformulation.

## 2. D-gap penalty reformulation

For constants $b>a>0$, define

$$
\phi_a(\lambda,\eta)
=\frac{1}{2a}\left(
\|\eta\|_2^2-\|\max(0,\eta-a\lambda)\|_2^2
\right),
$$

$$
\phi_b(\lambda,\eta)
=\frac{1}{2b}\left(
\|\eta\|_2^2-\|\max(0,\eta-b\lambda)\|_2^2
\right),
$$

where the maximum is applied elementwise. The D-gap function is

$$
\begin{aligned}
\phi_{ab}(\lambda,\eta)
&=\phi_a(\lambda,\eta)-\phi_b(\lambda,\eta)\\
&=\frac{b-a}{2ab}\|\eta\|_2^2
-\frac{1}{2a}\|\max(0,\eta-a\lambda)\|_2^2
+\frac{1}{2b}\|\max(0,\eta-b\lambda)\|_2^2.
\end{aligned}
$$

The properties used by Gapese are:

- $\phi_{ab}(\lambda,\eta)\ge0$;
- $\phi_{ab}(\lambda,\eta)=0$ if and only if $0\le\lambda\perp\eta\ge0$;
- $\phi_{ab}$ is semismoothly differentiable; and
- it is partially separable across the complementarity pairs.

Gapese replaces the complementarity constraint with the penalty term $\mu\phi_{ab}$, giving

$$
\begin{aligned}
P_{\mathrm{gap}}(\mu):\quad
\min_{x,\lambda,\eta}\quad
& J(x,\lambda,\eta)+\mu\phi_{ab}(\lambda,\eta)\\
\text{s.t.}\quad
& A_x x + A_\lambda\lambda + A_\eta\eta-b=0,\\
& C_x x + C_\lambda\lambda + C_\eta\eta-d\ge0,
\end{aligned}
$$

with $\mu>0$. The penalty parameter is increased when the penalized KKT system has converged but the D-gap residual remains above tolerance.

## 3. Interleaved complementarity variable

For each complementarity pair, define

$$
\xi_i=\begin{bmatrix}\lambda_i\\\eta_i\end{bmatrix},
$$

and interleave all pairs as

$$
\xi=
[\lambda_1,\eta_1,\ldots,\lambda_{n_\lambda},\eta_{n_\lambda}]^\top
=S\begin{bmatrix}\lambda\\\eta\end{bmatrix},
$$

where $S$ is a permutation matrix. The transformed quadratic and constraint data are

$$
Q_\xi=S^{-\top}
\begin{bmatrix}
Q_\lambda&0\\
0&Q_\eta
\end{bmatrix}S^{-1},
\qquad
r_\xi=S^{-\top}
\begin{bmatrix}r_\lambda\\r_\eta\end{bmatrix},
$$

$$
A_\xi=[A_\lambda\;A_\eta]S^{-1},
\qquad
C_\xi=[C_\lambda\;C_\eta]S^{-1}.
$$

The penalized problem becomes

$$
\begin{aligned}
\min_{x,\xi}\quad
& J(x,\xi)+\mu\phi_{ab}(\xi)\\
\text{s.t.}\quad
& A_xx+A_\xi\xi-b=0,\\
& C_xx+C_\xi\xi-d\ge0,
\end{aligned}
$$

with

$$
J(x,\xi)=
\frac12 x^\top Q_xx+r_x^\top x
+\frac12\xi^\top Q_\xi\xi+r_\xi^\top\xi.
$$

The implementation uses this interleaved ordering internally.

## 4. D-gap structure and Hessian modification

Partial separability gives

$$
\phi_{ab}(\xi)=\sum_{i=1}^{n_\lambda}\delta_{ab}(\xi_i),
$$

so a generalized Hessian has the block-diagonal structure

$$
\partial^2\phi_{ab}(\xi)
=\mathrm{diag}
\left(
\partial^2\delta_{ab}(\xi_1),\ldots,
\partial^2\delta_{ab}(\xi_{n_\lambda})
\right).
$$

For one pair $(\lambda_i,\eta_i)$, the $2\times2$ block is indefinite only in the strict region

$$
b\lambda_i>\eta_i>a\lambda_i.
$$

In that region,

$$
\partial^2\delta_{ab}(\xi_i)=
\begin{bmatrix}
-a&1\\
1&-1/b
\end{bmatrix}.
$$

Its negative eigenvalue is

$$
\kappa_{\min}
=-\frac12\left(a+\frac1b\right)
-\frac12\sqrt{\left(\frac1b-a\right)^2+4}<0.
$$

Gapese modifies only the strict indefinite blocks. Let

$$
\omega_i=
\begin{cases}
1,& b\lambda_i>\eta_i>a\lambda_i,\\
0,& \text{otherwise},
\end{cases}
$$

then

$$
B_{\mathrm{reg}}(\xi)
=-\kappa_{\min}
\,\mathrm{diag}(\omega_1I_2,\ldots,\omega_{n_\lambda}I_2),
$$

and the Newton approximation uses

$$
B(\xi)=\partial^2\phi_{ab}(\xi)+B_{\mathrm{reg}}(\xi)\succeq0.
$$

The strict inequalities above are important: boundary points use the adjacent positive-semidefinite branch rather than the indefinite branch.

## 5. Semismooth KKT system

Let $\gamma_b\in\mathbb{R}^{n_b}$ be the multiplier of the equality constraints and $\gamma_d\in\mathbb{R}^{n_d}$ the multiplier associated with the linear inequalities. Introduce the inequality slack $v_d\in\mathbb{R}^{n_d}$:

$$
C_xx+C_\xi\xi-d-v_d=0.
$$

For the default `fb_sigma = 0`, Gapese uses the standard Fisher-Burmeister mapping

$$
\psi(v,\gamma)=\sqrt{v^2+\gamma^2}-v-\gamma,
$$

which satisfies

$$
\psi(v,\gamma)=0
\quad\Longleftrightarrow\quad
0\le v\perp\gamma\ge0.
$$

The implementation also permits `fb_sigma > 0`, in which case it evaluates the smoothed mapping

$$
\psi_\sigma(v,\gamma)
=\sqrt{v^2+\gamma^2+\sigma^2}-v-\gamma.
$$

Applying the mapping elementwise gives $\Psi(v_d,\gamma_d)$. The semismooth residual is

$$
\mathcal{T}(x,\xi,v_d,\gamma_b,\gamma_d)=
\begin{bmatrix}
Q_xx+r_x+A_x^\top\gamma_b-C_x^\top\gamma_d\\
Q_\xi\xi+r_\xi+\mu\nabla\phi_{ab}(\xi)+A_\xi^\top\gamma_b-C_\xi^\top\gamma_d\\
A_xx+A_\xi\xi-b\\
C_xx+C_\xi\xi-d-v_d\\
\Psi(v_d,\gamma_d)
\end{bmatrix}.
$$

The implementation reports the infinity norm

$$
\|\mathcal{T}\|_\infty
$$

as the KKT residual.

## 6. Semismooth Newton direction

At each iteration, Gapese solves a regularized Newton system for

$$
\Delta z=
[\Delta x,\Delta\xi,\Delta v_d,\Delta\gamma_b,\Delta\gamma_d].
$$

With $D_v$ and $D_\gamma$ denoting diagonal generalized derivatives of the Fisher-Burmeister mapping, the linear system has the block form

$$
\begin{bmatrix}
Q_x+\nu_xI & 0 & 0 & A_x^\top & -C_x^\top\\
0 & Q_\xi+\mu B(\xi)+\nu_\xi I & 0 & A_\xi^\top & -C_\xi^\top\\
A_x & A_\xi & 0 & -\nu_bI & 0\\
C_x & C_\xi & -I & 0 & 0\\
0 & 0 & D_v-\nu_dI & 0 & D_\gamma-\nu_dI
\end{bmatrix}
\Delta z
=-\mathcal{T}.
$$

The regularization parameters correspond to the public options `nu_x`, `nu_xi`, `nu_b`, and `nu_d`.

The current C++ implementation assembles this system as an Eigen sparse matrix. It first attempts an Eigen `SparseLU` solve and uses `SparseQR` as a robustness fallback when the LU solve is unsuccessful or produces a non-finite solution.

The Newton step is rejected if its infinity norm is non-finite or reaches the `newton_step_max` threshold.

## 7. Merit-function globalization

Gapese globalizes the Newton step with an $\ell_1$ merit function

$$
\Theta(x,\xi,v_d,\gamma_b,\gamma_d;\mu)
=J(x,\xi)+\mu\phi_{ab}(\xi)+\beta\|M(x,\xi,v_d,\gamma_d)\|_1,
$$

where

$$
M(x,\xi,v_d,\gamma_d)=
\begin{bmatrix}
A_xx+A_\xi\xi-b\\
C_xx+C_\xi\xi-d-v_d\\
\Psi(v_d,\gamma_d)
\end{bmatrix}.
$$

For the current iterate and Newton direction, the directional model is

$$
D\Theta
=DJ+\mu D\phi_{ab}-\beta\|M\|_1.
$$

Starting from `beta_init`, the implementation increases $\beta$ when necessary so that it exceeds the trial bound implied by the descent condition with parameter `rho`.

The line search starts from $\alpha=1$ and checks the Armijo condition

$$
\Theta(z+\alpha\Delta z)
\le
\Theta(z)+\nu_D\alpha D\Theta.
$$

If the condition fails, the stepsize is contracted by

$$
\alpha\leftarrow\nu_\alpha\alpha
$$

until the condition is satisfied or `alpha_min` is reached. Failure to find an acceptable step at the minimum stepsize terminates the solve with failure status.

## 8. Penalty continuation and termination

The penalty parameter starts at

$$
\mu=\mu_{\min}.
$$

At each iteration, Gapese evaluates the quadratic objective, the D-gap residual $\phi_{ab}(\xi)$, and the KKT residual $\|\mathcal{T}\|_\infty$.

The termination logic is:

1. If `KKT_res <= tol_kkt` and `D_gap_res <= tol_dgap`, return success.
2. If the KKT tolerance is satisfied but the D-gap tolerance is not, increase

   $$
   \mu\leftarrow\min(\sigma_\mu\mu,\mu_{\max}),
   $$

   where $\sigma_\mu$ is the `mu_update` option and $\mu_{\max}$ is `mu_max`.

3. If the D-gap tolerance still fails when the maximum penalty parameter has been reached, terminate with failure.
4. Otherwise compute a Newton direction, perform the merit line search, and update all primal-dual variables.
5. Terminate with failure if the iteration limit, Newton-step threshold, linear solve, or line search prevents convergence.

Thus, `status == 1` is assigned only when **both** requested residual conditions are satisfied.

## 9. Initialization and warm starts

The required initial values are

$$
x^0,\qquad \lambda^0,\qquad \eta^0.
$$

The optional primal-dual values are

$$
v_d^0,\qquad \gamma_b^0,\qquad \gamma_d^0.
$$

If these optional values are omitted, Gapese initializes them as vectors of ones. The returned solution contains all six vectors, so the entire solution can be passed back as a warm start for a repeated solve with compatible dimensions.

## 10. Algorithm outline

A compact view of the implemented iteration is:

```text
input CCQP data and initial guess
        |
        v
interleave (lambda, eta) -> xi
        |
        v
initialize mu = mu_min
        |
        v
compute KKT and D-gap residuals
        |
        +-- both satisfy tolerances --> success
        |
        +-- KKT satisfied only --> increase mu
        |
        v
build modified D-gap Hessian B(xi)
        |
        v
assemble and solve semismooth Newton system
        |
        v
select beta and perform merit backtracking
        |
        v
update x, xi, v_d, gamma_b, gamma_d
        |
        +------------------------------> repeat
```

## References

1. F. Facchinei and J. S. Pang, *Finite-Dimensional Variational Inequalities and Complementarity Problems*, Springer, 2003.
2. K. Lin and T. Ohtsuka, “A gap penalty method for optimal control of linear complementarity systems,” *63rd IEEE Conference on Decision and Control*, 2024.
3. K. Lin and T. Ohtsuka, “A gap penalty reformulation for mathematical programming with complementarity constraints: Convergence analysis,” *IEEE Control Systems Letters*, vol. 9, pp. 925–930, 2025.
4. K. Lin and T. Ohtsuka, “A non-interior-point continuation method for the optimal control problem with equilibrium constraints,” *Automatica*, vol. 171, 111940, 2025.
5. J. Nocedal and S. J. Wright, *Numerical Optimization*, Springer, 2006.
6. J. M. Peng and Y. Yuan, “Unconstrained methods for generalized complementarity problems,” *Journal of Computational Mathematics*, pp. 253–264, 1997.
