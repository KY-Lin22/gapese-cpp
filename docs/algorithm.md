# Gapese algorithm

Gapese solves complementarity-constrained quadratic programs (CCQPs) using a gap penalty reformulation and a semismooth Newton method. This document summarizes the algorithm implemented in Gapese.

## 1. CCQP formulation

We consider the CCQP

$$
\begin{aligned}
\min_{x,\lambda,\eta}\quad
& \frac{1}{2}
\begin{bmatrix}
x \\
\lambda \\
\eta
\end{bmatrix}^{T}
\begin{bmatrix}
Q_x & 0 & 0 \\
0 & Q_{\lambda} & 0 \\
0 & 0 & Q_{\eta}
\end{bmatrix}
\begin{bmatrix}
x \\
\lambda \\
\eta
\end{bmatrix}
+
\begin{bmatrix}
r_x \\
r_{\lambda} \\
r_{\eta}
\end{bmatrix}^{T}
\begin{bmatrix}
x \\
\lambda \\
\eta
\end{bmatrix}
\\
\text{s.t.}\quad
& A_x x + A_{\lambda}\lambda + A_{\eta}\eta - b = 0, \\
& C_x x + C_{\lambda}\lambda + C_{\eta}\eta - d \geq 0, \\
& 0 \leq \lambda \perp \eta \geq 0.
\end{aligned}
$$

The decision variables are $x \in \mathbb{R}^{n_x}$, $\lambda \in \mathbb{R}^{n_\lambda}$, and $\eta \in \mathbb{R}^{n_\lambda}$. The cost function is formed by $Q_x$, $Q_{\lambda}$, $Q_{\eta}$ and $r_x$, $r_{\lambda}$, $r_{\eta}$; the equality constraints are formed by $A_x$, $A_{\lambda}$, $A_{\eta}$ and $b$; and the inequality constraints are formed by $C_x$, $C_{\lambda}$, $C_{\eta}$ and $d$.

A point $(x,\lambda,\eta)$ satisfying the equality, inequality, and complementarity constraints is a feasible point of the CCQP. Complementarity constraints violate standard constraint qualifications such as LICQ and MFCQ at feasible points, which motivates dedicated numerical methods for CCQPs.

## 2. Gap penalty reformulation for CCQP

### D-gap function and its key properties

Let $\lambda,\eta \in \mathbb{R}^{n_\lambda}$ and let $a,b$ satisfy $b>a>0$. Define

$$
\begin{aligned}
\varphi^{a}(\lambda,\eta)
&=
\frac{1}{2a}
\left(
\lVert \eta \rVert_2^2
-
\lVert \max(0,\eta-a\lambda) \rVert_2^2
\right),
\\
\varphi^{b}(\lambda,\eta)
&=
\frac{1}{2b}
\left(
\lVert \eta \rVert_2^2
-
\lVert \max(0,\eta-b\lambda) \rVert_2^2
\right).
\end{aligned}
$$

The D-gap function [8] is

$$
\begin{aligned}
\varphi^{ab}(\lambda,\eta)
&=
\varphi^{a}(\lambda,\eta)-\varphi^{b}(\lambda,\eta)
\\
&=
\frac{b-a}{2ab}\lVert \eta \rVert_2^2
-
\frac{1}{2a}\lVert \max(0,\eta-a\lambda) \rVert_2^2
+
\frac{1}{2b}\lVert \max(0,\eta-b\lambda) \rVert_2^2 .
\end{aligned}
$$

Its key properties are:

- **Non-negativity:** $\varphi^{ab}(\lambda,\eta)\geq 0$ for all $\lambda,\eta$.
- **Equivalence:** $\varphi^{ab}(\lambda,\eta)=0$ if and only if $0\leq\lambda\perp\eta\geq0$.
- **Differentiability:** $\varphi^{ab}(\lambda,\eta)$ is semismoothly differentiable.
- **Partial separability:**

$$
\varphi^{ab}(\lambda,\eta)
=
\sum_{i=1}^{n_\lambda}
\delta^{ab}(\lambda_i,\eta_i),
$$

where

$$
\delta^{ab}(\lambda_i,\eta_i)
=
\frac{b-a}{2ab}\eta_i^2
-
\frac{1}{2a}
\left\{\max(0,\eta_i-a\lambda_i)\right\}^2
+
\frac{1}{2b}
\left\{\max(0,\eta_i-b\lambda_i)\right\}^2 .
$$

### Gap penalty reformulation

Using the non-negativity and equivalence of the D-gap function, the complementarity constraints are replaced by a penalty term. For a penalty parameter $\mu>0$,

$$
\begin{aligned}
\mathcal{P}_{\mathrm{gap}}(\mu):\quad
\min_{x,\lambda,\eta}\quad
& \frac{1}{2}
\begin{bmatrix}
x \\
\lambda \\
\eta
\end{bmatrix}^{T}
\begin{bmatrix}
Q_x & 0 & 0 \\
0 & Q_{\lambda} & 0 \\
0 & 0 & Q_{\eta}
\end{bmatrix}
\begin{bmatrix}
x \\
\lambda \\
\eta
\end{bmatrix}
+
\begin{bmatrix}
r_x \\
r_{\lambda} \\
r_{\eta}
\end{bmatrix}^{T}
\begin{bmatrix}
x \\
\lambda \\
\eta
\end{bmatrix}
+
\mu\varphi^{ab}(\lambda,\eta)
\\
\text{s.t.}\quad
& A_x x + A_{\lambda}\lambda + A_{\eta}\eta - b = 0,
\\
& C_x x + C_{\lambda}\lambda + C_{\eta}\eta - d \geq 0.
\end{aligned}
$$

The solution of the original CCQP can be approached by solving a sequence of $\mathcal{P}_{\mathrm{gap}}(\mu)$ with increasing $\mu$ [5]. The main difficulty is the nonconvexity of $\varphi^{ab}$. Gapese exploits the sparsity and convexity structure of the D-gap function to mitigate this nonconvexity.

## 3. Sparsity and convexity structure of the D-gap function

### Variable transformation

Let $\xi \in \mathbb{R}^{2n_\lambda}$ be the element-wise concatenation of $\lambda$ and $\eta$:

$$
\xi
=
\left[
\xi_1^T,\,
\cdots,\,
\xi_i^T,\,
\cdots,\,
\xi_{n_\lambda}^T
\right]^T,
\qquad
\xi_i
=
\begin{bmatrix}
\lambda_i \\
\eta_i
\end{bmatrix}.
$$

Equivalently,

$$
\xi
=
S
\begin{bmatrix}
\lambda \\
\eta
\end{bmatrix},
$$

where $S\in\mathbb{R}^{2n_\lambda\times2n_\lambda}$ is nonsingular. The transformed gap penalty reformulation is

$$
\begin{aligned}
\mathcal{P}^{\xi}_{\mathrm{gap}}(\mu):\quad
\min_{x,\xi}\quad
& \frac{1}{2}
\begin{bmatrix}
x \\
\xi
\end{bmatrix}^{T}
\begin{bmatrix}
Q_x & 0 \\
0 & Q_{\xi}
\end{bmatrix}
\begin{bmatrix}
x \\
\xi
\end{bmatrix}
+
\begin{bmatrix}
r_x \\
r_{\xi}
\end{bmatrix}^{T}
\begin{bmatrix}
x \\
\xi
\end{bmatrix}
+
\mu\varphi^{ab}(\xi)
\\
\text{s.t.}\quad
& A_x x + A_{\xi}\xi - b = 0,
\\
& C_x x + C_{\xi}\xi - d \geq 0.
\end{aligned}
$$

The transformed problem data are

$$
\begin{aligned}
Q_{\xi}
&=
(S^{-1})^T
\begin{bmatrix}
Q_{\lambda} & 0 \\
0 & Q_{\eta}
\end{bmatrix}
S^{-1},
\\
r_{\xi}
&=
(S^{-1})^T
\begin{bmatrix}
r_{\lambda} \\
r_{\eta}
\end{bmatrix},
\\
A_{\xi}
&=
\begin{bmatrix}
A_{\lambda} & A_{\eta}
\end{bmatrix}
S^{-1},
\\
C_{\xi}
&=
\begin{bmatrix}
C_{\lambda} & C_{\eta}
\end{bmatrix}
S^{-1}.
\end{aligned}
$$

For simplicity, define

$$
J(x,\xi)
=
\frac{1}{2}
\begin{bmatrix}
x \\
\xi
\end{bmatrix}^{T}
\begin{bmatrix}
Q_x & 0 \\
0 & Q_{\xi}
\end{bmatrix}
\begin{bmatrix}
x \\
\xi
\end{bmatrix}
+
\begin{bmatrix}
r_x \\
r_{\xi}
\end{bmatrix}^{T}
\begin{bmatrix}
x \\
\xi
\end{bmatrix}.
$$

### Partial separability

The partial separability of the D-gap function implies that its generalized Hessian has a block-diagonal sparsity structure:

$$
\partial^2\varphi^{ab}(\xi)
=
\mathrm{diag}
\left\{
\partial^2\delta^{ab}(\xi_1),\,
\cdots,\,
\partial^2\delta^{ab}(\xi_{n_\lambda})
\right\}.
$$

Thus, negative eigenvalues can only originate from indefinite submatrices $\partial^2\delta^{ab}(\xi_i)$.

### Piecewise structure

The scalar subfunction can be written as

$$
\delta^{ab}(\xi_i)
=
\delta^{ab}(\lambda_i,\eta_i)
=
\begin{cases}
\dfrac{b-a}{2}\lambda_i^2,
&
\eta_i\geq b\lambda_i
\text{ and }
\eta_i\geq a\lambda_i,
\\[4pt]
-\dfrac{a}{2}\lambda_i^2
+\lambda_i\eta_i
-\dfrac{1}{2b}\eta_i^2,
&
b\lambda_i>\eta_i>a\lambda_i,
\\[4pt]
\dfrac{b-a}{2ab}\eta_i^2,
&
\eta_i\leq b\lambda_i
\text{ and }
\eta_i\leq a\lambda_i,
\\[4pt]
\dfrac{b}{2}\lambda_i^2
-\lambda_i\eta_i
+\dfrac{1}{2a}\eta_i^2,
&
b\lambda_i<\eta_i<a\lambda_i .
\end{cases}
$$

The submatrix $\partial^2\delta^{ab}(\xi_i)$ is indefinite only when

$$
b\lambda_i>\eta_i>a\lambda_i,
$$

and is positive semidefinite otherwise. In the indefinite region,

$$
\partial^2\delta^{ab}(\xi_i)
=
\begin{bmatrix}
-a & 1 \\
1 & -\dfrac{1}{b}
\end{bmatrix}.
$$

Its two eigenvalues are

$$
\begin{aligned}
\kappa_{\min}
&=
-\frac{1}{2}
\left(
a+\frac{1}{b}
\right)
-
\frac{1}{2}
\sqrt{
\left(
\frac{1}{b}-a
\right)^2+4
}
<0,
\\
\kappa_{\max}
&=
-\frac{1}{2}
\left(
a+\frac{1}{b}
\right)
+
\frac{1}{2}
\sqrt{
\left(
\frac{1}{b}-a
\right)^2+4
}
>0.
\end{aligned}
$$

### Hessian modification

Let

$$
\mathcal{I}
=
\{1,\ldots,n_\lambda\},
$$

$$
\mathcal{I}_{\mathrm{ind}}
=
\left\{
i\in\mathcal{I}
\mid
b\lambda_i>\eta_i>a\lambda_i
\right\},
\qquad
\mathcal{I}_{\mathrm{psd}}
=
\mathcal{I}\setminus\mathcal{I}_{\mathrm{ind}}.
$$

Gapese constructs a positive-semidefinite approximation $B(\xi)$ of the generalized Hessian by preserving the positive-semidefinite blocks and regularizing only the indefinite blocks:

$$
B(\xi)
=
\partial^2\varphi^{ab}(\xi)
+
B_{\mathrm{reg}}(\xi).
$$

The regularization matrix is

$$
B_{\mathrm{reg}}(\xi)
=
-\kappa_{\min}
\mathrm{diag}
\left\{
\omega_1 I_{2\times2},\,
\cdots,\,
\omega_{n_\lambda}I_{2\times2}
\right\},
$$

where

$$
\omega_i
=
\begin{cases}
0, & i\in\mathcal{I}_{\mathrm{psd}},\\
1, & i\in\mathcal{I}_{\mathrm{ind}}.
\end{cases}
$$

If $\mathcal{I}_{\mathrm{ind}}$ is empty, $B(\xi)$ equals the exact positive-semidefinite generalized Hessian. Otherwise, the modification regularizes only the indefinite $2\times2$ blocks using the pre-computed negative eigenvalue $\kappa_{\min}$.

## 4. Semismooth Newton method

For a fixed penalty parameter $\mu$, Gapese solves $\mathcal{P}^{\xi}_{\mathrm{gap}}(\mu)$ using a semismooth Newton method with the Hessian modification above.

### KKT conditions and their semismooth equation reformulation

Let $\gamma_b\in\mathbb{R}^{n_b}$ and $\gamma_d\in\mathbb{R}^{n_d}$ be the Lagrange multipliers associated with the equality and inequality constraints. The KKT conditions are

$$
\begin{aligned}
&
\begin{bmatrix}
Q_x & 0 \\
0 & Q_{\xi}
\end{bmatrix}
\begin{bmatrix}
x \\
\xi
\end{bmatrix}
+
\begin{bmatrix}
r_x \\
r_{\xi}
\end{bmatrix}
+
\begin{bmatrix}
0 \\
\mu(\nabla_{\xi}\varphi^{ab}(\xi))^T
\end{bmatrix}
+
\begin{bmatrix}
A_x^T \\
A_{\xi}^T
\end{bmatrix}
\gamma_b
-
\begin{bmatrix}
C_x^T \\
C_{\xi}^T
\end{bmatrix}
\gamma_d
=
0,
\\
&
A_x x + A_{\xi}\xi - b = 0,
\\
&
0\leq\gamma_d
\perp
C_x x + C_{\xi}\xi - d
\geq0.
\end{aligned}
$$

Let $v_d\in\mathbb{R}^{n_d}$ be an auxiliary variable for the inequality constraints. Using the Fisher--Burmeister function [1]

$$
\psi(p,q)
=
\sqrt{p^2+q^2}
-p-q,
$$

the KKT conditions can be rewritten as the semismooth system

$$
\mathbf{T}(x,\xi,v_d,\gamma_b,\gamma_d)
=
\begin{bmatrix}
Q_x x + r_x + A_x^T\gamma_b - C_x^T\gamma_d
\\
Q_{\xi}\xi + r_{\xi}
+\mu(\nabla_{\xi}\varphi^{ab}(\xi))^T
+A_{\xi}^T\gamma_b
-C_{\xi}^T\gamma_d
\\
A_x x + A_{\xi}\xi - b
\\
C_x x + C_{\xi}\xi - d - v_d
\\
\Psi(v_d,\gamma_d)
\end{bmatrix}
=
0.
$$

For a given $\mu$, the inner iteration satisfies the KKT tolerance when

$$
\left\lVert
\mathbf{T}
\left(
x^k,\xi^k,v_d^k,\gamma_b^k,\gamma_d^k
\right)
\right\rVert
\leq
\epsilon_T.
$$

### Newton system

For a given iterate $(x^k,\xi^k,v_d^k,\gamma_b^k,\gamma_d^k)$, the search direction $(\Delta x,\Delta\xi,\Delta v_d,\Delta\gamma_b,\Delta\gamma_d)$ is obtained from

$$
\begin{bmatrix}
Q_x+\nu_x I
& 0
& 0
& A_x^T
& -C_x^T
\\
0
& Q_{\xi}+\mu B^k+\nu_{\xi}I
& 0
& A_{\xi}^T
& -C_{\xi}^T
\\
A_x
& A_{\xi}
& 0
& -\nu_b I
& 0
\\
C_x
& C_{\xi}
& -I
& 0
& 0
\\
0
& 0
& \partial_{v_d}\Psi^k-\nu_d I
& 0
& \partial_{\gamma_d}\Psi^k-\nu_d I
\end{bmatrix}
\begin{bmatrix}
\Delta x
\\
\Delta\xi
\\
\Delta v_d
\\
\Delta\gamma_b
\\
\Delta\gamma_d
\end{bmatrix}
=
-\mathbf{T}
\left(
x^k,\xi^k,v_d^k,\gamma_b^k,\gamma_d^k
\right),
$$

where $B^k=B(\xi^k)$ and $\nu_x,\nu_{\xi},\nu_b,\nu_d\geq0$ are regularization parameters.

### Merit-function line search

The search direction is globalized using the $\ell_1$ non-differentiable exact penalty function [6,7]:

$$
\Theta(x,\xi,v_d,\gamma_b,\gamma_d,\mu)
=
J(x,\xi)
+
\mu\varphi^{ab}(\xi)
+
\beta
\left\lVert
M(x,\xi,v_d,\gamma_d)
\right\rVert_1,
$$

where

$$
M(x,\xi,v_d,\gamma_d)
=
\begin{bmatrix}
A_x x + A_{\xi}\xi - b
\\
C_x x + C_{\xi}\xi - d - v_d
\\
\Psi(v_d,\gamma_d)
\end{bmatrix}.
$$

For the Newton direction, the directional derivative is

$$
\mathbb{D}\Theta^k
=
\nabla_xJ^k\Delta x
+
\nabla_{\xi}J^k\Delta\xi
+
\mu\nabla_{\xi}\varphi^{ab}(\xi^k)\Delta\xi
-
\beta
\left\lVert
M^k
\right\rVert_1
\leq
-\rho\beta
\left\lVert
M^k
\right\rVert_1.
$$

The weighting parameter is chosen so that

$$
\beta
\geq
\beta_{\mathrm{trial}}
=
\frac{
\underbrace{
\left(
(x^k)^TQ_x+r_x^T
\right)
}_{\nabla_xJ^k}
\Delta x
+
\underbrace{
\left(
(\xi^k)^TQ_{\xi}+r_{\xi}^T
\right)
}_{\nabla_{\xi}J^k}
\Delta\xi
+
\mu\nabla_{\xi}\varphi^{ab}(\xi^k)\Delta\xi
}{
(1-\rho)
\left\lVert
M^k
\right\rVert_1
}.
$$

Starting from $\alpha_{\max}=1$, backtracking reduces the trial stepsize according to $\alpha\leftarrow\nu_{\alpha}\alpha$ until the Armijo condition is satisfied:

$$
\begin{aligned}
&
\Theta
\left(
x^k+\alpha\Delta x,\,
\xi^k+\alpha\Delta\xi,\,
v_d^k+\alpha\Delta v_d,\,
\gamma_b^k+\alpha\Delta\gamma_b,\,
\gamma_d^k+\alpha\Delta\gamma_d,\,
\mu
\right)
\\
&\qquad
\leq
\Theta
\left(
x^k,\,
\xi^k,\,
v_d^k,\,
\gamma_b^k,\,
\gamma_d^k,\,
\mu
\right)
+
\nu_D\alpha\mathbb{D}\Theta^k .
\end{aligned}
$$

The reference algorithm uses $\rho=0.1$, $\nu_{\alpha}=0.5$, and $\nu_D=10^{-4}$.

## 5. Overall algorithm

Given the CCQP data and an initial guess $(x^0,\lambda^0,\eta^0)$:

1. Set $k_{\max}$, $\epsilon_{\varphi}>0$, $\epsilon_T>0$, $\mu>0$, $\sigma_{\mu}\geq1$, and $b>a>0$, and pre-compute $\kappa_{\min}$.
2. Form $\xi^0$ and the transformed data $Q_{\xi}$, $r_{\xi}$, $A_{\xi}$, and $C_{\xi}$. Initialize $x^k=x^0$, $\xi^k=\xi^0$, $v_d^k=\mathbf{1}$, $\gamma_b^k=\mathbf{1}$, and $\gamma_d^k=\mathbf{1}$.
3. For $k=1,\ldots,k_{\max}$:
   1. If the KKT residual satisfies the tolerance:
      - If $\varphi^{ab}(\xi^k)\leq\epsilon_{\varphi}$, terminate with $(x^*,\lambda^*,\eta^*)$.
      - Otherwise update $\mu\leftarrow\sigma_{\mu}\mu$.
   2. Evaluate the search direction by solving the semismooth Newton system.
   3. Evaluate the merit weight $\beta$ and use backtracking line search to obtain $\alpha$.
   4. Update $x$, $\xi$, $v_d$, $\gamma_b$, and $\gamma_d$.

## References

1. F. Facchinei and J. S. Pang, *Finite-Dimensional Variational Inequalities and Complementarity Problems*. Springer, 2003.
2. S. Gros and M. Diehl, *Numerical Optimal Control (Draft)*, 2020.
3. K. Lin and T. Ohtsuka, “Dynamical System Approach for Optimal Control Problems with Equilibrium Constraints Using Gap-Constraint-Based Reformulation,” arXiv:2412.01326, 2024.
4. K. Lin and T. Ohtsuka, “A Gap Penalty Method for Optimal Control of Linear Complementarity Systems,” 63rd IEEE Conference on Decision and Control (CDC), pp. 880–887, 2024.
5. K. Lin and T. Ohtsuka, “A Gap Penalty Reformulation for Mathematical Programming with Complementarity Constraints: Convergence Analysis,” *IEEE Control System Letters*, vol. 9, pp. 925–930, 2025.
6. K. Lin and T. Ohtsuka, “A Non-Interior-Point Continuation Method for the Optimal Control Problem with Equilibrium Constraints,” *Automatica*, vol. 171, 111940, 2025.
7. J. Nocedal and S. J. Wright, *Numerical Optimization*. Springer, 2006.
8. J. M. Peng and Y. Yuan, “Unconstrained Methods for Generalized Complementarity Problems,” *Journal of Computational Mathematics*, pp. 253–264, 1997.
