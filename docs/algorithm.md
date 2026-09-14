# Gapese algorithm

Gapese solves complementarity-constrained quadratic programs (CCQPs) using a gap penalty reformulation and a semismooth Newton method. This document summarizes the algorithm implemented in Gapese.

## 1. CCQP formulation

We consider the CCQP

$$
\begin{aligned}
\min_{x,\lambda,\eta}\quad
& \frac{1}{2}
\begin{bmatrix}x\\ \lambda\\ \eta\end{bmatrix}^{T}
\begin{bmatrix}
Q_x & 0 & 0\\
0 & Q_{\lambda} & 0\\
0 & 0 & Q_{\eta}
\end{bmatrix}
\begin{bmatrix}x\\ \lambda\\ \eta\end{bmatrix}
+
\begin{bmatrix}r_x\\ r_{\lambda}\\ r_{\eta}\end{bmatrix}^{T}
\begin{bmatrix}x\\ \lambda\\ \eta\end{bmatrix} \\
\text{s.t.}\quad
& A_x x + A_{\lambda}\lambda + A_{\eta}\eta - b = 0, \\
& C_x x + C_{\lambda}\lambda + C_{\eta}\eta - d \ge 0, \\
& 0 \le \lambda \perp \eta \ge 0.
\end{aligned}
$$

The decision variables are $x \in \mathbb{R}^{n_x}$ and $\lambda,\eta \in \mathbb{R}^{n_\lambda}$. The equality and inequality constraints have dimensions $n_b$ and $n_d$, respectively. The notation $0 \le \lambda \perp \eta \ge 0$ means

$$
\lambda \ge 0,\qquad \eta \ge 0,\qquad \lambda^T\eta = 0.
$$

Complementarity constraints violate standard constraint qualifications such as LICQ and MFCQ at feasible points. This loss of constraint regularity is one reason that CCQPs require dedicated numerical methods rather than a direct application of standard nonlinear-programming methods.

## 2. Gap penalty reformulation for CCQP

### D-gap function and its key properties

Let $a$ and $b$ satisfy $b>a>0$. For $\lambda,\eta \in \mathbb{R}^{n_\lambda}$, define

$$
\begin{aligned}
\varphi^a(\lambda,\eta)
&= \frac{1}{2a}\left(\|\eta\|_2^2-\|\max(0,\eta-a\lambda)\|_2^2\right),\\
\varphi^b(\lambda,\eta)
&= \frac{1}{2b}\left(\|\eta\|_2^2-\|\max(0,\eta-b\lambda)\|_2^2\right),
\end{aligned}
$$

where the maximum is evaluated element-wise. The D-gap function [8] is

$$
\varphi^{ab}(\lambda,\eta)
= \varphi^a(\lambda,\eta)-\varphi^b(\lambda,\eta)
= \frac{b-a}{2ab}\|\eta\|_2^2
{}- \frac{1}{2a}\|\max(0,\eta-a\lambda)\|_2^2
{}+ \frac{1}{2b}\|\max(0,\eta-b\lambda)\|_2^2.
$$

The first three properties below are summarized in [1,3], while the partial separability follows the standard definition in [2]:

- **Non-negativity:** $\varphi^{ab}(\lambda,\eta)\ge 0$ for all $\lambda,\eta$.
- **Equivalence:** $\varphi^{ab}(\lambda,\eta)=0$ if and only if $0\le\lambda\perp\eta\ge0$.
- **Differentiability:** $\varphi^{ab}$ is semismoothly differentiable.
- **Partial separability:**

$$
\varphi^{ab}(\lambda,\eta)
= \sum_{i=1}^{n_\lambda}\delta^{ab}(\lambda_i,\eta_i),
$$

with

$$
\delta^{ab}(\lambda_i,\eta_i)
= \frac{b-a}{2ab}\eta_i^2
{}- \frac{1}{2a}\{\max(0,\eta_i-a\lambda_i)\}^2
{}+ \frac{1}{2b}\{\max(0,\eta_i-b\lambda_i)\}^2.
$$

### Gap penalty reformulation

Using the non-negativity and equivalence of the D-gap function, the complementarity constraints are replaced by a penalty term. For a penalty parameter $\mu>0$, the gap penalty reformulation is

$$
\begin{aligned}
\mathcal{P}_{\mathrm{gap}}(\mu):\qquad
\min_{x,\lambda,\eta}\quad
& \frac{1}{2}
\begin{bmatrix}x\\ \lambda\\ \eta\end{bmatrix}^{T}
\begin{bmatrix}
Q_x & 0 & 0\\
0 & Q_{\lambda} & 0\\
0 & 0 & Q_{\eta}
\end{bmatrix}
\begin{bmatrix}x\\ \lambda\\ \eta\end{bmatrix}
+
\begin{bmatrix}r_x\\ r_{\lambda}\\ r_{\eta}\end{bmatrix}^{T}
\begin{bmatrix}x\\ \lambda\\ \eta\end{bmatrix}
{}+ \mu\varphi^{ab}(\lambda,\eta) \\
\text{s.t.}\quad
& A_x x + A_{\lambda}\lambda + A_{\eta}\eta - b = 0, \\
& C_x x + C_{\lambda}\lambda + C_{\eta}\eta - d \ge 0.
\end{aligned}
$$

A solution of the original CCQP can be approached by solving a sequence of $\mathcal{P}_{\mathrm{gap}}(\mu)$ with increasing $\mu$ [5]. Compared with complementarity-penalty reformulations, this formulation does not retain the explicit inequalities $\lambda,\eta\ge0$. Compared with relaxation or smoothing reformulations, its constraint system remains differentiable and regular independently of $\mu$.

The remaining difficulty is the nonconvexity of $\varphi^{ab}$. Gapese exploits the sparsity and convexity structure of the D-gap function to construct a positive-semidefinite approximation of its generalized Hessian.

## 3. Sparsity and convexity structure of the D-gap function

### Variable transformation

Define the interleaved variable

$$
\xi
= \begin{bmatrix}\xi_1^T & \cdots & \xi_{n_\lambda}^T\end{bmatrix}^{T},
\qquad
\xi_i = \begin{bmatrix}\lambda_i\\ \eta_i\end{bmatrix},
$$

or equivalently

$$
\xi = S\begin{bmatrix}\lambda\\ \eta\end{bmatrix},
$$

where $S\in\mathbb{R}^{2n_\lambda\times 2n_\lambda}$ is nonsingular. The gap penalty problem can then be written as

$$
\begin{aligned}
\mathcal{P}^{\xi}_{\mathrm{gap}}(\mu):\qquad
\min_{x,\xi}\quad
& \frac{1}{2}
\begin{bmatrix}x\\ \xi\end{bmatrix}^{T}
\begin{bmatrix}Q_x & 0\\ 0 & Q_{\xi}\end{bmatrix}
\begin{bmatrix}x\\ \xi\end{bmatrix}
+
\begin{bmatrix}r_x\\ r_{\xi}\end{bmatrix}^{T}
\begin{bmatrix}x\\ \xi\end{bmatrix}
{}+ \mu\varphi^{ab}(\xi) \\
\text{s.t.}\quad
& A_x x + A_{\xi}\xi - b = 0, \\
& C_x x + C_{\xi}\xi - d \ge 0,
\end{aligned}
$$

where

$$
\begin{aligned}
Q_{\xi}
&= (S^{-1})^T
\begin{bmatrix}Q_{\lambda} & 0\\ 0 & Q_{\eta}\end{bmatrix}S^{-1},\\
r_{\xi}
&= (S^{-1})^T\begin{bmatrix}r_{\lambda}\\ r_{\eta}\end{bmatrix},\\
A_{\xi}
&= \begin{bmatrix}A_{\lambda} & A_{\eta}\end{bmatrix}S^{-1},\\
C_{\xi}
&= \begin{bmatrix}C_{\lambda} & C_{\eta}\end{bmatrix}S^{-1}.
\end{aligned}
$$

For convenience, denote the quadratic part of the transformed objective by

$$
J(x,\xi)
= \frac{1}{2}
\begin{bmatrix}x\\ \xi\end{bmatrix}^{T}
\begin{bmatrix}Q_x & 0\\ 0 & Q_{\xi}\end{bmatrix}
\begin{bmatrix}x\\ \xi\end{bmatrix}
+
\begin{bmatrix}r_x\\ r_{\xi}\end{bmatrix}^{T}
\begin{bmatrix}x\\ \xi\end{bmatrix}.
$$

### Partial separability

With the interleaved variable $\xi$, the partial separability of the D-gap function gives a block-diagonal generalized Hessian:

$$
\partial^2\varphi^{ab}(\xi)
= \operatorname{diag}\left\{
\partial^2\delta^{ab}(\xi_1),\ldots,
\partial^2\delta^{ab}(\xi_{n_\lambda})
\right\}.
$$

Therefore, negative eigenvalues can only originate from the $2\times2$ blocks $\partial^2\delta^{ab}(\xi_i)$ that are indefinite.

### Piecewise structure

Each scalar subfunction has the piecewise representation

$$
\delta^{ab}(\lambda_i,\eta_i)
=
\begin{cases}
\dfrac{b-a}{2}\lambda_i^2,
& \eta_i\ge b\lambda_i \text{ and } \eta_i\ge a\lambda_i,\\[4pt]
-\dfrac{a}{2}\lambda_i^2 + \lambda_i\eta_i - \dfrac{1}{2b}\eta_i^2,
& b\lambda_i>\eta_i>a\lambda_i,\\[4pt]
\dfrac{b-a}{2ab}\eta_i^2,
& \eta_i\le b\lambda_i \text{ and } \eta_i\le a\lambda_i,\\[4pt]
\dfrac{b}{2}\lambda_i^2 - \lambda_i\eta_i + \dfrac{1}{2a}\eta_i^2,
& b\lambda_i<\eta_i<a\lambda_i.
\end{cases}
$$

The block $\partial^2\delta^{ab}(\xi_i)$ is indefinite only in the region

$$
b\lambda_i>\eta_i>a\lambda_i,
$$

and is positive semidefinite otherwise. In the indefinite region,

$$
\partial^2\delta^{ab}(\xi_i)
=
\begin{bmatrix}
-a & 1\\
1 & -\dfrac{1}{b}
\end{bmatrix}.
$$

Its eigenvalues are

$$
\begin{aligned}
\kappa_{\min}
&= -\frac{1}{2}\left(a+\frac{1}{b}\right)
-\frac{1}{2}\sqrt{\left(\frac{1}{b}-a\right)^2+4}<0,\\
\kappa_{\max}
&= -\frac{1}{2}\left(a+\frac{1}{b}\right)
+\frac{1}{2}\sqrt{\left(\frac{1}{b}-a\right)^2+4}>0.
\end{aligned}
$$

### Hessian modification

Let

$$
\mathcal{I}_{\mathrm{ind}}
= \{i\mid b\lambda_i>\eta_i>a\lambda_i\},
\qquad
\mathcal{I}_{\mathrm{psd}}
= \{1,\ldots,n_\lambda\}\setminus\mathcal{I}_{\mathrm{ind}}.
$$

Gapese constructs a positive-semidefinite approximation $B(\xi)$ of the generalized Hessian by preserving the positive-semidefinite blocks and regularizing only the indefinite blocks:

$$
B(\xi)=\partial^2\varphi^{ab}(\xi)+B_{\mathrm{reg}}(\xi),
$$

with

$$
B_{\mathrm{reg}}(\xi)
= -\kappa_{\min}\operatorname{diag}
\left\{\omega_1 I_{2\times2},\ldots,\omega_{n_\lambda}I_{2\times2}\right\},
$$

where

$$
\omega_i=
\begin{cases}
0, & i\in\mathcal{I}_{\mathrm{psd}},\\
1, & i\in\mathcal{I}_{\mathrm{ind}}.
\end{cases}
$$

If $\mathcal{I}_{\mathrm{ind}}$ is empty, $B(\xi)$ equals the exact positive-semidefinite generalized Hessian. Otherwise, the modification shifts only the indefinite $2\times2$ blocks using the pre-computed value $\kappa_{\min}$, without requiring a factorization of $\partial^2\varphi^{ab}(\xi)$.

## 4. Semismooth Newton method

For a fixed penalty parameter $\mu$, Gapese solves $\mathcal{P}^{\xi}_{\mathrm{gap}}(\mu)$ with a semismooth Newton method using the Hessian modification above.

### KKT conditions and their semismooth equation reformulation

Let $\gamma_b\in\mathbb{R}^{n_b}$ and $\gamma_d\in\mathbb{R}^{n_d}$ be the Lagrange multipliers associated with the equality and inequality constraints. The KKT conditions are

$$
\begin{aligned}
&
\begin{bmatrix}Q_x & 0\\ 0 & Q_{\xi}\end{bmatrix}
\begin{bmatrix}x\\ \xi\end{bmatrix}
+
\begin{bmatrix}r_x\\ r_{\xi}\end{bmatrix}
+
\begin{bmatrix}0\\ \mu(\nabla_{\xi}\varphi^{ab}(\xi))^T\end{bmatrix}
+
\begin{bmatrix}A_x^T\\ A_{\xi}^T\end{bmatrix}\gamma_b
-
\begin{bmatrix}C_x^T\\ C_{\xi}^T\end{bmatrix}\gamma_d
=0,\\
& A_xx+A_{\xi}\xi-b=0,\\
& 0\le\gamma_d\perp C_xx+C_{\xi}\xi-d\ge0.
\end{aligned}
$$

Introduce the auxiliary variable $v_d\in\mathbb{R}^{n_d}$ and the Fisher--Burmeister function [1]

$$
\psi(p,q)=\sqrt{p^2+q^2}-p-q,
$$

which satisfies $\psi(p,q)=0$ if and only if $p\ge0$, $q\ge0$, and $pq=0$. Applying $\psi$ element-wise gives $\Psi(v_d,\gamma_d)$, and the KKT conditions can be written as the semismooth system

$$
\boldsymbol{T}(x,\xi,v_d,\gamma_b,\gamma_d)
=
\begin{bmatrix}
Q_xx+r_x+A_x^T\gamma_b-C_x^T\gamma_d\\
Q_{\xi}\xi+r_{\xi}+\mu(\nabla_{\xi}\varphi^{ab}(\xi))^T+A_{\xi}^T\gamma_b-C_{\xi}^T\gamma_d\\
A_xx+A_{\xi}\xi-b\\
C_xx+C_{\xi}\xi-d-v_d\\
\Psi(v_d,\gamma_d)
\end{bmatrix}
=0.
$$

For a given $\mu$, the inner iteration satisfies the KKT tolerance when

$$
\left\|\boldsymbol{T}(x^k,\xi^k,v_d^k,\gamma_b^k,\gamma_d^k)\right\|
\le \epsilon_T.
$$

### Newton system

At an iterate $(x^k,\xi^k,v_d^k,\gamma_b^k,\gamma_d^k)$, the search direction $(\Delta x,\Delta\xi,\Delta v_d,\Delta\gamma_b,\Delta\gamma_d)$ is obtained from

$$
\begin{bmatrix}
Q_x+\nu_x I & 0 & 0 & A_x^T & -C_x^T\\
0 & Q_{\xi}+\mu B^k+\nu_{\xi}I & 0 & A_{\xi}^T & -C_{\xi}^T\\
A_x & A_{\xi} & 0 & -\nu_b I & 0\\
C_x & C_{\xi} & -I & 0 & 0\\
0 & 0 & \partial_{v_d}\Psi^k-\nu_d I & 0 & \partial_{\gamma_d}\Psi^k-\nu_d I
\end{bmatrix}
\begin{bmatrix}
\Delta x\\
\Delta\xi\\
\Delta v_d\\
\Delta\gamma_b\\
\Delta\gamma_d
\end{bmatrix}
=
-\boldsymbol{T}(x^k,\xi^k,v_d^k,\gamma_b^k,\gamma_d^k),
$$

where $B^k=B(\xi^k)$ and $\nu_x,\nu_{\xi},\nu_b,\nu_d\ge0$ are regularization parameters.

### Merit-function line search

The search direction is globalized with an $\ell_1$ exact-penalty merit function [6,7]:

$$
\Theta(x,\xi,v_d,\gamma_b,\gamma_d,\mu)
=J(x,\xi)+\mu\varphi^{ab}(\xi)
+\beta\|M(x,\xi,v_d,\gamma_d)\|_1,
$$

where

$$
M(x,\xi,v_d,\gamma_d)
=
\begin{bmatrix}
A_xx+A_{\xi}\xi-b\\
C_xx+C_{\xi}\xi-d-v_d\\
\Psi(v_d,\gamma_d)
\end{bmatrix}.
$$

For the Newton direction, the directional derivative is

$$
\mathbb{D}\Theta^k
= \nabla_xJ^k\Delta x
{}+ \nabla_{\xi}J^k\Delta\xi
{}+ \mu\nabla_{\xi}\varphi^{ab}(\xi^k)\Delta\xi
{}- \beta\|M^k\|_1
\le -\rho\beta\|M^k\|_1.
$$

The weighting parameter is chosen so that

$$
\beta\ge\beta_{\mathrm{trial}}
=
\frac{
\underbrace{\left((x^k)^TQ_x+r_x^T\right)}_{\nabla_xJ^k}\Delta x
+
\underbrace{\left((\xi^k)^TQ_{\xi}+r_{\xi}^T\right)}_{\nabla_{\xi}J^k}\Delta\xi
+
\mu\nabla_{\xi}\varphi^{ab}(\xi^k)\Delta\xi
}{(1-\rho)\|M^k\|_1},
$$

with $\rho\in(0,1)$. Starting from $\alpha_{\max}=1$, backtracking reduces $\alpha$ by $\alpha\leftarrow\nu_{\alpha}\alpha$ until the Armijo condition is satisfied:

$$
\Theta(x^k+\alpha\Delta x,\xi^k+\alpha\Delta\xi,v_d^k+\alpha\Delta v_d,\gamma_b^k+\alpha\Delta\gamma_b,\gamma_d^k+\alpha\Delta\gamma_d,\mu)
\le
\Theta(x^k,\xi^k,v_d^k,\gamma_b^k,\gamma_d^k,\mu)
+\nu_D\alpha\mathbb{D}\Theta^k.
$$

The reference algorithm uses $\rho=0.1$, $\nu_{\alpha}=0.5$, and $\nu_D=10^{-4}$.

## 5. Overall algorithm

Given the CCQP data and an initial guess $(x^0,\lambda^0,\eta^0)$:

1. Choose $k_{\max}$, $\epsilon_{\varphi}>0$, $\epsilon_T>0$, $\mu>0$, $\sigma_{\mu}\ge1$, and $b>a>0$. Pre-compute $\kappa_{\min}$.
2. Form $\xi^0$ and the transformed data $Q_{\xi}$, $r_{\xi}$, $A_{\xi}$, and $C_{\xi}$. Initialize $x^k=x^0$, $\xi^k=\xi^0$, $v_d^k=\mathbf{1}$, $\gamma_b^k=\mathbf{1}$, and $\gamma_d^k=\mathbf{1}$.
3. For $k=1,\ldots,k_{\max}$:
   1. If the KKT residual satisfies $\|\boldsymbol{T}^k\|\le\epsilon_T$:
      - If $\varphi^{ab}(\xi^k)\le\epsilon_{\varphi}$, terminate with $(x^*,\lambda^*,\eta^*)$.
      - Otherwise update the penalty parameter as $\mu\leftarrow\sigma_{\mu}\mu$.
   2. Construct $B(\xi^k)$ and solve the semismooth Newton system for $(\Delta x,\Delta\xi,\Delta v_d,\Delta\gamma_b,\Delta\gamma_d)$.
   3. Choose the merit weight $\beta$ and perform backtracking line search to obtain $\alpha$.
   4. Update all primal, auxiliary, and dual variables with the step length $\alpha$.

## References

1. F. Facchinei and J. S. Pang, *Finite-Dimensional Variational Inequalities and Complementarity Problems*. Springer, 2003.
2. S. Gros and M. Diehl, *Numerical Optimal Control (Draft)*, 2020.
3. K. Lin and T. Ohtsuka, “Dynamical System Approach for Optimal Control Problems with Equilibrium Constraints Using Gap-Constraint-Based Reformulation,” arXiv:2412.01326, 2024.
4. K. Lin and T. Ohtsuka, “A Gap Penalty Method for Optimal Control of Linear Complementarity Systems,” 63rd IEEE Conference on Decision and Control (CDC), pp. 880–887, 2024.
5. K. Lin and T. Ohtsuka, “A Gap Penalty Reformulation for Mathematical Programming with Complementarity Constraints: Convergence Analysis,” *IEEE Control System Letters*, vol. 9, pp. 925–930, 2025.
6. K. Lin and T. Ohtsuka, “A Non-Interior-Point Continuation Method for the Optimal Control Problem with Equilibrium Constraints,” *Automatica*, vol. 171, 111940, 2025.
7. J. Nocedal and S. J. Wright, *Numerical Optimization*. Springer, 2006.
8. J. M. Peng and Y. Yuan, “Unconstrained Methods for Generalized Complementarity Problems,” *Journal of Computational Mathematics*, pp. 253–264, 1997.
