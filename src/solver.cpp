#include "gapese/solver.hpp"

#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gapese {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void require_matrix_size(const SparseMatrix& matrix, int rows, int cols, const char* name) {
    if (matrix.rows() != rows || matrix.cols() != cols) {
        throw std::invalid_argument(
            std::string(name) + " must have size " + std::to_string(rows) + "-by-" + std::to_string(cols));
    }
}

void require_vector_size(const Vector& vector, int size, const char* name) {
    if (vector.size() != size) {
        throw std::invalid_argument(std::string(name) + " must have length " + std::to_string(size));
    }
}

using SparseLUSolver = Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>>;

struct SolverWorkspace {
    explicit SolverWorkspace(const ProblemSize& problem)
        : qx_x(problem.n_x),
          qxi_xi(detail::transformed_dim(problem)),
          ax_x(problem.n_b),
          axi_xi(problem.n_b),
          cx_x(problem.n_d),
          cxi_xi(problem.n_d),
          dgap_gradient(detail::transformed_dim(problem)),
          dgap_blocks(static_cast<std::size_t>(problem.n_lambda)),
          fb_value(problem.n_d),
          fb_d_v(problem.n_d),
          fb_d_gamma(problem.n_d),
          kkt_residual(detail::kkt_dim(problem)),
          rhs(detail::kkt_dim(problem)),
          newton_step(detail::kkt_dim(problem)),
          qx_dx(problem.n_x),
          qxi_dxi(detail::transformed_dim(problem)),
          x_trial(problem.n_x),
          xi_trial(detail::transformed_dim(problem)),
          v_d_trial(problem.n_d),
          gamma_d_trial(problem.n_d) {}

    Vector qx_x;
    Vector qxi_xi;
    Vector ax_x;
    Vector axi_xi;
    Vector cx_x;
    Vector cxi_xi;

    Vector dgap_gradient;
    std::vector<std::array<double, 3>> dgap_blocks;
    Vector fb_value;
    Vector fb_d_v;
    Vector fb_d_gamma;

    Vector kkt_residual;
    Vector rhs;
    Vector newton_step;

    Vector qx_dx;
    Vector qxi_dxi;

    Vector x_trial;
    Vector xi_trial;
    Vector v_d_trial;
    Vector gamma_d_trial;
};

bool factor_and_solve_sparse_system(
    SparseLUSolver& lu,
    const SparseMatrix& matrix,
    const Vector& rhs,
    Vector& solution,
    Timing& timing) {
    auto phase_start = Clock::now();
    lu.factorize(matrix);
    timing.lu_factorize += elapsed_seconds(phase_start);
    if (lu.info() == Eigen::Success) {
        phase_start = Clock::now();
        solution = lu.solve(rhs);
        timing.lu_solve += elapsed_seconds(phase_start);
        if (lu.info() == Eigen::Success && solution.allFinite()) {
            return true;
        }
    }

    // Robust fallback for singular or nearly singular Newton systems.
    // Its cost is intentionally included in Timing::search_other.
    Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr;
    qr.compute(matrix);
    if (qr.info() != Eigen::Success) {
        return false;
    }
    solution = qr.solve(rhs);
    return qr.info() == Eigen::Success && solution.allFinite();
}

}  // namespace

Solver::Solver(ProblemSize problem, Options options)
    : problem_(problem),
      options_(options),
      dgap_(options.dgap_a, options.dgap_b),
      fb_(options.fb_sigma),
      kkt_(problem, options) {
    problem_.validate();
    options_.validate();

    if (options_.print_level > 0) {
        std::cout << "creating Gapese C++ solver...\n"
                  << "*--------------------- CCQP Problem Information -------------------*\n"
                  << "1. problem size\n"
                  << "number of variables (x): ................. " << problem_.n_x << '\n'
                  << "number of variables (lambda): ............ " << problem_.n_lambda << '\n'
                  << "number of equality constraint (b): ....... " << problem_.n_b << '\n'
                  << "number of inequality constraint (d): ..... " << problem_.n_d << '\n'
                  << "Done!\n";
    }
}

void Solver::validate_data(const CcqpData& data) const {
    const int nx = problem_.n_x;
    const int nl = problem_.n_lambda;
    const int nb = problem_.n_b;
    const int nd = problem_.n_d;

    require_matrix_size(data.Q_x, nx, nx, "Q_x");
    require_matrix_size(data.Q_lambda, nl, nl, "Q_lambda");
    require_matrix_size(data.Q_eta, nl, nl, "Q_eta");
    require_vector_size(data.r_x, nx, "r_x");
    require_vector_size(data.r_lambda, nl, "r_lambda");
    require_vector_size(data.r_eta, nl, "r_eta");

    require_matrix_size(data.A_x, nb, nx, "A_x");
    require_matrix_size(data.A_lambda, nb, nl, "A_lambda");
    require_matrix_size(data.A_eta, nb, nl, "A_eta");
    require_vector_size(data.b, nb, "b");

    require_matrix_size(data.C_x, nd, nx, "C_x");
    require_matrix_size(data.C_lambda, nd, nl, "C_lambda");
    require_matrix_size(data.C_eta, nd, nl, "C_eta");
    require_vector_size(data.d, nd, "d");
}

void Solver::validate_initial(const InitialGuess& initial) const {
    require_vector_size(initial.x, problem_.n_x, "Init.x");
    require_vector_size(initial.lambda, problem_.n_lambda, "Init.lambda");
    require_vector_size(initial.eta, problem_.n_lambda, "Init.eta");
    if (initial.v_d) require_vector_size(*initial.v_d, problem_.n_d, "Init.v_d");
    if (initial.gamma_b) require_vector_size(*initial.gamma_b, problem_.n_b, "Init.gamma_b");
    if (initial.gamma_d) require_vector_size(*initial.gamma_d, problem_.n_d, "Init.gamma_d");
}

Vector Solver::interleave(const Vector& lambda, const Vector& eta) const {
    Vector xi(detail::transformed_dim(problem_));
    for (int i = 0; i < problem_.n_lambda; ++i) {
        xi[2 * i] = lambda[i];
        xi[2 * i + 1] = eta[i];
    }
    return xi;
}

void Solver::deinterleave(const Vector& xi, Vector& lambda, Vector& eta) const {
    lambda.resize(problem_.n_lambda);
    eta.resize(problem_.n_lambda);
    for (int i = 0; i < problem_.n_lambda; ++i) {
        lambda[i] = xi[2 * i];
        eta[i] = xi[2 * i + 1];
    }
}

detail::TransformedData Solver::transform_data(const CcqpData& data) const {
    const int nl = problem_.n_lambda;
    const int nxi = detail::transformed_dim(problem_);

    detail::TransformedData transformed;

    std::vector<Triplet> q_triplets;
    q_triplets.reserve(static_cast<std::size_t>(data.Q_lambda.nonZeros() + data.Q_eta.nonZeros()));
    for (int col = 0; col < data.Q_lambda.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(data.Q_lambda, col); it; ++it) {
            q_triplets.emplace_back(2 * it.row(), 2 * it.col(), it.value());
        }
    }
    for (int col = 0; col < data.Q_eta.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(data.Q_eta, col); it; ++it) {
            q_triplets.emplace_back(2 * it.row() + 1, 2 * it.col() + 1, it.value());
        }
    }
    transformed.Q_xi.resize(nxi, nxi);
    transformed.Q_xi.setFromTriplets(q_triplets.begin(), q_triplets.end());
    transformed.Q_xi.makeCompressed();

    transformed.r_xi = Vector::Zero(nxi);
    for (int i = 0; i < nl; ++i) {
        transformed.r_xi[2 * i] = data.r_lambda[i];
        transformed.r_xi[2 * i + 1] = data.r_eta[i];
    }

    std::vector<Triplet> a_triplets;
    a_triplets.reserve(static_cast<std::size_t>(data.A_lambda.nonZeros() + data.A_eta.nonZeros()));
    for (int col = 0; col < data.A_lambda.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(data.A_lambda, col); it; ++it) {
            a_triplets.emplace_back(it.row(), 2 * it.col(), it.value());
        }
    }
    for (int col = 0; col < data.A_eta.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(data.A_eta, col); it; ++it) {
            a_triplets.emplace_back(it.row(), 2 * it.col() + 1, it.value());
        }
    }
    transformed.A_xi.resize(problem_.n_b, nxi);
    transformed.A_xi.setFromTriplets(a_triplets.begin(), a_triplets.end());
    transformed.A_xi.makeCompressed();

    std::vector<Triplet> c_triplets;
    c_triplets.reserve(static_cast<std::size_t>(data.C_lambda.nonZeros() + data.C_eta.nonZeros()));
    for (int col = 0; col < data.C_lambda.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(data.C_lambda, col); it; ++it) {
            c_triplets.emplace_back(it.row(), 2 * it.col(), it.value());
        }
    }
    for (int col = 0; col < data.C_eta.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(data.C_eta, col); it; ++it) {
            c_triplets.emplace_back(it.row(), 2 * it.col() + 1, it.value());
        }
    }
    transformed.C_xi.resize(problem_.n_d, nxi);
    transformed.C_xi.setFromTriplets(c_triplets.begin(), c_triplets.end());
    transformed.C_xi.makeCompressed();

    return transformed;
}

SolveResult Solver::solve(const CcqpData& data, const InitialGuess& initial) {
    const auto total_start = Clock::now();
    validate_data(data);
    validate_initial(initial);
    const detail::TransformedData transformed = transform_data(data);

    Vector x = initial.x;
    Vector xi = interleave(initial.lambda, initial.eta);
    Vector v_d = initial.v_d.value_or(Vector::Ones(problem_.n_d));
    Vector gamma_b = initial.gamma_b.value_or(Vector::Ones(problem_.n_b));
    Vector gamma_d = initial.gamma_d.value_or(Vector::Ones(problem_.n_d));

    SolverWorkspace workspace(problem_);

    double mu = options_.mu_min;
    Timing timing;

    const auto initial_kkt_start = Clock::now();
    const auto initial_assembly_start = Clock::now();
    detail::KKTWorkspace kkt_workspace = kkt_.initialize_matrix(data, transformed);
    timing.kkt_assembly += elapsed_seconds(initial_assembly_start);

    SparseLUSolver lu;
    const auto analyze_start = Clock::now();
    lu.analyzePattern(kkt_workspace.matrix);
    timing.lu_analyze += elapsed_seconds(analyze_start);
    timing.search_direction += elapsed_seconds(initial_kkt_start);

    double newton_norm = std::numeric_limits<double>::quiet_NaN();
    double alpha = std::numeric_limits<double>::quiet_NaN();
    double beta = std::numeric_limits<double>::quiet_NaN();
    double objective = 0.0;
    double dgap_residual = 0.0;
    double kkt_norm = 0.0;
    int status = 0;
    int line_search_status = 0;
    int iteration = 0;
    std::string message = "- Solver fails: solver terminated before setting a status";

    const int nx = problem_.n_x;
    const int nxi = detail::transformed_dim(problem_);
    const int nb = problem_.n_b;
    const int nd = problem_.n_d;
    const int row_eq = nx + nxi;
    const int row_ineq = row_eq + nb;

    for (int k = 0; k <= options_.max_iter; ++k) {
        iteration = k;

        const auto termination_start = Clock::now();
        // Compute each sparse matrix-vector product for the current iterate once,
        // then reuse it in the objective, KKT residual, and line search.
        workspace.qx_x = data.Q_x * x;
        workspace.qxi_xi = transformed.Q_xi * xi;
        workspace.ax_x = data.A_x * x;
        workspace.axi_xi = transformed.A_xi * xi;
        workspace.cx_x = data.C_x * x;
        workspace.cxi_xi = transformed.C_xi * xi;
        dgap_.evaluate(xi, dgap_residual, workspace.dgap_gradient, workspace.dgap_blocks);
        fb_.evaluate(v_d, gamma_d, workspace.fb_value, workspace.fb_d_v, workspace.fb_d_gamma);
        objective =
            0.5 * x.dot(workspace.qx_x) + data.r_x.dot(x) +
            0.5 * xi.dot(workspace.qxi_xi) + transformed.r_xi.dot(xi);
        kkt_.residual(
            mu,
            workspace.qx_x,
            workspace.qxi_xi,
            workspace.ax_x,
            workspace.axi_xi,
            workspace.cx_x,
            workspace.cxi_xi,
            workspace.dgap_gradient,
            workspace.fb_value,
            v_d,
            gamma_b,
            gamma_d,
            data,
            transformed,
            workspace.kkt_residual);
        kkt_norm = workspace.kkt_residual.lpNorm<Eigen::Infinity>();

        if (options_.print_level == 2) {
            if (k % 10 == 0) {
                std::cout << "------------------------------------------------------------------------------------------\n"
                          << " Iter |    mu    | D-gap res | KKT res  | Newton step | stepsize |   beta   |   cost\n";
            }
            std::cout << std::setw(5) << k << " | " << std::scientific << std::setprecision(2)
                      << std::setw(8) << mu << " | " << std::setw(9) << dgap_residual << " | "
                      << std::setw(8) << kkt_norm << " | " << std::setw(11) << newton_norm << " | "
                      << std::setw(8) << alpha << " | " << std::setw(8) << beta << " | "
                      << std::setw(9) << objective << '\n';
        }

        if (kkt_norm <= options_.tol_kkt) {
            if (dgap_residual <= options_.tol_dgap) {
                status = 1;
                message = "- Solver succeeds: the KKT and D-gap residual satisfy tolerance";
                timing.check_termination += elapsed_seconds(termination_start);
                break;
            }
            if (mu >= options_.mu_max) {
                status = 0;
                message = "- Solver fails: the maximum penalty parameter exceeded";
                timing.check_termination += elapsed_seconds(termination_start);
                break;
            }
            if (k == options_.max_iter) {
                status = 0;
                message = "- Solver fails: the maximum number of iteration exceeded before D-gap convergence";
                timing.check_termination += elapsed_seconds(termination_start);
                break;
            }

            mu = std::min(options_.mu_update * mu, options_.mu_max);
            timing.check_termination += elapsed_seconds(termination_start);
            continue;
        }

        if (k == options_.max_iter) {
            status = 0;
            message = "- Solver fails: the maximum number of iteration exceeded";
            timing.check_termination += elapsed_seconds(termination_start);
            break;
        }
        timing.check_termination += elapsed_seconds(termination_start);

        const auto search_start = Clock::now();
        const auto assembly_start = Clock::now();
        kkt_.update_matrix(
            kkt_workspace, mu, workspace.dgap_blocks, workspace.fb_d_v, workspace.fb_d_gamma);
        timing.kkt_assembly += elapsed_seconds(assembly_start);

        workspace.rhs = -workspace.kkt_residual;
        if (!factor_and_solve_sparse_system(
                lu, kkt_workspace.matrix, workspace.rhs, workspace.newton_step, timing)) {
            status = 0;
            message = "- Solver fails: KKT linear solve failed";
            timing.search_direction += elapsed_seconds(search_start);
            break;
        }

        newton_norm = workspace.newton_step.lpNorm<Eigen::Infinity>();
        if (!std::isfinite(newton_norm) || newton_norm >= options_.newton_step_max) {
            status = 0;
            message = "- Solver fails: the Newton step is too large or non-finite";
            timing.search_direction += elapsed_seconds(search_start);
            break;
        }

        int offset = 0;
        const auto dx = workspace.newton_step.segment(offset, nx);
        offset += nx;
        const auto dxi = workspace.newton_step.segment(offset, nxi);
        offset += nxi;
        const auto dv_d = workspace.newton_step.segment(offset, nd);
        offset += nd;
        const auto dgamma_b = workspace.newton_step.segment(offset, nb);
        offset += nb;
        const auto dgamma_d = workspace.newton_step.segment(offset, nd);
        timing.search_direction += elapsed_seconds(search_start);

        const auto line_search_start = Clock::now();
        workspace.qx_dx = data.Q_x * dx;
        workspace.qxi_dxi = transformed.Q_xi * dxi;

        const double d_objective =
            workspace.qx_x.dot(dx) + data.r_x.dot(dx) +
            workspace.qxi_xi.dot(dxi) + transformed.r_xi.dot(dxi);
        const double objective_curvature =
            dx.dot(workspace.qx_dx) + dxi.dot(workspace.qxi_dxi);
        const double d_dgap = workspace.dgap_gradient.dot(dxi);
        const int violation_size = nb + 2 * nd;
        const double violation_norm = violation_size > 0
            ? workspace.kkt_residual.tail(violation_size).lpNorm<1>()
            : 0.0;

        beta = options_.beta_init;
        if (violation_norm > std::numeric_limits<double>::epsilon()) {
            const double beta_trial =
                (d_objective + mu * d_dgap) / ((1.0 - options_.rho) * violation_norm);
            if (std::isfinite(beta_trial) && beta < beta_trial) {
                beta = beta_trial + 1.0;
            }
        }

        const double theta = objective + mu * dgap_residual + beta * violation_norm;
        const double d_theta = d_objective + mu * d_dgap - beta * violation_norm;

        alpha = 1.0;
        line_search_status = 0;

        while (true) {
            workspace.xi_trial = xi + alpha * dxi;
            workspace.v_d_trial = v_d + alpha * dv_d;
            workspace.gamma_d_trial = gamma_d + alpha * dgamma_d;

            const double objective_trial =
                objective + alpha * d_objective +
                0.5 * alpha * alpha * objective_curvature;
            const double dgap_trial = dgap_.value(workspace.xi_trial);

            double violation_trial_norm = 0.0;
            // The equality/inequality trial residuals are affine in alpha.
            // Reuse the corresponding Newton equations instead of multiplying
            // A/C by every trial point during backtracking.
            for (int i = 0; i < nb; ++i) {
                const double eq_direction = -workspace.kkt_residual[row_eq + i] +
                    options_.nu_b * dgamma_b[i];
                const double value = workspace.kkt_residual[row_eq + i] + alpha * eq_direction;
                violation_trial_norm += std::abs(value);
            }
            for (int i = 0; i < nd; ++i) {
                const double value = (1.0 - alpha) * workspace.kkt_residual[row_ineq + i];
                violation_trial_norm += std::abs(value);
            }
            violation_trial_norm += fb_.l1_norm(workspace.v_d_trial, workspace.gamma_d_trial);

            const double theta_trial =
                objective_trial + mu * dgap_trial + beta * violation_trial_norm;

            if (theta_trial <= theta + alpha * options_.nu_D * d_theta) {
                line_search_status = 1;
                break;
            }
            if (alpha <= options_.alpha_min * (1.0 + std::numeric_limits<double>::epsilon())) {
                break;
            }
            alpha = std::max(options_.nu_alpha * alpha, options_.alpha_min);
        }

        if (line_search_status == 0) {
            status = 0;
            message = "- Solver fails: the min stepsize reached";
            timing.step_size += elapsed_seconds(line_search_start);
            break;
        }

        workspace.x_trial = x + alpha * dx;
        x.swap(workspace.x_trial);
        xi.swap(workspace.xi_trial);
        v_d.swap(workspace.v_d_trial);
        gamma_b += alpha * dgamma_b;
        gamma_d.swap(workspace.gamma_d_trial);
        timing.step_size += elapsed_seconds(line_search_start);
    }

    timing.total = elapsed_seconds(total_start);
    timing.search_other = std::max(
        0.0,
        timing.search_direction - timing.kkt_assembly - timing.lu_analyze -
            timing.lu_factorize - timing.lu_solve);
    timing.other = std::max(0.0, timing.total - timing.check_termination - timing.search_direction - timing.step_size);

    SolveResult result;
    result.solution.x = x;
    deinterleave(xi, result.solution.lambda, result.solution.eta);
    result.solution.v_d = v_d;
    result.solution.gamma_b = gamma_b;
    result.solution.gamma_d = gamma_d;

    result.info.cost = objective;
    result.info.dgap_res = dgap_residual;
    result.info.kkt_res = kkt_norm;
    result.info.mu = mu;
    result.info.iter_num = iteration;
    result.info.status = status;
    result.info.message = message;
    result.info.timing = timing;
    result.info.line_search_status = line_search_status;

    if (options_.print_level > 0) {
        std::cout << "********************** Solution Information *********************\n"
                  << "1. Terminal Status\n" << message << '\n'
                  << "2. Iteration Process Message\n"
                  << "- Iterations: .........." << iteration << '\n'
                  << "- Elapsed time: ........" << std::fixed << std::setprecision(6) << timing.total << "s\n"
                  << "- Average time: ........" << 1000.0 * timing.total / std::max(iteration, 1) << " ms/Iter\n"
                  << "3. Solution Message\n"
                  << std::scientific << std::setprecision(3)
                  << "- Cost: ................" << objective << ";\n"
                  << "- D-gap residual: ......" << dgap_residual << ";\n"
                  << "- KKT residual: ........" << kkt_norm << ";\n"
                  << "*****************************************************************\n";
    }

    return result;
}

}  // namespace gapese
