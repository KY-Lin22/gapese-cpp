#include "gapese/c_api.h"

#include "gapese/solver.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct gapese_solver_handle {
    std::unique_ptr<gapese::Solver> solver;
    gapese::ProblemSize problem;
};

namespace {

void copy_text(const std::string& text, char* destination, int capacity) {
    if (destination == nullptr || capacity <= 0) return;
    const int count = std::min<int>(capacity - 1, static_cast<int>(text.size()));
    if (count > 0) std::memcpy(destination, text.data(), static_cast<std::size_t>(count));
    destination[count] = '\0';
}

gapese::Options to_cpp_options(const gapese_options& value) {
    gapese::Options options;
    options.print_level = value.print_level;
    options.dgap_a = value.dgap_a;
    options.dgap_b = value.dgap_b;
    options.fb_sigma = value.fb_sigma;
    options.max_iter = value.max_iter;
    options.tol_kkt = value.tol_kkt;
    options.tol_dgap = value.tol_dgap;
    options.newton_step_max = value.newton_step_max;
    options.mu_min = value.mu_min;
    options.mu_max = value.mu_max;
    options.mu_update = value.mu_update;
    options.nu_x = value.nu_x;
    options.nu_xi = value.nu_xi;
    options.nu_b = value.nu_b;
    options.nu_d = value.nu_d;
    options.beta_init = value.beta_init;
    options.rho = value.rho;
    options.alpha_min = value.alpha_min;
    options.nu_alpha = value.nu_alpha;
    options.nu_D = value.nu_D;
    return options;
}

gapese::SparseMatrix to_sparse(const gapese_csc_view& view) {
    if (view.rows < 0 || view.cols < 0 || view.nnz < 0) {
        throw std::invalid_argument("invalid CSC matrix dimensions");
    }
    if (view.col_ptr == nullptr || (view.nnz > 0 && (view.row_idx == nullptr || view.values == nullptr))) {
        throw std::invalid_argument("invalid CSC matrix pointers");
    }

    std::vector<gapese::Triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(view.nnz));
    for (int col = 0; col < view.cols; ++col) {
        for (int p = view.col_ptr[col]; p < view.col_ptr[col + 1]; ++p) {
            triplets.emplace_back(view.row_idx[p], col, view.values[p]);
        }
    }
    gapese::SparseMatrix result(view.rows, view.cols);
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();
    return result;
}

gapese::Vector to_vector(const gapese_vector_view& view) {
    if (view.size < 0 || (view.size > 0 && view.values == nullptr)) {
        throw std::invalid_argument("invalid vector view");
    }
    gapese::Vector result(view.size);
    for (int i = 0; i < view.size; ++i) result[i] = view.values[i];
    return result;
}

gapese::CcqpData to_data(const gapese_ccqp_view& value) {
    gapese::CcqpData data;
    data.Q_x = to_sparse(value.Q_x);
    data.Q_lambda = to_sparse(value.Q_lambda);
    data.Q_eta = to_sparse(value.Q_eta);
    data.r_x = to_vector(value.r_x);
    data.r_lambda = to_vector(value.r_lambda);
    data.r_eta = to_vector(value.r_eta);
    data.A_x = to_sparse(value.A_x);
    data.A_lambda = to_sparse(value.A_lambda);
    data.A_eta = to_sparse(value.A_eta);
    data.b = to_vector(value.b);
    data.C_x = to_sparse(value.C_x);
    data.C_lambda = to_sparse(value.C_lambda);
    data.C_eta = to_sparse(value.C_eta);
    data.d = to_vector(value.d);
    return data;
}

gapese::InitialGuess to_initial(const gapese_initial_view& value) {
    gapese::InitialGuess initial;
    initial.x = to_vector(value.x);
    initial.lambda = to_vector(value.lambda);
    initial.eta = to_vector(value.eta);
    if (value.has_v_d) initial.v_d = to_vector(value.v_d);
    if (value.has_gamma_b) initial.gamma_b = to_vector(value.gamma_b);
    if (value.has_gamma_d) initial.gamma_d = to_vector(value.gamma_d);
    return initial;
}

void copy_vector(const gapese::Vector& source, double* destination) {
    if (source.size() > 0 && destination == nullptr) {
        throw std::invalid_argument("solution output buffer is null");
    }
    for (int i = 0; i < source.size(); ++i) destination[i] = source[i];
}

}  // namespace

extern "C" {

gapese_options gapese_default_options(void) {
    const gapese::Options source;
    gapese_options options{};
    options.print_level = source.print_level;
    options.dgap_a = source.dgap_a;
    options.dgap_b = source.dgap_b;
    options.fb_sigma = source.fb_sigma;
    options.max_iter = source.max_iter;
    options.tol_kkt = source.tol_kkt;
    options.tol_dgap = source.tol_dgap;
    options.newton_step_max = source.newton_step_max;
    options.mu_min = source.mu_min;
    options.mu_max = source.mu_max;
    options.mu_update = source.mu_update;
    options.nu_x = source.nu_x;
    options.nu_xi = source.nu_xi;
    options.nu_b = source.nu_b;
    options.nu_d = source.nu_d;
    options.beta_init = source.beta_init;
    options.rho = source.rho;
    options.alpha_min = source.alpha_min;
    options.nu_alpha = source.nu_alpha;
    options.nu_D = source.nu_D;
    return options;
}

gapese_solver_handle* gapese_create(
    gapese_problem_size problem,
    gapese_options options,
    char* error,
    int error_capacity) {
    try {
        auto handle = std::make_unique<gapese_solver_handle>();
        handle->problem = {problem.n_x, problem.n_lambda, problem.n_b, problem.n_d};
        handle->solver = std::make_unique<gapese::Solver>(handle->problem, to_cpp_options(options));
        copy_text("", error, error_capacity);
        return handle.release();
    } catch (const std::exception& exception) {
        copy_text(exception.what(), error, error_capacity);
        return nullptr;
    }
}

void gapese_destroy(gapese_solver_handle* handle) {
    delete handle;
}

int gapese_solve(
    gapese_solver_handle* handle,
    const gapese_ccqp_view* data,
    const gapese_initial_view* initial,
    gapese_solution_buffer* solution,
    gapese_info_c* info,
    char* message,
    int message_capacity,
    char* error,
    int error_capacity) {
    try {
        if (handle == nullptr || data == nullptr || initial == nullptr || solution == nullptr || info == nullptr) {
            throw std::invalid_argument("gapese_solve received a null required pointer");
        }

        const gapese::SolveResult result = handle->solver->solve(to_data(*data), to_initial(*initial));
        copy_vector(result.solution.x, solution->x);
        copy_vector(result.solution.lambda, solution->lambda);
        copy_vector(result.solution.eta, solution->eta);
        copy_vector(result.solution.v_d, solution->v_d);
        copy_vector(result.solution.gamma_b, solution->gamma_b);
        copy_vector(result.solution.gamma_d, solution->gamma_d);

        info->cost = result.info.cost;
        info->dgap_res = result.info.dgap_res;
        info->kkt_res = result.info.kkt_res;
        info->mu = result.info.mu;
        info->iter_num = result.info.iter_num;
        info->status = result.info.status;
        info->line_search_status = result.info.line_search_status;
        info->time_check_termination = result.info.timing.check_termination;
        info->time_search_direction = result.info.timing.search_direction;
        info->time_step_size = result.info.timing.step_size;
        info->time_other = result.info.timing.other;
        info->time_total = result.info.timing.total;

        copy_text(result.info.message, message, message_capacity);
        copy_text("", error, error_capacity);
        return 0;
    } catch (const std::exception& exception) {
        copy_text(exception.what(), error, error_capacity);
        return -1;
    }
}

}  // extern "C"
