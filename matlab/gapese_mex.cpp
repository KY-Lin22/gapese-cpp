#include "mex.h"

#include "gapese/solver.hpp"

#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using gapese::CcqpData;
using gapese::InitialGuess;
using gapese::Options;
using gapese::ProblemSize;
using gapese::SolveResult;
using gapese::Solver;
using gapese::SparseMatrix;
using gapese::Triplet;
using gapese::Vector;

std::unordered_map<std::uint64_t, std::unique_ptr<Solver>>& registry() {
    static std::unordered_map<std::uint64_t, std::unique_ptr<Solver>> value;
    return value;
}

std::uint64_t& next_handle() {
    static std::uint64_t value = 1;
    return value;
}

void cleanup_registry() {
    registry().clear();
    while (mexIsLocked()) mexUnlock();
}

const double* real_data(const mxArray* array) {
#if MX_HAS_INTERLEAVED_COMPLEX
    return mxGetDoubles(array);
#else
    return mxGetPr(array);
#endif
}

double* writable_real_data(mxArray* array) {
#if MX_HAS_INTERLEAVED_COMPLEX
    return mxGetDoubles(array);
#else
    return mxGetPr(array);
#endif
}

const mxArray* field(const mxArray* structure, const char* name, bool required = true) {
    if (!mxIsStruct(structure) || mxGetNumberOfElements(structure) != 1) {
        throw std::invalid_argument("expected a scalar MATLAB struct");
    }
    const mxArray* value = mxGetField(structure, 0, name);
    if (required && value == nullptr) {
        throw std::invalid_argument(std::string("missing struct field: ") + name);
    }
    return value;
}

double scalar_field(const mxArray* structure, const char* name) {
    const mxArray* value = field(structure, name);
    if (!mxIsDouble(value) || mxIsComplex(value) || mxGetNumberOfElements(value) != 1) {
        throw std::invalid_argument(std::string(name) + " must be a real scalar");
    }
    return mxGetScalar(value);
}

int integer_field(const mxArray* structure, const char* name) {
    const double value = scalar_field(structure, name);
    if (!std::isfinite(value) || value != static_cast<int>(value)) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    return static_cast<int>(value);
}

SparseMatrix matrix_from_mx(const mxArray* array, const char* name) {
    if (!mxIsDouble(array) || mxIsComplex(array) || mxGetNumberOfDimensions(array) != 2) {
        throw std::invalid_argument(std::string(name) + " must be a real double matrix");
    }
    const mwSize rows_mw = mxGetM(array);
    const mwSize cols_mw = mxGetN(array);
    if (rows_mw > static_cast<mwSize>(std::numeric_limits<int>::max()) ||
        cols_mw > static_cast<mwSize>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(name) + " dimensions exceed Eigen integer limits");
    }
    const int rows = static_cast<int>(rows_mw);
    const int cols = static_cast<int>(cols_mw);
    std::vector<Triplet> triplets;

    if (mxIsSparse(array)) {
        const mwIndex* col_ptr = mxGetJc(array);
        const mwIndex* row_idx = mxGetIr(array);
        const double* values = real_data(array);
        const mwIndex nnz = col_ptr[cols_mw];
        if (nnz > static_cast<mwIndex>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument(std::string(name) + " has too many nonzeros");
        }
        triplets.reserve(static_cast<std::size_t>(nnz));
        for (mwIndex col = 0; col < cols_mw; ++col) {
            for (mwIndex p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
                if (values[p] != 0.0) {
                    triplets.emplace_back(static_cast<int>(row_idx[p]), static_cast<int>(col), values[p]);
                }
            }
        }
    } else {
        const double* values = real_data(array);
        triplets.reserve(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                const double value = values[row + rows * col];
                if (value != 0.0) triplets.emplace_back(row, col, value);
            }
        }
    }

    SparseMatrix matrix(rows, cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
    return matrix;
}

Vector vector_from_mx(const mxArray* array, const char* name) {
    if (!mxIsDouble(array) || mxIsComplex(array) || mxGetNumberOfDimensions(array) != 2) {
        throw std::invalid_argument(std::string(name) + " must be a real double vector");
    }
    const mwSize rows = mxGetM(array);
    const mwSize cols = mxGetN(array);
    if (!(rows == 1 || cols == 1 || rows * cols == 0)) {
        throw std::invalid_argument(std::string(name) + " must be a vector");
    }
    const mwSize size_mw = mxGetNumberOfElements(array);
    if (size_mw > static_cast<mwSize>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(name) + " is too long");
    }
    Vector vector = Vector::Zero(static_cast<int>(size_mw));
    if (mxIsSparse(array)) {
        const mwIndex* col_ptr = mxGetJc(array);
        const mwIndex* row_idx = mxGetIr(array);
        const double* values = real_data(array);
        if (cols == 1) {
            for (mwIndex p = col_ptr[0]; p < col_ptr[1]; ++p) vector[static_cast<int>(row_idx[p])] = values[p];
        } else {
            for (mwIndex col = 0; col < cols; ++col) {
                for (mwIndex p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
                    vector[static_cast<int>(col)] = values[p];
                }
            }
        }
    } else {
        const double* values = real_data(array);
        for (int i = 0; i < vector.size(); ++i) vector[i] = values[i];
    }
    return vector;
}

ProblemSize problem_from_mx(const mxArray* structure) {
    ProblemSize problem;
    problem.n_x = integer_field(structure, "n_x");
    problem.n_lambda = integer_field(structure, "n_lambda");
    problem.n_b = integer_field(structure, "n_b");
    problem.n_d = integer_field(structure, "n_d");
    return problem;
}

Options options_from_mx(const mxArray* structure) {
    if (!mxIsStruct(structure) || mxGetNumberOfElements(structure) != 1) {
        throw std::invalid_argument("options must be a scalar MATLAB struct");
    }

    static const std::unordered_set<std::string> canonical_names = {
        "print_level",
        "dgap_a",
        "dgap_b",
        "fb_sigma",
        "max_iter",
        "tol_kkt",
        "tol_dgap",
        "newton_step_max",
        "mu_min",
        "mu_max",
        "mu_update",
        "nu_x",
        "nu_xi",
        "nu_b",
        "nu_d",
        "beta_init",
        "rho",
        "alpha_min",
        "nu_alpha",
        "nu_D"};

    const int field_count = mxGetNumberOfFields(structure);
    for (int i = 0; i < field_count; ++i) {
        const char* name = mxGetFieldNameByNumber(structure, i);
        if (name == nullptr || canonical_names.find(name) == canonical_names.end()) {
            throw std::invalid_argument(
                std::string("unknown Gapese option: ") + (name == nullptr ? "<unnamed>" : name));
        }
    }

    Options options;
    if (field(structure, "print_level", false)) options.print_level = integer_field(structure, "print_level");
    if (field(structure, "dgap_a", false)) options.dgap_a = scalar_field(structure, "dgap_a");
    if (field(structure, "dgap_b", false)) options.dgap_b = scalar_field(structure, "dgap_b");
    if (field(structure, "fb_sigma", false)) options.fb_sigma = scalar_field(structure, "fb_sigma");
    if (field(structure, "max_iter", false)) options.max_iter = integer_field(structure, "max_iter");
    if (field(structure, "tol_kkt", false)) options.tol_kkt = scalar_field(structure, "tol_kkt");
    if (field(structure, "tol_dgap", false)) options.tol_dgap = scalar_field(structure, "tol_dgap");
    if (field(structure, "newton_step_max", false)) {
        options.newton_step_max = scalar_field(structure, "newton_step_max");
    }
    if (field(structure, "mu_min", false)) options.mu_min = scalar_field(structure, "mu_min");
    if (field(structure, "mu_max", false)) options.mu_max = scalar_field(structure, "mu_max");
    if (field(structure, "mu_update", false)) options.mu_update = scalar_field(structure, "mu_update");
    if (field(structure, "nu_x", false)) options.nu_x = scalar_field(structure, "nu_x");
    if (field(structure, "nu_xi", false)) options.nu_xi = scalar_field(structure, "nu_xi");
    if (field(structure, "nu_b", false)) options.nu_b = scalar_field(structure, "nu_b");
    if (field(structure, "nu_d", false)) options.nu_d = scalar_field(structure, "nu_d");
    if (field(structure, "beta_init", false)) options.beta_init = scalar_field(structure, "beta_init");
    if (field(structure, "rho", false)) options.rho = scalar_field(structure, "rho");
    if (field(structure, "alpha_min", false)) options.alpha_min = scalar_field(structure, "alpha_min");
    if (field(structure, "nu_alpha", false)) options.nu_alpha = scalar_field(structure, "nu_alpha");
    if (field(structure, "nu_D", false)) options.nu_D = scalar_field(structure, "nu_D");
    options.validate();
    return options;
}

CcqpData data_from_mx(const mxArray* structure) {
    CcqpData data;
    data.Q_x = matrix_from_mx(field(structure, "Q_x"), "Q_x");
    data.Q_lambda = matrix_from_mx(field(structure, "Q_lambda"), "Q_lambda");
    data.Q_eta = matrix_from_mx(field(structure, "Q_eta"), "Q_eta");
    data.r_x = vector_from_mx(field(structure, "r_x"), "r_x");
    data.r_lambda = vector_from_mx(field(structure, "r_lambda"), "r_lambda");
    data.r_eta = vector_from_mx(field(structure, "r_eta"), "r_eta");
    data.A_x = matrix_from_mx(field(structure, "A_x"), "A_x");
    data.A_lambda = matrix_from_mx(field(structure, "A_lambda"), "A_lambda");
    data.A_eta = matrix_from_mx(field(structure, "A_eta"), "A_eta");
    data.b = vector_from_mx(field(structure, "b"), "b");
    data.C_x = matrix_from_mx(field(structure, "C_x"), "C_x");
    data.C_lambda = matrix_from_mx(field(structure, "C_lambda"), "C_lambda");
    data.C_eta = matrix_from_mx(field(structure, "C_eta"), "C_eta");
    data.d = vector_from_mx(field(structure, "d"), "d");
    return data;
}

InitialGuess initial_from_mx(const mxArray* structure) {
    InitialGuess initial;
    initial.x = vector_from_mx(field(structure, "x"), "Init.x");
    initial.lambda = vector_from_mx(field(structure, "lambda"), "Init.lambda");
    initial.eta = vector_from_mx(field(structure, "eta"), "Init.eta");
    if (const mxArray* value = field(structure, "v_d", false)) initial.v_d = vector_from_mx(value, "Init.v_d");
    if (const mxArray* value = field(structure, "gamma_b", false)) initial.gamma_b = vector_from_mx(value, "Init.gamma_b");
    if (const mxArray* value = field(structure, "gamma_d", false)) initial.gamma_d = vector_from_mx(value, "Init.gamma_d");
    return initial;
}

std::uint64_t handle_from_mx(const mxArray* array) {
    if (!mxIsUint64(array) || mxGetNumberOfElements(array) != 1) {
        throw std::invalid_argument("solver handle must be a uint64 scalar");
    }
    return *static_cast<const std::uint64_t*>(mxGetData(array));
}

Solver& solver_from_handle(std::uint64_t handle) {
    const auto iterator = registry().find(handle);
    if (iterator == registry().end()) throw std::invalid_argument("invalid or destroyed Gapese MEX handle");
    return *iterator->second;
}

mxArray* vector_to_mx(const Vector& vector) {
    mxArray* array = mxCreateDoubleMatrix(static_cast<mwSize>(vector.size()), 1, mxREAL);
    double* values = writable_real_data(array);
    for (int i = 0; i < vector.size(); ++i) values[i] = vector[i];
    return array;
}

mxArray* solution_to_mx(const SolveResult& result) {
    const char* names[] = {"x", "lambda", "eta", "v_d", "gamma_b", "gamma_d"};
    mxArray* structure = mxCreateStructMatrix(1, 1, 6, names);
    mxSetField(structure, 0, "x", vector_to_mx(result.solution.x));
    mxSetField(structure, 0, "lambda", vector_to_mx(result.solution.lambda));
    mxSetField(structure, 0, "eta", vector_to_mx(result.solution.eta));
    mxSetField(structure, 0, "v_d", vector_to_mx(result.solution.v_d));
    mxSetField(structure, 0, "gamma_b", vector_to_mx(result.solution.gamma_b));
    mxSetField(structure, 0, "gamma_d", vector_to_mx(result.solution.gamma_d));
    return structure;
}

mxArray* info_to_mx(const SolveResult& result) {
    const char* time_names[] = {
        "check_termination", "search_direction", "kkt_assembly", "lu_analyze", "lu_factorize",
        "lu_solve", "search_other", "step_size", "else", "total"};
    mxArray* time = mxCreateStructMatrix(1, 1, 10, time_names);
    mxSetField(time, 0, "check_termination", mxCreateDoubleScalar(result.info.timing.check_termination));
    mxSetField(time, 0, "search_direction", mxCreateDoubleScalar(result.info.timing.search_direction));
    mxSetField(time, 0, "kkt_assembly", mxCreateDoubleScalar(result.info.timing.kkt_assembly));
    mxSetField(time, 0, "lu_analyze", mxCreateDoubleScalar(result.info.timing.lu_analyze));
    mxSetField(time, 0, "lu_factorize", mxCreateDoubleScalar(result.info.timing.lu_factorize));
    mxSetField(time, 0, "lu_solve", mxCreateDoubleScalar(result.info.timing.lu_solve));
    mxSetField(time, 0, "search_other", mxCreateDoubleScalar(result.info.timing.search_other));
    mxSetField(time, 0, "step_size", mxCreateDoubleScalar(result.info.timing.step_size));
    mxSetField(time, 0, "else", mxCreateDoubleScalar(result.info.timing.other));
    mxSetField(time, 0, "total", mxCreateDoubleScalar(result.info.timing.total));

    const char* names[] = {
        "cost", "D_gap_res", "KKT_res", "mu", "iter_num", "status", "msg", "Time", "line_search_status"};
    mxArray* structure = mxCreateStructMatrix(1, 1, 9, names);
    mxSetField(structure, 0, "cost", mxCreateDoubleScalar(result.info.cost));
    mxSetField(structure, 0, "D_gap_res", mxCreateDoubleScalar(result.info.dgap_res));
    mxSetField(structure, 0, "KKT_res", mxCreateDoubleScalar(result.info.kkt_res));
    mxSetField(structure, 0, "mu", mxCreateDoubleScalar(result.info.mu));
    mxSetField(structure, 0, "iter_num", mxCreateDoubleScalar(result.info.iter_num));
    mxSetField(structure, 0, "status", mxCreateDoubleScalar(result.info.status));
    mxSetField(structure, 0, "msg", mxCreateString(result.info.message.c_str()));
    mxSetField(structure, 0, "Time", time);
    mxSetField(structure, 0, "line_search_status", mxCreateDoubleScalar(result.info.line_search_status));
    return structure;
}

std::string command_from_mx(const mxArray* array) {
    if (!mxIsChar(array)) throw std::invalid_argument("first input must be a command string");
    char* text = mxArrayToString(array);
    if (text == nullptr) throw std::runtime_error("failed to read command string");
    std::string command(text);
    mxFree(text);
    return command;
}

}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    static bool registered_cleanup = false;
    if (!registered_cleanup) {
        mexAtExit(cleanup_registry);
        registered_cleanup = true;
    }

    try {
        if (nrhs < 1) throw std::invalid_argument("gapese_mex requires a command");
        const std::string command = command_from_mx(prhs[0]);

        if (command == "create") {
            if (nrhs != 3 || nlhs > 1) throw std::invalid_argument("create usage: handle = gapese_mex('create', Prob, Option)");
            auto solver = std::make_unique<Solver>(problem_from_mx(prhs[1]), options_from_mx(prhs[2]));
            const std::uint64_t handle = next_handle()++;
            registry().emplace(handle, std::move(solver));
            mexLock();
            plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
            *static_cast<std::uint64_t*>(mxGetData(plhs[0])) = handle;
            return;
        }

        if (command == "solve") {
            if (nrhs != 4 || nlhs > 2) throw std::invalid_argument("solve usage: [Sol, Info] = gapese_mex('solve', handle, CCQP, Init)");
            Solver& solver = solver_from_handle(handle_from_mx(prhs[1]));
            const SolveResult result = solver.solve(data_from_mx(prhs[2]), initial_from_mx(prhs[3]));
            if (nlhs >= 1) plhs[0] = solution_to_mx(result);
            if (nlhs >= 2) plhs[1] = info_to_mx(result);
            return;
        }

        if (command == "destroy") {
            if (nrhs != 2 || nlhs != 0) throw std::invalid_argument("destroy usage: gapese_mex('destroy', handle)");
            const std::uint64_t handle = handle_from_mx(prhs[1]);
            const std::size_t erased = registry().erase(handle);
            if (erased == 0) throw std::invalid_argument("invalid or already destroyed Gapese MEX handle");
            mexUnlock();
            return;
        }

        if (command == "clear") {
            cleanup_registry();
            return;
        }

        throw std::invalid_argument("unknown gapese_mex command: " + command);
    } catch (const std::exception& exception) {
        mexErrMsgIdAndTxt("gapese:mex", "%s", exception.what());
    }
}
