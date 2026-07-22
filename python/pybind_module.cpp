#include "gapese/solver.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

const py::handle dict_value(const py::dict& dictionary, const char* key) {
    if (!dictionary.contains(key)) {
        throw std::invalid_argument(std::string("missing dictionary field: ") + key);
    }
    return dictionary[key];
}

template <typename T>
T get_alias(const py::dict& dictionary, std::initializer_list<const char*> keys, T default_value) {
    for (const char* key : keys) {
        if (dictionary.contains(key)) return py::cast<T>(dictionary[key]);
    }
    return default_value;
}

gapese::SparseMatrix sparse_from_python(const py::handle& object, const char* name) {
    if (py::isinstance<py::float_>(object) || py::isinstance<py::int_>(object)) {
        gapese::SparseMatrix matrix(1, 1);
        const double value = py::cast<double>(object);
        if (value != 0.0) matrix.insert(0, 0) = value;
        matrix.makeCompressed();
        return matrix;
    }

    if (py::hasattr(object, "tocsc") && py::hasattr(object, "shape")) {
        py::object csc = object.attr("tocsc")();
        if (py::hasattr(csc, "sum_duplicates")) csc.attr("sum_duplicates")();
        if (py::hasattr(csc, "sort_indices")) csc.attr("sort_indices")();

        const auto shape = py::cast<std::pair<int, int>>(csc.attr("shape"));
        auto data = py::array_t<double, py::array::forcecast>::ensure(csc.attr("data"));
        auto indices = py::array_t<long long, py::array::forcecast>::ensure(csc.attr("indices"));
        auto indptr = py::array_t<long long, py::array::forcecast>::ensure(csc.attr("indptr"));
        if (!data || !indices || !indptr) {
            throw std::invalid_argument(std::string(name) + " has invalid CSC arrays");
        }

        const auto data_view = data.unchecked<1>();
        const auto indices_view = indices.unchecked<1>();
        const auto indptr_view = indptr.unchecked<1>();
        std::vector<gapese::Triplet> triplets;
        triplets.reserve(static_cast<std::size_t>(data_view.shape(0)));
        for (int col = 0; col < shape.second; ++col) {
            for (long long p = indptr_view(col); p < indptr_view(col + 1); ++p) {
                const double value = data_view(p);
                if (value != 0.0) {
                    triplets.emplace_back(static_cast<int>(indices_view(p)), col, value);
                }
            }
        }
        gapese::SparseMatrix matrix(shape.first, shape.second);
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        matrix.makeCompressed();
        return matrix;
    }

    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(object);
    if (!array || array.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be a scalar, a 2-D NumPy array, or a SciPy sparse matrix");
    }
    const int rows = static_cast<int>(array.shape(0));
    const int cols = static_cast<int>(array.shape(1));
    const auto view = array.unchecked<2>();
    std::vector<gapese::Triplet> triplets;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const double value = view(row, col);
            if (value != 0.0) triplets.emplace_back(row, col, value);
        }
    }
    gapese::SparseMatrix matrix(rows, cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
    return matrix;
}

gapese::Vector vector_from_python(const py::handle& object, const char* name) {
    if (py::isinstance<py::float_>(object) || py::isinstance<py::int_>(object)) {
        return gapese::Vector::Constant(1, py::cast<double>(object));
    }
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(object);
    if (!array) {
        throw std::invalid_argument(std::string(name) + " must be convertible to a real NumPy array");
    }
    if (array.ndim() == 1) {
        gapese::Vector vector(array.shape(0));
        const auto view = array.unchecked<1>();
        for (py::ssize_t i = 0; i < array.shape(0); ++i) vector[static_cast<int>(i)] = view(i);
        return vector;
    }
    if (array.ndim() == 2 && (array.shape(0) == 1 || array.shape(1) == 1 || array.size() == 0)) {
        gapese::Vector vector(array.size());
        const auto view = array.unchecked<2>();
        if (array.shape(1) == 1) {
            for (py::ssize_t i = 0; i < array.shape(0); ++i) vector[static_cast<int>(i)] = view(i, 0);
        } else {
            for (py::ssize_t i = 0; i < array.shape(1); ++i) vector[static_cast<int>(i)] = view(0, i);
        }
        return vector;
    }
    throw std::invalid_argument(std::string(name) + " must be a scalar or vector-like NumPy array");
}

py::array_t<double> vector_to_numpy(const gapese::Vector& vector) {
    py::array_t<double> array(vector.size());
    auto view = array.mutable_unchecked<1>();
    for (int i = 0; i < vector.size(); ++i) view(i) = vector[i];
    return array;
}

gapese::ProblemSize problem_from_dict(const py::dict& dictionary) {
    return {
        py::cast<int>(dict_value(dictionary, "n_x")),
        py::cast<int>(dict_value(dictionary, "n_lambda")),
        py::cast<int>(dict_value(dictionary, "n_b")),
        py::cast<int>(dict_value(dictionary, "n_d"))};
}

gapese::Options options_from_dict(const py::dict& dictionary) {
    gapese::Options options;
    options.print_level = get_alias<int>(dictionary, {"print_level"}, options.print_level);
    options.dgap_a = get_alias<double>(dictionary, {"dgap_a", "D_gap_param_a"}, options.dgap_a);
    options.dgap_b = get_alias<double>(dictionary, {"dgap_b", "D_gap_param_b"}, options.dgap_b);
    options.fb_sigma = get_alias<double>(dictionary, {"fb_sigma", "FB_sigma"}, options.fb_sigma);
    options.max_iter = get_alias<int>(dictionary, {"max_iter"}, options.max_iter);
    options.tol_kkt = get_alias<double>(dictionary, {"tol_kkt", "tol_KKT_res"}, options.tol_kkt);
    options.tol_dgap = get_alias<double>(dictionary, {"tol_dgap", "tol_D_gap_res"}, options.tol_dgap);
    options.newton_step_max = get_alias<double>(dictionary, {"newton_step_max", "Newton_step_norm"}, options.newton_step_max);
    options.mu_min = get_alias<double>(dictionary, {"mu_min"}, options.mu_min);
    options.mu_max = get_alias<double>(dictionary, {"mu_max"}, options.mu_max);
    options.mu_update = get_alias<double>(dictionary, {"mu_update"}, options.mu_update);
    options.nu_x = get_alias<double>(dictionary, {"nu_x"}, options.nu_x);
    options.nu_xi = get_alias<double>(dictionary, {"nu_xi"}, options.nu_xi);
    options.nu_b = get_alias<double>(dictionary, {"nu_b"}, options.nu_b);
    options.nu_d = get_alias<double>(dictionary, {"nu_d"}, options.nu_d);
    options.beta_init = get_alias<double>(dictionary, {"beta_init"}, options.beta_init);
    options.rho = get_alias<double>(dictionary, {"rho"}, options.rho);
    options.alpha_min = get_alias<double>(dictionary, {"alpha_min"}, options.alpha_min);
    options.nu_alpha = get_alias<double>(dictionary, {"nu_alpha"}, options.nu_alpha);
    options.nu_D = get_alias<double>(dictionary, {"nu_D"}, options.nu_D);
    return options;
}

gapese::CcqpData data_from_dict(const py::dict& dictionary) {
    gapese::CcqpData data;
    data.Q_x = sparse_from_python(dict_value(dictionary, "Q_x"), "Q_x");
    data.Q_lambda = sparse_from_python(dict_value(dictionary, "Q_lambda"), "Q_lambda");
    data.Q_eta = sparse_from_python(dict_value(dictionary, "Q_eta"), "Q_eta");
    data.r_x = vector_from_python(dict_value(dictionary, "r_x"), "r_x");
    data.r_lambda = vector_from_python(dict_value(dictionary, "r_lambda"), "r_lambda");
    data.r_eta = vector_from_python(dict_value(dictionary, "r_eta"), "r_eta");
    data.A_x = sparse_from_python(dict_value(dictionary, "A_x"), "A_x");
    data.A_lambda = sparse_from_python(dict_value(dictionary, "A_lambda"), "A_lambda");
    data.A_eta = sparse_from_python(dict_value(dictionary, "A_eta"), "A_eta");
    data.b = vector_from_python(dict_value(dictionary, "b"), "b");
    data.C_x = sparse_from_python(dict_value(dictionary, "C_x"), "C_x");
    data.C_lambda = sparse_from_python(dict_value(dictionary, "C_lambda"), "C_lambda");
    data.C_eta = sparse_from_python(dict_value(dictionary, "C_eta"), "C_eta");
    data.d = vector_from_python(dict_value(dictionary, "d"), "d");
    return data;
}

gapese::InitialGuess initial_from_dict(const py::dict& dictionary) {
    gapese::InitialGuess initial;
    initial.x = vector_from_python(dict_value(dictionary, "x"), "x");
    initial.lambda = vector_from_python(dict_value(dictionary, "lambda"), "lambda");
    initial.eta = vector_from_python(dict_value(dictionary, "eta"), "eta");
    if (dictionary.contains("v_d")) initial.v_d = vector_from_python(dictionary["v_d"], "v_d");
    if (dictionary.contains("gamma_b")) initial.gamma_b = vector_from_python(dictionary["gamma_b"], "gamma_b");
    if (dictionary.contains("gamma_d")) initial.gamma_d = vector_from_python(dictionary["gamma_d"], "gamma_d");
    return initial;
}

py::dict options_to_dict(const gapese::Options& options) {
    py::dict result;
    result["print_level"] = options.print_level;
    result["dgap_a"] = options.dgap_a;
    result["dgap_b"] = options.dgap_b;
    result["fb_sigma"] = options.fb_sigma;
    result["max_iter"] = options.max_iter;
    result["tol_kkt"] = options.tol_kkt;
    result["tol_dgap"] = options.tol_dgap;
    result["newton_step_max"] = options.newton_step_max;
    result["mu_min"] = options.mu_min;
    result["mu_max"] = options.mu_max;
    result["mu_update"] = options.mu_update;
    result["nu_x"] = options.nu_x;
    result["nu_xi"] = options.nu_xi;
    result["nu_b"] = options.nu_b;
    result["nu_d"] = options.nu_d;
    result["beta_init"] = options.beta_init;
    result["rho"] = options.rho;
    result["alpha_min"] = options.alpha_min;
    result["nu_alpha"] = options.nu_alpha;
    result["nu_D"] = options.nu_D;
    return result;
}

py::tuple result_to_python(const gapese::SolveResult& result) {
    py::dict solution;
    solution["x"] = vector_to_numpy(result.solution.x);
    solution["lambda"] = vector_to_numpy(result.solution.lambda);
    solution["eta"] = vector_to_numpy(result.solution.eta);
    solution["v_d"] = vector_to_numpy(result.solution.v_d);
    solution["gamma_b"] = vector_to_numpy(result.solution.gamma_b);
    solution["gamma_d"] = vector_to_numpy(result.solution.gamma_d);

    py::dict timing;
    timing["check_termination"] = result.info.timing.check_termination;
    timing["search_direction"] = result.info.timing.search_direction;
    timing["step_size"] = result.info.timing.step_size;
    timing["else"] = result.info.timing.other;
    timing["total"] = result.info.timing.total;

    py::dict info;
    info["cost"] = result.info.cost;
    info["D_gap_res"] = result.info.dgap_res;
    info["KKT_res"] = result.info.kkt_res;
    info["mu"] = result.info.mu;
    info["iter_num"] = result.info.iter_num;
    info["status"] = result.info.status;
    info["msg"] = result.info.message;
    info["Time"] = timing;
    info["line_search_status"] = result.info.line_search_status;
    return py::make_tuple(solution, info);
}

class PythonSolver {
public:
    PythonSolver(const py::dict& problem, const py::dict& options)
        : solver_(problem_from_dict(problem), options_from_dict(options)) {}

    py::tuple solve_ccqp(const py::dict& data, const py::dict& initial) {
        const gapese::CcqpData cpp_data = data_from_dict(data);
        const gapese::InitialGuess cpp_initial = initial_from_dict(initial);
        gapese::SolveResult result;
        {
            py::gil_scoped_release release;
            result = solver_.solve(cpp_data, cpp_initial);
        }
        return result_to_python(result);
    }

private:
    gapese::Solver solver_;
};

}  // namespace

PYBIND11_MODULE(gapese, module) {
    module.doc() = "Stateful C++ implementation of the Gapese CCQP solver";
    module.attr("__version__") = "0.1.0";

    module.def("default_options", []() { return options_to_dict(gapese::Options{}); });

    py::class_<PythonSolver>(module, "Solver")
        .def(py::init<const py::dict&, const py::dict&>(), py::arg("problem"), py::arg("options") = py::dict())
        .def("solve_ccqp", &PythonSolver::solve_ccqp, py::arg("ccqp"), py::arg("initial"));
}
