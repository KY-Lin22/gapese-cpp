#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <optional>
#include <string>

namespace gapese {

using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using Vector = Eigen::VectorXd;
using Triplet = Eigen::Triplet<double, int>;

struct ProblemSize {
    int n_x = 0;
    int n_lambda = 0;
    int n_b = 0;
    int n_d = 0;

    void validate() const;
};

struct CcqpData {
    SparseMatrix Q_x;
    SparseMatrix Q_lambda;
    SparseMatrix Q_eta;
    Vector r_x;
    Vector r_lambda;
    Vector r_eta;

    SparseMatrix A_x;
    SparseMatrix A_lambda;
    SparseMatrix A_eta;
    Vector b;

    SparseMatrix C_x;
    SparseMatrix C_lambda;
    SparseMatrix C_eta;
    Vector d;
};

struct InitialGuess {
    Vector x;
    Vector lambda;
    Vector eta;
    std::optional<Vector> v_d;
    std::optional<Vector> gamma_b;
    std::optional<Vector> gamma_d;
};

struct Solution {
    Vector x;
    Vector lambda;
    Vector eta;
    Vector v_d;
    Vector gamma_b;
    Vector gamma_d;
};

struct Timing {
    double check_termination = 0.0;
    double search_direction = 0.0;
    double kkt_assembly = 0.0;
    double kkt_analyze = 0.0;
    double kkt_factorize = 0.0;
    double kkt_solve = 0.0;
    double search_other = 0.0;
    double step_size = 0.0;
    double other = 0.0;
    double total = 0.0;
};

struct Info {
    double cost = 0.0;
    double dgap_res = 0.0;
    double kkt_res = 0.0;
    double mu = 0.0;
    int iter_num = 0;
    int status = 0;
    std::string message;
    Timing timing;
    int line_search_status = 0;
};

struct SolveResult {
    Solution solution;
    Info info;
};

}  // namespace gapese
