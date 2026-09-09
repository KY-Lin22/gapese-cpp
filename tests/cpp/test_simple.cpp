#include "gapese/solver.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

gapese::SparseMatrix sparse_from_triplets(
    int rows,
    int cols,
    std::initializer_list<gapese::Triplet> triplets) {
    gapese::SparseMatrix matrix(rows, cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
    return matrix;
}

}  // namespace

int main() {
    gapese::ProblemSize problem{2, 1, 1, 2};
    gapese::Options options;
    options.print_level = 0;
    options.dgap_a = 0.5;
    options.dgap_b = 2.0;
    options.tol_kkt = 1e-5;
    options.tol_dgap = 1e-5;
    options.mu_min = 10.0;
    options.mu_max = 1e6;

    gapese::CcqpData data;
    data.Q_x = sparse_from_triplets(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
    data.Q_lambda = sparse_from_triplets(1, 1, {{0, 0, 1.0}});
    data.Q_eta = sparse_from_triplets(1, 1, {{0, 0, 1.0}});
    data.r_x = (gapese::Vector(2) << -1.0, -1.0).finished();
    data.r_lambda = gapese::Vector::Constant(1, -1.0);
    data.r_eta = gapese::Vector::Constant(1, -1.0);
    data.A_x = sparse_from_triplets(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}});
    data.A_lambda = gapese::SparseMatrix(1, 1);
    data.A_eta = gapese::SparseMatrix(1, 1);
    data.b = gapese::Vector::Constant(1, 1.0);
    data.C_x = sparse_from_triplets(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
    data.C_lambda = gapese::SparseMatrix(2, 1);
    data.C_eta = gapese::SparseMatrix(2, 1);
    data.d = gapese::Vector::Zero(2);

    gapese::InitialGuess initial;
    initial.x = (gapese::Vector(2) << 0.25, -0.5).finished();
    initial.lambda = gapese::Vector::Constant(1, 0.3);
    initial.eta = gapese::Vector::Constant(1, 0.7);

    gapese::Solver solver(problem, options);
    const gapese::SolveResult result = solver.solve(data, initial);

    std::cout << "status=" << result.info.status
              << " iter=" << result.info.iter_num
              << " cost=" << result.info.cost
              << " dgap=" << result.info.dgap_res
              << " kkt=" << result.info.kkt_res << '\n';

    if (result.info.status != 1) return 1;
    if (result.info.kkt_res > options.tol_kkt * 1.01) return 2;
    if (result.info.dgap_res > options.tol_dgap * 1.01) return 3;
    if (std::abs(result.solution.x.sum() - 1.0) > 1e-7) return 4;
    if ((result.solution.x.array() < -1e-7).any()) return 5;
    // Verify stateful repeated solve with primal-dual warm start.
    gapese::InitialGuess warm;
    warm.x = result.solution.x;
    warm.lambda = result.solution.lambda;
    warm.eta = result.solution.eta;
    warm.v_d = result.solution.v_d;
    warm.gamma_b = result.solution.gamma_b;
    warm.gamma_d = result.solution.gamma_d;
    const gapese::SolveResult warm_result = solver.solve(data, warm);
    if (warm_result.info.status != 1) return 6;
    if ((warm_result.solution.x - result.solution.x).lpNorm<Eigen::Infinity>() > 1e-7) return 7;

    return 0;
}
