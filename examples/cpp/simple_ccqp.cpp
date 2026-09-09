#include "gapese/solver.hpp"

#include <initializer_list>
#include <iostream>

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
    // Problem dimensions: x in R^2, one complementarity pair,
    // one equality constraint, and two inequality constraints.
    gapese::ProblemSize problem{2, 1, 1, 2};

    gapese::Options options;
    options.print_level = 0;
    options.dgap_a = 0.5;
    options.dgap_b = 2.0;
    options.tol_kkt = 1e-5;
    options.tol_dgap = 1e-5;
    options.mu_min = 10.0;
    options.mu_max = 1e6;

    gapese::CcqpData ccqp;
    ccqp.Q_x = sparse_from_triplets(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
    ccqp.Q_lambda = sparse_from_triplets(1, 1, {{0, 0, 1.0}});
    ccqp.Q_eta = sparse_from_triplets(1, 1, {{0, 0, 1.0}});
    ccqp.r_x = (gapese::Vector(2) << -1.0, -1.0).finished();
    ccqp.r_lambda = gapese::Vector::Constant(1, -1.0);
    ccqp.r_eta = gapese::Vector::Constant(1, -1.0);

    ccqp.A_x = sparse_from_triplets(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}});
    ccqp.A_lambda = gapese::SparseMatrix(1, 1);
    ccqp.A_eta = gapese::SparseMatrix(1, 1);
    ccqp.b = gapese::Vector::Constant(1, 1.0);

    ccqp.C_x = sparse_from_triplets(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
    ccqp.C_lambda = gapese::SparseMatrix(2, 1);
    ccqp.C_eta = gapese::SparseMatrix(2, 1);
    ccqp.d = gapese::Vector::Zero(2);

    gapese::InitialGuess initial;
    initial.x = (gapese::Vector(2) << 0.25, -0.5).finished();
    initial.lambda = gapese::Vector::Constant(1, 0.3);
    initial.eta = gapese::Vector::Constant(1, 0.7);

    gapese::Solver solver(problem, options);
    const gapese::SolveResult result = solver.solve(ccqp, initial);

    std::cout << "status: " << result.info.status << '\n';
    std::cout << "message: " << result.info.message << '\n';
    std::cout << "x: " << result.solution.x.transpose() << '\n';
    std::cout << "lambda: " << result.solution.lambda.transpose() << '\n';
    std::cout << "eta: " << result.solution.eta.transpose() << '\n';
    std::cout << "KKT residual: " << result.info.kkt_res << '\n';
    std::cout << "D-gap residual: " << result.info.dgap_res << '\n';

    return result.info.status == 1 ? 0 : 1;
}
