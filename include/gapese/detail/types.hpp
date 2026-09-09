#pragma once

#include "gapese/types.hpp"

namespace gapese::detail {

struct TransformedData {
    SparseMatrix Q_xi;
    Vector r_xi;
    SparseMatrix A_xi;
    SparseMatrix C_xi;
};

inline int transformed_dim(const ProblemSize& problem) noexcept {
    return 2 * problem.n_lambda;
}

inline int kkt_dim(const ProblemSize& problem) noexcept {
    return problem.n_x + transformed_dim(problem) + problem.n_b + 2 * problem.n_d;
}

}  // namespace gapese::detail
