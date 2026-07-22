#include "gapese/kkt.hpp"

#include <stdexcept>
#include <vector>

namespace gapese {
namespace {

void add_sparse_block(
    std::vector<Triplet>& triplets,
    const SparseMatrix& matrix,
    int row_offset,
    int col_offset,
    double scale = 1.0) {
    for (int col = 0; col < matrix.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(matrix, col); it; ++it) {
            const double value = scale * it.value();
            if (value != 0.0) {
                triplets.emplace_back(row_offset + it.row(), col_offset + it.col(), value);
            }
        }
    }
}

void add_transpose_block(
    std::vector<Triplet>& triplets,
    const SparseMatrix& matrix,
    int row_offset,
    int col_offset,
    double scale = 1.0) {
    for (int col = 0; col < matrix.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(matrix, col); it; ++it) {
            const double value = scale * it.value();
            if (value != 0.0) {
                triplets.emplace_back(row_offset + it.col(), col_offset + it.row(), value);
            }
        }
    }
}

void add_constant_diagonal(
    std::vector<Triplet>& triplets,
    int row_offset,
    int col_offset,
    int size,
    double value) {
    if (value == 0.0) return;
    for (int i = 0; i < size; ++i) {
        triplets.emplace_back(row_offset + i, col_offset + i, value);
    }
}

void add_vector_diagonal(
    std::vector<Triplet>& triplets,
    int row_offset,
    int col_offset,
    const Vector& values,
    double shift = 0.0) {
    for (int i = 0; i < values.size(); ++i) {
        const double value = values[i] + shift;
        if (value != 0.0) {
            triplets.emplace_back(row_offset + i, col_offset + i, value);
        }
    }
}

}  // namespace

KKTAssembler::KKTAssembler(
    ProblemSize problem,
    Options options,
    const DGapEvaluator& dgap,
    const FBEvaluator& fb)
    : problem_(problem), options_(options), dgap_(dgap), fb_(fb) {}

Vector KKTAssembler::residual(
    double mu,
    const Vector& x,
    const Vector& xi,
    const Vector& v_d,
    const Vector& gamma_b,
    const Vector& gamma_d,
    const CcqpData& data,
    const TransformedData& transformed) const {
    const int nx = problem_.n_x;
    const int nxi = problem_.n_xi();
    const int nb = problem_.n_b;
    const int nd = problem_.n_d;

    Vector result(problem_.kkt_dim());
    int offset = 0;

    result.segment(offset, nx) =
        data.Q_x * x + data.r_x + data.A_x.transpose() * gamma_b - data.C_x.transpose() * gamma_d;
    offset += nx;

    result.segment(offset, nxi) =
        transformed.Q_xi * xi + transformed.r_xi + mu * dgap_.gradient(xi) +
        transformed.A_xi.transpose() * gamma_b - transformed.C_xi.transpose() * gamma_d;
    offset += nxi;

    result.segment(offset, nb) = data.A_x * x + transformed.A_xi * xi - data.b;
    offset += nb;

    result.segment(offset, nd) = data.C_x * x + transformed.C_xi * xi - data.d - v_d;
    offset += nd;

    result.segment(offset, nd) = fb_.value(v_d, gamma_d);
    return result;
}

SparseMatrix KKTAssembler::matrix(
    double mu,
    const Vector& xi,
    const Vector& v_d,
    const Vector& gamma_d,
    const CcqpData& data,
    const TransformedData& transformed) const {
    const int nx = problem_.n_x;
    const int nxi = problem_.n_xi();
    const int nb = problem_.n_b;
    const int nd = problem_.n_d;
    const int dim = problem_.kkt_dim();

    const int col_x = 0;
    const int col_xi = col_x + nx;
    const int col_vd = col_xi + nxi;
    const int col_gb = col_vd + nd;
    const int col_gd = col_gb + nb;

    const int row_x = 0;
    const int row_xi = row_x + nx;
    const int row_eq = row_xi + nxi;
    const int row_ineq = row_eq + nb;
    const int row_fb = row_ineq + nd;

    const SparseMatrix B = dgap_.modified_hessian(xi);
    const auto [d_v, d_gamma] = fb_.jacobian_diagonals(v_d, gamma_d);

    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(
        data.Q_x.nonZeros() + transformed.Q_xi.nonZeros() + B.nonZeros() +
        2 * data.A_x.nonZeros() + 2 * transformed.A_xi.nonZeros() +
        2 * data.C_x.nonZeros() + 2 * transformed.C_xi.nonZeros() +
        nx + nxi + nb + 3 * nd));

    add_sparse_block(triplets, data.Q_x, row_x, col_x);
    add_constant_diagonal(triplets, row_x, col_x, nx, options_.nu_x);
    add_transpose_block(triplets, data.A_x, row_x, col_gb);
    add_transpose_block(triplets, data.C_x, row_x, col_gd, -1.0);

    add_sparse_block(triplets, transformed.Q_xi, row_xi, col_xi);
    add_sparse_block(triplets, B, row_xi, col_xi, mu);
    add_constant_diagonal(triplets, row_xi, col_xi, nxi, options_.nu_xi);
    add_transpose_block(triplets, transformed.A_xi, row_xi, col_gb);
    add_transpose_block(triplets, transformed.C_xi, row_xi, col_gd, -1.0);

    add_sparse_block(triplets, data.A_x, row_eq, col_x);
    add_sparse_block(triplets, transformed.A_xi, row_eq, col_xi);
    add_constant_diagonal(triplets, row_eq, col_gb, nb, -options_.nu_b);

    add_sparse_block(triplets, data.C_x, row_ineq, col_x);
    add_sparse_block(triplets, transformed.C_xi, row_ineq, col_xi);
    add_constant_diagonal(triplets, row_ineq, col_vd, nd, -1.0);

    add_vector_diagonal(triplets, row_fb, col_vd, d_v, -options_.nu_d);
    add_vector_diagonal(triplets, row_fb, col_gd, d_gamma, -options_.nu_d);

    SparseMatrix kkt(dim, dim);
    kkt.setFromTriplets(triplets.begin(), triplets.end());
    kkt.makeCompressed();
    return kkt;
}

Vector KKTAssembler::violation(
    const Vector& x,
    const Vector& xi,
    const Vector& v_d,
    const Vector& gamma_d,
    const CcqpData& data,
    const TransformedData& transformed) const {
    const int nb = problem_.n_b;
    const int nd = problem_.n_d;
    Vector result(nb + 2 * nd);
    int offset = 0;
    result.segment(offset, nb) = data.A_x * x + transformed.A_xi * xi - data.b;
    offset += nb;
    result.segment(offset, nd) = data.C_x * x + transformed.C_xi * xi - data.d - v_d;
    offset += nd;
    result.segment(offset, nd) = fb_.value(v_d, gamma_d);
    return result;
}

}  // namespace gapese
