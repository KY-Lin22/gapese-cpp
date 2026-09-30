#include "gapese/detail/kkt.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace gapese::detail {
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

int value_index(const SparseMatrix& matrix, int row, int col) {
    const int begin = matrix.outerIndexPtr()[col];
    const int end = matrix.outerIndexPtr()[col + 1];
    const int* first = matrix.innerIndexPtr() + begin;
    const int* last = matrix.innerIndexPtr() + end;
    const int* position = std::lower_bound(first, last, row);
    if (position == last || *position != row) {
        throw std::logic_error("KKT structural entry was not reserved");
    }
    return begin + static_cast<int>(position - first);
}

}  // namespace

KKTAssembler::KKTAssembler(ProblemSize problem, Options options)
    : problem_(problem), options_(options) {}

void KKTAssembler::residual(
    double mu,
    const Vector& qx_x,
    const Vector& qxi_xi,
    const Vector& ax_x,
    const Vector& axi_xi,
    const Vector& cx_x,
    const Vector& cxi_xi,
    const Vector& dgap_gradient,
    const Vector& fb_value,
    const Vector& v_d,
    const Vector& gamma_b,
    const Vector& gamma_d,
    const CcqpData& data,
    const TransformedData& transformed,
    Vector& result) const {
    const int nx = problem_.n_x;
    const int nxi = transformed_dim(problem_);
    const int nb = problem_.n_b;
    const int nd = problem_.n_d;

    if (result.size() != kkt_dim(problem_)) {
        result.resize(kkt_dim(problem_));
    }

    int offset = 0;
    auto stationarity_x = result.segment(offset, nx);
    stationarity_x = qx_x + data.r_x;
    stationarity_x.noalias() += data.A_x.transpose() * gamma_b;
    stationarity_x.noalias() -= data.C_x.transpose() * gamma_d;
    offset += nx;

    auto stationarity_xi = result.segment(offset, nxi);
    stationarity_xi = qxi_xi + transformed.r_xi + mu * dgap_gradient;
    stationarity_xi.noalias() += transformed.A_xi.transpose() * gamma_b;
    stationarity_xi.noalias() -= transformed.C_xi.transpose() * gamma_d;
    offset += nxi;

    result.segment(offset, nb) = ax_x + axi_xi - data.b;
    offset += nb;

    result.segment(offset, nd) = cx_x + cxi_xi - data.d - v_d;
    offset += nd;

    result.segment(offset, nd) = fb_value;
}

KKTWorkspace KKTAssembler::initialize_matrix(
    const CcqpData& data,
    const TransformedData& transformed) const {
    const int nx = problem_.n_x;
    const int nxi = transformed_dim(problem_);
    const int nb = problem_.n_b;
    const int nd = problem_.n_d;
    const int nl = problem_.n_lambda;
    const int dim = kkt_dim(problem_);

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

    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(
        data.Q_x.nonZeros() + transformed.Q_xi.nonZeros() +
        2 * data.A_x.nonZeros() + 2 * transformed.A_xi.nonZeros() +
        2 * data.C_x.nonZeros() + 2 * transformed.C_xi.nonZeros() +
        nx + nxi + nb + 3 * nd));

    // Constant KKT blocks. These values are written once for the whole solve.
    add_sparse_block(triplets, data.Q_x, row_x, col_x);
    add_constant_diagonal(triplets, row_x, col_x, nx, options_.nu_x);
    add_transpose_block(triplets, data.A_x, row_x, col_gb);
    add_transpose_block(triplets, data.C_x, row_x, col_gd, -1.0);

    add_sparse_block(triplets, transformed.Q_xi, row_xi, col_xi);
    add_constant_diagonal(triplets, row_xi, col_xi, nxi, options_.nu_xi);
    add_transpose_block(triplets, transformed.A_xi, row_xi, col_gb);
    add_transpose_block(triplets, transformed.C_xi, row_xi, col_gd, -1.0);

    add_sparse_block(triplets, data.A_x, row_eq, col_x);
    add_sparse_block(triplets, transformed.A_xi, row_eq, col_xi);
    add_constant_diagonal(triplets, row_eq, col_gb, nb, -options_.nu_b);

    add_sparse_block(triplets, data.C_x, row_ineq, col_x);
    add_sparse_block(triplets, transformed.C_xi, row_ineq, col_xi);
    add_constant_diagonal(triplets, row_ineq, col_vd, nd, -1.0);

    // -nu_d is the constant part of the two FB diagonal blocks.
    add_constant_diagonal(triplets, row_fb, col_vd, nd, -options_.nu_d);
    add_constant_diagonal(triplets, row_fb, col_gd, nd, -options_.nu_d);

    KKTWorkspace workspace;
    workspace.matrix.resize(dim, dim);
    workspace.matrix.setFromTriplets(triplets.begin(), triplets.end());

    // Reserve the full structural pattern of the nonlinear blocks. Explicit
    // zeros are intentional: the numerical values may switch between zero and
    // nonzero as the D-gap region or FB generalized derivative changes.
    for (int i = 0; i < nl; ++i) {
        const int l = 2 * i;
        const int e = 2 * i + 1;
        workspace.matrix.coeffRef(row_xi + l, col_xi + l) += 0.0;
        workspace.matrix.coeffRef(row_xi + l, col_xi + e) += 0.0;
        workspace.matrix.coeffRef(row_xi + e, col_xi + l) += 0.0;
        workspace.matrix.coeffRef(row_xi + e, col_xi + e) += 0.0;
    }
    for (int i = 0; i < nd; ++i) {
        workspace.matrix.coeffRef(row_fb + i, col_vd + i) += 0.0;
        workspace.matrix.coeffRef(row_fb + i, col_gd + i) += 0.0;
    }
    workspace.matrix.makeCompressed();

    workspace.dgap_indices.resize(static_cast<std::size_t>(nl));
    workspace.dgap_base_values.resize(static_cast<std::size_t>(nl));
    const double* values = workspace.matrix.valuePtr();
    for (int i = 0; i < nl; ++i) {
        const int l = 2 * i;
        const int e = 2 * i + 1;
        auto& indices = workspace.dgap_indices[static_cast<std::size_t>(i)];
        indices = {
            value_index(workspace.matrix, row_xi + l, col_xi + l),
            value_index(workspace.matrix, row_xi + l, col_xi + e),
            value_index(workspace.matrix, row_xi + e, col_xi + l),
            value_index(workspace.matrix, row_xi + e, col_xi + e)};

        auto& base = workspace.dgap_base_values[static_cast<std::size_t>(i)];
        base = {values[indices[0]], values[indices[1]], values[indices[2]], values[indices[3]]};
    }

    workspace.fb_v_indices.resize(static_cast<std::size_t>(nd));
    workspace.fb_gamma_indices.resize(static_cast<std::size_t>(nd));
    workspace.fb_v_base_values.resize(static_cast<std::size_t>(nd));
    workspace.fb_gamma_base_values.resize(static_cast<std::size_t>(nd));
    for (int i = 0; i < nd; ++i) {
        const int v_index = value_index(workspace.matrix, row_fb + i, col_vd + i);
        const int gamma_index = value_index(workspace.matrix, row_fb + i, col_gd + i);
        workspace.fb_v_indices[static_cast<std::size_t>(i)] = v_index;
        workspace.fb_gamma_indices[static_cast<std::size_t>(i)] = gamma_index;
        workspace.fb_v_base_values[static_cast<std::size_t>(i)] = values[v_index];
        workspace.fb_gamma_base_values[static_cast<std::size_t>(i)] = values[gamma_index];
    }

    return workspace;
}

void KKTAssembler::update_matrix(
    KKTWorkspace& workspace,
    double mu,
    const std::vector<std::array<double, 3>>& dgap_blocks,
    const Vector& fb_d_v,
    const Vector& fb_d_gamma) const {
    const int nl = problem_.n_lambda;
    const int nd = problem_.n_d;
    if (dgap_blocks.size() != static_cast<std::size_t>(nl)) {
        throw std::invalid_argument("D-gap block count is incorrect for KKT update");
    }
    if (fb_d_v.size() != nd || fb_d_gamma.size() != nd) {
        throw std::invalid_argument("FB derivative vectors have incorrect length for KKT update");
    }

    double* values = workspace.matrix.valuePtr();
    for (int i = 0; i < nl; ++i) {
        const auto& block = dgap_blocks[static_cast<std::size_t>(i)];
        const auto& indices = workspace.dgap_indices[static_cast<std::size_t>(i)];
        const auto& base = workspace.dgap_base_values[static_cast<std::size_t>(i)];
        values[indices[0]] = base[0] + mu * block[0];
        values[indices[1]] = base[1] + mu * block[1];
        values[indices[2]] = base[2] + mu * block[1];
        values[indices[3]] = base[3] + mu * block[2];
    }

    for (int i = 0; i < nd; ++i) {
        const std::size_t index = static_cast<std::size_t>(i);
        values[workspace.fb_v_indices[index]] = workspace.fb_v_base_values[index] + fb_d_v[i];
        values[workspace.fb_gamma_indices[index]] = workspace.fb_gamma_base_values[index] + fb_d_gamma[i];
    }
}

}  // namespace gapese::detail
