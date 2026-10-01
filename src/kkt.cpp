#include "gapese/detail/kkt.hpp"

#include <Eigen/OrderingMethods>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gapese::detail {
namespace {

void add_upper_sparse_block(
    std::vector<Triplet>& triplets,
    const SparseMatrix& matrix,
    int row_offset,
    int col_offset,
    double scale = 1.0) {
    for (int col = 0; col < matrix.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(matrix, col); it; ++it) {
            const int row = row_offset + it.row();
            const int out_col = col_offset + it.col();
            const double value = scale * it.value();
            if (row <= out_col && value != 0.0) {
                triplets.emplace_back(row, out_col, value);
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
    int offset,
    int size,
    double value) {
    if (value == 0.0) return;
    for (int i = 0; i < size; ++i) {
        triplets.emplace_back(offset + i, offset + i, value);
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

std::pair<int, int> permuted_upper_coordinate(
    const std::vector<int>& old_to_new,
    int row,
    int col) {
    int new_row = old_to_new[static_cast<std::size_t>(row)];
    int new_col = old_to_new[static_cast<std::size_t>(col)];
    if (new_row > new_col) std::swap(new_row, new_col);
    return {new_row, new_col};
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
    const int dim = reduced_kkt_dim(problem_);

    const int off_x = 0;
    const int off_xi = off_x + nx;
    const int off_gb = off_xi + nxi;
    const int off_gd = off_gb + nb;

    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(
        data.Q_x.nonZeros() + transformed.Q_xi.nonZeros() +
        data.A_x.nonZeros() + transformed.A_xi.nonZeros() +
        data.C_x.nonZeros() + transformed.C_xi.nonZeros() +
        nx + nxi + nb + nd + 3 * nl));

    // Store only the upper triangle of the symmetric reduced KKT matrix.
    add_upper_sparse_block(triplets, data.Q_x, off_x, off_x);
    add_constant_diagonal(triplets, off_x, nx, options_.nu_x);
    add_transpose_block(triplets, data.A_x, off_x, off_gb);
    add_transpose_block(triplets, data.C_x, off_x, off_gd, -1.0);

    add_upper_sparse_block(triplets, transformed.Q_xi, off_xi, off_xi);
    add_constant_diagonal(triplets, off_xi, nxi, options_.nu_xi);
    add_transpose_block(triplets, transformed.A_xi, off_xi, off_gb);
    add_transpose_block(triplets, transformed.C_xi, off_xi, off_gd, -1.0);

    add_constant_diagonal(triplets, off_gb, nb, -options_.nu_b);

    SparseMatrix unpermuted(dim, dim);
    unpermuted.setFromTriplets(triplets.begin(), triplets.end());

    // Reserve all diagonals and the complete nonlinear structural pattern.
    for (int i = 0; i < dim; ++i) {
        unpermuted.coeffRef(i, i) += 0.0;
    }
    for (int i = 0; i < nl; ++i) {
        const int l = off_xi + 2 * i;
        const int e = l + 1;
        unpermuted.coeffRef(l, l) += 0.0;
        unpermuted.coeffRef(l, e) += 0.0;
        unpermuted.coeffRef(e, e) += 0.0;
    }
    for (int i = 0; i < nd; ++i) {
        unpermuted.coeffRef(off_gd + i, off_gd + i) += 0.0;
    }
    unpermuted.makeCompressed();

    // AMD sees the complete symmetric structural pattern. Values are all one
    // so explicit structural zeros from the nonlinear blocks are preserved.
    std::vector<Triplet> pattern_triplets;
    pattern_triplets.reserve(static_cast<std::size_t>(
        2 * unpermuted.nonZeros() - dim));
    for (int col = 0; col < unpermuted.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(unpermuted, col); it; ++it) {
            pattern_triplets.emplace_back(it.row(), it.col(), 1.0);
            if (it.row() != it.col()) {
                pattern_triplets.emplace_back(it.col(), it.row(), 1.0);
            }
        }
    }
    SparseMatrix symmetric_pattern(dim, dim);
    symmetric_pattern.setFromTriplets(pattern_triplets.begin(), pattern_triplets.end());
    symmetric_pattern.makeCompressed();

    // Eigen ordering functors return the inverse permutation (see
    // SimplicialCholeskyBase::ordering).  QDLDL factors P K P^T, so convert
    // AMD's output to the forward old-to-new permutation used below.
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> inverse_permutation(dim);
    Eigen::AMDOrdering<int> ordering;
    ordering(symmetric_pattern, inverse_permutation);
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> permutation =
        inverse_permutation.inverse();

    KKTWorkspace workspace;
    workspace.old_to_new.resize(static_cast<std::size_t>(dim));
    for (int old_index = 0; old_index < dim; ++old_index) {
        workspace.old_to_new[static_cast<std::size_t>(old_index)] = permutation.indices()[old_index];
    }

    std::vector<Triplet> permuted_triplets;
    permuted_triplets.reserve(static_cast<std::size_t>(unpermuted.nonZeros()));
    for (int col = 0; col < unpermuted.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(unpermuted, col); it; ++it) {
            auto coordinate = permuted_upper_coordinate(workspace.old_to_new, it.row(), it.col());
            permuted_triplets.emplace_back(coordinate.first, coordinate.second, it.value());
        }
    }
    workspace.matrix.resize(dim, dim);
    workspace.matrix.setFromTriplets(permuted_triplets.begin(), permuted_triplets.end());
    workspace.matrix.makeCompressed();

    workspace.dgap_indices.resize(static_cast<std::size_t>(nl));
    workspace.dgap_base_values.resize(static_cast<std::size_t>(nl));
    const double* values = workspace.matrix.valuePtr();
    for (int i = 0; i < nl; ++i) {
        const int l = off_xi + 2 * i;
        const int e = l + 1;
        const auto ll = permuted_upper_coordinate(workspace.old_to_new, l, l);
        const auto le = permuted_upper_coordinate(workspace.old_to_new, l, e);
        const auto ee = permuted_upper_coordinate(workspace.old_to_new, e, e);

        auto& indices = workspace.dgap_indices[static_cast<std::size_t>(i)];
        indices = {
            value_index(workspace.matrix, ll.first, ll.second),
            value_index(workspace.matrix, le.first, le.second),
            value_index(workspace.matrix, ee.first, ee.second)};

        auto& base = workspace.dgap_base_values[static_cast<std::size_t>(i)];
        base = {values[indices[0]], values[indices[1]], values[indices[2]]};
    }

    workspace.fb_diag_indices.resize(static_cast<std::size_t>(nd));
    for (int i = 0; i < nd; ++i) {
        const auto diagonal = permuted_upper_coordinate(
            workspace.old_to_new, off_gd + i, off_gd + i);
        workspace.fb_diag_indices[static_cast<std::size_t>(i)] =
            value_index(workspace.matrix, diagonal.first, diagonal.second);
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
        values[indices[2]] = base[2] + mu * block[2];
    }

    for (int i = 0; i < nd; ++i) {
        const double d_v = fb_d_v[i] - options_.nu_d;
        const double d_gamma = fb_d_gamma[i] - options_.nu_d;
        values[workspace.fb_diag_indices[static_cast<std::size_t>(i)]] = -d_gamma / d_v;
    }
}

void KKTAssembler::reduced_rhs(
    const Vector& full_residual,
    const Vector& fb_d_v,
    Vector& rhs) const {
    const int nx = problem_.n_x;
    const int nxi = transformed_dim(problem_);
    const int nb = problem_.n_b;
    const int nd = problem_.n_d;
    const int row_eq = nx + nxi;
    const int row_ineq = row_eq + nb;
    const int row_fb = row_ineq + nd;

    if (full_residual.size() != kkt_dim(problem_)) {
        throw std::invalid_argument("Full KKT residual has incorrect length");
    }
    if (fb_d_v.size() != nd) {
        throw std::invalid_argument("FB derivative vector has incorrect length");
    }
    if (rhs.size() != reduced_kkt_dim(problem_)) {
        rhs.resize(reduced_kkt_dim(problem_));
    }

    rhs.segment(0, nx + nxi) = -full_residual.segment(0, nx + nxi);
    rhs.segment(nx + nxi, nb) = -full_residual.segment(row_eq, nb);
    for (int i = 0; i < nd; ++i) {
        const double d_v = fb_d_v[i] - options_.nu_d;
        rhs[nx + nxi + nb + i] =
            full_residual[row_ineq + i] + full_residual[row_fb + i] / d_v;
    }
}

}  // namespace gapese::detail
