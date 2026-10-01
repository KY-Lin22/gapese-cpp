#pragma once

#include "gapese/options.hpp"
#include "gapese/detail/types.hpp"

#include <array>
#include <vector>

namespace gapese::detail {

struct KKTWorkspace {
    // AMD-permuted upper-triangular reduced KKT matrix in compressed CSC form.
    SparseMatrix matrix;

    // Symmetric permutation used as K_perm = P * K * P^T.  For an original
    // index i, old_to_new[i] gives its position in the permuted system.
    std::vector<int> old_to_new;

    // For each complementarity pair, indices into matrix.valuePtr() for the
    // upper-triangular (lambda,lambda), (lambda,eta), (eta,eta) entries.
    std::vector<std::array<int, 3>> dgap_indices;
    std::vector<std::array<double, 3>> dgap_base_values;

    // One reduced FB Schur-complement diagonal per inequality constraint.
    std::vector<int> fb_diag_indices;
};

class KKTAssembler {
public:
    KKTAssembler(ProblemSize problem, Options options);

    void residual(
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
        Vector& result) const;

    KKTWorkspace initialize_matrix(
        const CcqpData& data,
        const TransformedData& transformed) const;

    void update_matrix(
        KKTWorkspace& workspace,
        double mu,
        const std::vector<std::array<double, 3>>& dgap_blocks,
        const Vector& fb_d_v,
        const Vector& fb_d_gamma) const;

    void reduced_rhs(
        const Vector& full_residual,
        const Vector& fb_d_v,
        Vector& rhs) const;

private:
    ProblemSize problem_;
    Options options_;
};

}  // namespace gapese::detail
