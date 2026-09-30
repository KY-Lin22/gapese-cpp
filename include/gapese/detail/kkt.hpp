#pragma once

#include "gapese/options.hpp"
#include "gapese/detail/types.hpp"

#include <array>
#include <vector>

namespace gapese::detail {

struct KKTWorkspace {
    SparseMatrix matrix;

    // For each complementarity pair, indices into matrix.valuePtr() for
    // (lambda,lambda), (lambda,eta), (eta,lambda), (eta,eta).
    std::vector<std::array<int, 4>> dgap_indices;
    std::vector<std::array<double, 4>> dgap_base_values;

    // Indices into matrix.valuePtr() for the two FB diagonal blocks.
    std::vector<int> fb_v_indices;
    std::vector<int> fb_gamma_indices;
    std::vector<double> fb_v_base_values;
    std::vector<double> fb_gamma_base_values;
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

private:
    ProblemSize problem_;
    Options options_;
};

}  // namespace gapese::detail
