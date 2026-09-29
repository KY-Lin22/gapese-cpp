#pragma once

#include "gapese/detail/dgap.hpp"
#include "gapese/detail/fb.hpp"
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
    KKTAssembler(ProblemSize problem, Options options, const DGapEvaluator& dgap, const FBEvaluator& fb);

    Vector residual(
        double mu,
        const Vector& x,
        const Vector& xi,
        const Vector& v_d,
        const Vector& gamma_b,
        const Vector& gamma_d,
        const CcqpData& data,
        const TransformedData& transformed) const;

    KKTWorkspace initialize_matrix(
        const CcqpData& data,
        const TransformedData& transformed) const;

    void update_matrix(
        KKTWorkspace& workspace,
        double mu,
        const Vector& xi,
        const Vector& v_d,
        const Vector& gamma_d) const;

    Vector violation(
        const Vector& x,
        const Vector& xi,
        const Vector& v_d,
        const Vector& gamma_d,
        const CcqpData& data,
        const TransformedData& transformed) const;

private:
    ProblemSize problem_;
    Options options_;
    const DGapEvaluator& dgap_;
    const FBEvaluator& fb_;
};

}  // namespace gapese::detail
