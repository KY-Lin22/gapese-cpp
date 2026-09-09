#pragma once

#include "gapese/detail/dgap.hpp"
#include "gapese/detail/fb.hpp"
#include "gapese/options.hpp"
#include "gapese/detail/types.hpp"

namespace gapese::detail {

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

    SparseMatrix matrix(
        double mu,
        const Vector& xi,
        const Vector& v_d,
        const Vector& gamma_d,
        const CcqpData& data,
        const TransformedData& transformed) const;

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
