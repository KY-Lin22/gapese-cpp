#pragma once

#include "gapese/detail/dgap.hpp"
#include "gapese/detail/fb.hpp"
#include "gapese/detail/kkt.hpp"
#include "gapese/detail/types.hpp"
#include "gapese/options.hpp"
#include "gapese/types.hpp"

namespace gapese {

class Solver {
public:
    Solver(ProblemSize problem, Options options = {});

    const ProblemSize& problem() const noexcept { return problem_; }
    const Options& options() const noexcept { return options_; }

    SolveResult solve(const CcqpData& data, const InitialGuess& initial);

private:
    ProblemSize problem_;
    Options options_;
    detail::DGapEvaluator dgap_;
    detail::FBEvaluator fb_;
    detail::KKTAssembler kkt_;

    void validate_data(const CcqpData& data) const;
    void validate_initial(const InitialGuess& initial) const;
    detail::TransformedData transform_data(const CcqpData& data) const;
    Vector interleave(const Vector& lambda, const Vector& eta) const;
    void deinterleave(const Vector& xi, Vector& lambda, Vector& eta) const;
};

}  // namespace gapese
