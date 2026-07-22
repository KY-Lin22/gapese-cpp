#pragma once

#include "gapese/dgap.hpp"
#include "gapese/fb.hpp"
#include "gapese/kkt.hpp"
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
    DGapEvaluator dgap_;
    FBEvaluator fb_;
    KKTAssembler kkt_;

    void validate_data(const CcqpData& data) const;
    void validate_initial(const InitialGuess& initial) const;
    TransformedData transform_data(const CcqpData& data) const;
    Vector interleave(const Vector& lambda, const Vector& eta) const;
    void deinterleave(const Vector& xi, Vector& lambda, Vector& eta) const;
};

}  // namespace gapese
