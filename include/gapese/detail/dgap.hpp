#pragma once

#include "gapese/types.hpp"

#include <array>
#include <vector>

namespace gapese::detail {

class DGapEvaluator {
public:
    DGapEvaluator(double a, double b);

    double value(const Vector& xi) const;
    Vector gradient(const Vector& xi) const;
    SparseMatrix modified_hessian(const Vector& xi) const;
    std::array<double, 3> modified_hessian_block(double lambda, double eta) const;

    // Evaluate all D-gap quantities needed by one Newton iteration in a
    // single pass over xi. Output storage is resized only when necessary and
    // can therefore be reused across iterations.
    void evaluate(
        const Vector& xi,
        double& value,
        Vector& gradient,
        std::vector<std::array<double, 3>>& modified_hessian_blocks) const;

    double a() const noexcept { return a_; }
    double b() const noexcept { return b_; }
    double kappa_min() const noexcept { return kappa_min_; }

private:
    double a_;
    double b_;
    double kappa_min_;
};

}  // namespace gapese::detail
