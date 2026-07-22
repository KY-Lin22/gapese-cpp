#pragma once

#include "gapese/types.hpp"

namespace gapese {

class DGapEvaluator {
public:
    DGapEvaluator(double a, double b);

    double value(const Vector& xi) const;
    Vector gradient(const Vector& xi) const;
    SparseMatrix modified_hessian(const Vector& xi) const;

    double a() const noexcept { return a_; }
    double b() const noexcept { return b_; }
    double kappa_min() const noexcept { return kappa_min_; }

private:
    double a_;
    double b_;
    double kappa_min_;
};

}  // namespace gapese
