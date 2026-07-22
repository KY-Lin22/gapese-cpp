#pragma once

#include "gapese/types.hpp"

#include <utility>

namespace gapese {

class FBEvaluator {
public:
    explicit FBEvaluator(double sigma) : sigma_(sigma) {}

    Vector value(const Vector& v, const Vector& gamma) const;
    std::pair<Vector, Vector> jacobian_diagonals(const Vector& v, const Vector& gamma) const;

    double sigma() const noexcept { return sigma_; }

private:
    double sigma_;
};

}  // namespace gapese
