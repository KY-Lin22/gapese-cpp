#pragma once

#include "gapese/types.hpp"

#include <utility>

namespace gapese::detail {

class FBEvaluator {
public:
    explicit FBEvaluator(double sigma) : sigma_(sigma) {}

    Vector value(const Vector& v, const Vector& gamma) const;
    std::pair<Vector, Vector> jacobian_diagonals(const Vector& v, const Vector& gamma) const;

    // Evaluate the FB residual and both generalized-Jacobian diagonals using
    // one square root per component. Output vectors are reusable workspaces.
    void evaluate(
        const Vector& v,
        const Vector& gamma,
        Vector& psi,
        Vector& d_v,
        Vector& d_gamma) const;

    // L1 norm of the FB residual without materializing a temporary vector.
    double l1_norm(const Vector& v, const Vector& gamma) const;

    double sigma() const noexcept { return sigma_; }

private:
    double sigma_;
};

}  // namespace gapese::detail
