#include "gapese/detail/fb.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace gapese::detail {

Vector FBEvaluator::value(const Vector& v, const Vector& gamma) const {
    if (v.size() != gamma.size()) {
        throw std::invalid_argument("FB arguments must have equal length");
    }

    Vector psi(v.size());
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        psi[i] = std::sqrt(v[i] * v[i] + gamma[i] * gamma[i] + sigma_ * sigma_) - v[i] - gamma[i];
    }
    return psi;
}

std::pair<Vector, Vector> FBEvaluator::jacobian_diagonals(const Vector& v, const Vector& gamma) const {
    if (v.size() != gamma.size()) {
        throw std::invalid_argument("FB arguments must have equal length");
    }

    Vector d_v(v.size());
    Vector d_gamma(v.size());
    const double epsilon = std::numeric_limits<double>::epsilon();

    for (Eigen::Index i = 0; i < v.size(); ++i) {
        const double denominator = std::sqrt(v[i] * v[i] + gamma[i] * gamma[i] + sigma_ * sigma_);
        double scaled_v = 0.0;
        double scaled_gamma = 0.0;
        if (denominator > epsilon) {
            scaled_v = v[i] / denominator;
            scaled_gamma = gamma[i] / denominator;
        }
        d_v[i] = scaled_v - 1.0;
        d_gamma[i] = scaled_gamma - 1.0;
    }

    return {d_v, d_gamma};
}

void FBEvaluator::evaluate(
    const Vector& v,
    const Vector& gamma,
    Vector& psi,
    Vector& d_v,
    Vector& d_gamma) const {
    if (v.size() != gamma.size()) {
        throw std::invalid_argument("FB arguments must have equal length");
    }
    if (psi.size() != v.size()) psi.resize(v.size());
    if (d_v.size() != v.size()) d_v.resize(v.size());
    if (d_gamma.size() != v.size()) d_gamma.resize(v.size());

    const double sigma_sq = sigma_ * sigma_;
    const double epsilon = std::numeric_limits<double>::epsilon();
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        const double denominator = std::sqrt(v[i] * v[i] + gamma[i] * gamma[i] + sigma_sq);
        psi[i] = denominator - v[i] - gamma[i];

        double scaled_v = 0.0;
        double scaled_gamma = 0.0;
        if (denominator > epsilon) {
            scaled_v = v[i] / denominator;
            scaled_gamma = gamma[i] / denominator;
        }
        d_v[i] = scaled_v - 1.0;
        d_gamma[i] = scaled_gamma - 1.0;
    }
}

double FBEvaluator::l1_norm(const Vector& v, const Vector& gamma) const {
    if (v.size() != gamma.size()) {
        throw std::invalid_argument("FB arguments must have equal length");
    }

    const double sigma_sq = sigma_ * sigma_;
    double result = 0.0;
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        const double psi = std::sqrt(v[i] * v[i] + gamma[i] * gamma[i] + sigma_sq) - v[i] - gamma[i];
        result += std::abs(psi);
    }
    return result;
}

}  // namespace gapese::detail
