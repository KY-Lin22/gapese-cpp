#include "gapese/detail/dgap.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace gapese::detail {

DGapEvaluator::DGapEvaluator(double a, double b)
    : a_(a),
      b_(b),
      kappa_min_(-0.5 * (a + 1.0 / b) - 0.5 * std::sqrt(std::pow(1.0 / b - a, 2.0) + 4.0)) {
    if (!(b_ > a_ && a_ > 0.0)) {
        throw std::invalid_argument("D-gap parameters must satisfy b > a > 0");
    }
}

double DGapEvaluator::value(const Vector& xi) const {
    if (xi.size() % 2 != 0) {
        throw std::invalid_argument("xi must have even length");
    }

    double eta_sq = 0.0;
    double ya_sq = 0.0;
    double yb_sq = 0.0;
    for (Eigen::Index i = 0; i < xi.size() / 2; ++i) {
        const double lambda = xi[2 * i];
        const double eta = xi[2 * i + 1];
        const double ya = std::max(0.0, eta - a_ * lambda);
        const double yb = std::max(0.0, eta - b_ * lambda);
        eta_sq += eta * eta;
        ya_sq += ya * ya;
        yb_sq += yb * yb;
    }

    return (b_ - a_) / (2.0 * a_ * b_) * eta_sq - ya_sq / (2.0 * a_) + yb_sq / (2.0 * b_);
}

Vector DGapEvaluator::gradient(const Vector& xi) const {
    if (xi.size() % 2 != 0) {
        throw std::invalid_argument("xi must have even length");
    }

    Vector gradient = Vector::Zero(xi.size());
    for (Eigen::Index i = 0; i < xi.size() / 2; ++i) {
        const double lambda = xi[2 * i];
        const double eta = xi[2 * i + 1];
        const double ya = std::max(0.0, eta - a_ * lambda);
        const double yb = std::max(0.0, eta - b_ * lambda);

        gradient[2 * i] = ya - yb;
        gradient[2 * i + 1] = (b_ - a_) / (a_ * b_) * eta - ya / a_ + yb / b_;
    }
    return gradient;
}

std::array<double, 3> DGapEvaluator::modified_hessian_block(double lambda, double eta) const {
    const bool region_1 = (eta >= b_ * lambda) && (eta >= a_ * lambda);
    const bool region_2 = (b_ * lambda > eta) && (eta > a_ * lambda);
    const bool region_3 = (eta <= b_ * lambda) && (eta <= a_ * lambda);
    const double regularization = -kappa_min_;

    if (region_1) {
        return {b_ - a_, 0.0, 0.0};
    }
    if (region_2) {
        return {-a_ + regularization, 1.0, -1.0 / b_ + regularization};
    }
    if (region_3) {
        return {0.0, 0.0, (b_ - a_) / (a_ * b_)};
    }
    return {b_, -1.0, 1.0 / a_};
}

void DGapEvaluator::evaluate(
    const Vector& xi,
    double& value,
    Vector& gradient,
    std::vector<std::array<double, 3>>& modified_hessian_blocks) const {
    if (xi.size() % 2 != 0) {
        throw std::invalid_argument("xi must have even length");
    }

    const Eigen::Index pair_count = xi.size() / 2;
    if (gradient.size() != xi.size()) {
        gradient.resize(xi.size());
    }
    if (modified_hessian_blocks.size() != static_cast<std::size_t>(pair_count)) {
        modified_hessian_blocks.resize(static_cast<std::size_t>(pair_count));
    }

    double eta_sq = 0.0;
    double ya_sq = 0.0;
    double yb_sq = 0.0;
    for (Eigen::Index i = 0; i < pair_count; ++i) {
        const double lambda = xi[2 * i];
        const double eta = xi[2 * i + 1];
        const double eta_minus_a_lambda = eta - a_ * lambda;
        const double eta_minus_b_lambda = eta - b_ * lambda;
        const double ya = std::max(0.0, eta_minus_a_lambda);
        const double yb = std::max(0.0, eta_minus_b_lambda);

        eta_sq += eta * eta;
        ya_sq += ya * ya;
        yb_sq += yb * yb;

        gradient[2 * i] = ya - yb;
        gradient[2 * i + 1] = (b_ - a_) / (a_ * b_) * eta - ya / a_ + yb / b_;

        auto& block = modified_hessian_blocks[static_cast<std::size_t>(i)];
        if (eta_minus_b_lambda >= 0.0 && eta_minus_a_lambda >= 0.0) {
            block = {b_ - a_, 0.0, 0.0};
        } else if (eta_minus_b_lambda < 0.0 && eta_minus_a_lambda > 0.0) {
            const double regularization = -kappa_min_;
            block = {-a_ + regularization, 1.0, -1.0 / b_ + regularization};
        } else if (eta_minus_b_lambda <= 0.0 && eta_minus_a_lambda <= 0.0) {
            block = {0.0, 0.0, (b_ - a_) / (a_ * b_)};
        } else {
            block = {b_, -1.0, 1.0 / a_};
        }
    }

    value = (b_ - a_) / (2.0 * a_ * b_) * eta_sq - ya_sq / (2.0 * a_) + yb_sq / (2.0 * b_);
}

SparseMatrix DGapEvaluator::modified_hessian(const Vector& xi) const {
    if (xi.size() % 2 != 0) {
        throw std::invalid_argument("xi must have even length");
    }

    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(4 * xi.size() / 2));

    for (int i = 0; i < xi.size() / 2; ++i) {
        const auto block = modified_hessian_block(xi[2 * i], xi[2 * i + 1]);
        const double h11 = block[0];
        const double h12 = block[1];
        const double h22 = block[2];

        const int l = 2 * i;
        const int e = 2 * i + 1;
        if (h11 != 0.0) triplets.emplace_back(l, l, h11);
        if (h22 != 0.0) triplets.emplace_back(e, e, h22);
        if (h12 != 0.0) {
            triplets.emplace_back(l, e, h12);
            triplets.emplace_back(e, l, h12);
        }
    }

    SparseMatrix hessian(xi.size(), xi.size());
    hessian.setFromTriplets(triplets.begin(), triplets.end());
    hessian.makeCompressed();
    return hessian;
}

}  // namespace gapese::detail
