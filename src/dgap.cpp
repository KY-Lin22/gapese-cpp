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

SparseMatrix DGapEvaluator::modified_hessian(const Vector& xi) const {
    if (xi.size() % 2 != 0) {
        throw std::invalid_argument("xi must have even length");
    }

    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(4 * xi.size() / 2));
    const double regularization = -kappa_min_;

    for (int i = 0; i < xi.size() / 2; ++i) {
        const double lambda = xi[2 * i];
        const double eta = xi[2 * i + 1];

        const bool region_1 = (eta >= b_ * lambda) && (eta >= a_ * lambda);
        const bool region_2 = (b_ * lambda > eta) && (eta > a_ * lambda);
        const bool region_3 = (eta <= b_ * lambda) && (eta <= a_ * lambda);

        double h11 = 0.0;
        double h12 = 0.0;
        double h22 = 0.0;

        if (region_1) {
            h11 = b_ - a_;
        } else if (region_2) {
            h11 = -a_ + regularization;
            h12 = 1.0;
            h22 = -1.0 / b_ + regularization;
        } else if (region_3) {
            h22 = (b_ - a_) / (a_ * b_);
        } else {
            h11 = b_;
            h12 = -1.0;
            h22 = 1.0 / a_;
        }

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
