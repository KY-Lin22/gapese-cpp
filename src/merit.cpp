#include "gapese/detail/merit.hpp"

namespace gapese::detail {

double qp_objective(
    const Vector& x,
    const Vector& xi,
    const CcqpData& data,
    const TransformedData& transformed) {
    return 0.5 * x.dot(data.Q_x * x) + data.r_x.dot(x) +
           0.5 * xi.dot(transformed.Q_xi * xi) + transformed.r_xi.dot(xi);
}

double qp_directional_derivative(
    const Vector& x,
    const Vector& xi,
    const Vector& dx,
    const Vector& dxi,
    const CcqpData& data,
    const TransformedData& transformed) {
    return (data.Q_x * x + data.r_x).dot(dx) +
           (transformed.Q_xi * xi + transformed.r_xi).dot(dxi);
}

}  // namespace gapese::detail
