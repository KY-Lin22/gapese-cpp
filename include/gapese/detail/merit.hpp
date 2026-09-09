#pragma once

#include "gapese/detail/types.hpp"

namespace gapese::detail {

double qp_objective(
    const Vector& x,
    const Vector& xi,
    const CcqpData& data,
    const TransformedData& transformed);

double qp_directional_derivative(
    const Vector& x,
    const Vector& xi,
    const Vector& dx,
    const Vector& dxi,
    const CcqpData& data,
    const TransformedData& transformed);

}  // namespace gapese::detail
