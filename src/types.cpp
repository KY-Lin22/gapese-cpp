#include "gapese/types.hpp"

#include <stdexcept>

namespace gapese {

void ProblemSize::validate() const {
    if (n_x <= 0) {
        throw std::invalid_argument("n_x must be a positive integer");
    }
    if (n_lambda <= 0) {
        throw std::invalid_argument("n_lambda must be a positive integer");
    }
    if (n_b < 0 || n_d < 0) {
        throw std::invalid_argument("n_b and n_d must be nonnegative integers");
    }
}

}  // namespace gapese
