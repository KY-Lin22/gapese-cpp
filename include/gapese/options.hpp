#pragma once

#include <stdexcept>
#include <string>

namespace gapese {

struct Options {
    int print_level = 2;

    double dgap_a = 0.1;
    double dgap_b = 10.0;
    double fb_sigma = 0.0;

    int max_iter = 500;
    double tol_kkt = 1e-4;
    double tol_dgap = 1e-4;
    double newton_step_max = 1e10;

    double mu_min = 1.0;
    double mu_max = 1e6;
    double mu_update = 2.0;

    double nu_x = 1e-8;
    double nu_xi = 1e-8;
    double nu_b = 1e-8;
    double nu_d = 1e-8;

    double beta_init = 1.0;
    double rho = 0.1;
    double alpha_min = 1e-16;
    double nu_alpha = 0.5;
    double nu_D = 1e-4;

    void validate() const {
        if (print_level < 0 || print_level > 2) {
            throw std::invalid_argument("print_level must be 0, 1, or 2");
        }
        if (!(dgap_b > dgap_a && dgap_a > 0.0)) {
            throw std::invalid_argument("D-gap parameters must satisfy b > a > 0");
        }
        if (fb_sigma < 0.0) {
            throw std::invalid_argument("fb_sigma must be nonnegative");
        }
        if (max_iter <= 0) {
            throw std::invalid_argument("max_iter must be positive");
        }
        if (tol_kkt <= 0.0 || tol_dgap <= 0.0 || newton_step_max <= 0.0) {
            throw std::invalid_argument("tolerances and newton_step_max must be positive");
        }
        if (!(mu_max >= mu_min && mu_min > 0.0)) {
            throw std::invalid_argument("mu_max and mu_min must satisfy mu_max >= mu_min > 0");
        }
        if (mu_update <= 1.0) {
            throw std::invalid_argument("mu_update must be greater than 1");
        }
        if (nu_x < 0.0 || nu_xi < 0.0 || nu_b < 0.0 || nu_d < 0.0) {
            throw std::invalid_argument("regularization parameters must be nonnegative");
        }
        if (beta_init <= 0.0) {
            throw std::invalid_argument("beta_init must be positive");
        }
        if (!(rho > 0.0 && rho < 1.0)) {
            throw std::invalid_argument("rho must lie in (0,1)");
        }
        if (!(alpha_min > 0.0 && alpha_min < 1.0 && nu_alpha > 0.0 && nu_alpha < 1.0)) {
            throw std::invalid_argument("alpha_min and nu_alpha must lie in (0,1)");
        }
        if (!(nu_D > 0.0 && nu_D < 1.0)) {
            throw std::invalid_argument("nu_D must lie in (0,1)");
        }
    }
};

}  // namespace gapese
