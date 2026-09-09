#include "gapese/options.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

struct DoubleOptionField {
    const char* name;
    double gapese::Options::*member;
};

template <typename Modify>
bool expect_invalid(const char* name, Modify&& modify) {
    gapese::Options options;
    modify(options);
    try {
        options.validate();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << name
                  << ": expected std::invalid_argument, got " << error.what() << '\n';
        return false;
    }

    std::cerr << "[FAIL] " << name << ": expected std::invalid_argument\n";
    return false;
}

}  // namespace

int main() {
    int failures = 0;

    // The default configuration must always be valid.
    try {
        gapese::Options{}.validate();
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] default options: " << error.what() << '\n';
        ++failures;
    }

    // Every floating-point option must reject NaN and infinities.
    constexpr std::array<DoubleOptionField, 18> double_fields{{
        {"dgap_a", &gapese::Options::dgap_a},
        {"dgap_b", &gapese::Options::dgap_b},
        {"fb_sigma", &gapese::Options::fb_sigma},
        {"tol_kkt", &gapese::Options::tol_kkt},
        {"tol_dgap", &gapese::Options::tol_dgap},
        {"newton_step_max", &gapese::Options::newton_step_max},
        {"mu_min", &gapese::Options::mu_min},
        {"mu_max", &gapese::Options::mu_max},
        {"mu_update", &gapese::Options::mu_update},
        {"nu_x", &gapese::Options::nu_x},
        {"nu_xi", &gapese::Options::nu_xi},
        {"nu_b", &gapese::Options::nu_b},
        {"nu_d", &gapese::Options::nu_d},
        {"beta_init", &gapese::Options::beta_init},
        {"rho", &gapese::Options::rho},
        {"alpha_min", &gapese::Options::alpha_min},
        {"nu_alpha", &gapese::Options::nu_alpha},
        {"nu_D", &gapese::Options::nu_D},
    }};

    constexpr std::array<double, 3> nonfinite_values{{
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    }};
    constexpr std::array<const char*, 3> nonfinite_names{{"NaN", "+Inf", "-Inf"}};

    for (const auto& field : double_fields) {
        for (std::size_t i = 0; i < nonfinite_values.size(); ++i) {
            const std::string case_name = std::string(field.name) + " = " + nonfinite_names[i];
            if (!expect_invalid(case_name.c_str(), [&](gapese::Options& options) {
                    options.*(field.member) = nonfinite_values[i];
                })) {
                ++failures;
            }
        }
    }

    // Integer ranges.
    failures += !expect_invalid("print_level < 0", [](gapese::Options& o) { o.print_level = -1; });
    failures += !expect_invalid("print_level > 2", [](gapese::Options& o) { o.print_level = 3; });
    failures += !expect_invalid("max_iter <= 0", [](gapese::Options& o) { o.max_iter = 0; });

    // D-gap and Fisher--Burmeister parameters.
    failures += !expect_invalid("dgap_a <= 0", [](gapese::Options& o) { o.dgap_a = 0.0; });
    failures += !expect_invalid("dgap_b <= dgap_a", [](gapese::Options& o) {
        o.dgap_a = 0.5;
        o.dgap_b = 0.5;
    });
    failures += !expect_invalid("fb_sigma < 0", [](gapese::Options& o) { o.fb_sigma = -1.0; });

    // Termination and Newton-step parameters.
    failures += !expect_invalid("tol_kkt <= 0", [](gapese::Options& o) { o.tol_kkt = 0.0; });
    failures += !expect_invalid("tol_dgap <= 0", [](gapese::Options& o) { o.tol_dgap = 0.0; });
    failures += !expect_invalid("newton_step_max <= 0", [](gapese::Options& o) { o.newton_step_max = 0.0; });

    // Penalty update parameters.
    failures += !expect_invalid("mu_min <= 0", [](gapese::Options& o) { o.mu_min = 0.0; });
    failures += !expect_invalid("mu_max < mu_min", [](gapese::Options& o) {
        o.mu_min = 2.0;
        o.mu_max = 1.0;
    });
    failures += !expect_invalid("mu_update <= 1", [](gapese::Options& o) { o.mu_update = 1.0; });

    // Regularization and merit parameters.
    failures += !expect_invalid("nu_x < 0", [](gapese::Options& o) { o.nu_x = -1.0; });
    failures += !expect_invalid("nu_xi < 0", [](gapese::Options& o) { o.nu_xi = -1.0; });
    failures += !expect_invalid("nu_b < 0", [](gapese::Options& o) { o.nu_b = -1.0; });
    failures += !expect_invalid("nu_d < 0", [](gapese::Options& o) { o.nu_d = -1.0; });
    failures += !expect_invalid("beta_init <= 0", [](gapese::Options& o) { o.beta_init = 0.0; });
    failures += !expect_invalid("rho <= 0", [](gapese::Options& o) { o.rho = 0.0; });
    failures += !expect_invalid("rho >= 1", [](gapese::Options& o) { o.rho = 1.0; });
    failures += !expect_invalid("alpha_min <= 0", [](gapese::Options& o) { o.alpha_min = 0.0; });
    failures += !expect_invalid("alpha_min >= 1", [](gapese::Options& o) { o.alpha_min = 1.0; });
    failures += !expect_invalid("nu_alpha <= 0", [](gapese::Options& o) { o.nu_alpha = 0.0; });
    failures += !expect_invalid("nu_alpha >= 1", [](gapese::Options& o) { o.nu_alpha = 1.0; });
    failures += !expect_invalid("nu_D <= 0", [](gapese::Options& o) { o.nu_D = 0.0; });
    failures += !expect_invalid("nu_D >= 1", [](gapese::Options& o) { o.nu_D = 1.0; });

    if (failures != 0) {
        std::cerr << failures << " option validation test(s) failed\n";
        return 1;
    }

    return 0;
}
