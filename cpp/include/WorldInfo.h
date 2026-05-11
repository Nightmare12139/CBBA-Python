#pragma once

#include <array>
#include <cmath>

namespace cbba {

struct WorldInfo {
    std::array<double, 2> limit_x{0.0, 100.0};
    std::array<double, 2> limit_y{0.0, 100.0};
    std::array<double, 2> limit_z{0.0, 0.0};
    double distance_max = 0.0;

    WorldInfo() = default;
    WorldInfo(const std::array<double, 2>& x, const std::array<double, 2>& y, const std::array<double, 2>& z)
        : limit_x(x), limit_y(y), limit_z(z) {
        const double dx = limit_x[1] - limit_x[0];
        const double dy = limit_y[1] - limit_y[0];
        const double dz = limit_z[1] - limit_z[0];
        distance_max = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
};

}  // namespace cbba
