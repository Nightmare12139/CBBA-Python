#pragma once

#include "Planner.h"

namespace cbba {

class EuclideanPlanner final : public Planner {
public:
    double distance(const Position& from, const Position& to) const override {
        const double dx = from.x - to.x;
        const double dy = from.y - to.y;
        const double dz = from.z - to.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
};

}  // namespace cbba
