#pragma once

#include <cmath>

namespace cbba {

struct Position {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

class Planner {
public:
    virtual ~Planner() = default;

    virtual double distance(const Position& from, const Position& to) const = 0;

    double travel_time(const Position& from, const Position& to, double speed) const {
        if (speed <= 0.0) {
            return 0.0;
        }
        return distance(from, to) / speed;
    }
};

}  // namespace cbba
