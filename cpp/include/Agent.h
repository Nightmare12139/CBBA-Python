#pragma once

namespace cbba {

struct Agent {
    int agent_id = 0;
    int agent_type = 0;
    double availability = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double nom_velocity = 0.0;
    double fuel = 0.0;
};

}  // namespace cbba
