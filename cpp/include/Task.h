#pragma once

namespace cbba {

struct Task {
    int task_id = 0;
    int task_type = 0;
    double task_value = 0.0;
    double start_time = 0.0;
    double end_time = 0.0;
    double duration = 0.0;
    double discount = 0.1;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

}  // namespace cbba
