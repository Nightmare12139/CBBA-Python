#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Planner.h"

namespace cbba {

struct AgentDefaults {
    double nom_velocity = 1.0;
};

struct TaskDefaults {
    double task_value = 100.0;
    double start_time = 0.0;
    double end_time = 0.0;
    double duration = 0.0;
    double discount = 0.1;
};

struct PlannerConfig {
    std::string type = "euclidean";
};

struct DmgConfig {
    bool enabled = false;
    std::string mode = "exp";
    double penalty_coef = 0.2;
    int max_tasks = 0;
};

struct Config {
    std::vector<std::string> agent_types;
    std::vector<std::string> task_types;
    AgentDefaults quad_default;
    AgentDefaults car_default;
    TaskDefaults track_default;
    TaskDefaults rescue_default;
    PlannerConfig planner;
    DmgConfig dmg;
};

Config load_config(const std::string& path);
std::shared_ptr<Planner> create_planner(const Config& config);

}  // namespace cbba
