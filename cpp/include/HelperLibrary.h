#pragma once

#include <vector>

#include "Agent.h"
#include "Config.h"
#include "Task.h"
#include "WorldInfo.h"

namespace cbba {

void create_agents_and_tasks(int num_agents, int num_tasks, const WorldInfo& world,
                             const Config& config, std::vector<Agent>& agents, std::vector<Task>& tasks);

void create_agents_and_tasks_homogeneous(int num_agents, int num_tasks, const WorldInfo& world,
                                         const Config& config, std::vector<Agent>& agents, std::vector<Task>& tasks);

}  // namespace cbba
