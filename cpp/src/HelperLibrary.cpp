#include "HelperLibrary.h"

#include <algorithm>
#include <random>

namespace cbba {
namespace {

int index_of(const std::vector<std::string>& values, const std::string& target, int fallback = 0) {
    auto it = std::find(values.begin(), values.end(), target);
    if (it == values.end()) {
        return fallback;
    }
    return static_cast<int>(std::distance(values.begin(), it));
}

double random_double(std::mt19937& rng, double min_value, double max_value) {
    std::uniform_real_distribution<double> dist(min_value, max_value);
    return dist(rng);
}

}  // namespace

void create_agents_and_tasks(int num_agents, int num_tasks, const WorldInfo& world,
                             const Config& config, std::vector<Agent>& agents, std::vector<Task>& tasks) {
    agents.clear();
    tasks.clear();

    Agent quad_default;
    quad_default.agent_type = index_of(config.agent_types, "quad");
    quad_default.nom_velocity = config.quad_default.nom_velocity;

    Agent car_default;
    car_default.agent_type = index_of(config.agent_types, "car");
    car_default.nom_velocity = config.car_default.nom_velocity;

    Task track_default;
    track_default.task_type = index_of(config.task_types, "track");
    track_default.task_value = config.track_default.task_value;
    track_default.start_time = config.track_default.start_time;
    track_default.end_time = config.track_default.end_time;
    track_default.duration = config.track_default.duration;

    Task rescue_default;
    rescue_default.task_type = index_of(config.task_types, "rescue", track_default.task_type);
    rescue_default.task_value = config.rescue_default.task_value;
    rescue_default.start_time = config.rescue_default.start_time;
    rescue_default.end_time = config.rescue_default.end_time;
    rescue_default.duration = config.rescue_default.duration;

    std::mt19937 rng(42);

    for (int idx_agent = 0; idx_agent < num_agents; ++idx_agent) {
        Agent agent = (static_cast<double>(idx_agent) / num_agents <= 0.5) ? quad_default : car_default;
        agent.agent_id = idx_agent;
        agent.x = random_double(rng, world.limit_x[0], world.limit_x[1]);
        agent.y = random_double(rng, world.limit_y[0], world.limit_y[1]);
        agent.z = 0.0;
        agents.push_back(agent);
    }

    const double max_end = std::max(config.track_default.end_time, config.rescue_default.end_time);
    const double max_duration = std::max(config.track_default.duration, config.rescue_default.duration);
    const double latest_start = std::max(0.0, max_end - max_duration);

    for (int idx_task = 0; idx_task < num_tasks; ++idx_task) {
        Task task = (static_cast<double>(idx_task) / num_tasks <= 0.5) ? track_default : rescue_default;
        task.task_id = idx_task;
        task.x = random_double(rng, world.limit_x[0], world.limit_x[1]);
        task.y = random_double(rng, world.limit_y[0], world.limit_y[1]);
        task.z = 0.0;
        task.start_time = random_double(rng, 0.0, latest_start);
        task.end_time = task.start_time + task.duration;
        tasks.push_back(task);
    }
}

void create_agents_and_tasks_homogeneous(int num_agents, int num_tasks, const WorldInfo& world,
                                         const Config& config, std::vector<Agent>& agents, std::vector<Task>& tasks) {
    agents.clear();
    tasks.clear();

    Agent quad_default;
    quad_default.agent_type = index_of(config.agent_types, "quad");
    quad_default.nom_velocity = config.quad_default.nom_velocity;

    Task track_default;
    track_default.task_type = index_of(config.task_types, "track");
    track_default.task_value = config.track_default.task_value;
    track_default.start_time = 0.0;
    track_default.end_time = 0.0;
    track_default.duration = 0.0;

    std::mt19937 rng(42);

    for (int idx_agent = 0; idx_agent < num_agents; ++idx_agent) {
        Agent agent = quad_default;
        agent.agent_id = idx_agent;
        agent.x = random_double(rng, world.limit_x[0], world.limit_x[1]);
        agent.y = random_double(rng, world.limit_y[0], world.limit_y[1]);
        agent.z = 0.0;
        agents.push_back(agent);
    }

    for (int idx_task = 0; idx_task < num_tasks; ++idx_task) {
        Task task = track_default;
        task.task_id = idx_task;
        task.x = random_double(rng, world.limit_x[0], world.limit_x[1]);
        task.y = random_double(rng, world.limit_y[0], world.limit_y[1]);
        task.z = 0.0;
        tasks.push_back(task);
    }
}

}  // namespace cbba
