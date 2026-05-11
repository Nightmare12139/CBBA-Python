#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "CBBA.h"
#include "Config.h"
#include "HelperLibrary.h"
#include "WorldInfo.h"

int main(int argc, char** argv) {
    std::filesystem::path config_path;
    if (argc > 1) {
        config_path = argv[1];
    } else {
        const auto exe_dir = std::filesystem::absolute(std::filesystem::path(argv[0])).parent_path();
        config_path = exe_dir / ".." / ".." / "config_example_cpp_01.json";
    }

    cbba::Config config = cbba::load_config(config_path.string());
    auto planner = cbba::create_planner(config);
    cbba::CBBA solver(config, planner);

    cbba::WorldInfo world({-2.0, 2.5}, {-1.5, 5.5}, {0.0, 20.0});

    const int num_agents = 5;
    const int num_tasks = 10;
    const int max_depth = num_tasks;

    std::vector<cbba::Agent> agents;
    std::vector<cbba::Task> tasks;
    cbba::create_agents_and_tasks(num_agents, num_tasks, world, config, agents, tasks);

    const bool time_window_flag = true;
    const auto result = solver.solve(agents, tasks, world, max_depth, time_window_flag);

    const auto& path_list = result.first;
    const auto& times_list = result.second;

    std::cout << "DMG enabled: " << (config.dmg.enabled ? "true" : "false")
              << ", mode: " << config.dmg.mode
              << ", penalty coef: " << config.dmg.penalty_coef
              << ", max tasks: " << config.dmg.max_tasks << "\n";

    for (size_t i = 0; i < path_list.size(); ++i) {
        std::cout << "Agent " << i << " tasks: ";
        for (const int task_id : path_list[i]) {
            std::cout << task_id << " ";
        }
        std::cout << "\n";

        std::cout << "Agent " << i << " times: ";
        for (const double time_value : times_list[i]) {
            std::cout << time_value << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
