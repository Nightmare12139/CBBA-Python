#pragma once

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "Agent.h"
#include "Config.h"
#include "Planner.h"
#include "Task.h"
#include "WorldInfo.h"

namespace cbba {

class CBBA {
public:
    CBBA(const Config& config, std::shared_ptr<Planner> planner);

    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<double>>> solve(
        const std::vector<Agent>& agents,
        const std::vector<Task>& tasks,
        const WorldInfo& world,
        int max_depth,
        bool time_window_flag);

private:
    struct ScoreInfo {
        double score = 0.0;
        double min_start = 0.0;
        double max_start = 0.0;
    };

    void settings(const std::vector<Agent>& agents,
                  const std::vector<Task>& tasks,
                  const WorldInfo& world,
                  int max_depth,
                  bool time_window_flag);

    bool bundle(int idx_agent);
    void bundle_remove(int idx_agent);
    bool bundle_add(int idx_agent);
    std::vector<std::vector<int>> communicate(const std::vector<std::vector<int>>& time_mat, int iter_idx);

    struct BidResult {
        std::vector<int> best_indices;
        std::vector<double> task_times;
        std::vector<std::vector<int>> feasibility;
    };

    BidResult compute_bid(int idx_agent, std::vector<std::vector<int>> feasibility);
    ScoreInfo scoring_compute_score(int idx_agent,
                                    const Task& task_current,
                                    const Task* task_prev,
                                    std::optional<double> time_prev,
                                    const Task* task_next,
                                    std::optional<double> time_next,
                                    int task_count_after);

    double apply_dmg(double reward, int task_count_after) const;
    int count_assigned(const std::vector<int>& values) const;
    Task lookup_task(int task_id) const;
    int index_of(const std::vector<std::string>& values, const std::string& target) const;

    Config config_;
    std::shared_ptr<Planner> planner_;

    int num_agents_ = 0;
    int num_tasks_ = 0;
    int max_depth_ = 0;
    bool time_window_flag_ = false;
    bool duration_flag_ = false;

    std::vector<std::string> agent_types_;
    std::vector<std::string> task_types_;

    std::array<double, 2> space_limit_x_{};
    std::array<double, 2> space_limit_y_{};
    std::array<double, 2> space_limit_z_{};

    std::vector<double> time_interval_list_;

    std::vector<int> agent_index_list_;
    std::vector<std::vector<int>> bundle_list_;
    std::vector<std::vector<int>> path_list_;
    std::vector<std::vector<double>> times_list_;
    std::vector<std::vector<double>> scores_list_;
    std::vector<std::vector<double>> bid_list_;
    std::vector<std::vector<int>> winners_list_;
    std::vector<std::vector<double>> winner_bid_list_;
    std::vector<std::vector<int>> graph_;
    std::vector<std::vector<int>> compatibility_mat_;

    std::vector<Agent> agents_;
    std::vector<Task> tasks_;
    WorldInfo world_;
};

}  // namespace cbba
