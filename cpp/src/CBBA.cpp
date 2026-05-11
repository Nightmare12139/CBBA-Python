#include "CBBA.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <stdexcept>

namespace cbba {

CBBA::CBBA(const Config& config, std::shared_ptr<Planner> planner)
    : config_(config), planner_(std::move(planner)) {
    agent_types_ = config_.agent_types;
    task_types_ = config_.task_types;

    time_interval_list_ = {
        std::min(config_.track_default.start_time, config_.rescue_default.start_time),
        std::max(config_.track_default.end_time, config_.rescue_default.end_time)
    };

    duration_flag_ = (std::min(config_.track_default.duration, config_.rescue_default.duration) > 0.0);

    compatibility_mat_.assign(agent_types_.size(), std::vector<int>(task_types_.size(), 0));

    const int quad_index = index_of(agent_types_, "quad");
    const int car_index = index_of(agent_types_, "car");
    const int track_index = index_of(task_types_, "track");
    const int rescue_index = index_of(task_types_, "rescue");

    if (quad_index >= 0 && track_index >= 0) {
        compatibility_mat_[quad_index][track_index] = 1;
    }
    if (car_index >= 0 && rescue_index >= 0) {
        compatibility_mat_[car_index][rescue_index] = 1;
    }
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<double>>> CBBA::solve(
    const std::vector<Agent>& agents,
    const std::vector<Task>& tasks,
    const WorldInfo& world,
    int max_depth,
    bool time_window_flag) {
    settings(agents, tasks, world, max_depth, time_window_flag);

    int iter_idx = 1;
    std::vector<std::vector<int>> time_mat(num_agents_, std::vector<int>(num_agents_, 0));
    int iter_prev = 0;
    bool done_flag = false;

    while (!done_flag) {
        time_mat = communicate(time_mat, iter_idx);

        for (int idx_agent = 0; idx_agent < num_agents_; ++idx_agent) {
            const bool new_bid_flag = bundle(idx_agent);
            if (new_bid_flag) {
                iter_prev = iter_idx;
            }
        }

        if ((iter_idx - iter_prev) > (2 * num_agents_)) {
            std::cerr << "Algorithm did not converge within expected iterations (iter="
                      << iter_idx << ", last_update=" << iter_prev << ")." << std::endl;
            done_flag = true;
        } else if ((iter_idx - iter_prev) > num_agents_) {
            done_flag = true;
        } else {
            ++iter_idx;
        }
    }

    for (int n = 0; n < num_agents_; ++n) {
        for (int m = 0; m < max_depth_; ++m) {
            if (bundle_list_[n][m] == -1) {
                break;
            }
            bundle_list_[n][m] = tasks_[bundle_list_[n][m]].task_id;
        }
        for (int m = 0; m < max_depth_; ++m) {
            if (path_list_[n][m] == -1) {
                break;
            }
            path_list_[n][m] = tasks_[path_list_[n][m]].task_id;
        }
    }

    for (auto& path : path_list_) {
        path.erase(std::remove(path.begin(), path.end(), -1), path.end());
    }
    for (auto& bundle : bundle_list_) {
        bundle.erase(std::remove(bundle.begin(), bundle.end(), -1), bundle.end());
    }
    for (auto& times : times_list_) {
        times.erase(std::remove(times.begin(), times.end(), -1.0), times.end());
    }
    for (auto& scores : scores_list_) {
        scores.erase(std::remove(scores.begin(), scores.end(), -1.0), scores.end());
    }

    return {path_list_, times_list_};
}

void CBBA::settings(const std::vector<Agent>& agents,
                    const std::vector<Task>& tasks,
                    const WorldInfo& world,
                    int max_depth,
                    bool time_window_flag) {
    num_agents_ = static_cast<int>(agents.size());
    num_tasks_ = static_cast<int>(tasks.size());
    max_depth_ = max_depth;
    time_window_flag_ = time_window_flag;

    agents_ = agents;
    tasks_ = tasks;
    world_ = world;

    space_limit_x_ = world_.limit_x;
    space_limit_y_ = world_.limit_y;
    space_limit_z_ = world_.limit_z;

    graph_.assign(num_agents_, std::vector<int>(num_agents_, 1));
    for (int i = 0; i < num_agents_; ++i) {
        graph_[i][i] = 0;
    }

    bundle_list_.assign(num_agents_, std::vector<int>(max_depth_, -1));
    path_list_.assign(num_agents_, std::vector<int>(max_depth_, -1));
    times_list_.assign(num_agents_, std::vector<double>(max_depth_, -1.0));
    scores_list_.assign(num_agents_, std::vector<double>(max_depth_, -1.0));

    bid_list_.assign(num_agents_, std::vector<double>(num_tasks_, -1.0));
    winners_list_.assign(num_agents_, std::vector<int>(num_tasks_, -1));
    winner_bid_list_.assign(num_agents_, std::vector<double>(num_tasks_, -1.0));

    agent_index_list_.clear();
    for (int n = 0; n < num_agents_; ++n) {
        agent_index_list_.push_back(agents_[n].agent_id);
    }
}

bool CBBA::bundle(int idx_agent) {
    bundle_remove(idx_agent);
    return bundle_add(idx_agent);
}

void CBBA::bundle_remove(int idx_agent) {
    bool out_bid_for_task = false;
    for (int idx = 0; idx < max_depth_; ++idx) {
        if (bundle_list_[idx_agent][idx] < 0) {
            break;
        }
        if (winners_list_[idx_agent][bundle_list_[idx_agent][idx]] != agent_index_list_[idx_agent]) {
            out_bid_for_task = true;
        }

        if (out_bid_for_task) {
            if (winners_list_[idx_agent][bundle_list_[idx_agent][idx]] == agent_index_list_[idx_agent]) {
                winners_list_[idx_agent][bundle_list_[idx_agent][idx]] = -1;
                winner_bid_list_[idx_agent][bundle_list_[idx_agent][idx]] = -1.0;
            }

            const int task_id = bundle_list_[idx_agent][idx];
            auto& path = path_list_[idx_agent];
            auto it = std::find(path.begin(), path.end(), task_id);
            if (it != path.end()) {
                const auto idx_remove = static_cast<int>(std::distance(path.begin(), it));
                path.erase(path.begin() + idx_remove);
                path.push_back(-1);
                times_list_[idx_agent].erase(times_list_[idx_agent].begin() + idx_remove);
                times_list_[idx_agent].push_back(-1.0);
                scores_list_[idx_agent].erase(scores_list_[idx_agent].begin() + idx_remove);
                scores_list_[idx_agent].push_back(-1.0);
            }

            bundle_list_[idx_agent][idx] = -1;
        }
    }
}

bool CBBA::bundle_add(int idx_agent) {
    const double epsilon = 1e-5;
    bool new_bid_flag = false;

    bool bundle_full_flag = (std::find(bundle_list_[idx_agent].begin(), bundle_list_[idx_agent].end(), -1) ==
                             bundle_list_[idx_agent].end());

    std::vector<std::vector<int>> feasibility(num_tasks_, std::vector<int>(max_depth_ + 1, 1));

    while (!bundle_full_flag) {
        const auto bid_result = compute_bid(idx_agent, feasibility);
        feasibility = bid_result.feasibility;

        double value_max = -1.0;
        int best_task = -1;
        std::vector<int> tied_tasks;

        for (int idx_task = 0; idx_task < num_tasks_; ++idx_task) {
            const double bid_value = bid_list_[idx_agent][idx_task];
            const double winner_bid = winner_bid_list_[idx_agent][idx_task];
            const int winner = winners_list_[idx_agent][idx_task];
            const bool improved = (bid_value - winner_bid) > epsilon;
            const bool tied = std::abs(bid_value - winner_bid) <= epsilon;
            const bool tie_break = agent_index_list_[idx_agent] < winner;
            const bool eligible = improved || (tied && tie_break);
            const double value = eligible ? bid_value : 0.0;

            if (value > value_max) {
                value_max = value;
                best_task = idx_task;
                tied_tasks.clear();
                tied_tasks.push_back(idx_task);
            } else if (std::abs(value - value_max) <= epsilon && value > 0.0) {
                tied_tasks.push_back(idx_task);
            }
        }

        if (value_max > 0.0 && best_task >= 0) {
            new_bid_flag = true;
            if (tied_tasks.size() > 1) {
                double earliest = std::numeric_limits<double>::max();
                for (const int task_idx : tied_tasks) {
                    if (tasks_[task_idx].start_time < earliest) {
                        earliest = tasks_[task_idx].start_time;
                        best_task = task_idx;
                    }
                }
            }

            winners_list_[idx_agent][best_task] = agents_[idx_agent].agent_id;
            winner_bid_list_[idx_agent][best_task] = bid_list_[idx_agent][best_task];

            path_list_[idx_agent].insert(path_list_[idx_agent].begin() + bid_result.best_indices[best_task], best_task);
            path_list_[idx_agent].pop_back();
            times_list_[idx_agent].insert(times_list_[idx_agent].begin() + bid_result.best_indices[best_task],
                                          bid_result.task_times[best_task]);
            times_list_[idx_agent].pop_back();
            scores_list_[idx_agent].insert(scores_list_[idx_agent].begin() + bid_result.best_indices[best_task],
                                           bid_list_[idx_agent][best_task]);
            scores_list_[idx_agent].pop_back();

            const int length = count_assigned(bundle_list_[idx_agent]);
            bundle_list_[idx_agent][length] = best_task;

            for (int i = 0; i < num_tasks_; ++i) {
                feasibility[i].insert(feasibility[i].begin() + bid_result.best_indices[best_task],
                                      feasibility[i][bid_result.best_indices[best_task]]);
                feasibility[i].pop_back();
            }
        } else {
            break;
        }

        bundle_full_flag = (std::find(bundle_list_[idx_agent].begin(), bundle_list_[idx_agent].end(), -1) ==
                             bundle_list_[idx_agent].end());
    }

    return new_bid_flag;
}

std::vector<std::vector<int>> CBBA::communicate(const std::vector<std::vector<int>>& time_mat, int iter_idx) {
    auto time_mat_new = time_mat;
    const auto old_z = winners_list_;
    const auto old_y = winner_bid_list_;
    auto z = old_z;
    auto y = old_y;

    const double epsilon = 1e-5;
    const auto format_state = [&](const std::string& context, int i, int k, int j) {
        return context + " (i=" + std::to_string(i) + ", k=" + std::to_string(k) +
               ", j=" + std::to_string(j) + ", z=" + std::to_string(z[i][j]) +
               ", old_z=" + std::to_string(old_z[k][j]) + ").";
    };

    for (int k = 0; k < num_agents_; ++k) {
        for (int i = 0; i < num_agents_; ++i) {
            if (graph_[k][i] == 1) {
                for (int j = 0; j < num_tasks_; ++j) {
                    if (old_z[k][j] == k) {
                        if (z[i][j] == i) {
                            if ((old_y[k][j] - y[i][j]) > epsilon) {
                                z[i][j] = old_z[k][j];
                                y[i][j] = old_y[k][j];
                            } else if (std::abs(old_y[k][j] - y[i][j]) <= epsilon) {
                                if (z[i][j] > old_z[k][j]) {
                                    z[i][j] = old_z[k][j];
                                    y[i][j] = old_y[k][j];
                                }
                            }
                        } else if (z[i][j] == k) {
                            z[i][j] = old_z[k][j];
                            y[i][j] = old_y[k][j];
                        } else if (z[i][j] > -1) {
                            if (time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]) {
                                z[i][j] = old_z[k][j];
                                y[i][j] = old_y[k][j];
                            } else if ((old_y[k][j] - y[i][j]) > epsilon) {
                                z[i][j] = old_z[k][j];
                                y[i][j] = old_y[k][j];
                            } else if (std::abs(old_y[k][j] - y[i][j]) <= epsilon) {
                                if (z[i][j] > old_z[k][j]) {
                                    z[i][j] = old_z[k][j];
                                    y[i][j] = old_y[k][j];
                                }
                            }
                        } else if (z[i][j] == -1) {
                            z[i][j] = old_z[k][j];
                            y[i][j] = old_y[k][j];
                        } else {
                            throw std::runtime_error(format_state("Unexpected winner value while sender owns task",
                                                                 i, k, j));
                        }
                    } else if (old_z[k][j] == i) {
                        if (z[i][j] == i) {
                            continue;
                        } else if (z[i][j] == k) {
                            z[i][j] = -1;
                            y[i][j] = -1;
                        } else if (z[i][j] > -1) {
                            if (time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]) {
                                z[i][j] = -1;
                                y[i][j] = -1;
                            }
                        } else if (z[i][j] == -1) {
                            continue;
                        } else {
                            throw std::runtime_error(format_state("Unexpected winner value while sender assigns to receiver",
                                                                 i, k, j));
                        }
                    } else if (old_z[k][j] > -1) {
                        if (z[i][j] == i) {
                            if (time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]) {
                                if ((old_y[k][j] - y[i][j]) > epsilon) {
                                    z[i][j] = old_z[k][j];
                                    y[i][j] = old_y[k][j];
                                } else if (std::abs(old_y[k][j] - y[i][j]) <= epsilon) {
                                    if (z[i][j] > old_z[k][j]) {
                                        z[i][j] = old_z[k][j];
                                        y[i][j] = old_y[k][j];
                                    }
                                }
                            }
                        } else if (z[i][j] == k) {
                            if (time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]) {
                                z[i][j] = old_z[k][j];
                                y[i][j] = old_y[k][j];
                            } else {
                                z[i][j] = -1;
                                y[i][j] = -1;
                            }
                        } else if (z[i][j] == old_z[k][j]) {
                            if (time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]) {
                                z[i][j] = old_z[k][j];
                                y[i][j] = old_y[k][j];
                            }
                        } else if (z[i][j] > -1) {
                            if (time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]) {
                                if (time_mat[k][old_z[k][j]] >= time_mat_new[i][old_z[k][j]]) {
                                    z[i][j] = old_z[k][j];
                                    y[i][j] = old_y[k][j];
                                } else if (time_mat[k][old_z[k][j]] < time_mat_new[i][old_z[k][j]]) {
                                    z[i][j] = -1;
                                    y[i][j] = -1;
                                }
                            } else {
                                if (time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]) {
                                    if ((old_y[k][j] - y[i][j]) > epsilon) {
                                        z[i][j] = old_z[k][j];
                                        y[i][j] = old_y[k][j];
                                    } else if (std::abs(old_y[k][j] - y[i][j]) <= epsilon) {
                                        if (z[i][j] > old_z[k][j]) {
                                            z[i][j] = old_z[k][j];
                                            y[i][j] = old_y[k][j];
                                        }
                                    }
                                }
                            }
                        } else if (z[i][j] == -1) {
                            if (time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]) {
                                z[i][j] = old_z[k][j];
                                y[i][j] = old_y[k][j];
                            }
                        } else {
                            throw std::runtime_error(format_state("Unexpected winner value while sender reports other winner",
                                                                 i, k, j));
                        }
                    } else if (old_z[k][j] == -1) {
                        if (z[i][j] == i) {
                            continue;
                        } else if (z[i][j] == k) {
                            z[i][j] = old_z[k][j];
                            y[i][j] = old_y[k][j];
                        } else if (z[i][j] > -1) {
                            if (time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]) {
                                z[i][j] = old_z[k][j];
                                y[i][j] = old_y[k][j];
                            }
                        } else if (z[i][j] == -1) {
                            continue;
                        } else {
                            throw std::runtime_error(format_state("Unexpected winner value while no winner reported",
                                                                 i, k, j));
                        }
                    } else {
                        throw std::runtime_error(format_state("Unexpected winner value in communication", i, k, j));
                    }
                }

                for (int n = 0; n < num_agents_; ++n) {
                    if ((n != i) && (time_mat_new[i][n] < time_mat[k][n])) {
                        time_mat_new[i][n] = time_mat[k][n];
                    }
                }
                time_mat_new[i][k] = iter_idx;
            }
        }
    }

    winners_list_ = z;
    winner_bid_list_ = y;
    return time_mat_new;
}

CBBA::BidResult CBBA::compute_bid(int idx_agent, std::vector<std::vector<int>> feasibility) {
    const auto empty_it = std::find(path_list_[idx_agent].begin(), path_list_[idx_agent].end(), -1);
    if (empty_it == path_list_[idx_agent].end()) {
        return {{}, {}, {}};
    }
    const int empty_index = static_cast<int>(std::distance(path_list_[idx_agent].begin(), empty_it));

    bid_list_[idx_agent].assign(num_tasks_, -1.0);
    std::vector<int> best_indices(num_tasks_, -1);
    std::vector<double> task_times(num_tasks_, -2.0);

    const int task_count_after = count_assigned(path_list_[idx_agent]) + 1;

    for (int idx_task = 0; idx_task < num_tasks_; ++idx_task) {
        if (compatibility_mat_[agents_[idx_agent].agent_type][tasks_[idx_task].task_type] > 0) {
            auto path_begin = path_list_[idx_agent].begin();
            auto path_end = path_list_[idx_agent].begin() + empty_index;
            if (std::find(path_begin, path_end, idx_task) == path_end) {
                double best_bid = 0.0;
                int best_index = -1;
                double best_time = -2.0;

                for (int j = 0; j <= empty_index; ++j) {
                    if (feasibility[idx_task][j] == 1) {
                        bool skip_flag = false;
                        const Task* task_prev = nullptr;
                        std::optional<double> time_prev;
                        const Task* task_next = nullptr;
                        std::optional<double> time_next;

                        if (j > 0) {
                            task_prev = &tasks_[path_list_[idx_agent][j - 1]];
                            time_prev = times_list_[idx_agent][j - 1];
                        }
                        if (j < empty_index) {
                            task_next = &tasks_[path_list_[idx_agent][j]];
                            time_next = times_list_[idx_agent][j];
                        }

                        const auto score_info = scoring_compute_score(idx_agent, tasks_[idx_task],
                                                                     task_prev, time_prev, task_next, time_next,
                                                                     task_count_after);

                        if (time_window_flag_) {
                            if (score_info.min_start > score_info.max_start) {
                                skip_flag = true;
                                feasibility[idx_task][j] = 0;
                            }

                            if (!skip_flag && score_info.score > best_bid) {
                                best_bid = score_info.score;
                                best_index = j;
                                best_time = score_info.min_start;
                            }
                        } else {
                            if (score_info.score > best_bid) {
                                best_bid = score_info.score;
                                best_index = j;
                                best_time = 0.0;
                            }
                        }
                    }
                }

                if (best_bid > 0.0) {
                    bid_list_[idx_agent][idx_task] = best_bid;
                    best_indices[idx_task] = best_index;
                    task_times[idx_task] = best_time;
                }
            }
        }
    }

    return {best_indices, task_times, feasibility};
}

CBBA::ScoreInfo CBBA::scoring_compute_score(int idx_agent,
                                            const Task& task_current,
                                            const Task* task_prev,
                                            std::optional<double> time_prev,
                                            const Task* task_next,
                                            std::optional<double> time_next,
                                            int task_count_after) {
    if (!planner_) {
        throw std::runtime_error("Planner is not configured.");
    }

    ScoreInfo result;
    const Agent& agent = agents_[idx_agent];
    const Position agent_pos{agent.x, agent.y, agent.z};
    const Position current_pos{task_current.x, task_current.y, task_current.z};

    if (!task_prev) {
        const double dt = planner_->travel_time(agent_pos, current_pos, agent.nom_velocity);
        result.min_start = std::max(task_current.start_time, agent.availability + dt);
    } else {
        const Position prev_pos{task_prev->x, task_prev->y, task_prev->z};
        const double dt = planner_->travel_time(prev_pos, current_pos, agent.nom_velocity);
        result.min_start = std::max(task_current.start_time, time_prev.value_or(0.0) + task_prev->duration + dt);
    }

    if (!task_next) {
        result.max_start = task_current.end_time;
    } else {
        const Position next_pos{task_next->x, task_next->y, task_next->z};
        const double dt = planner_->travel_time(current_pos, next_pos, agent.nom_velocity);
        result.max_start = std::min(task_current.end_time, time_next.value_or(0.0) - task_current.duration - dt);
    }

    double reward = 0.0;
    if (time_window_flag_) {
        reward = task_current.task_value * std::exp((-task_current.discount) *
                                                   (result.min_start - task_current.start_time));
    } else {
        const double dt_current = planner_->travel_time(agent_pos, current_pos, agent.nom_velocity);
        reward = task_current.task_value * std::exp((-task_current.discount) * dt_current);
    }

    result.score = apply_dmg(reward, task_count_after);
    return result;
}

double CBBA::apply_dmg(double reward, int task_count_after) const {
    if (!config_.dmg.enabled || reward <= 0.0) {
        return reward;
    }

    const int task_index = std::max(0, task_count_after - 1);
    double factor = 1.0;

    if (config_.dmg.mode == "linear") {
        factor = std::max(0.0, 1.0 - config_.dmg.penalty_coef * task_index);
    } else if (config_.dmg.mode == "inverse") {
        factor = 1.0 / (1.0 + config_.dmg.penalty_coef * task_index);
    } else {
        factor = std::exp(-config_.dmg.penalty_coef * task_index);
    }

    if (config_.dmg.max_tasks > 0 && task_count_after > config_.dmg.max_tasks) {
        const int overflow = task_count_after - config_.dmg.max_tasks;
        factor *= std::exp(-config_.dmg.penalty_coef * overflow);
    }

    return reward * factor;
}

int CBBA::count_assigned(const std::vector<int>& values) const {
    return static_cast<int>(std::count_if(values.begin(), values.end(), [](int value) { return value > -1; }));
}

Task CBBA::lookup_task(int task_id) const {
    for (const auto& task : tasks_) {
        if (task.task_id == task_id) {
            return task;
        }
    }
    throw std::runtime_error("Task not found: " + std::to_string(task_id));
}

int CBBA::index_of(const std::vector<std::string>& values, const std::string& target) const {
    auto it = std::find(values.begin(), values.end(), target);
    if (it == values.end()) {
        return -1;
    }
    return static_cast<int>(std::distance(values.begin(), it));
}

}  // namespace cbba
