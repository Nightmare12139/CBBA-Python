#include "Config.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "EuclideanPlanner.h"

namespace cbba {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open config file: " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string extract_block(const std::string& content, const std::string& key, char open_char, char close_char) {
    const std::string token = "\"" + key + "\"";
    const auto key_pos = content.find(token);
    if (key_pos == std::string::npos) {
        return {};
    }
    const auto colon_pos = content.find(':', key_pos + token.size());
    if (colon_pos == std::string::npos) {
        return {};
    }
    const auto open_pos = content.find(open_char, colon_pos + 1);
    if (open_pos == std::string::npos) {
        return {};
    }
    int depth = 0;
    for (size_t i = open_pos; i < content.size(); ++i) {
        if (content[i] == open_char) {
            ++depth;
        } else if (content[i] == close_char) {
            --depth;
            if (depth == 0) {
                return content.substr(open_pos + 1, i - open_pos - 1);
            }
        }
    }
    return {};
}

std::string extract_object(const std::string& content, const std::string& key) {
    return extract_block(content, key, '{', '}');
}

std::string extract_array(const std::string& content, const std::string& key) {
    return extract_block(content, key, '[', ']');
}

std::string escape_regex(const std::string& input) {
    const std::string metacharacters = R"(\.^$|()[]{}*+?)";
    std::string escaped;
    escaped.reserve(input.size() * 2);
    for (const char c : input) {
        if (metacharacters.find(c) != std::string::npos) {
            escaped.push_back('\\');
        }
        escaped.push_back(c);
    }
    return escaped;
}

std::string read_string(const std::string& content, const std::string& key, const std::string& fallback) {
    const std::string escaped_key = escape_regex(key);
    const std::regex pattern("\\\"" + escaped_key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
        return match[1].str();
    }
    return fallback;
}

bool read_bool(const std::string& content, const std::string& key, bool fallback) {
    const std::string escaped_key = escape_regex(key);
    const std::regex pattern("\\\"" + escaped_key + "\\\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
        return match[1].str() == "true";
    }
    return fallback;
}

double read_number(const std::string& content, const std::string& key, double fallback) {
    const std::string escaped_key = escape_regex(key);
    const std::regex pattern("\\\"" + escaped_key + "\\\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
        return std::stod(match[1].str());
    }
    return fallback;
}

int read_int(const std::string& content, const std::string& key, int fallback) {
    const double value = read_number(content, key, static_cast<double>(fallback));
    return static_cast<int>(value);
}

std::vector<std::string> read_string_array(const std::string& content) {
    std::vector<std::string> values;
    const std::regex pattern("\\\"([^\\\"]+)\\\"");
    for (auto it = std::sregex_iterator(content.begin(), content.end(), pattern); it != std::sregex_iterator(); ++it) {
        values.push_back((*it)[1].str());
    }
    return values;
}

}  // namespace

Config load_config(const std::string& path) {
    const std::string content = read_file(path);
    Config config;

    const auto agent_block = extract_array(content, "AGENT_TYPES");
    config.agent_types = read_string_array(agent_block);
    if (config.agent_types.empty()) {
        throw std::runtime_error("AGENT_TYPES is missing or empty in config.");
    }

    const auto task_block = extract_array(content, "TASK_TYPES");
    config.task_types = read_string_array(task_block);
    if (config.task_types.empty()) {
        throw std::runtime_error("TASK_TYPES is missing or empty in config.");
    }

    const auto quad_block = extract_object(content, "QUAD_DEFAULT");
    config.quad_default.nom_velocity = read_number(quad_block, "NOM_VELOCITY", config.quad_default.nom_velocity);

    const auto car_block = extract_object(content, "CAR_DEFAULT");
    config.car_default.nom_velocity = read_number(car_block, "NOM_VELOCITY", config.car_default.nom_velocity);

    const auto track_block = extract_object(content, "TRACK_DEFAULT");
    config.track_default.task_value = read_number(track_block, "TASK_VALUE", config.track_default.task_value);
    config.track_default.start_time = read_number(track_block, "START_TIME", config.track_default.start_time);
    config.track_default.end_time = read_number(track_block, "END_TIME", config.track_default.end_time);
    config.track_default.duration = read_number(track_block, "DURATION", config.track_default.duration);

    const auto rescue_block = extract_object(content, "RESCUE_DEFAULT");
    config.rescue_default.task_value = read_number(rescue_block, "TASK_VALUE", config.rescue_default.task_value);
    config.rescue_default.start_time = read_number(rescue_block, "START_TIME", config.rescue_default.start_time);
    config.rescue_default.end_time = read_number(rescue_block, "END_TIME", config.rescue_default.end_time);
    config.rescue_default.duration = read_number(rescue_block, "DURATION", config.rescue_default.duration);

    const auto planner_block = extract_object(content, "PLANNER");
    if (!planner_block.empty()) {
        config.planner.type = read_string(planner_block, "TYPE", config.planner.type);
    }

    const auto dmg_block = extract_object(content, "DMG");
    if (!dmg_block.empty()) {
        config.dmg.enabled = read_bool(dmg_block, "ENABLED", config.dmg.enabled);
        config.dmg.mode = read_string(dmg_block, "MODE", config.dmg.mode);
        config.dmg.penalty_coef = read_number(dmg_block, "PENALTY_COEF", config.dmg.penalty_coef);
        config.dmg.max_tasks = read_int(dmg_block, "MAX_TASKS", config.dmg.max_tasks);
    }

    return config;
}

std::shared_ptr<Planner> create_planner(const Config& config) {
    if (config.planner.type == "euclidean") {
        return std::make_shared<EuclideanPlanner>();
    }
    throw std::runtime_error("Unsupported planner type: " + config.planner.type);
}

}  // namespace cbba
