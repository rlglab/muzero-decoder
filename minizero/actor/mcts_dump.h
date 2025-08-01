#pragma once

#include "environment.h"
#include "mcts.h"
#include <iostream>
#include <string>
#include <vector>

namespace minizero::actor {

class MCTSDump {
public:
    void dumpMCTS(const Environment& env, MCTSNode* root);

protected:
    void dumpMCTSInfo(MCTSNode* node, std::ostream& dump, Environment env, MCTSNode* root, std::vector<Action> actions, bool is_legal = true, bool is_already_terminal = false, const bool plot_all_child = false);
    void setDumpId(const Environment& env);
    void reset();
    void checkConfig() const;
    std::string getCurrentTreeName() const { return "game" + std::to_string(game_id_) + "_move" + std::to_string(tree_id_); }

    std::string dir_name_;
    int game_id_;
    int tree_id_;
    int tree_step_;
    std::ofstream tree_dump_;
    std::ofstream move_dump_;
    std::ofstream terminal_dump_;
    std::ofstream illegal_dump_;
    std::vector<Action> prev_actions_;
};

} // namespace minizero::actor
