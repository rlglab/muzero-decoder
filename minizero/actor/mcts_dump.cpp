#include "mcts_dump.h"
#include <algorithm>
#include <csignal>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace minizero::actor {

void MCTSDump::dumpMCTS(const Environment& env, MCTSNode* root)
{
    if (!tree_dump_.is_open()) { reset(); }
    setDumpId(env);

    std::stringstream dump;
    dumpMCTSInfo(root, dump, env, root, env.getActionHistory(), true, false, false);

    std::stringstream ss;
    ss << "tree " << getCurrentTreeName() << "\n";
    ss << dump.str() << "\n\n";

    tree_dump_ << ss.str() << std::flush;
}

void MCTSDump::dumpMCTSInfo(MCTSNode* node, std::ostream& dump, Environment env, MCTSNode* root, std::vector<Action> actions, bool is_legal, bool is_already_terminal, const bool plot_all_child /* = false */)
{
    dump << int(node - root);
    dump << '[';
    bool has_child = false;
    for (int i = 0; i < node->getNumChildren(); i++) {
        if (!plot_all_child && node->getChild(i)->getCount() == 0) continue;
        dump << int(node->getChild(i) - root) << ',';
        has_child = true;
    }
    if (has_child) dump.seekp(-1, std::ios::cur);
    dump << ']';
    dump << '{';
    bool legal_act = true;
    if (node == root) {
        dump << "ROOT" << ',';
    } else {
        legal_act = env.act(node->getAction());
        actions.push_back(node->getAction());
        std::string action_string = node->getAction().toConsoleString();
        dump << (node->getAction().getPlayer() == minizero::env::Player::kPlayer1 ? 'B' : 'W') << '[' << action_string << ']' << ',';
    }

    if (config::actor_dump_board_in_mcts_node) {
        std::string env_str = env.toString();
        std::replace(env_str.begin(), env_str.end(), '\n', ',');
        dump << env_str << ',';
    }
    dump << "q=" << node->getMean() << ',';
    dump << "n=" << node->getCount() << ',';
    dump << "p=" << node->getPolicy() << ',';
    dump << "v=" << node->getValue() << ',';
    dump << "r=" << node->getReward();
    dump << '}';

    // node visual info
    bool is_terminal = env.isTerminal();
    dump << '{';
    dump << "color=" << (legal_act ? (is_terminal ? "lightblue" : "white") : "lightpink");
    dump << '}';
    std::stringstream dump_data_ss;
    dump_data_ss << getCurrentTreeName() << ":";
    for (Action a : actions) { dump_data_ss << " " << a.toConsoleString(); }
    std::string dump_data = dump_data_ss.str();
    move_dump_ << dump_data << std::endl;
    if (!(is_legal & legal_act)) { illegal_dump_ << dump_data << std::endl; }
    if (is_already_terminal | is_terminal) { terminal_dump_ << dump_data << std::endl; }

    for (int i = 0; i < node->getNumChildren(); i++) {
        if (!plot_all_child && node->getChild(i)->getCount() == 0) continue;
        dumpMCTSInfo(node->getChild(i), dump << std::endl, env, root, actions, is_legal & legal_act, is_already_terminal | is_terminal, plot_all_child);
    }
}

void MCTSDump::setDumpId(const Environment& env)
{
    std::vector<Action> curr_actions = env.getActionHistory();

    bool is_common = true;
    for (size_t i = 0; i < std::min(prev_actions_.size(), curr_actions.size()); i++) {
        if (prev_actions_[i].getActionID() != curr_actions[i].getActionID()) {
            is_common = false;
            break;
        }
    }

    if (prev_actions_.size() == curr_actions.size() && is_common) {
        // same game, same move, new step
        tree_step_++;
    } else if (prev_actions_.size() < curr_actions.size() && is_common) {
        // same game, new move
        tree_id_ = curr_actions.size();
        tree_step_ = 0;
    } else if (prev_actions_.size() > curr_actions.size() || !is_common) {
        // new game
        game_id_++;
        tree_id_ = 0;
        tree_step_ = 0;
    }

    prev_actions_ = std::move(curr_actions);
}

void MCTSDump::reset()
{
    checkConfig();
    std::string dump_dir_name = config::zero_training_directory.size() ? config::zero_training_directory : "./tree_dump";
    if (!std::filesystem::exists(dump_dir_name)) { std::filesystem::create_directory(dump_dir_name); }

    // std::time_t now = std::time(nullptr);
    // std::stringstream ss;
    // ss << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
    // std::string time_str = ss.str();
    // dir_name_ = dump_dir_name + "/" + time_str;
    // std::filesystem::create_directory(dir_name_);
    dir_name_ = dump_dir_name;

    tree_dump_.open(dir_name_ + "/tree_dump.txt");
    if (!tree_dump_.is_open()) {
        std::cerr << "Failed to open " << dir_name_ + "/tree_dump.txt" << std::endl;
        exit(1);
    }
    move_dump_.open(dir_name_ + "/move_dump.txt");
    if (!move_dump_.is_open()) {
        std::cerr << "Failed to open " << dir_name_ + "/move_dump.txt" << std::endl;
        exit(1);
    }
    terminal_dump_.open(dir_name_ + "/terminal_dump.txt");
    if (!terminal_dump_.is_open()) {
        std::cerr << "Failed to open " << dir_name_ + "/terminal_dump.txt" << std::endl;
        exit(1);
    }
    illegal_dump_.open(dir_name_ + "/illegal_dump.txt");
    if (!illegal_dump_.is_open()) {
        std::cerr << "Failed to open " << dir_name_ + "/illegal_dump.txt" << std::endl;
        exit(1);
    }

    game_id_ = 0;
    tree_id_ = 0;
    tree_step_ = -1;
    prev_actions_.clear();
}

void MCTSDump::checkConfig() const
{
    if (config::zero_num_threads > 1) {
        std::cerr << "zero_num_threads should be 1 when dump MCTS" << std::endl;
        exit(1);
    }
    if (config::zero_num_parallel_games > 1) {
        std::cerr << "zero_num_parallel_games should be 1 when dump MCTS" << std::endl;
        exit(1);
    }
}

} // namespace minizero::actor
