#include "decoder_dump.h"
#include "actor_group.h"
#include "configuration.h"
#include "console.h"
#include "create_actor.h"
#include "create_network.h"
#include "git_info.h"
#include "mode_handler.h"
#include "ostream_redirector.h"
#include "random.h"
#include "sgf_loader.h"
#include "zero_server.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <set>
#include <sstream>
#include <string>
#include <torch/cuda.h>
#include <vector>

namespace minizero::utils {

void runDecoderAnalysis()
{
    std::shared_ptr<network::MuZeroNetwork> muzero_network = std::static_pointer_cast<network::MuZeroNetwork>(network::createNetwork(config::nn_file_name, 0));
    std::ifstream sgf_file(config::decoder_sgf_file_path);
    std::string sgf;
    Environment env;
    float mse_sum = 0;
    size_t mse_count = 0;
    while (std::getline(sgf_file, sgf)) {
        EnvironmentLoader env_loader;
        env_loader.loadFromString(sgf);
#if ATARI
        env.reset(std::stoi(env_loader.getTag("SD")));
        const size_t batch_size = 256;
        std::vector<std::vector<float>> batch_features;

        const auto& action_pairs = env_loader.getActionPairs();
        for (size_t n = 0; n < action_pairs.size(); ++n) {
            env.act(action_pairs[n].first);
            batch_features.push_back(env.getFeatures());

            if (batch_features.size() == batch_size || n == action_pairs.size() - 1) {
                std::vector<int> batch_ids;
                for (const auto& feat : batch_features) {
                    batch_ids.push_back(muzero_network->pushBackInitialData(feat));
                }
                std::vector<std::shared_ptr<network::NetworkOutput>> outputs = muzero_network->initialInference();

                for (size_t i = 0; i < batch_ids.size(); ++i) {
                    int batch_id = batch_ids[i];
                    const std::vector<float>& input_features = batch_features[i];
                    auto muzero_output = std::static_pointer_cast<minizero::network::MuZeroNetworkOutput>(outputs[batch_id]);
                    const std::vector<float>& decoder_output = muzero_output->decoder_output_;

                    int R_offset = -3 * minizero::env::atari::kAtariResolution * minizero::env::atari::kAtariResolution;
                    int G_offset = -2 * minizero::env::atari::kAtariResolution * minizero::env::atari::kAtariResolution;
                    int B_offset = -1 * minizero::env::atari::kAtariResolution * minizero::env::atari::kAtariResolution;
                    for (int j = 0; j < minizero::env::atari::kAtariResolution * minizero::env::atari::kAtariResolution; ++j) {
                        float R_error = input_features[input_features.size() + R_offset + j] - decoder_output[decoder_output.size() + R_offset + j];
                        mse_sum += (R_error * R_error);
                        mse_count += 1;
                        float G_error = input_features[input_features.size() + G_offset + j] - decoder_output[decoder_output.size() + G_offset + j];
                        mse_sum += (G_error * G_error);
                        mse_count += 1;
                        float B_error = input_features[input_features.size() + B_offset + j] - decoder_output[decoder_output.size() + B_offset + j];
                        mse_sum += (B_error * B_error);
                        mse_count += 1;
                    }
                }

                batch_features.clear();
            }
        }
#elif GO
        env.reset();
        const size_t batch_size = 256;
        std::vector<std::vector<float>> batch_features;

        const auto& action_pairs = env_loader.getActionPairs();
        for (size_t n = 0; n < action_pairs.size(); ++n) {
            env.act(action_pairs[n].first);
            batch_features.push_back(env.getFeatures());

            if (batch_features.size() == batch_size || n == action_pairs.size() - 1) {
                std::vector<int> batch_ids;
                for (const auto& feat : batch_features) {
                    batch_ids.push_back(muzero_network->pushBackInitialData(feat));
                }
                std::vector<std::shared_ptr<network::NetworkOutput>> outputs = muzero_network->initialInference();

                for (size_t i = 0; i < batch_ids.size(); ++i) {
                    int batch_id = batch_ids[i];
                    const std::vector<float>& input_features = batch_features[i];
                    auto muzero_output = std::static_pointer_cast<minizero::network::MuZeroNetworkOutput>(outputs[batch_id]);
                    const std::vector<float>& decoder_output = muzero_output->decoder_output_;

                    bool is_black_turn = input_features.back() == 0;
                    int black_start = is_black_turn ? 0 : config::env_board_size * config::env_board_size;
                    int white_start = is_black_turn ? config::env_board_size * config::env_board_size : 0;
                    for (int j = 0; j < config::env_board_size * config::env_board_size; ++j) {
                        float black_error = input_features[black_start + j] - decoder_output[black_start + j];
                        mse_sum += (black_error * black_error);
                        mse_count += 1;
                        float white_error = input_features[white_start + j] - decoder_output[white_start + j];
                        mse_sum += (white_error * white_error);
                        mse_count += 1;
                    }
                }

                batch_features.clear();
            }
        }
#else
        assert(false);
#endif
        std::cerr << ".";
    }
    std::cerr << std::endl;
    std::cout << (mse_sum / mse_count) << std::endl;
}

void runDecoderDumpTool()
{
    // varibles
    int zero_num_games_per_iteration = config::zero_num_games_per_iteration;
    int decoder_eval_sample_num = config::decoder_eval_sample_num;
    int learner_muzero_unrolling_step = config::learner_muzero_unrolling_step;
    bool is_decoder = false;

    std::ifstream sgfFile(config::decoder_sgf_file_path);
    std::string sgf_string;
    std::string model_info = get_model_info();

    Environment env;
    EnvironmentLoader env_loader;
    std::shared_ptr<network::MuZeroNetwork> muzero_network = nullptr;
    std::shared_ptr<network::Network> network = network::createNetwork(config::nn_file_name, 0); // TODO: use more GPUs
    bool is_atari = (network->getNetworkTypeName() == "muzero_atari");
    if (network->getNetworkTypeName() == "muzero" || is_atari) {
        std::ofstream outFile(config::decoder_out_file_path + "/" + model_info + "-target_dump.txt");
        std::vector<std::ofstream> outFile_vec;
        muzero_network = std::static_pointer_cast<network::MuZeroNetwork>(network);
        std::vector<int> randon_indices = generate_randoms(zero_num_games_per_iteration, decoder_eval_sample_num);
        int game_cnt_ = 1;
        int random_idx = 0;
        int max_game_length = 0;
        std::string unroll_label;

        if (learner_muzero_unrolling_step == -1) {
            while (getline(sgfFile, sgf_string)) {
                if (game_cnt_ == randon_indices[random_idx]) {
                    env_loader.loadFromString(sgf_string);
                    if (max_game_length < static_cast<int>(env_loader.getActionPairs().size())) { max_game_length = env_loader.getActionPairs().size(); }
                    ++random_idx;
                }
                ++game_cnt_;
            }
            sgfFile.clear();                 // Clear any error flags
            sgfFile.seekg(0, std::ios::beg); // Move the file position to the beginning
            unroll_label = "to end";
        } else {
            max_game_length = learner_muzero_unrolling_step;
            unroll_label = std::to_string(learner_muzero_unrolling_step);
        }

        std::cerr << "max unroll length:" << max_game_length << std::endl;

        for (int i = 0; i < max_game_length; i++) {
            outFile_vec.push_back(std::ofstream(config::decoder_out_file_path + "/" + model_info + "-unroll_" + std::to_string(i + 1) + "_dump.txt"));
            outFile_vec.back() << format_toString(unroll_label, decoder_eval_sample_num, i + 1);
        }
        outFile << format_toString(unroll_label, decoder_eval_sample_num);

        game_cnt_ = 1;
        random_idx = 0;
        while (getline(sgfFile, sgf_string)) {
            if (game_cnt_ == randon_indices[random_idx]) {
                outFile << "Game " << std::to_string(randon_indices[random_idx]) << ':' << std::endl;

                for (int k = 0; k < max_game_length; ++k) {
                    outFile_vec[k] << "Game " << std::to_string(randon_indices[random_idx]) << ':' << std::endl;
                }

                env_loader.loadFromString(sgf_string);
                env.reset();

                muzero_network->pushBackInitialData(env.getFeatures());
                std::shared_ptr<network::NetworkOutput> network_output = muzero_network->initialInference()[0];
                std::shared_ptr<network::MuZeroNetworkOutput> zero_output = std::static_pointer_cast<minizero::network::MuZeroNetworkOutput>(network_output);
                std::vector<float> hidden_state = zero_output->hidden_state_;
                std::vector<float> decoder_output = zero_output->decoder_output_;
                is_decoder = decoder_output.size() != 0;
                // print target data
                std::vector<float> env_feature = env.getFeatures();
                outFile << "step " << 0 << ": "
                        << "NONE,NONE," << vector_toString(env_feature) << std::endl;
                dump_target(outFile, EvalOutput(zero_output->policy_, zero_output->value_, hidden_state, decoder_output));

                for (size_t i = 0; i < env_loader.getActionPairs().size(); ++i) { // unroll to the end
                    std::vector<EvalOutput> evalout_vec;
                    int unroll_num;
                    if ((i + learner_muzero_unrolling_step > env_loader.getActionPairs().size()) || (learner_muzero_unrolling_step == -1)) {
                        unroll_num = env_loader.getActionPairs().size() - i;
                    } else {
                        unroll_num = learner_muzero_unrolling_step;
                    }

                    for (int k = 0; k < unroll_num; ++k) {
                        auto action = env_loader.getActionPairs()[i + k].first;
                        muzero_network->pushBackRecurrentData(hidden_state, env.getActionFeatures(action));                                                                         // dynamics
                        std::shared_ptr<network::NetworkOutput> network_output_unroll = muzero_network->recurrentInference()[0];                                                    // dynamics
                        std::shared_ptr<network::MuZeroNetworkOutput> zero_output_unroll = std::static_pointer_cast<minizero::network::MuZeroNetworkOutput>(network_output_unroll); // dynamics
                        hidden_state = zero_output_unroll->hidden_state_;                                                                                                           // hidden state
                        std::vector<float> policy_unroll = zero_output_unroll->policy_;
                        float value_unroll = zero_output_unroll->value_;
                        std::vector<float> decoder_output_unroll;
                        if (is_decoder) { decoder_output_unroll = zero_output_unroll->decoder_output_; }
                        evalout_vec.push_back(EvalOutput(policy_unroll, value_unroll, hidden_state, decoder_output_unroll));
                    }
                    // print unroll data
                    for (size_t k = 0; k < evalout_vec.size(); ++k) {
                        outFile_vec[k] << "step " << i + 1 + k << ":" << std::endl;
                        dump_unroll(outFile_vec[k], evalout_vec[k], k + 1);
                    }
                    evalout_vec.clear();

                    auto action = env_loader.getActionPairs()[i].first;
                    env.act(action);                                                                                // action to env
                    muzero_network->pushBackInitialData(env.getFeatures());                                         // representation
                    network_output = muzero_network->initialInference()[0];                                         // representation
                    zero_output = std::static_pointer_cast<minizero::network::MuZeroNetworkOutput>(network_output); // representation
                    hidden_state = zero_output->hidden_state_;
                    if (is_decoder) { decoder_output = zero_output->decoder_output_; }
                    // print target data
                    std::vector<float> env_feature = env.getFeatures();
                    if (is_atari)
                        outFile << "step " << i + 1 << ": " << action.getActionID() << ","
                                << "NONE"
                                << "," << vector_toString(env_feature) << std::endl;
                    else
                        outFile << "step " << i + 1 << ": " << action.getActionID() << ","
                                << utils::SGFLoader::actionIDToBoardCoordinateString(action.getActionID(), config::env_board_size)
                                << "," << vector_toString(env_feature) << std::endl;
                    dump_target(outFile, EvalOutput(zero_output->policy_, zero_output->value_, hidden_state, decoder_output));
                }
                ++random_idx;
            }
            ++game_cnt_;
        }

        // close files
        outFile.close();
        for (int i = 0; i < max_game_length; i++) {
            outFile_vec[i].close();
        }
    } else {
        std::cerr << "This tool is for muzero." << std::endl;
    }
}

std::string format_toString(std::string unroll_num, int decoder_eval_sample_num)
{
    std::ostringstream oss;
    oss << "Unroll number:" << unroll_num << std::endl;
    oss << "Sample number:" << decoder_eval_sample_num << std::endl;
    oss << "Annotation:" << std::endl;
    oss << "step i: [action ID],[action coordinate],[observation features]" << std::endl;
    oss << "[H] "
        << "[hidden state]" << std::endl;
    oss << "[D] "
        << "[decode output]" << std::endl;
    oss << "[P] "
        << "[policy]" << std::endl;
    oss << "[V] "
        << "[value]" << std::endl;
    oss << std::endl;
    return oss.str();
}

std::string format_toString(std::string unroll_num, int decoder_eval_sample_num, int k)
{
    std::ostringstream oss;
    oss << "Unroll number:" << unroll_num << std::endl;
    oss << "Sample number:" << decoder_eval_sample_num << std::endl;
    oss << "Annotation:" << std::endl;
    oss << "step i:" << std::endl;
    oss << "[U" << k << " H] "
        << "[unroll " << k << " hidden state]" << std::endl;
    oss << "[U" << k << " D] "
        << "[unroll " << k << " decode output]" << std::endl;
    oss << "[U" << k << " P] "
        << "[unroll " << k << " policy]" << std::endl;
    oss << "[U" << k << " V] "
        << "[unroll " << k << " value]" << std::endl;
    oss << std::endl;
    return oss.str();
}

std::string vector_toString(std::vector<float>& v1)
{
    std::ostringstream oss;
    if (v1.size() > 0) {
        for (size_t i = 0; i < v1.size() - 1; ++i) {
            oss << v1[i] << " ";
        }
        oss << v1[v1.size() - 1];
    }
    return oss.str();
}

void dump_target(std::ofstream& outFile, EvalOutput evaloutput)
{
    outFile << "[H] " << vector_toString(evaloutput.hidden_state) << std::endl;
    outFile << "[D] " << vector_toString(evaloutput.board_feature) << std::endl;
    outFile << "[P] " << vector_toString(evaloutput.policy) << std::endl;
    outFile << "[V] " << evaloutput.value << std::endl;
}

void dump_unroll(std::ofstream& outFile, EvalOutput evaloutput, int k)
{
    outFile << "[U" << k << " H] " << vector_toString(evaloutput.hidden_state) << std::endl;
    outFile << "[U" << k << " D] " << vector_toString(evaloutput.board_feature) << std::endl;
    outFile << "[U" << k << " P] " << vector_toString(evaloutput.policy) << std::endl;
    outFile << "[U" << k << " V] " << evaloutput.value << std::endl;
}

std::vector<int> generate_randoms(int zero_num_games_per_iteration, int decoder_eval_sample_num)
{
    std::vector<int> randomNumbers;

    for (int i = 1; i <= zero_num_games_per_iteration; ++i) {
        randomNumbers.push_back(i);
    }

    std::shuffle(randomNumbers.begin(), randomNumbers.end(), utils::Random::generator_);
    std::vector<int> selectedNumbers(randomNumbers.begin(), randomNumbers.begin() + decoder_eval_sample_num);
    std::sort(selectedNumbers.begin(), selectedNumbers.end());
    return selectedNumbers;
}

std::string get_model_info()
{
    std::string part1 = replace_string(config::nn_file_name.substr(config::nn_file_name.rfind('/') + 1), ".pt", "");
    std::string part2 = replace_string(config::decoder_sgf_file_path.substr(config::decoder_sgf_file_path.rfind('/') + 1), ".sgf", "");
    std::string model_info = std::string(GIT_SHORT_HASH) + "-" + part1 + "-sgf_" + part2;
    std::cerr << model_info << std::endl;
    return model_info;
}

std::string replace_string(std::string s1, std::string w1, std::string r1)
{
    size_t pos = s1.find(w1);
    if (pos != std::string::npos) {
        s1.replace(pos, w1.length(), r1);
    }
    return s1;
}

void runPlotter()
{
#if ATARI
    std::cerr << "This tool is for board games." << std::endl;
#else
    namespace py = pybind11;
    py::initialize_interpreter();
    py::module plot_board = py::module::import("tools.plot_board");
    py::object PlotBoard = plot_board.attr("PlotBoard");

    std::shared_ptr<network::Network> network = network::createNetwork(config::nn_file_name, 0);
    std::shared_ptr<network::MuZeroNetwork> muzero_network = std::static_pointer_cast<network::MuZeroNetwork>(network);

    auto stones = [](const std::vector<float>& features, char who) -> std::vector<float> {
        int num_each_channel = features.size() / Environment().getNumInputChannels();
        std::vector<float> black, white;
        if (features.back() < 0.5) { // assume black
            black.insert(black.end(), features.begin(), features.begin() + num_each_channel);
            white.insert(white.end(), features.begin() + num_each_channel, features.begin() + num_each_channel * 2);
        } else {
            white.insert(white.end(), features.begin(), features.begin() + num_each_channel);
            black.insert(black.end(), features.begin() + num_each_channel, features.begin() + num_each_channel * 2);
        }
        return who == 'B' ? black : white;
    };

    std::string game = Environment().name();
    game = game.substr(0, game.find('_'));

    std::map<std::string, std::shared_ptr<network::MuZeroNetworkOutput>> network_output_cache;

    std::string line;
    while (std::getline(std::cin, line)) {
        std::string name = line.substr(0, line.find(':'));
        size_t num_env_steps = std::stoi(name.substr(name.find("move") + 4));
        std::stringstream moves(line.substr(line.find(':') + 1));
        std::vector<Action> actions;
        std::string turn = "B";
        std::string moves_name = name;
        for (std::string a; moves >> a; turn = (turn == "B") ? "W" : "B") {
            actions.push_back(Action({turn, a}));
            moves_name += '_';
            moves_name += a;
        }
        Environment env;
        for (size_t i = 0; i < num_env_steps; i++) {
            env.act(actions[i]);
        }

        // exact environment
        Environment env_full = env;
        for (size_t i = num_env_steps; i < actions.size(); i++) {
            env_full.act(actions[i]);
        }
        std::cout << env_full.toString();
        auto features = env_full.getFeatures();
        std::cout << config::decoder_out_file_path << "/" << (moves_name + "_env.png") << std::endl;
        PlotBoard.attr("plot_board")(game, env.getBoardSize(), py::cast(stones(features, 'B')), py::cast(stones(features, 'W')));
        PlotBoard.attr("savefig")(config::decoder_out_file_path, moves_name + "_env.png", game);

        // decoder output
        std::shared_ptr<network::MuZeroNetworkOutput> zero_output;
        int fast_forward = -1, offset = 0;
        std::string test_moves_name = moves_name;
        while (test_moves_name.find("move") != std::string::npos && fast_forward == -1) {
            auto it = network_output_cache.find(test_moves_name);
            if (it != network_output_cache.end()) {
                fast_forward = actions.size() - offset;
                zero_output = it->second;
                std::cerr << "cache hit: " << test_moves_name << std::endl;
            } else {
                test_moves_name = test_moves_name.substr(0, test_moves_name.find_last_of('_'));
                offset++;
            }
        }
        float policy = 0.0 / 0.0, value = 0.0 / 0.0;
        if (!zero_output) {
            std::cerr << "run initial" << std::endl;
            muzero_network->pushBackInitialData(env.getFeatures());
            zero_output = std::static_pointer_cast<minizero::network::MuZeroNetworkOutput>(muzero_network->initialInference()[0]);
            value = zero_output->value_;
        }
        for (size_t i = (fast_forward != -1 ? fast_forward : num_env_steps); i < actions.size(); i++) {
            std::cerr << "run recurrent: " << actions[i].toConsoleString() << std::endl;
            policy = zero_output->policy_[actions[i].getActionID()];
            muzero_network->pushBackRecurrentData(zero_output->hidden_state_, env.getActionFeatures(actions[i]));
            zero_output = std::static_pointer_cast<minizero::network::MuZeroNetworkOutput>(muzero_network->recurrentInference()[0]);
            value = zero_output->value_;
        }
        std::cerr << (actions.size() ? actions.back().toConsoleString() : "INIT") << " policy: " << policy << " value: " << value << std::endl;
        network_output_cache.insert({moves_name, zero_output}); // save output for fast forward
        auto& decoder = zero_output->decoder_output_;
        std::cout << config::decoder_out_file_path << "/" << (moves_name + "_decoder.png") << std::endl;
        PlotBoard.attr("plot_board")(game, env.getBoardSize(), py::cast(stones(decoder, 'B')), py::cast(stones(decoder, 'W')));
        PlotBoard.attr("savefig")(config::decoder_out_file_path, moves_name + "_decoder.png", game);
    }
#endif
}

void runTreeDumpAnalysis()
{
    std::ifstream terminal_dump(config::decoder_sgf_file_path + "/terminal_dump.txt");
    std::map<std::string, std::vector<std::string>> terminals;
    std::string terminal;
    while (std::getline(terminal_dump, terminal)) {
        std::string name = terminal.substr(0, terminal.find(':'));
        std::string moves = terminal.substr(terminal.find(':') + 2);
        if (!terminals.count(name)) terminals.insert({name, {}});
        terminals[name].push_back(moves);
    }
    std::ifstream illegal_dump(config::decoder_sgf_file_path + "/illegal_dump.txt");
    std::map<std::string, std::vector<std::string>> illegals;
    std::string illegal;
    while (std::getline(illegal_dump, illegal)) {
        std::string name = illegal.substr(0, illegal.find(':'));
        std::string moves = illegal.substr(illegal.find(':') + 2);
        if (!illegals.count(name)) illegals.insert({name, {}});
        illegals[name].push_back(moves);
    }

    std::set<std::string> names;
    for (const auto& terminal : terminals) {
        names.insert(terminal.first);
    }
    for (const auto& illegal : illegals) {
        names.insert(illegal.first);
    }

    for (auto name : names) {
        std::set<std::string> all;
        if (terminals.count(name)) {
            for (const std::string& term : terminals[name]) {
                all.insert(term);
            }
        }
        if (illegals.count(name)) {
            for (const std::string& ill : illegals[name]) {
                all.insert(ill);
            }
        }

        int num_exact_terminal = 0, num_cover_terminal = 0;
        if (terminals.count(name)) {
            for (const std::string& term : terminals[name]) {
                bool is_cover = false;
                for (const std::string& check : all) { // check term is not covered by any other
                    if (term.size() > check.size() && term.find(check) == 0) {
                        is_cover = true;
                        break;
                    }
                }
                if (is_cover) {
                    num_cover_terminal++;
                } else {
                    num_exact_terminal++;
                }
            }
        }
        int num_exact_illegal = 0, num_cover_illegal = 0;
        if (illegals.count(name)) {
            for (const std::string& ill : illegals[name]) {
                bool is_cover = false;
                for (const std::string& check : all) { // check ill is not covered by any other
                    if (ill.size() > check.size() && ill.find(check) == 0) {
                        is_cover = true;
                        break;
                    }
                }
                if (is_cover) {
                    num_cover_illegal++;
                } else {
                    num_exact_illegal++;
                }
            }
        }
        std::cout << name << ": " << num_exact_terminal << ' ' << num_exact_illegal << ' ' << (all.size() - num_exact_terminal) << std::endl;
    }
}

} // namespace minizero::utils
