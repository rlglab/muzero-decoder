#pragma once

#include <memory>
#include <string>
#include <vector>

namespace minizero::utils {

class EvalOutput {
public:
    std::vector<float> policy;
    float value;
    std::vector<float> hidden_state;
    std::vector<float> board_feature;

    EvalOutput(const std::vector<float>& _policy, float _value, const std::vector<float>& _hidden_state, const std::vector<float>& _board_feature)
        : policy(_policy), value(_value), hidden_state(_hidden_state), board_feature(_board_feature)
    {
    }
};

void runDecoderAnalysis();
void runDecoderDumpTool();
std::string format_toString(std::string unroll_num, int decoder_eval_sample_num);
std::string format_toString(std::string unroll_num, int decoder_eval_sample_num, int k);
std::string vector_toString(std::vector<float>& v1);
std::string replace_string(std::string s1, std::string w1, std::string r1);
std::string get_model_info();
std::vector<int> generate_randoms(int zero_num_games_per_iteration, int decoder_eval_sample_num);
void dump_target(std::ofstream& outFile, EvalOutput evaloutput);
void dump_unroll(std::ofstream& outFile, EvalOutput evaloutput, int k);
void runPlotter();
void runTreeDumpAnalysis();

} // namespace minizero::utils
