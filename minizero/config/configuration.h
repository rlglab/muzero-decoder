#pragma once

#include "configure_loader.h"
#include <string>

namespace minizero::config {

// program parameters
extern int program_seed;
extern bool program_auto_seed;
extern bool program_quiet;

// actor parameters
extern int actor_num_simulation;
extern float actor_mcts_puct_base;
extern float actor_mcts_puct_init;
extern float actor_mcts_reward_discount;
extern int actor_mcts_think_batch_size;
extern float actor_mcts_think_time_limit;
extern bool actor_mcts_value_rescale;
extern char actor_mcts_value_flipping_player;
extern bool actor_select_action_by_count;
extern bool actor_select_action_by_softmax_count;
extern float actor_select_action_softmax_temperature;
extern bool actor_select_action_softmax_temperature_decay;
extern bool actor_use_random_rotation_features;
extern bool actor_use_dirichlet_noise;
extern float actor_dirichlet_noise_alpha;
extern float actor_dirichlet_noise_epsilon;
extern bool actor_use_gumbel;
extern bool actor_use_gumbel_noise;
extern int actor_gumbel_sample_size;
extern float actor_gumbel_sigma_visit_c;
extern float actor_gumbel_sigma_scale_c;
extern float actor_resign_threshold;
extern bool actor_dump_mcts_tree;
extern bool actor_dump_board_in_mcts_node;

// zero parameters
extern int zero_num_threads;
extern int zero_num_parallel_games;
extern int zero_server_port;
extern std::string zero_training_directory;
extern int zero_num_games_per_iteration;
extern int zero_start_iteration;
extern int zero_end_iteration;
extern int zero_replay_buffer;
extern float zero_disable_resign_ratio;
extern int zero_actor_intermediate_sequence_length;
extern std::string zero_actor_ignored_command;
extern bool zero_server_accept_different_model_games;

// learner parameters
extern bool learner_use_per;
extern float learner_per_alpha;
extern float learner_per_init_beta;
extern bool learner_per_beta_anneal;
extern int learner_training_step;
extern int learner_training_display_step;
extern int learner_batch_size;
extern int learner_muzero_unrolling_step;
extern int learner_n_step_return;
extern float learner_learning_rate;
extern float learner_momentum;
extern float learner_weight_decay;
extern float learner_value_loss_scale;
extern bool learner_use_state_consistency;
extern bool learner_use_decoder;
extern int learner_num_thread;

// network parameters
extern std::string nn_file_name;
extern int nn_num_blocks;
extern int nn_num_hidden_channels;
extern int nn_num_value_hidden_channels;
extern int nn_state_consistency_proj_hid;
extern int nn_state_consistency_proj_out;
extern int nn_state_consistency_pred_hid;
extern int nn_state_consistency_pred_out;
extern std::string nn_type_name;

// decoder parameters
extern std::string decoder_sgf_file_path;
extern std::string decoder_out_file_path;
extern int decoder_eval_sample_num;
extern bool decoder_output_at_inference;
extern float decoder_loss_scale;
extern float decoder_clip_grad_value;

// environment parameters
extern int env_board_size;
extern std::string env_atari_rom_dir;
extern std::string env_atari_name;
extern float env_go_komi;
extern std::string env_go_ko_rule;
extern bool env_killallgo_use_seki;
extern std::string env_gomoku_rule;
extern bool env_gomoku_exactly_five_stones;
extern bool env_hex_use_swap_rule;
extern int env_rubiks_scramble_rotate;

void setConfiguration(ConfigureLoader& cl);

} // namespace minizero::config
