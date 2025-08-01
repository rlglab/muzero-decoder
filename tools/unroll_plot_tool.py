#!/usr/bin/env python
from tools.plot_board import PlotBoard
from minizero.network.py.create_network import create_network
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
# change sys path to load minizero and its package
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# import pybind library


def get_black_white_boards(board_state):
    channelsize = board_state.shape[0]
    colors = {'black': 0, 'white': 1} if (board_state[channelsize - 2].mean() > board_state[channelsize - 1].mean()) else {'black': 1, 'white': 0}
    black_board = board_state[colors['black']].reshape(-1)
    white_board = board_state[colors['white']].reshape(-1)
    return black_board.tolist(), white_board.tolist()


class Model:

    def __init__(self, devive):
        self.network = None
        self.device = devive

    def load_model(self, nn_file_name):
        self.network = torch.jit.load(nn_file_name, map_location=torch.device('cpu'))
        self.network.to(self.device)
        self.network.eval()


class DecoderDump:
    def __init__(self, out_dir, sgf_file, nn_file_name, game_type, unroll_history_num=6):
        self.sgf_file = sgf_file
        self.nn_file_name = nn_file_name
        self.game_type = game_type
        self.num_input_channels = py.get_nn_num_input_channels()
        self.input_channel_height = py.get_nn_input_channel_height()
        self.input_channel_width = py.get_nn_input_channel_width()
        self.num_hidden_channels = py.get_nn_num_hidden_channels()
        self.hidden_channel_height = py.get_nn_hidden_channel_height()
        self.hidden_channel_width = py.get_nn_hidden_channel_width()
        self.num_action_feature_channels = py.get_nn_num_action_feature_channels()
        self.feature_shape = (1, self.num_input_channels, self.input_channel_height, self.input_channel_width)
        self.action_feature_shape = (1, self.num_action_feature_channels, self.hidden_channel_height, self.hidden_channel_width)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Model(self.device)

        self.dump_num = 5
        self.process_batch_size = 4
        self.game_length = None
        self.Envs = []
        self.out_dir = out_dir
        self.unroll_one_outputs = {"decoder_output": [], "observation": [], "representation": [], "ids": [], "action": []}
        self.representation_outputs = {"value": [], "reward": []}
        self.dynamics_outputs = {"value": [], "reward": [], "total_reward": []}
        self.answers = {"value": [], "reward": [], "total_reward": []}
        self.k_dynamics_outputs = {"value": [], "reward": [], "total_reward": []}
        self.exist_reward = False
        self.unroll_history_num = unroll_history_num
        self.k_hidden_states_hisory = []
        self.k_network_output_hisory = []
        for _ in range(unroll_history_num):
            self.k_hidden_states_hisory.append(None)
            self.k_network_output_hisory.append(None)

    def clean_output(self):
        for key, _ in self.unroll_one_outputs.items():
            self.unroll_one_outputs[key] = []

    class Env_unit:
        def __init__(self, env, action_id, end_action_id):
            self.env = env
            self.action_id = action_id
            self.end_action_id = end_action_id
            self.hidden_state = None

        def act(self):
            action_ID_ = -1

            if (self.is_action_legal()):

                action = self.get_action()
                self.env.act(action)
                self.action_id += 1
                action_ID_ = action.get_action_id()

            return action_ID_

        def is_action_legal(self):
            if (self.action_id <= self.end_action_id - 1):
                return True
            else:
                return False

        def get_action(self):
            if (self.is_action_legal()):
                return env_loader.get_action(self.action_id)
            else:
                return None

        def update_hidden_state(self, new_hidden_state):
            self.hidden_state = new_hidden_state

    def get_reshaped_tensor(self, input_reshape_, input_feature_):
        feature = torch.tensor(input_feature_).to(self.device)
        features_reshaped = feature.view(input_reshape_)
        return features_reshaped

    def invert_value(self, value):
        epsilon = 0.001
        sign_value = 1.0 if value > 0.0 else (0.0 if value == 0.0 else -1.0)
        return sign_value * (((math.sqrt(1 + 4 * epsilon * (abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)

    def accumulate_output_array(self, input_array):
        start_value = -py.get_nn_discrete_value_size() // 2
        weighted_sum = 0.0
        for value_ in input_array:
            start_value += 1
            weighted_sum += (value_ * start_value)

        return self.invert_value(weighted_sum)

    def print_nokey_error(self):
        print()
        print("The model does not have key decoder_output")
        print("Please recreate the model with the key decoder_output")
        print()
        exit(0)

    def create_new_k_hisory(self):
        new_k_hisory = []
        for _ in range(self.unroll_history_num):
            new_k_hisory.append(None)
        return new_k_hisory

    def unroll_one_line(self, start_id, end_action_id):

        if self.model.network is None:
            self.model.load_model(self.nn_file_name)

        env = py.Env()
        if self.game_type == 'atari':
            seed = int(env_loader.get_tag("SD"))
            print(f'seed: {seed}')
            env.reset(seed)
            self.exist_reward = True
        tmp_id = 0
        while (tmp_id != start_id):
            env.act(env_loader.get_action(tmp_id))
            tmp_id += 1

        print(f'start_id: {start_id}, end_id: {end_action_id}')
        env_unit = self.Env_unit(env, start_id, end_action_id)
        feature = self.get_reshaped_tensor(self.feature_shape, env_unit.env.get_features())
        network_output = self.model.network(feature)

        # the values/rewards of the initial state
        value_ = self.accumulate_output_array(network_output["value"][0].cpu().detach().numpy())
        self.representation_outputs["value"].append(value_)

        env_unit.update_hidden_state(network_output["hidden_state"])
        self.k_hidden_states_hisory[-1] = network_output["hidden_state"]

        decode_output = None
        if "decoder_output" in network_output:
            decode_output = network_output["decoder_output"]
        else:
            self.print_nokey_error()

        if is_plot:
            self.plot_single_feature(env_unit.env.get_features(), f'observation_{env_unit.action_id}')
            self.plot_single_network_output(decode_output[0], f'representation_{env_unit.action_id}')

        # the values/rewards of the initial state
        self.answers["value"].append(0)
        if self.exist_reward:
            reward = env_unit.env.get_reward()
            self.answers["reward"].append(reward)
            self.answers["total_reward"].append(reward)
            self.dynamics_outputs["total_reward"].append(0)
            self.k_dynamics_outputs["total_reward"].append(0)

        while (env_unit.is_action_legal()):
            action_feature = env_unit.env.get_action_features(env_unit.get_action())
            action_feature = self.get_reshaped_tensor(self.action_feature_shape, action_feature)

            # dynamics
            network_output = self.model.network.recurrent_inference(env_unit.hidden_state, action_feature)
            if self.game_type == 'atari':
                value_ = self.accumulate_output_array(network_output["value"][0].cpu().detach().numpy())
            else:
                value_ = network_output["value"].cpu().detach().numpy()
                value_ = value_[0][0]
            self.dynamics_outputs["value"].append(value_)
            if self.exist_reward:
                reward_ = self.accumulate_output_array(network_output["reward"][0].cpu().detach().numpy())
                self.dynamics_outputs["reward"].append(reward_)
                self.dynamics_outputs["total_reward"].append(self.dynamics_outputs["total_reward"][-1] + reward_)

            hidden_state = network_output["hidden_state"]
            decode_output = network_output["decoder_output"]
            self.plot_single_network_output(decode_output[0], f'dynamics/dynamics_{str(env_unit.action_id + 1)}')

            self.unroll_one_outputs["ids"].append(env_unit.action_id + 1)
            env_unit.update_hidden_state(hidden_state)

            new_k_hidden_states_hisory = self.create_new_k_hisory()
            new_k_network_output_hisory = self.create_new_k_hisory()
            for i in range(self.unroll_history_num - 1):
                if self.k_hidden_states_hisory[i + 1] is not None:
                    network_output_ = self.model.network.recurrent_inference(self.k_hidden_states_hisory[i + 1], action_feature)
                    new_k_network_output_hisory[i] = network_output_
                    new_k_hidden_states_hisory[i] = network_output_["hidden_state"]

            # observation
            self.unroll_one_outputs["action"].append(env_unit.act())
            feature = env_unit.env.get_features()

            value = env_loader.get_action_pairs_value(env_unit.action_id - 1)
            self.answers["value"].append(value)
            if self.exist_reward:
                reward = env_unit.env.get_reward()
                self.answers["reward"].append(reward)
                self.answers["total_reward"].append(self.answers["total_reward"][-1] + reward)

            if is_plot:
                self.plot_single_feature(feature, f'observation/observation_{env_unit.action_id}')

            # representation
            feature = self.get_reshaped_tensor(self.feature_shape, feature)
            network_output = self.model.network.initial_inference(feature)
            new_k_hidden_states_hisory[-1] = network_output["hidden_state"]
            self.k_hidden_states_hisory = new_k_hidden_states_hisory
            self.k_network_output_hisory = new_k_network_output_hisory

            if self.game_type == 'atari':
                value_ = self.accumulate_output_array(network_output["value"][0].cpu().detach().numpy())
            else:
                value_ = network_output["value"].cpu().detach().numpy()
                value_ = value_[0][0]
            self.representation_outputs["value"].append(value_)

            decode_output = network_output["decoder_output"]
            if is_plot:
                self.plot_single_network_output(decode_output[0], f'representation/representation_{env_unit.action_id}')
                if self.k_hidden_states_hisory[0] is not None:
                    if self.exist_reward:
                        network_output_ = self.k_network_output_hisory[0]
                        reward_ = self.accumulate_output_array(network_output_["reward"][0].cpu().detach().numpy())
                        self.k_dynamics_outputs["reward"].append(reward_)
                        self.k_dynamics_outputs["total_reward"].append(self.k_dynamics_outputs["total_reward"][-1] + reward_)
                    unroll_k_decode_output = self.model.network.forward_decoder(self.k_hidden_states_hisory[0])
                    self.plot_single_network_output(unroll_k_decode_output[0], f'dynamics_k/dynamics_k_{env_unit.action_id}')
            if is_plot:
                print(f'\rPloting observation/representation/dynamics in action {env_unit.action_id}', end='')
        print()
        print('Output value/(reward) txt log files')

        # remove values of the initial state
        self.answers["value"] = self.answers["value"][1:]
        self.answers["reward"] = self.answers["reward"][1:]
        self.answers["total_reward"] = self.answers["total_reward"][1:]
        self.dynamics_outputs["total_reward"] = self.dynamics_outputs["total_reward"][1:]
        self.k_dynamics_outputs["total_reward"] = self.k_dynamics_outputs["total_reward"][1:]
        self.representation_outputs["value"] = self.representation_outputs["value"][1:]

        self.write_to_file(self.answers["value"], 'answser_value.txt')
        if self.exist_reward:
            self.write_to_file(self.answers["reward"], 'answser_reward.txt')
            self.write_to_file(self.answers["total_reward"], 'answser_total_reward.txt')
            self.write_to_file(self.k_dynamics_outputs["reward"], 'k_dynamics_reward.txt')
            self.write_to_file(self.k_dynamics_outputs["total_reward"], 'k_dynamics_total_reward.txt')

        self.write_to_file(self.representation_outputs["value"], 'representation_value.txt')
        self.write_to_file(self.dynamics_outputs["value"], 'dynamics_value.txt')
        if self.exist_reward:
            self.write_to_file(self.dynamics_outputs["reward"], 'dynamics_reward.txt')
            self.write_to_file(self.dynamics_outputs["total_reward"], 'dynamics_total_reward.txt')

    def write_to_file(self, content, output_name):
        with open(f'{self.out_dir}/{output_name}', 'w') as f:
            for c in content:
                f.write(str(c) + '\n')

    def plot_single_feature(self, feature, output_name):
        c = self.num_input_channels
        h = self.input_channel_height
        w = self.input_channel_width

        feature = np.array(feature)
        feature = feature.reshape(c, h, w)
        self.plot_game(feature, c)

        if self.game_type == 'atari':
            plt.axis('off')
        plt.savefig(f'{self.out_dir}/{output_name}.png')
        plt.close()

    def plot_single_network_output(self, feature, output_name):
        if self.game_type == 'atari':
            c = 3
        else:
            c = self.num_input_channels
        h = self.input_channel_height
        w = self.input_channel_width

        feature = feature.cpu().detach().numpy().reshape(c, h, w)
        self.plot_game(feature, c)

        if self.game_type == 'atari':
            plt.axis('off')
        plt.savefig(f'{self.out_dir}/{output_name}.png')
        plt.close()

    def get_black_white_boards(self, board_state):
        channelsize = board_state.shape[0]
        colors = {'black': 0, 'white': 1} if (board_state[channelsize - 2].mean() > board_state[channelsize - 1].mean()) else {'black': 1, 'white': 0}
        black_board = board_state[colors['black']].reshape(-1)
        white_board = board_state[colors['white']].reshape(-1)
        return black_board, white_board

    def plot_game(self, feature, c):
        if self.game_type == 'atari':
            image_r = feature[c - 3].clip(0, 1)
            image_g = feature[c - 2].clip(0, 1)
            image_b = feature[c - 1].clip(0, 1)
            plt.imshow((np.dstack((image_r, image_g, image_b)) * 255.999).astype(np.uint8))
        else:
            blackboard, whiteboard = self.get_black_white_boards(feature)
            PlotBoard.plot_board(self.game_type, self.input_channel_height, blackboard, whiteboard)


def exist_dir(path):
    if not os.path.isdir(f'{path}'):
        os.mkdir(f'{path}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-out_dir', dest='out_dir', type=str, default="", help='output directory (default: decoder_output-start_[start_id]-end_[end_id])')
    parser.add_argument('-conf_file', dest='conf_file', type=str, default="", help='path of config file')
    parser.add_argument('-nn_file_name', dest='nn_file_name', type=str, default="", help='path of model file')
    parser.add_argument('-game', dest='game_type', type=str, default="othello", help='game type (default: othello)')
    parser.add_argument('-sgf', dest='sgf_file', type=str, default="", help='path of sgf file')
    parser.add_argument('-l', dest='line_num', type=int, default=1, help='using which line in the input sgf file (default: 1)')
    parser.add_argument('-game_len', dest='game_len', type=int, default=0, help='max game length (default: none)')
    parser.add_argument('-start', dest='start_id', type=int, default=0, help='start action index (default: 0)')
    parser.add_argument('-plot', dest='is_plot', type=bool, default=True, help='True: plot game observation, False: not output plot (default: True)')
    parser.add_argument('-k', dest='k', type=int, default=5, help='unroll history num (default: 5)')
    args = parser.parse_args()

    game_type = args.game_type
    conf_file = args.conf_file
    nn_file_name = args.nn_file_name
    sgf_file = args.sgf_file
    start_id = args.start_id
    line_num = args.line_num
    is_plot = args.is_plot
    k = args.k + 1

    if game_type and conf_file and nn_file_name and sgf_file:

        # catch ModuleNotFoundError
        try:
            _temps = __import__(f'build.{game_type}', globals(), locals(), ['minizero_py'], 0)
        except ModuleNotFoundError:
            print(f'Require to build the game {game_type} first')

        py = _temps.minizero_py
        py.load_config_file(conf_file)
        env_loader = py.EnvLoader()

        out_dir = None
        if args.out_dir == "":
            end_id = str(start_id + args.game_len) if args.game_len != 0 else 'all'
            out_dir = f'decoder_output-start_{start_id}-end_{end_id}'
        else:
            out_dir = args.out_dir
        if out_dir:
            exist_dir(out_dir)

        exist_dir(f'{out_dir}/observation')
        exist_dir(f'{out_dir}/representation')
        exist_dir(f'{out_dir}/dynamics')
        exist_dir(f'{out_dir}/dynamics_k')

        decoder_dump = DecoderDump(out_dir, sgf_file, nn_file_name, game_type, k)
        with open(sgf_file, 'r') as file:
            l = 1
            for line in file:
                if (l == line_num):
                    line = line.strip()
                    env_loader.load_from_string(line)
                    end_id = start_id + args.game_len
                    action_pairs_size = env_loader.get_action_pairs_size()
                    end_id = min(action_pairs_size, end_id) if args.game_len != 0 else action_pairs_size
                    decoder_dump.unroll_one_line(start_id, end_id)
                    exit(0)
                l += 1
            if (l != line_num):
                print("given line is out of range")
    else:
        print()
        print("please input game type, config file, nn file name, sgf file")
        parser.print_help()
        exit(1)
