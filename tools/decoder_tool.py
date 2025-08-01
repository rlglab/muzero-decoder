#!/usr/bin/env python
from tools.plot_board import PlotBoard
from minizero.network.py.create_network import create_network
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
    def __init__(self, out_dir, sgf_file, nn_file_name):
        self.sgf_file = sgf_file
        self.nn_file_name = nn_file_name
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

    def unroll_one_line_pca(self, start_id, end_action_id):

        if self.model.network is None:
            self.model.load_model(self.nn_file_name)

        env = py.Env()
        tmp_id = 0
        while (tmp_id != start_id):
            env.act(env_loader.get_action(tmp_id))
            tmp_id += 1

        print(f'start_id: {start_id} ')
        env_unit = self.Env_unit(env, start_id, end_action_id)
        feature = self.get_reshaped_tensor(self.feature_shape, env_unit.env.get_features())
        network_output = self.model.network(feature)
        env_unit.update_hidden_state(network_output["hidden_state"])

        hidden_state_rep_list = []
        hidden_state_dyn_list = []

        while (env_unit.is_action_legal()):
            action_feature = env_unit.env.get_action_features(env_unit.get_action())
            action_feature = self.get_reshaped_tensor(self.action_feature_shape, action_feature)

            network_output = self.model.network.recurrent_inference(env_unit.hidden_state, action_feature)
            hidden_state = network_output["hidden_state"]
            # decode_output = network_output["decoder_output"]
            # self.unroll_one_outputs["decoder_output"].append(decode_output[0])
            self.unroll_one_outputs["ids"].append(env_unit.action_id + 1)
            env_unit.update_hidden_state(hidden_state)
            hidden_state_dyn_list.append(hidden_state.cpu().detach().numpy().flatten())

            self.unroll_one_outputs["action"].append(env_unit.act())
            feature = env_unit.env.get_features()
            feature = self.get_reshaped_tensor(self.feature_shape, feature)
            network_output = self.model.network.initial_inference(feature)
            # decode_output = network_output["decoder_output"]
            hidden_state_0 = network_output["hidden_state"]
            hidden_state_rep_list.append(hidden_state_0.cpu().detach().numpy().flatten())

            self.unroll_one_outputs["observation"].append(feature[0])
            # self.unroll_one_outputs["representation"].append(decode_output[0])

        hidden_state_rep_len = len(hidden_state_rep_list)
        hidden_state_rep_arr = np.array(hidden_state_rep_list)
        hidden_state_dyn_arr = np.array(hidden_state_dyn_list)

        combined_hidden_state = np.concatenate((hidden_state_rep_arr, hidden_state_dyn_arr), axis=0)
        pca = PCA(n_components=3)
        combined_hidden_state_3d = pca.fit_transform(combined_hidden_state)

        grip_len = 10
        # hidden_state_dyn_3d = combined_hidden_state_3d[:hidden_state_rep_len][:grip_len]
        # hidden_state_rep_3d = combined_hidden_state_3d[hidden_state_rep_len:][:grip_len]
        hidden_state_dyn_3d = combined_hidden_state_3d[:hidden_state_rep_len][::4]
        hidden_state_rep_3d = combined_hidden_state_3d[hidden_state_rep_len:][::4]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(hidden_state_dyn_3d[:, 0], hidden_state_dyn_3d[:, 1], hidden_state_dyn_3d[:, 2], c='green', marker='o', label='dynamics')
        ax.plot(hidden_state_rep_3d[:, 0], hidden_state_rep_3d[:, 1], hidden_state_rep_3d[:, 2], c='blue', marker='o', label='representation')

        for i, (xi, yi, zi) in enumerate(zip(hidden_state_dyn_3d[:, 0], hidden_state_dyn_3d[:, 1], hidden_state_dyn_3d[:, 2])):
            ax.text(xi, yi, zi, str(i + 1), fontsize=12)

        for i, (xi, yi, zi) in enumerate(zip(hidden_state_rep_3d[:, 0], hidden_state_rep_3d[:, 1], hidden_state_rep_3d[:, 2])):
            ax.text(xi, yi, zi, str(i + 1), fontsize=12)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.title('PCA of Hidden States')
        plt.legend()
        plt.savefig(f'pca.png')

    def unroll_one_line(self, start_id, end_action_id):

        if self.model.network is None:
            self.model.load_model(self.nn_file_name)

        env = py.Env()
        tmp_id = 0
        while (tmp_id != start_id):
            env.act(env_loader.get_action(tmp_id))
            tmp_id += 1

        print(f'start_id: {start_id} ')
        env_unit = self.Env_unit(env, start_id, end_action_id)
        feature = self.get_reshaped_tensor(self.feature_shape, env_unit.env.get_features())
        network_output = self.model.network(feature)
        env_unit.update_hidden_state(network_output["hidden_state"])

        while (env_unit.is_action_legal()):
            action_feature = env_unit.env.get_action_features(env_unit.get_action())
            action_feature = self.get_reshaped_tensor(self.action_feature_shape, action_feature)

            network_output = self.model.network.recurrent_inference(env_unit.hidden_state, action_feature)
            hidden_state = network_output["hidden_state"]
            decode_output = network_output["decoder_output"]
            self.unroll_one_outputs["decoder_output"].append(decode_output[0])
            self.unroll_one_outputs["ids"].append(env_unit.action_id + 1)
            env_unit.update_hidden_state(hidden_state)

            self.unroll_one_outputs["action"].append(env_unit.act())
            feature = env_unit.env.get_features()
            feature = self.get_reshaped_tensor(self.feature_shape, feature)
            network_output = self.model.network.initial_inference(feature)
            decode_output = network_output["decoder_output"]

            self.unroll_one_outputs["observation"].append(feature[0])
            self.unroll_one_outputs["representation"].append(decode_output[0])

            if (len(self.unroll_one_outputs["ids"]) == self.dump_num):
                output_prefix_ = f'{self.out_dir}/step_{self.unroll_one_outputs["ids"][0]}_{self.unroll_one_outputs["ids"][-1]}'
                self.plot_multiple_plots(output_prefix_)
                self.clean_output()

        if (len(self.unroll_one_outputs["ids"]) != 0):
            output_prefix_ = f'{self.out_dir}/step_{self.unroll_one_outputs["ids"][0]}_{self.unroll_one_outputs["ids"][-1]}'
            self.plot_multiple_plots(output_prefix_)
            self.clean_output()

    def plot_output_state(self, output_state, output_name_prefix):
        if (game_type == 'atari'):
            PlotBoard.plot_atari_state(output_state, self.num_input_channels, self.input_channel_height, self.input_channel_width)
            PlotBoard.savefig(f'{output_name_prefix}.png')
        else:
            blackboard, whiteboard = get_black_white_boards(output_state)
            PlotBoard.plot_board(game_type, self.input_channel_height, blackboard, whiteboard)
            PlotBoard.savefig(f'{output_name_prefix}.png')

    def plot_multiple_plots(self, prefix_name):
        if (game_type == 'atari'):
            self.plot_atari_multiple_plots(prefix_name)
        else:
            self.plot_board_game_multiple_plots(prefix_name)

    def plot_board_game_multiple_plots(self, prefix_name):
        tmp_names_ = [f'{self.out_dir}/obs_tmp.png', f'{self.out_dir}/rep_tmp.png', f'{self.out_dir}/dyn_tmp.png']
        h_tmp_names_ = []
        i = 0
        for obs, rep, dyn, id_, action_ID_ in zip(self.unroll_one_outputs["observation"], self.unroll_one_outputs["representation"],
                                                  self.unroll_one_outputs["decoder_output"], self.unroll_one_outputs["ids"], self.unroll_one_outputs["action"]):
            self.plot_single_board_game(obs, tmp_names_[0], addtext=[f'obs:, act ID: {action_ID_}'])
            self.plot_single_board_game(rep, tmp_names_[1], addtext=[f'rep: step {id_}'])
            self.plot_single_board_game(dyn, tmp_names_[2], addtext=[f'dyn: step {id_}'])
            h_tmp_names_.append(f'{self.out_dir}/tmp_{i}.png')
            PlotBoard.merge_images_vertical(tmp_names_, h_tmp_names_[-1])
            i += 1

        PlotBoard.merge_images_horizontal(h_tmp_names_, f'{prefix_name}.png')

    def plot_single_board_game(self, output_state, output_name, addtext=None):
        blackboard, whiteboard = get_black_white_boards(output_state)
        PlotBoard.plot_board(game_type, self.input_channel_height, blackboard, whiteboard, addtext=addtext)
        PlotBoard.savefig(f'{output_name}')

    def plot_atari_multiple_plots(self, prefix_name):

        plt.figure(figsize=(50, 30))
        subtext_list_ = []
        for id_, action_ID_ in zip(self.unroll_one_outputs["ids"], self.unroll_one_outputs["action"]):
            subtext_list_.append(f'obs, act ID: {action_ID_}')
        PlotBoard.plot_single_atari_state(self.unroll_one_outputs["observation"], 3, 0, self.dump_num, self.num_input_channels, self.input_channel_height, self.input_channel_width, subtext_list_)

        subtext_list_ = []
        for id_ in self.unroll_one_outputs["ids"]:
            subtext_list_.append(f'rep: step {id_}')
        PlotBoard.plot_single_atari_state(self.unroll_one_outputs["representation"], 3, 1, self.dump_num, self.num_input_channels, self.input_channel_height, self.input_channel_width, subtext_list_)

        subtext_list_ = []
        for id_ in self.unroll_one_outputs["ids"]:
            subtext_list_.append(f'dyn: step {id_}')
        PlotBoard.plot_single_atari_state(self.unroll_one_outputs["decoder_output"], 3, 2, self.dump_num, self.num_input_channels, self.input_channel_height, self.input_channel_width, subtext_list_)

        PlotBoard.savefig(f'{prefix_name}.png')

    def init_envs(self, initial_env, start, length):
        Envs = []
        env = initial_env
        for action_id in range(start, start + length):
            if action_id < self.game_length:
                Envs.append(self.Env_unit(env, action_id, self.game_length))
                env = py.copy_env(env)
                action = env_loader.get_action(action_id)
                env.act(action)
        return Envs

    def clean_Envs(self, Envs):
        Envs = [Env_ for Env_ in Envs if Env_.env is not None]
        return Envs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-out_dir', dest='out_dir', type=str, default="decoder_output", help='output directory (default: decoder_output)')
    parser.add_argument('-conf_file', dest='conf_file', type=str, default="", help='path of config file')
    parser.add_argument('-nn_file_name', dest='nn_file_name', type=str, default="", help='path of model file')
    parser.add_argument('-game', dest='game_type', type=str, default="othello", help='game type (default: othello)')
    parser.add_argument('-sgf', dest='sgf_file', type=str, default="", help='path of sgf file')
    parser.add_argument('-l', dest='line_num', type=int, default=1, help='using which line in the input sgf file (default: 1)')
    parser.add_argument('-game_len', dest='game_len', type=int, default=0, help='max game length (default: none)')
    parser.add_argument('-s', dest='start_id', type=int, default=0, help='start action index (default: 0)')
    parser.add_argument('-mode', dest='mode', type=str, default="pca", help='mode: pca, plot_board (default: pca)')
    args = parser.parse_args()

    game_type = args.game_type
    conf_file = args.conf_file
    nn_file_name = args.nn_file_name
    sgf_file = args.sgf_file
    start_id = args.start_id
    line_num = args.line_num
    mode = *args.mode

    _temps = __import__(f'build.{game_type}', globals(), locals(), ['minizero_py'], 0)
    py = _temps.minizero_py
    py.load_config_file(conf_file)
    env_loader = py.EnvLoader()

    out_dir = args.out_dir
    if out_dir:
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

    decoder_dump = DecoderDump(out_dir, sgf_file, nn_file_name)
    if game_type and conf_file and nn_file_name and sgf_file:
        with open(sgf_file, 'r') as file:
            l = 1
            for line in file:
                if (l == line_num):
                    line = line.strip()
                    env_loader.load_from_string(line)
                    end_id = start_id + args.game_len
                    end_id = min(env_loader.get_action_pairs_size(), end_id) if end_id != 0 else env_loader.get_action_pairs_size()
                    if mode == 'pca':
                        decoder_dump.unroll_one_line_pca(start_id, end_id)
                    elif mode == 'plot_board':
                        decoder_dump.unroll_one_line(start_id, end_id)
                    break
                l += 1
    if (l != line_num):
        print("given line is out of range")
