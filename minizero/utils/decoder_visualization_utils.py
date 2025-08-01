import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os


class PlottingUtils:
    def __init__(self, training_dir, display_interval):
        self.cm_black = LinearSegmentedColormap.from_list('custom_map', [(0, 'yellow'), (0.5, 'black'), (1, 'white')], N=256)
        self.cm_white = LinearSegmentedColormap.from_list('custom_map', [(0, 'yellow'), (0.5, 'white'), (1, 'black')], N=256)
        self.loss_decoder_showplt = []
        self.training_dir = training_dir
        self.display_interval = display_interval
        if not os.path.exists(f"{training_dir}/decoder"):
            os.makedirs(f"{training_dir}/decoder")

    def plot_states(self, features, decoder_output, iter, training_step, py):
        if (self.display_interval != 0 and iter % self.display_interval == 0):
            n = py.get_muzero_unrolling_step() + 1
            c = py.get_nn_num_decoder_output_channels()
            h = py.get_nn_input_channel_height()
            w = py.get_nn_input_channel_width()
            plt.figure(figsize=(50, 30))
            for i in range(n):
                if not py.get_game_name().startswith('atari_'):
                    self.plot_single_board_state(features[0][i], decoder_output[i][0], i, n, c, h, w)
                else:
                    self.plot_single_atari_state(features[0][i], decoder_output[i][0], i, n, c, h, w)
            plt.savefig(f'{self.training_dir}/decoder/visualization_{iter}_{training_step}.png')
            plt.close()

    def plot_single_board_state(self, feature, decoder_output, i, n, c, h, w):
        cmap = self.cm_black if (feature[c - 2][0][0] == 1) else self.cm_white
        ax1 = plt.subplot(2, n, i + 1)
        stones = feature[1].cpu().detach().numpy().reshape(h, w) + feature[0].cpu().detach().numpy().reshape(h, w) * 0.5
        plt.imshow(stones, cmap=cmap, vmin=0, vmax=1)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        if i == 0:
            plt.text(0, -1, 'observation:', fontsize=60, color='black')

        ax2 = plt.subplot(2, n, n + i + 1)
        stones = decoder_output[1].cpu().detach().numpy().reshape(h, w) + decoder_output[0].cpu().detach().numpy().reshape(h, w) * 0.5
        plt.imshow(stones, cmap=cmap, vmin=0, vmax=1)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        if i == 0:
            plt.text(0, -1, 'decoder:', fontsize=60, color='black')
        else:
            plt.text(0, -1, f'unroll_{i}', fontsize=60, color='black')

    def plot_single_atari_state(self, feature, decoder_output, i, n, c, h, w):
        ax1 = plt.subplot(2, n, i + 1)
        image_r = feature[c - 3].cpu().detach().numpy().clip(0, 1).reshape(h, w)
        image_g = feature[c - 2].cpu().detach().numpy().clip(0, 1).reshape(h, w)
        image_b = feature[c - 1].cpu().detach().numpy().clip(0, 1).reshape(h, w)
        plt.imshow((np.dstack((image_r, image_g, image_b)) * 255.999).astype(np.uint8))
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        if i == 0:
            plt.text(2, -5, 'observation:', fontsize=60, color='black')
        else:
            plt.text(2, -5, f'unroll_{i}', fontsize=60, color='black')

        ax2 = plt.subplot(2, n, n + i + 1)
        image_r = decoder_output[c - 3].cpu().detach().numpy().clip(0, 1).reshape(h, w)
        image_g = decoder_output[c - 2].cpu().detach().numpy().clip(0, 1).reshape(h, w)
        image_b = decoder_output[c - 1].cpu().detach().numpy().clip(0, 1).reshape(h, w)
        plt.imshow((np.dstack((image_r, image_g, image_b)) * 255.999).astype(np.uint8))
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        if i == 0:
            plt.text(2, -5, 'decoder:', fontsize=60, color='black')
