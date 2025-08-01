#!/usr/bin/env python
from minizero.network.py.create_network import create_network
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs, flush=True)


class Model:
    def __init__(self):
        self.training_step = 0
        self.network = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.scheduler = None

    def load_model(self, model_file):
        self.training_step = 0
        self.network = create_network(py.get_game_name(),
                                      py.get_nn_num_input_channels(),
                                      py.get_nn_input_channel_height(),
                                      py.get_nn_input_channel_width(),
                                      py.get_nn_num_hidden_channels(),
                                      py.get_nn_hidden_channel_height(),
                                      py.get_nn_hidden_channel_width(),
                                      py.get_nn_num_action_feature_channels(),
                                      py.get_nn_num_blocks(),
                                      py.get_nn_action_size(),
                                      py.get_nn_num_value_hidden_channels(),
                                      py.get_nn_discrete_value_size(),
                                      py.get_nn_state_consistency_params(),
                                      py.get_nn_num_decoder_output_channels(),
                                      py.get_decoder_output_at_inference(),
                                      py.get_nn_type_name())
        self.network.to(self.device)
        self.optimizer = optim.SGD(self.network.parameters(),
                                   lr=py.get_learning_rate(),
                                   momentum=py.get_momentum(),
                                   weight_decay=py.get_weight_decay())
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000000, gamma=0.1)

        if model_file:
            snapshot = torch.load(f"{model_file}", map_location=torch.device('cpu'))
            self.training_step = snapshot['training_step']
            self.network.load_state_dict(snapshot['network'])
            self.optimizer.load_state_dict(snapshot['optimizer'])
            self.optimizer.param_groups[0]["lr"] = py.get_learning_rate()
            self.scheduler.load_state_dict(snapshot['scheduler'])

        # for multi-gpu
        self.network = nn.DataParallel(self.network)


if __name__ == '__main__':

    if len(sys.argv) >= 3:
        game_type = sys.argv[1]
        conf_file_name = sys.argv[2]
        model_file = sys.argv[3]
        output_suffix = "_fixed"

        # import pybind library
        _temps = __import__(f'build.{game_type}', globals(), locals(), ['minizero_py'], 0)
        py = _temps.minizero_py
    else:
        eprint()
        eprint("Usage:")
        eprint("python recreate_model.py game_type conf_file model_pkl_file")
        eprint()
        exit(0)

    py.load_config_file(conf_file_name)
    model = Model()
    model.load_model(model_file)
    output_model_file = model_file.replace('.pkl', output_suffix + '.pt')
    eprint(f"output path: {output_model_file}")
    torch.jit.script(model.network.module).save(f"{output_model_file}")
