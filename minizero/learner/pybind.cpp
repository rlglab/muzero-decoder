#include "configuration.h"
#include "data_loader.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;
using namespace minizero;

std::shared_ptr<Environment> kEnvInstance;

Environment& getEnvInstance()
{
    if (!kEnvInstance) { kEnvInstance = std::make_shared<Environment>(); }
    return *kEnvInstance;
}

PYBIND11_MODULE(minizero_py, m)
{
    m.def("load_config_file", [](std::string file_name) {
        minizero::env::setUpEnv();
        minizero::config::ConfigureLoader cl;
        minizero::config::setConfiguration(cl);
        bool success = cl.loadFromFile(file_name);
        if (success) { kEnvInstance = std::make_shared<Environment>(); }
        return success;
    });
    m.def("load_config_string", [](std::string conf_str) {
        minizero::config::ConfigureLoader cl;
        minizero::config::setConfiguration(cl);
        bool success = cl.loadFromString(conf_str);
        if (success) { kEnvInstance = std::make_shared<Environment>(); }
        return success;
    });
    m.def("use_gumbel", []() { return config::actor_use_gumbel; });
    m.def("get_zero_replay_buffer", []() { return config::zero_replay_buffer; });
    m.def("use_per", []() { return config::learner_use_per; });
    m.def("get_training_step", []() { return config::learner_training_step; });
    m.def("get_training_display_step", []() { return config::learner_training_display_step; });
    m.def("get_batch_size", []() { return config::learner_batch_size; });
    m.def("get_muzero_unrolling_step", []() { return config::learner_muzero_unrolling_step; });
    m.def("get_n_step_return", []() { return config::learner_n_step_return; });
    m.def("get_learning_rate", []() { return config::learner_learning_rate; });
    m.def("get_momentum", []() { return config::learner_momentum; });
    m.def("get_weight_decay", []() { return config::learner_weight_decay; });
    m.def("get_value_loss_scale", []() { return config::learner_value_loss_scale; });
    m.def("get_game_name", []() { return getEnvInstance().name(); });
    m.def("get_nn_num_input_channels", []() { return getEnvInstance().getNumInputChannels(); });
    m.def("get_nn_input_channel_height", []() { return getEnvInstance().getInputChannelHeight(); });
    m.def("get_nn_input_channel_width", []() { return getEnvInstance().getInputChannelWidth(); });
    m.def("get_nn_num_hidden_channels", []() { return config::nn_num_hidden_channels; });
    m.def("get_nn_hidden_channel_height", []() { return getEnvInstance().getHiddenChannelHeight(); });
    m.def("get_nn_hidden_channel_width", []() { return getEnvInstance().getHiddenChannelWidth(); });
    m.def("get_nn_num_action_feature_channels", []() { return getEnvInstance().getNumActionFeatureChannels(); });
    m.def("get_nn_num_blocks", []() { return config::nn_num_blocks; });
    m.def("get_nn_action_size", []() { return getEnvInstance().getPolicySize(); });
    m.def("get_nn_num_value_hidden_channels", []() { return config::nn_num_value_hidden_channels; });
    m.def("get_nn_discrete_value_size", []() { return kEnvInstance->getDiscreteValueSize(); });
    m.def("use_state_consistency", []() { return config::learner_use_state_consistency; });
    m.def("get_nn_state_consistency_params", []() -> py::object {
        if (!config::learner_use_state_consistency) { return py::none(); }
        return py::make_tuple(config::nn_state_consistency_proj_hid, config::nn_state_consistency_proj_out, config::nn_state_consistency_pred_hid, config::nn_state_consistency_pred_out);
    });
    m.def("use_decoder", []() { return config::learner_use_decoder; });
    m.def("get_nn_num_decoder_output_channels", []() { return getEnvInstance().getNumDecoderOutputChannels(); });
    m.def("get_decoder_output_at_inference", []() { return config::decoder_output_at_inference; });
    m.def("get_decoder_loss_scale", []() { return config::decoder_loss_scale; });
    m.def("get_decoder_clip_grad_value", []() { return config::decoder_clip_grad_value; });
    m.def("get_nn_type_name", []() { return config::nn_type_name; });

    py::class_<learner::DataLoader>(m, "DataLoader")
        .def(py::init<std::string>())
        .def("initialize", &learner::DataLoader::initialize)
        .def("load_data_from_file", &learner::DataLoader::loadDataFromFile, py::call_guard<py::gil_scoped_release>())
        .def(
            "update_priority", [](learner::DataLoader& data_loader, py::array_t<int>& sampled_index, py::array_t<float>& batch_values) {
                data_loader.updatePriority(static_cast<int*>(sampled_index.request().ptr), static_cast<float*>(batch_values.request().ptr));
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "sample_data", [](learner::DataLoader& data_loader, py::array_t<float>& features, py::array_t<float>& action_features, py::array_t<float>& policy, py::array_t<float>& value, py::array_t<float>& reward, py::array_t<float>& loss_scale, py::array_t<int>& sampled_index) {
                data_loader.getSharedData()->getDataPtr()->features_ = static_cast<float*>(features.request().ptr);
                data_loader.getSharedData()->getDataPtr()->action_features_ = static_cast<float*>(action_features.request().ptr);
                data_loader.getSharedData()->getDataPtr()->policy_ = static_cast<float*>(policy.request().ptr);
                data_loader.getSharedData()->getDataPtr()->value_ = static_cast<float*>(value.request().ptr);
                data_loader.getSharedData()->getDataPtr()->reward_ = static_cast<float*>(reward.request().ptr);
                data_loader.getSharedData()->getDataPtr()->loss_scale_ = static_cast<float*>(loss_scale.request().ptr);
                data_loader.getSharedData()->getDataPtr()->sampled_index_ = static_cast<int*>(sampled_index.request().ptr);
                data_loader.sampleData();
            },
            py::call_guard<py::gil_scoped_release>());

    py::enum_<env::Player>(m, "Player")
        .value("kPlayerNone", env::Player::kPlayerNone)
        .value("kPlayer1", env::Player::kPlayer1)
        .value("kPlayer2", env::Player::kPlayer2)
        .value("kPlayerSize", env::Player::kPlayerSize)
        .export_values();

    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def(py::init<int, env::Player>())
        .def("next_player", &Action::nextPlayer)
        .def("get_action_id", &Action::getActionID)
        .def("get_player", &Action::getPlayer);

    py::class_<Environment>(m, "Env")
        .def(py::init<>())
        .def("to_string", &Environment::toString)
        .def("is_terminal", &Environment::isTerminal)
        .def("act", py::overload_cast<const Action&>(&Environment::act))
        .def("act", py::overload_cast<const std::vector<std::string>&>(&Environment::act))
        .def("get_legal_actions", &Environment::getLegalActions)
        .def("get_features", [](Environment& env) {
            return env.getFeatures();
        })
        .def("get_reward", &Environment::getReward)
        .def("get_action_features", [](Environment& env, Action action) { return env.getActionFeatures(action); })
#if ATARI
        .def("reset", py::overload_cast<int>(&Environment::reset));
#else
        .def("reset", py::overload_cast<>(&Environment::reset));
#endif

    py::class_<EnvironmentLoader>(m, "EnvLoader")
        .def(py::init<>())
        .def("load_from_string", py::overload_cast<const std::string&>(&EnvironmentLoader::loadFromString))
        .def("get_action", [](EnvironmentLoader& loader, int pos) { return loader.getActionPairs()[pos].first; })
        .def("get_action_pairs_size", [](EnvironmentLoader& loader) { return loader.getActionPairs().size(); })
        .def("to_string", &EnvironmentLoader::toString)
        .def("get_tag", &EnvironmentLoader::getTag)
        .def("get_policy", &EnvironmentLoader::getPolicy)
        .def("get_value", &EnvironmentLoader::getValue)
        .def("get_reward", &EnvironmentLoader::getReward)
        .def("get_action_pairs_value", [](EnvironmentLoader& loader, int pos) { return loader.getActionPairs()[pos].second["V"]; });

    m.def("copy_env", [](Environment env) {
        Environment env_copy_ = env;
        return env_copy_;
    });
}
