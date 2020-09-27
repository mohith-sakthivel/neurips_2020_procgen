from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

from models.impala_cnn_torch import ConvSequence

torch, nn = try_import_torch()


class DeepMDP_Impala(TorchModelV2, nn.Module):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self._embed_shape = (shape[0], shape[1], shape[2])
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)

        self.rew_hid = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        self.rew_out = nn.Linear(in_features=256, out_features=num_outputs)
        self.trans_conv = nn.Conv2d(in_channels=shape[0], out_channels=shape[0]*num_outputs, kernel_size=3, padding=1, stride=1)
        
    def get_embedding(self, x):
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        return x

    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._embed = self.get_embedding(input_dict["obs"].float())
        x = torch.flatten(self._embed, start_dim=1)
        x = nn.functional.relu(x)
        x = self.hidden_fc(x)
        x = nn.functional.relu(x)
        logits = self.logits_fc(x)
        value = self.value_fc(x)
        self._value = value.squeeze(1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value

    def aux_loss(self, input_dict):
        action = input_dict["actions"]
        sample_no = torch.arange(len(action))
        next_state = nn.functional.relu(self.trans_conv(self._embed))
        next_state = next_state.view((-1, self.num_outputs,) + self._embed_shape)
        next_state_target = self.get_embedding(input_dict["new_obs"].float())
        trans_model_error = nn.functional.mse_loss(next_state[sample_no, action],
                                                   next_state_target,
                                                   reduction='none')
        trans_model_error = torch.flatten(trans_model_error, start_dim=1).mean(dim=1)
        
        x = torch.flatten(self._embed, start_dim=1)
        x = nn.functional.relu(self.rew_hid(x))
        reward = self.rew_out(x)
        reward_error = nn.functional.mse_loss(reward[sample_no, action],
                                              input_dict["rewards"],
                                              reduction='none')
        return reward_error


ModelCatalog.register_custom_model("deepmdp_impala_torch", DeepMDP_Impala)