from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ICMImpalaCNN(TorchModelV2, nn.Module):
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

        # c, h, w = obs_space.shape     # Frame stack
        h, w, c = obs_space.shape
        shape = (c, h, w)
        embed_size = 256

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc_1 = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=embed_size)
        # self.hidden_fc_2 = nn.Linear(in_features=256, out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        # ICM Layers
        self.idm_hidden = nn.Linear(in_features=embed_size * 2, out_features=256)
        self.idm_logits = nn.Linear(in_features=256, out_features=num_outputs)
        self.fdm_hidden = nn.Linear(in_features=embed_size + num_outputs, out_features=256)
        self.fdm_output = nn.Linear(in_features=256, out_features=embed_size)
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._obs_embed = self.embedding(input_dict["obs"])
        x = self._obs_embed
        # x = self.hidden_fc_2(self._obs_embed)
        # x = nn.functional.relu(x)
        logits = self.logits_fc(x)
        value = self.value_fc(x)
        self._value = value.squeeze(1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value

    def embedding(self, obs):
        x = obs.float()
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.hidden_fc_1(x)
        return nn.functional.relu(x)
        # return x

    def intrinsic_reward(self, input_dict):
        with torch.no_grad():
            mask = [not i for i in input_dict["dones"]]
            obs = self.embedding(input_dict["obs"])
            next_obs_t = obs[1:][mask[:-1]]
            if mask[-1]:
                next_obs_t = torch.cat((next_obs_t, self.embedding(input_dict["new_obs"][-1:])), dim=0)
            obs = obs[mask]
            act_one_hot = nn.functional.one_hot(input_dict["actions"][mask].long(),
                                                num_classes=self.action_space.n)
            in_rew = torch.zeros(len(mask), dtype=torch.float32, device=obs.device)

            x = torch.cat((obs, act_one_hot.float()), dim=1)
            x = self.fdm_hidden(x)
            x = nn.functional.relu(x)
            next_obs = self.fdm_output(x)
            in_rew[mask] = torch.mean(torch.pow((next_obs_t - next_obs), 2), dim=1)
            return in_rew.cpu().numpy()

    def get_aux_loss(self, input_dict):
        assert self._obs_embed is not None, "must call forward() first"
        mask = [not i for i in input_dict["dones"]]
        obs = self._obs_embed[mask]
        with torch.no_grad():
            next_obs_t = self._obs_embed[1:][mask[:-1]]
            if mask[-1]:
                next_obs_t = torch.cat((next_obs_t, self.embedding(input_dict["new_obs"][-1:])), dim=0)
        # with torch.no_grad():
        #     next_obs_t = self.embedding(input_dict["new_obs"][mask].float())
        act = input_dict["actions"][mask].long()
        act_one_hot = nn.functional.one_hot(act, num_classes=self.action_space.n).float()
        fdm_loss = torch.zeros(len(mask), dtype=torch.float32, device=obs.device)
        idm_loss = torch.zeros(len(mask), dtype=torch.float32, device=obs.device)

        fdm_x = torch.cat((obs.detach(), act_one_hot.detach()), dim=1)
        fdm_x = self.fdm_hidden(fdm_x)
        fdm_x = nn.functional.relu(fdm_x)
        next_obs = self.fdm_output(fdm_x)
        fdm_loss[mask] = 0.5 * torch.mean(torch.pow((next_obs_t.detach() - next_obs), 2), dim=1)

        idm_x = torch.cat((obs, next_obs), dim=1)
        idm_x = self.idm_hidden(idm_x)
        idm_x = nn.functional.relu(idm_x)
        idm_x = self.idm_logits(idm_x)
        log_probs = nn.functional.log_softmax(idm_x, dim=1)
        idm_loss[mask] = nn.functional.nll_loss(log_probs, act, reduction='none')

        return fdm_loss+idm_loss


ModelCatalog.register_custom_model("icm_impala_cnn_torch", ICMImpalaCNN)