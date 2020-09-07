from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
from torch import embedding

torch, nn = try_import_torch()


def layer_norm(x):
    mean = torch.mean(x, 1, keepdim=True)
    var = torch.var(x, 1, keepdim=True)
    return (x-mean)/(torch.sqrt(var) + 1e-8)


class ConvLayer(nn.Module):
    def __init__(self, input_shape, out_channels, kernel, stride, padding=0):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.kernel= kernel
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels,
                              kernel_size=self.kernel, stride=self.stride, padding=self.padding)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        out_h = int((((h + 2 * self.padding - (self.kernel - 1) - 1) / self.stride) + 1)//1)
        out_w = int((((w + 2 * self.padding - (self.kernel - 1) - 1) / self.stride) + 1)//1)
        return (self._out_channels, out_h, out_w)


class MohithCNN(TorchModelV2, nn.Module):
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
        c, h, w = obs_space.shape
        shape = (c, h, w)
        embed_size = 256

        conv_seqs = []
        for out_channels, kernel, stride in zip([32, 64, 64], [8, 4, 3], [4, 2, 1]):
            conv_seq = ConvLayer(shape, out_channels, kernel, stride)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc_1 = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=embed_size)
        self.hidden_fc_2 = nn.Linear(in_features=256, out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        self.bn = nn.BatchNorm1d(embed_size)
        # ICM Layers
        self.idm_hidden = nn.Linear(in_features=embed_size * 2, out_features=256)
        self.idm_logits = nn.Linear(in_features=256, out_features=num_outputs)
        self.fdm_hidden = nn.Linear(in_features=embed_size + num_outputs, out_features=256)
        self.fdm_output = nn.Linear(in_features=256, out_features=embed_size)

    def embedding(self, obs):
        x = obs.float()
        x = x / 255.0  # scale to 0-1
        # x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.hidden_fc_1(x)
        x =  self.bn(x) if x.shape[0]>1 else x
        return layer_norm(x)
        # return x

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = self.embedding(input_dict["obs"])
        x = self.hidden_fc_2(x)
        x = nn.functional.relu(x)
        logits = self.logits_fc(x)
        value = self.value_fc(x)
        self._value = value.squeeze(1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value

    def intrinsic_reward(self, input_dict):
        with torch.no_grad():
            obs = self.embedding(input_dict["obs"].float())
            next_obs_t = self.embedding(input_dict["new_obs"].float())
            act_one_hot = nn.functional.one_hot(input_dict["actions"].long(),
                                                num_classes=self.action_space.n)
            x = torch.cat((obs, act_one_hot.float()), dim=1)
            x = self.fdm_hidden(x)
            x = nn.functional.relu(x)
            next_obs = self.fdm_output(x)
            return torch.mean(torch.pow((next_obs_t - next_obs), 2), dim=1).cpu().numpy()

    def icm_losses(self, input_dict):
        mask = [not i for i in input_dict["dones"]]
        obs = self.embedding(input_dict["obs"][mask].float())
        next_obs_t = self.embedding(input_dict["new_obs"][mask].float())
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

        return fdm_loss, idm_loss


ModelCatalog.register_custom_model("mohith_cnn_torch", MohithCNN)