from algorithms.custom_ppo.custom_ppo import DEFAULT_CONFIG

import logging

from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from algorithms.custom_ppo.custom_ppo_tf_policy import postprocess_ppo_gae, \
    setup_config
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, \
    LearningRateSchedule
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.torch_ops import sequence_mask
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class PPOLoss:
    def __init__(self,
                 dist_class,
                 model,
                 value_targets,
                 advantages,
                 actions,
                 prev_logits,
                 prev_actions_logp,
                 vf_preds,
                 curr_action_dist,
                 value_fn,
                 cur_kl_coeff,
                 valid_mask,
                 aux_loss,
                 aux_loss_coeff=0,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True):
        """Constructs the loss for Proximal Policy Objective.

        Arguments:
            dist_class: action distribution class for logits.
            value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            prev_logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            prev_actions_logp (Placeholder): Placeholder for prob output from
                previous model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from previous model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            valid_mask (Tensor): A bool mask of valid input elements (#2992).
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
        """
        if valid_mask is not None:
            num_valid = torch.sum(valid_mask)

            def reduce_mean_valid(t):
                return torch.sum(t * valid_mask) / num_valid

        else:

            def reduce_mean_valid(t):
                return torch.mean(t)

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        adv_std, adv_mean = torch.std_mean(advantages)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        logp_ratio = torch.exp(
            curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            advantages * logp_ratio,
            advantages * torch.clamp(logp_ratio, 1 - clip_param,
                                     1 + clip_param))
        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        if aux_loss is not None:
            self.aux_loss = reduce_mean_valid(aux_loss)

        if use_gae:
            vf_loss1 = torch.pow(value_fn - value_targets, 2.0)
            vf_clipped = vf_preds + torch.clamp(value_fn - vf_preds,
                                                -vf_clip_param, vf_clip_param)
            vf_loss2 = torch.pow(vf_clipped - value_targets, 2.0)
            vf_loss = torch.max(vf_loss1, vf_loss2)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            loss = (-surrogate_loss + cur_kl_coeff * action_kl +
                    vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = 0.0
            loss = (-surrogate_loss + cur_kl_coeff * action_kl - entropy_coeff * curr_entropy)
        if aux_loss is not None:
            loss += aux_loss_coeff * aux_loss
        loss = reduce_mean_valid(loss)   
        self.loss = loss


def ppo_surrogate_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    mask = None
    if state:
        max_seq_len = torch.max(train_batch["seq_lens"])
        mask = sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = torch.reshape(mask, [-1])

    if policy.config["use_aux_loss"]:
        aux_input = {
                    "new_obs": train_batch[SampleBatch.NEXT_OBS],
                    "actions": train_batch[SampleBatch.ACTIONS],
                    "dones": train_batch["dones"]
                    }
        aux_loss = policy.model.get_aux_loss(aux_input)
        train_batch["aux_loss"] = aux_loss
    
    policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        train_batch.get("aux_loss", None),
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        aux_loss_coeff=policy.config["aux_loss_coeff"],
        use_gae=policy.config["use_gae"],
    )

    return policy.loss_obj.loss


def kl_and_loss_stats(policy, train_batch):
    stats = {
        "cur_kl_coeff": policy.kl_coeff,
        "cur_lr": policy.cur_lr,
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function(),
            framework="torch"),
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": policy.entropy_coeff,
    }
    if policy.config["use_aux_loss"]:
        stats["aux_loss"] = policy.loss_obj.aux_loss
    return stats


def vf_preds_fetches(policy, input_dict, state_batches, model, action_dist):
    """Adds value function outputs to experience train_batches."""
    extra_dict = { SampleBatch.VF_PREDS: policy.model.value_function(),}
    return extra_dict


class KLCoeffMixin:
    def __init__(self, config):
        # KL Coefficient.
        self.kl_coeff = config["kl_coeff"]
        self.kl_target = config["kl_target"]

    def update_kl(self, sampled_kl):
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff *= 0.5
        return self.kl_coeff


class ValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        if config["use_gae"]:

            def value(ob, prev_action, prev_reward, *state):
                model_out, _ = self.model({
                    SampleBatch.CUR_OBS: self._convert_to_tensor([ob]),
                    SampleBatch.PREV_ACTIONS: self._convert_to_tensor(
                        [prev_action]),
                    SampleBatch.PREV_REWARDS: self._convert_to_tensor(
                        [prev_reward]),
                    "is_training": False,
                }, [self._convert_to_tensor(s) for s in state],
                                          self._convert_to_tensor([1]))
                return self.model.value_function()[0]

        else:

            def value(ob, prev_action, prev_reward, *state):
                return 0.0

        self._value = value


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def optimizer(policy, config):
    """Custom PyTorch optimizer to use."""
    return torch.optim.Adam(policy.model.parameters(),
                            lr=config["lr"], eps=1e-5)


CustomPPOTorchPolicy = build_torch_policy(
    name="CustomPPOTorchPolicy",
    get_default_config=lambda: DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    optimizer_fn=optimizer,
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=postprocess_ppo_gae,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    after_init=setup_mixins,
    mixins=[KLCoeffMixin, ValueNetworkMixin])
