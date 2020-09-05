from os import stat
from algorithms.custom_ppo.custom_ppo import DEFAULT_CONFIG
import logging
import numpy as np
from ray.rllib.evaluation.postprocessing import discount
import ray
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.ppo.ppo_tf_policy import postprocess_ppo_gae, \
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
                 idm_loss,
                 fdm_loss,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 idm_loss_coeff=0,
                 fdm_loss_coeff=0,
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

        if idm_loss is not None:
            self.idm_loss = reduce_mean_valid(idm_loss)
            self.fdm_loss = reduce_mean_valid(fdm_loss)

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
            loss = -surrogate_loss + cur_kl_coeff * action_kl - entropy_coeff * curr_entropy
    
        if idm_loss is not None:
            loss += idm_loss_coeff * idm_loss + fdm_loss_coeff * fdm_loss

        loss = reduce_mean_valid(loss)        
        self.loss = loss

        # def calc_mean(t):
        #         return torch.mean(t).item()
        # losses = [surrogate_loss, action_kl, vf_loss, curr_entropy, idm_loss, fdm_loss]
        # losses = [*map(calc_mean, losses)]
        # coeff = [-1, cur_kl_coeff, vf_loss_coeff, -entropy_coeff, idm_loss_coeff, fdm_loss_coeff]
        # l = [round(lo*cf, 4) for lo, cf in zip(losses, coeff)]
        # print(len(losses), len(coeff))
        # print("Policy Loss: {}\t Action_KL: {}\t Value Function: {}\t Entorpy: {}\t IDM: {} FDM: {}".format(*l))


def ppo_surrogate_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    mask = None
    if state:
        max_seq_len = torch.max(train_batch["seq_lens"])
        mask = sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = torch.reshape(mask, [-1])

    if policy.config["use_intrinsic_rew"]:
        icm_input = {
                    "obs": train_batch[SampleBatch.CUR_OBS],
                    "new_obs": train_batch[SampleBatch.NEXT_OBS],
                    "actions": train_batch[SampleBatch.ACTIONS],
                    "dones": train_batch["dones"]
                    }
        fdm_loss, idm_loss = policy.model.icm_losses(icm_input)
    else:
        fdm_loss, idm_loss = None, None

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
        idm_loss,
        fdm_loss,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        idm_loss_coeff=policy.config["idm_loss_coeff"],
        fdm_loss_coeff=policy.config["fdm_loss_coeff"],
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
    if policy.config["use_intrinsic_rew"]:
        stats["idm_loss"] = policy.loss_obj.idm_loss
        stats["fdm_loss"] = policy.loss_obj.fdm_loss
    return stats


class PostProcessing:
    """ Constant definitions for postprocessing with intrinsic rewards. """
    ADVANTAGES = "advantages"
    VALUE_TARGETS = "value_targets"
    INTRINSIC_REWARDS = "intrinsic_rewards"


def vf_preds_fetches(policy, input_dict, state_batches, model, action_dist):
    """Adds value function outputs to experience train_batches."""
    return {
        SampleBatch.VF_PREDS: policy.model.value_function()
    }


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


def postprocess_ppo_gae(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)

    in_rews = np.zeros(len(sample_batch["dones"]))
    if policy.config["use_intrinsic_rew"]:
        masks = [not i for i in sample_batch["dones"]]
        in_rews = np.zeros(len(masks))
        obs = sample_batch[SampleBatch.CUR_OBS][masks]
        next_obs = sample_batch[SampleBatch.NEXT_OBS][masks]
        act = sample_batch[SampleBatch.ACTIONS][masks]
        in_rews_batch = {
                        SampleBatch.CUR_OBS: obs,
                        SampleBatch.NEXT_OBS: next_obs,
                        SampleBatch.ACTIONS: act,
                        }
        in_rews[masks] = policy.model.intrinsic_reward(policy._lazy_tensor_dict(in_rews_batch))
    sample_batch[PostProcessing.INTRINSIC_REWARDS] = in_rews

    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return batch


def compute_advantages(rollout,
                       last_r,
                       gamma=0.9,
                       lambda_=1.0,
                       use_gae=True,
                       use_critic=True):
    """
    Given a rollout, compute its value targets and the advantage.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory
        last_r (float): Value estimation for last observation
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estimation
        use_critic (bool): Whether to use critic (value estimates). Setting
                           this to False will use 0 as baseline.

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    traj = {}
    trajsize = len(rollout[SampleBatch.ACTIONS])
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    assert SampleBatch.VF_PREDS in rollout or not use_critic, \
        "use_critic=True but values not found"
    assert use_critic or not use_gae, \
        "Can't use gae without using a value function"
    if use_gae:
        vpred_t = np.concatenate(
            [rollout[SampleBatch.VF_PREDS],
             np.array([last_r])])
        delta_t = (
                   traj[SampleBatch.REWARDS] +
                   0.01 * traj[PostProcessing.INTRINSIC_REWARDS] +
                   gamma * vpred_t[1:] - vpred_t[:-1])
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        traj[Postprocessing.ADVANTAGES] = discount(delta_t, gamma * lambda_)
        traj[Postprocessing.VALUE_TARGETS] = (
            traj[Postprocessing.ADVANTAGES] +
            traj[SampleBatch.VF_PREDS]).copy().astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch.REWARDS] + 0.01 * traj[PostProcessing.INTRINSIC_REWARDS],
             np.array([last_r])])
        discounted_returns = discount(rewards_plus_v,
                                      gamma)[:-1].copy().astype(np.float32)

        if use_critic:
            traj[Postprocessing.
                 ADVANTAGES] = discounted_returns - rollout[SampleBatch.
                                                            VF_PREDS]
            traj[Postprocessing.VALUE_TARGETS] = discounted_returns
        else:
            traj[Postprocessing.ADVANTAGES] = discounted_returns
            traj[Postprocessing.VALUE_TARGETS] = np.zeros_like(
                traj[Postprocessing.ADVANTAGES])

    traj[Postprocessing.ADVANTAGES] = traj[
        Postprocessing.ADVANTAGES].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)


CustomPPOTorchPolicy = build_torch_policy(
    name="CustomPPOTorchPolicy",
    get_default_config=lambda: DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=postprocess_ppo_gae,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    after_init=setup_mixins,
    mixins=[KLCoeffMixin, ValueNetworkMixin])