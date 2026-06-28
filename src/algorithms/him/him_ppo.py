"""HIM PPO: standard rsl_rl PPO + an auxiliary state-estimator update.

The estimator lives inside the HIM actor model (``self.actor.estimator``) and is
optimized independently (its own Adam, its own loss). Because the estimator's
output is detached inside the actor, the PPO graph and the estimator graph never
share gradients; the only coupling is the (optional) adaptive learning rate,
which we forward to the estimator to mirror HIMLoco's behavior.

Data flow for the estimator supervision (per transition t):
  - encoder input  : flattened proprio history at t     -> observations[history_group]
  - target frame   : single proprio frame at t+1        -> next_observations[current_group]
  - velocity target: true base lin vel at t+1           -> next_observations[vel_group]
  - validity mask  : transitions that did NOT terminate -> (1 - dones)

We run the estimator pass BEFORE ``super().update()`` because the base update
clears the rollout storage at the end.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from .him_storage import HIMRolloutStorage


class HIMPPO(PPO):
  def __init__(
    self,
    *args,
    estimator_vel_group: str = "estimator_vel",
    estimator_history_group: str = "proprio_history",
    estimator_current_group: str = "proprio_current",
    **kwargs,
  ) -> None:
    self.estimator_vel_group = estimator_vel_group
    self.estimator_history_group = estimator_history_group
    self.estimator_current_group = estimator_current_group
    super().__init__(*args, **kwargs)
    # The estimator is owned by the actor model.
    self.estimator = self.actor.estimator

  # Rollout ---------------------------------------------------------------

  def process_env_step(
    self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
  ) -> None:
    # ``obs`` here is s_{t+1}. Record it as the transition's next observation.
    self.transition.next_observations = obs
    super().process_env_step(obs, rewards, dones, extras)

  # Update ----------------------------------------------------------------

  def _update_estimator(self) -> tuple[float, float]:
    history_group = self.estimator_history_group
    current_group = self.estimator_current_group
    vel_group = self.estimator_vel_group

    mean_est, mean_swap, n = 0.0, 0.0, 0
    generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    for batch in generator:
      obs_history = batch.observations[history_group]
      next_obs = batch.next_observations
      next_frame = next_obs[current_group]
      next_vel = next_obs[vel_group]
      valid_mask = (1.0 - batch.dones.float()).view(-1)

      est_loss, swap_loss = self.estimator.update(
        obs_history=obs_history,
        next_obs_frame=next_frame,
        next_vel=next_vel,
        valid_mask=valid_mask,
      )
      mean_est += est_loss
      mean_swap += swap_loss
      n += 1

    if n > 0:
      mean_est /= n
      mean_swap /= n
    return mean_est, mean_swap

  def compute_returns(self, obs) -> None:
    super().compute_returns(obs)
    # No return clamp — HIMLoco parity (clamp was distorting advantages).

  def update(self) -> dict[str, float]:
    est_loss, swap_loss = self._update_estimator()
    loss_dict = self._ppo_update_combined_clip()
    loss_dict["estimation"] = est_loss
    loss_dict["swap"] = swap_loss
    return loss_dict

  def _ppo_update_combined_clip(self) -> dict[str, float]:
    """PPO update identical to base PPO.update() except grad clip is combined.

    HIMLoco clips actor_critic.parameters() together as one norm; the base
    rsl_rl PPO clips actor and critic separately. Combined clipping means that
    when critic gradients are large, actor gradients (including std_param) are
    proportionally compressed, protecting std from going negative.
    """
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_entropy = 0
    mean_rnd_loss = 0 if self.rnd else None
    mean_symmetry_loss = 0 if self.symmetry else None
    num_finite_updates = 0  # count only batches where loss was finite

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    else:
      generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

    for batch in generator:
      original_batch_size = batch.observations.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

      if self.symmetry and self.symmetry["use_data_augmentation"]:
        data_augmentation_func = self.symmetry["data_augmentation_func"]
        batch.observations, batch.actions = data_augmentation_func(
          env=self.symmetry["_env"], obs=batch.observations, actions=batch.actions,
        )
        num_aug = int(batch.observations.batch_size[0] / original_batch_size)
        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
        batch.values = batch.values.repeat(num_aug, 1)
        batch.advantages = batch.advantages.repeat(num_aug, 1)
        batch.returns = batch.returns.repeat(num_aug, 1)

      self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
      distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

      if self.symmetry:
        if not self.symmetry["use_data_augmentation"]:
          data_augmentation_func = self.symmetry["data_augmentation_func"]
          batch.observations, _ = data_augmentation_func(obs=batch.observations, actions=None, env=self.symmetry["_env"])
        mean_actions = self.actor(batch.observations.detach().clone())
        action_mean_orig = mean_actions[:original_batch_size]
        _, actions_mean_symm = data_augmentation_func(obs=None, actions=action_mean_orig, env=self.symmetry["_env"])
        mse_loss = torch.nn.MSELoss()
        symmetry_loss = mse_loss(mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:])
        if self.symmetry["use_mirror_loss"]:
          loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
        else:
          symmetry_loss = symmetry_loss.detach()

      # Guard: skip mini-batch if loss is non-finite (hfield overflow artifact).
      # NaN/inf loss → NaN gradients → NaN parameters → training permanently corrupted.
      if not torch.isfinite(loss):
        continue

      num_finite_updates += 1

      if self.rnd:
        with torch.no_grad():
          rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
          rnd_state = self.rnd.state_normalizer(rnd_state)
        predicted_embedding = self.rnd.predictor(rnd_state)
        target_embedding = self.rnd.target(rnd_state).detach()
        rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()

      # Combined clip: actor+critic as one norm (mirrors HIMLoco clip_grad_norm_(actor_critic.parameters())).
      # When critic gradient is large, actor std_param is proportionally compressed, preventing std < 0.
      # Exclude estimator params (independent optimizer; PPO loss does not flow there).
      actor_params = [
        p for name, p in self.actor.named_parameters()
        if not name.startswith("estimator.")
      ]
      nn.utils.clip_grad_norm_(
        actor_params + list(self.critic.parameters()),
        self.max_grad_norm,
      )
      self.optimizer.step()
      if self.rnd_optimizer:
        self.rnd_optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    num_updates = max(1, num_finite_updates)  # avoid divide-by-zero if all batches were non-finite
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates

    self.storage.clear()

    loss_dict: dict[str, float] = {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
    }
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss
    return loss_dict

  # Save / load -----------------------------------------------------------

  def save(self) -> dict:
    saved = super().save()
    saved["estimator_optimizer_state_dict"] = self.estimator.optimizer.state_dict()
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    out = super().load(loaded_dict, load_cfg, strict)
    # Estimator weights ride along inside actor_state_dict (estimator is a
    # submodule of the actor). Only its optimizer needs explicit restore.
    if load_cfg is None or load_cfg.get("optimizer", True):
      if "estimator_optimizer_state_dict" in loaded_dict:
        self.estimator.optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
    return out

  # Construction ----------------------------------------------------------

  @staticmethod
  def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "HIMPPO":
    """Mirror PPO.construct_algorithm but use HIMRolloutStorage.

    Resolves actor/critic/algorithm classes via ``class_name`` exactly like the
    base, so the HIM actor model and HIMPPO algorithm are selected from config.
    """
    alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
    actor_class = resolve_callable(cfg["actor"].pop("class_name"))
    critic_class = resolve_callable(cfg["critic"].pop("class_name"))

    default_sets = ["actor", "critic"]
    if cfg["algorithm"].get("rnd_cfg") is not None:
      default_sets.append("rnd_state")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

    actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
    print(f"Actor Model: {actor}")
    if cfg["algorithm"].pop("share_cnn_encoders", None):
      cfg["critic"]["cnns"] = actor.cnns
    critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
    print(f"Critic Model: {critic}")

    storage = HIMRolloutStorage(
      "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
    )

    alg = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
    return alg
