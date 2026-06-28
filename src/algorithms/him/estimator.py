"""Hybrid Internal Model (HIM) estimator.

Ported from HIMLoco (rsl_rl/rsl_rl/modules/him_estimator.py) and adapted to the
mjlab/rsl_rl data flow:

- ``update()`` takes explicit tensors (obs_history, next single-step obs frame,
  next true base linear velocity, optional valid-mask) instead of slicing a
  monolithic ``next_critic_obs`` buffer. This decouples the estimator from a
  particular observation layout.
- The encoder consumes a *flattened* history of shape ``[B, T * S]`` where ``T``
  is ``temporal_steps`` and ``S`` is ``num_one_step_obs``. mjlab stacks history
  oldest->newest; the estimator only needs the full window, so ordering does not
  matter for the encoder, but the *target* network must receive the genuine
  next-step single frame (provided explicitly by the algorithm).

Original reference: Long et al., "Hybrid Internal Model", ICLR 2024.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_activation(act_name: str) -> nn.Module:
  acts = {
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "relu": nn.ReLU(),
    "crelu": nn.ReLU(),
    "silu": nn.SiLU(),
    "lrelu": nn.LeakyReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
  }
  if act_name not in acts:
    raise ValueError(f"Invalid activation function: {act_name}")
  return acts[act_name]


@torch.no_grad()
def sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
  """Sinkhorn-Knopp normalization (SwAV-style soft cluster assignment)."""
  q = torch.exp(out / eps).T
  K, B = q.shape[0], q.shape[1]
  q /= q.sum()
  for _ in range(iters):
    q /= torch.sum(q, dim=1, keepdim=True)
    q /= K
    q /= torch.sum(q, dim=0, keepdim=True)
    q /= B
  return (q * B).T


class HIMEstimator(nn.Module):
  """Estimates base linear velocity + an implicit latent (system response).

  The encoder maps a flattened observation history to ``[vel(3) | latent(L)]``.
  The latent is trained with contrastive (swapped-prediction) learning against a
  target network that encodes the next single-step observation, while the
  velocity head is trained with MSE against the true next base linear velocity.
  """

  def __init__(
    self,
    temporal_steps: int,
    num_one_step_obs: int,
    enc_hidden_dims: tuple[int, ...] = (128, 64, 16),
    tar_hidden_dims: tuple[int, ...] = (128, 64),
    activation: str = "elu",
    learning_rate: float = 1e-3,
    max_grad_norm: float = 10.0,
    num_prototype: int = 32,
    temperature: float = 3.0,
  ) -> None:
    super().__init__()
    act = get_activation(activation)

    self.temporal_steps = temporal_steps
    self.num_one_step_obs = num_one_step_obs
    self.num_latent = enc_hidden_dims[-1]
    self.max_grad_norm = max_grad_norm
    self.temperature = temperature

    # Encoder: flattened history -> [vel(3) | latent(L)].
    enc_input_dim = temporal_steps * num_one_step_obs
    enc_layers: list[nn.Module] = []
    for i in range(len(enc_hidden_dims) - 1):
      enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[i]), act]
      enc_input_dim = enc_hidden_dims[i]
    enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
    self.encoder = nn.Sequential(*enc_layers)

    # Target: single next-step obs -> latent(L).
    tar_input_dim = num_one_step_obs
    tar_layers: list[nn.Module] = []
    for i in range(len(tar_hidden_dims)):
      tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[i]), act]
      tar_input_dim = tar_hidden_dims[i]
    tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
    self.target = nn.Sequential(*tar_layers)

    # Prototypes for contrastive clustering.
    self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

    self.learning_rate = learning_rate
    self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

  # Inference -------------------------------------------------------------

  def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached (vel, normalized latent) for use by the policy."""
    parts = self.encoder(obs_history.detach())
    vel, z = parts[..., :3], parts[..., 3:]
    z = F.normalize(z, dim=-1, p=2)
    return vel.detach(), z.detach()

  def encode(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    parts = self.encoder(obs_history)
    vel, z = parts[..., :3], parts[..., 3:]
    z = F.normalize(z, dim=-1, p=2)
    return vel, z

  # Training --------------------------------------------------------------

  def update(
    self,
    obs_history: torch.Tensor,
    next_obs_frame: torch.Tensor,
    next_vel: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    lr: float | None = None,
  ) -> tuple[float, float]:
    """One optimization step of the estimator.

    Args:
      obs_history: flattened history ``[B, T*S]`` at step t (encoder input).
      next_obs_frame: single-step observation ``[B, S]`` at step t+1 (target input).
      next_vel: true base linear velocity ``[B, 3]`` at step t+1 (MSE target).
      valid_mask: optional ``[B]`` (or ``[B,1]``) float/bool mask zeroing out
        transitions that crossed an episode boundary (where t+1 is from a fresh
        episode and the "successor" relationship is invalid).
      lr: Deprecated, ignored. Estimator uses fixed LR from __init__ (HIMLoco parity).
    """
    # NOTE: lr parameter is intentionally ignored. HIMLoco's estimator uses a
    # fixed 1e-3 LR independent of PPO's adaptive LR. Param kept for backward
    # compatibility with old callers.
    del lr

    obs_history = obs_history.detach()
    next_obs_frame = next_obs_frame.detach()
    next_vel = next_vel.detach()

    # Build combined mask: episode-boundary mask AND physics-validity mask.
    # hfield collision overflow produces non-physical velocities (>>20 m/s or NaN/inf)
    # that would cause estimation_loss to spike to astronomical values.
    vel_finite = torch.isfinite(next_vel).all(dim=-1)          # [B]
    vel_reasonable = (next_vel.norm(dim=-1) < 5.0)              # [B] Boying max design ~1.5 m/s; 5 m/s guards hfield collision overflow
    physics_valid = (vel_finite & vel_reasonable).float()        # [B]
    if valid_mask is not None:
      m = valid_mask.float().view(-1) * physics_valid
    else:
      m = physics_valid
    denom = torch.clamp(m.sum(), min=1.0)

    z_s_full = self.encoder(obs_history)
    pred_vel, z_s = z_s_full[..., :3], z_s_full[..., 3:]
    z_t = self.target(next_obs_frame)

    z_s = F.normalize(z_s, dim=-1, p=2)
    z_t = F.normalize(z_t, dim=-1, p=2)

    # Keep prototypes on the unit sphere.
    with torch.no_grad():
      w = self.proto.weight.data.clone()
      w = F.normalize(w, dim=-1, p=2)
      self.proto.weight.copy_(w)

    score_s = z_s @ self.proto.weight.T
    score_t = z_t @ self.proto.weight.T

    with torch.no_grad():
      q_s = sinkhorn(score_s)
      q_t = sinkhorn(score_t)

    log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
    log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

    swap_per_sample = -0.5 * (q_s * log_p_t + q_t * log_p_s).sum(dim=-1)
    swap_loss = (swap_per_sample * m).sum() / denom
    vel_per_sample = F.mse_loss(pred_vel, next_vel, reduction="none").mean(dim=-1)
    estimation_loss = (vel_per_sample * m).sum() / denom

    losses = estimation_loss + swap_loss

    self.optimizer.zero_grad()
    losses.backward()
    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
    self.optimizer.step()

    return estimation_loss.item(), swap_loss.item()
