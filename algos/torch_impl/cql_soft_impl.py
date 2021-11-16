import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch import Parameter
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.algos.torch.cql_impl import CQLImpl, DiscreteCQLImpl


class CQLSoftImpl(CQLImpl):

    _alpha_learning_rate: float
    _alpha_optim_factory: OptimizerFactory
    _initial_alpha: float
    _alpha_threshold: float
    _conservative_weight: float
    _n_action_samples: int
    _soft_q_backup: bool
    _log_alpha: Optional[Parameter]
    _alpha_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        temp_learning_rate: float,
        alpha_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        alpha_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        target_reduction_type: str,
        initial_temperature: float,
        initial_alpha: float,
        alpha_threshold: float,
        conservative_weight: float,
        n_action_samples: int,
        soft_q_backup: bool,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            temp_learning_rate=temp_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=temp_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            target_reduction_type=target_reduction_type,
            initial_temperature=initial_temperature,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._alpha_learning_rate = alpha_learning_rate
        self._alpha_optim_factory = alpha_optim_factory
        self._initial_alpha = initial_alpha
        self._alpha_threshold = alpha_threshold
        self._conservative_weight = conservative_weight
        self._n_action_samples = n_action_samples
        self._soft_q_backup = soft_q_backup

        # initialized in build
        self._log_alpha = None
        self._alpha_optim = None

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor, obs_tp1: torch.Tensor
    ) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        assert self._log_alpha is not None

        policy_values_t = self._compute_policy_is_values(obs_t, obs_t)
        policy_values_tp1 = self._compute_policy_is_values(obs_tp1, obs_t)
        random_values = self._compute_random_is_values(obs_t)

        # compute logsumexp
        # (n critics, batch, 3 * n samples) -> (n critics, batch, 1)
        target_values = torch.cat(
            [policy_values_t, policy_values_tp1, random_values], dim=2
        )

        softmax_nn = torch.nn.Softmax(dim=2) ### Just change here soft max
        softmax_weight = softmax_nn(target_values)
        softmax = torch.sum(softmax_weight * target_values, dim=2, keepdim=True) 

        # estimate action-values for data actions
        data_values = self._q_func(obs_t, act_t, "none")

        loss = softmax.mean(dim=0).mean() - data_values.mean(dim=0).mean()
        scaled_loss = self._conservative_weight * loss

        # clip for stability
        clipped_alpha = self._log_alpha().exp().clamp(0, 1e6)[0][0]

        return clipped_alpha * (scaled_loss - self._alpha_threshold)


class DiscreteCQLSoftImpl(DiscreteCQLImpl):
    _alpha: float

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        n_critics: int,
        target_reduction_type: str,
        alpha: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            n_critics=n_critics,
            target_reduction_type=target_reduction_type,
            use_gpu=use_gpu,
            scaler=scaler,
            reward_scaler=reward_scaler,
        )
        self._alpha = alpha

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        # compute logsumexp
        policy_values = self._q_func(obs_t)
        # logsumexp = torch.logsumexp(policy_values, dim=1, keepdim=True)


        softmax_nn = torch.nn.Softmax(dim=1)
        softmax_weight = softmax_nn(policy_values)
        softmax = torch.sum(softmax_weight * policy_values, dim=1, keepdim=True) 


        # estimate action-values under data distribution
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        data_values = (self._q_func(obs_t) * one_hot).sum(dim=1, keepdim=True)

        return (softmax - data_values).mean()
