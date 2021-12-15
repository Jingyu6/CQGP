import copy
from typing import Optional, Sequence, Tuple, Union, Dict

import numpy as np
from d3rlpy.dataset import TransitionMiniBatch
from sklearn.gaussian_process import GaussianProcessRegressor # type: ignore
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel, WhiteKernel # type: ignore

from .base import SklearnImplBase

class GPQLBuffer:
    _input_size: int
    _action_size: int
    _observation_shape: Sequence[int]

    _max_buffer_size: int
    _cur_size: int
    _cur_ptr: int

    def __init__(
        self, 
        observation_shape: Sequence[int],
        action_size: int,
        max_buffer_size: int
    ):
        self._action_size = action_size # one hot encoding
        self._observation_shape = observation_shape
        self._input_size = action_size + sum([d for d in observation_shape])

        self._max_buffer_size = max_buffer_size
        self._cur_size = 0
        self._cur_ptr = 0
        self._input_buffer = np.zeros((max_buffer_size, self._input_size), dtype=np.float32)
        self._output_buffer = np.zeros((max_buffer_size, 1), dtype=np.float32)

        self._action_scaler = 1

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._input_buffer[:self._cur_size], self._output_buffer[:self._cur_size]

    def add_points(self, states: np.ndarray, actions: np.ndarray, q_values: np.ndarray) -> None:
        batch_size = states.shape[0]
        if batch_size == 0:
            return

        cap = self._cur_ptr + batch_size
        if cap <= self._max_buffer_size:
            input_values = self.stitch_state_action(states, actions)
            self._input_buffer[self._cur_ptr:cap] = input_values
            self._output_buffer[self._cur_ptr:cap] = q_values
            
            self._cur_ptr = cap % self._max_buffer_size
            self._cur_size = min(self._max_buffer_size, self._cur_size + batch_size)
        else:
            tail_num = self._max_buffer_size - self._cur_ptr

            self.add_points(states[:tail_num], actions[:tail_num], q_values[:tail_num])
            self.add_points(states[tail_num:], actions[tail_num:], q_values[tail_num:])

    def stitch_state_action(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        batch_size = states.shape[0]

        one_hot_actions = np.zeros((batch_size, self._action_size), dtype=np.float32)
        one_hot_actions[np.arange(batch_size), actions] = self._action_scaler
        flattened_states = states.reshape(batch_size, -1)
        stitch_input = np.concatenate((states, one_hot_actions), axis=-1)

        return stitch_input

    def is_full(self):
        return self._cur_size == self._max_buffer_size

    def __str__(self):
        s = 'GPQLBuffer: buffer_size={}, cur_size={}'.format(self._max_buffer_size, self._cur_size)
        if self._cur_size > 0:
            s += ', target_mean={}, target_std={}'.format(
                np.mean(self._output_buffer[:self._cur_size]), np.std(self._output_buffer[:self._cur_size]))
        return s

class GPQLFullBuffer:
    _input_size: int
    _action_size: int
    _observation_shape: Sequence[int]

    _max_buffer_size: int
    _cur_size: int
    _cur_ptr: int

    def __init__(
        self, 
        observation_shape: Sequence[int],
        action_size: int,
        max_buffer_size: int
    ):
        self._action_size = action_size # one hot encoding
        self._observation_shape = observation_shape
        self._input_size = action_size + sum([d for d in observation_shape])

        self._max_buffer_size = max_buffer_size
        self._cur_size = 0
        self._cur_ptr = 0

        self._input_buffer = np.zeros((max_buffer_size, self._input_size), dtype=np.float32)
        self._next_obs_buffer = np.zeros((max_buffer_size, sum([d for d in observation_shape])), dtype=np.float32)
        self._rewards_buffer = np.zeros((max_buffer_size, 1), dtype=np.float32)
        self._terminal_buffer = np.zeros((max_buffer_size, 1), dtype=np.float32)
        self._output_buffer = np.zeros((max_buffer_size, 1), dtype=np.float32)

        self._action_scaler = 0.5

    def get_training_data(self) -> Dict[str, np.ndarray]:
        return dict(
            input_data=self._input_buffer[:self._cur_size],
            next_obs_data=self._next_obs_buffer[:self._cur_size],
            reward_data=self._rewards_buffer[:self._cur_size],
            terminal_data=self._terminal_buffer[:self._cur_size],
            output_data=self._output_buffer[:self._cur_size]
        )

    def add_points(
        self, 
        states: np.ndarray, 
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        terminals: np.ndarray,
        q_values: np.ndarray
    ) -> None:
        batch_size = states.shape[0]
        if batch_size == 0:
            return

        cap = self._cur_ptr + batch_size
        if cap <= self._max_buffer_size:
            input_values = self.stitch_state_action(states, actions)
            
            self._input_buffer[self._cur_ptr:cap] = input_values
            self._next_obs_buffer[self._cur_ptr:cap] = next_states
            self._rewards_buffer[self._cur_ptr:cap] = rewards
            self._terminal_buffer[self._cur_ptr:cap] = terminals
            self._output_buffer[self._cur_ptr:cap] = q_values

            self._cur_ptr = cap % self._max_buffer_size
            self._cur_size = min(self._max_buffer_size, self._cur_size + batch_size)
        else:
            tail_num = self._max_buffer_size - self._cur_ptr

            self.add_points(
                states[:tail_num], 
                actions[:tail_num], 
                rewards[:tail_num], 
                next_states[:tail_num], 
                terminals[:tail_num], 
                q_values[:tail_num]
            )
            self.add_points(
                states[tail_num:], 
                actions[tail_num:], 
                rewards[tail_num:], 
                next_states[tail_num:], 
                terminals[tail_num:], 
                q_values[tail_num:]
            )

    def update_targets(
        self,
        targets: np.ndarray
    ) -> None:
        assert targets.shape[0] <= self._cur_size, 'Target len too large'
        self._output_buffer[:self._cur_size] = targets

    def stitch_state_action(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        batch_size = states.shape[0]

        one_hot_actions = np.zeros((batch_size, self._action_size), dtype=np.float32)
        one_hot_actions[np.arange(batch_size), actions] = self._action_scaler
        flattened_states = states.reshape(batch_size, -1)
        stitch_input = np.concatenate((states, one_hot_actions), axis=-1)

        return stitch_input

    def is_full(self):
        return self._cur_size == self._max_buffer_size

    def clean(self):
        self._cur_ptr = 0
        self._cur_size = 0

    def __str__(self):
        s = 'GPQLFullBuffer: buffer_size={}, cur_size={}'.format(self._max_buffer_size, self._cur_size)
        if self._cur_size > 0:
            s += ', target_mean={}, target_std={}'.format(
                np.mean(self._output_buffer[:self._cur_size]), np.std(self._output_buffer[:self._cur_size]))
        return s

class GPQFunction:

    _buffer: GPQLBuffer
    _gp: GaussianProcessRegressor
    _action_size: int
    _observation_shape: Sequence[int]
    _cur_update_cnt: int
    _update_interval: int

    def __init__(self,
        action_size: int, 
        observation_shape: Sequence[int],
        max_buffer_size: int = 2000
    ):
        self._buffer = GPQLBuffer(observation_shape, action_size, max_buffer_size)
        self._gp = GaussianProcessRegressor(
            kernel=ConstantKernel(constant_value=1.0, constant_value_bounds='fixed')
                * RationalQuadratic(length_scale=2.0, alpha=2.0, length_scale_bounds='fixed', alpha_bounds='fixed')
                + WhiteKernel(noise_level=0.01, noise_level_bounds='fixed'),
            normalize_y=True
        )

        self._action_size = action_size
        self._observation_shape = observation_shape
        self._cur_update_cnt = 0
        self._update_interval = 2

    def __call__(
        self, 
        states: np.ndarray, 
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """ return shape [B, A] """
        batch_size = states.shape[0]
        
        all_actions = np.tile(np.arange(self._action_size), reps=batch_size)
        duplicate_states = np.repeat(states, repeats=self._action_size, axis=0)

        if return_std:
            q_values_mean, q_values_std = self.evaluate(states=duplicate_states, actions=all_actions, return_std=True)
            return q_values_mean.reshape(batch_size, -1), q_values_std.reshape(batch_size, -1)
        else:
            q_values = self.evaluate(states=duplicate_states, actions=all_actions, return_std=False)
            assert isinstance(q_values, np.ndarray)
            return q_values.reshape(batch_size, -1)

    def add_points(self, states: np.ndarray, actions: np.ndarray, q_values: np.ndarray) -> None:
        self._buffer.add_points(states, actions, q_values)

    def evaluate(
        self, 
        states: np.ndarray, 
        actions: np.ndarray,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """ return shape [B] """
        input_values = self._buffer.stitch_state_action(states, actions)
        return self._gp.predict(input_values, return_std=return_std)

    def update(self) -> None:
        self._cur_update_cnt += 1
        if self._cur_update_cnt >= self._update_interval:
            input_values, output_values = self._buffer.get_training_data()
            # print(self._buffer)
            self._gp.fit(input_values, output_values)
            self._cur_update_cnt = 0

class GPMultiUpdateQFunction:

    _buffer: GPQLFullBuffer
    _gp: GaussianProcessRegressor
    _action_size: int
    _observation_shape: Sequence[int]
    _cur_update_cnt: int
    _update_interval: int
    _update_steps: int
    _gamma: float

    def __init__(self,
        action_size: int, 
        observation_shape: Sequence[int],
        max_buffer_size: int = 500,
        gamma: float = 0.99
    ):
        self._buffer = GPQLFullBuffer(observation_shape, action_size, max_buffer_size)
        self._gp = GaussianProcessRegressor(
            kernel=ConstantKernel(constant_value=1.0, constant_value_bounds='fixed')
                * RationalQuadratic(length_scale=3.0, alpha=3.0, length_scale_bounds='fixed', alpha_bounds='fixed')
                + WhiteKernel(noise_level=0.01, noise_level_bounds='fixed'),
            normalize_y=True
        )

        self._action_size = action_size
        self._observation_shape = observation_shape
        self._gamma = gamma
        
        self._cur_update_cnt = 0
        self._update_interval = 20
        self._update_steps = 30

        self._iter_cnt = 0

    def __call__(
        self, 
        states: np.ndarray, 
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """ return shape [B, A] """
        batch_size = states.shape[0]
        
        all_actions = np.tile(np.arange(self._action_size), reps=batch_size)
        duplicate_states = np.repeat(states, repeats=self._action_size, axis=0)

        if return_std:
            q_values_mean, q_values_std = self.evaluate(states=duplicate_states, actions=all_actions, return_std=True)
            return q_values_mean.reshape(batch_size, -1), q_values_std.reshape(batch_size, -1)
        else:
            q_values = self.evaluate(states=duplicate_states, actions=all_actions, return_std=False)
            assert isinstance(q_values, np.ndarray)
            return q_values.reshape(batch_size, -1)

    def add_points(
        self, 
        states: np.ndarray, 
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        terminals: np.ndarray,
        q_values: np.ndarray
    ) -> None:
        if not self._buffer.is_full():
            self._buffer.add_points(
                states,
                actions,
                rewards,
                next_states,
                terminals,
                q_values
            )

    def evaluate(
        self, 
        states: np.ndarray, 
        actions: np.ndarray,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """ return shape [B] """
        input_values = self._buffer.stitch_state_action(states, actions)
        return self._gp.predict(input_values, return_std=return_std)

    def _update_buffer_target(self) -> None:
        d_map = self._buffer.get_training_data()

        state_action_pairs = d_map['input_data']
        next_states = d_map['next_obs_data']
        rewards = d_map['reward_data']
        terminals = d_map['terminal_data']

        cur_q_estimate = self._gp.predict(state_action_pairs, return_std=False)
        cur_q_estimate = cur_q_estimate.reshape(-1, 1)

        next_q_mean, next_q_std = self(next_states, return_std=True)

        next_q_estimate = next_q_mean - next_q_std
        bellman_target = rewards + self._gamma * next_q_estimate.max(-1, keepdims=True) * (1 - terminals)

        adaptive_lr = 0.03
        new_target = cur_q_estimate + adaptive_lr * (bellman_target - cur_q_estimate)
        self._buffer.update_targets(new_target)

    def update(self) -> None:
        self._cur_update_cnt += 1
        if self._cur_update_cnt >= self._update_interval and self._buffer.is_full():
            for _ in range(self._update_steps):
                self._update_buffer_target()
                d_map = self._buffer.get_training_data()
                input_values, output_values = d_map['input_data'], d_map['output_data']
                self._gp.fit(input_values, output_values)
            self._cur_update_cnt = 0

class GPQLImpl(SklearnImplBase):

    _gamma: float
    _q_func: Optional[GPQFunction]
    _action_size: int
    _observation_shape: Sequence[int]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        gamma: float,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size
        )
        self._observation_shape = observation_shape
        self._action_size = action_size
        self._gamma = gamma

        # initialized in build
        self._q_func = None

    def build(self) -> None:
        self._build_network()

    def _build_network(self) -> None:
        self._q_func = GPQFunction(
            action_size=self._action_size, 
            observation_shape=self._observation_shape
        )

    def _predict_value(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        with_std: bool,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        assert self._q_func is not None
        return self._q_func.evaluate(states, actions, with_std)

    def update(self, batch: TransitionMiniBatch) -> float:
        assert self._q_func is not None
        
        # compute target
        targets = self.compute_target(batch)
        # compute loss
        loss = self.compute_loss(batch, targets)
        # add to buffer
        self._q_func.add_points(batch.observations, batch.actions, targets)
        # compute posterior
        self._q_func.update()

        return loss

    def compute_target(self, batch: TransitionMiniBatch) -> np.ndarray:
        assert self._q_func is not None
        
        rewards = batch.rewards
        next_q_values_mean, next_q_values_std = self._q_func(batch.next_observations, return_std=True)

        """
        # we clip our standard deviation so that the target will not blow up
        clipped_next_q_values_std = np.clip(
            next_q_values_std, 
            0, # make sure it is positive 
            next_q_values_mean * 0.2 # we don't want to penalize it too much
        )

        next_conservative_q_values = next_q_values_mean - next_q_values_std
        """
        next_conservative_q_values = next_q_values_mean
        next_max_conservative_q_values = next_conservative_q_values.max(-1, keepdims=True)

        return rewards + self._gamma * next_max_conservative_q_values

    def compute_loss(
        self,
        batch: TransitionMiniBatch,
        targets: np.ndarray,
    ) -> float:
        assert self._q_func is not None
        # get current mean Q(s, a)
        estimated_q_values = self._q_func.evaluate(batch.observations, batch.actions)
        # compute the mse
        return np.mean((estimated_q_values - targets) ** 2)

    def _predict_best_action(self, x: np.ndarray) -> np.ndarray:
        assert self._q_func is not None
        q_values = self._q_func(x)
        assert isinstance(q_values, np.ndarray)
        return q_values.argmax(axis=-1)

    def _sample_action(self, x: np.ndarray) -> np.ndarray:
        return self._predict_best_action(x)

class GPMultiUpdateQLImpl(SklearnImplBase):

    _gamma: float
    _q_func: Optional[GPMultiUpdateQFunction]
    _max_buffer_size: int
    _action_size: int
    _observation_shape: Sequence[int]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        gamma: float,
        max_buffer_size: int = 500
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size
        )
        self._observation_shape = observation_shape
        self._action_size = action_size
        self._gamma = gamma
        self._max_buffer_size = max_buffer_size

        # initialized in build
        self._q_func = None

    def build(self) -> None:
        self._build_network()

    def _build_network(self) -> None:
        self._q_func = GPMultiUpdateQFunction(
            action_size=self._action_size, 
            observation_shape=self._observation_shape,
            gamma=self._gamma,
            max_buffer_size=self._max_buffer_size
        )

    def _predict_value(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        with_std: bool,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        assert self._q_func is not None
        return self._q_func.evaluate(states, actions, with_std)

    def update(self, batch: TransitionMiniBatch) -> float:
        assert self._q_func is not None

        # compute target
        targets = self.compute_target(batch)
        # compute loss
        loss = self.compute_loss(batch, targets)
        # add to buffer
        self._q_func.add_points(
            batch.observations, 
            batch.actions, 
            batch.rewards,
            batch.next_observations,
            batch.terminals,
            targets
        )
        # compute posterior
        self._q_func.update()

        return loss

    def compute_target(self, batch: TransitionMiniBatch) -> np.ndarray:
        assert self._q_func is not None
        
        rewards = batch.rewards
        terminals = batch.terminals

        next_q_values = self._q_func(batch.next_observations, return_std=False)

        return rewards + self._gamma * next_q_values.max(-1, keepdims=True) * (1 - terminals)

    def compute_loss(
        self,
        batch: TransitionMiniBatch,
        targets: np.ndarray,
    ) -> float:
        assert self._q_func is not None
        # get current mean Q(s, a)
        estimated_q_values = self._q_func.evaluate(batch.observations, batch.actions)
        # compute the mse
        return np.mean((estimated_q_values - targets) ** 2)

    def _predict_best_action(self, x: np.ndarray) -> np.ndarray:
        assert self._q_func is not None
        q_values_mean, q_values_std = self._q_func(x, return_std=True)
        return (q_values_mean - q_values_std).argmax(axis=-1)

    def _sample_action(self, x: np.ndarray) -> np.ndarray:
        return self._predict_best_action(x)
