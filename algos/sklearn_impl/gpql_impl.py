import copy
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from d3rlpy.dataset import TransitionMiniBatch
from sklearn.gaussian_process import GaussianProcessRegressor # type: ignore
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel # type: ignore

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
        one_hot_actions[np.arange(batch_size), actions] = 1
        flattened_states = states.reshape(batch_size, -1)
        stitch_input = np.concatenate((states, one_hot_actions), axis=-1)

        return stitch_input


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
        max_buffer_size: int = 5000
    ):
        self._buffer = GPQLBuffer(observation_shape, action_size, max_buffer_size)
        self._gp = GaussianProcessRegressor(
            kernel=RBF(),
            normalize_y=True
        )

        self._action_size = action_size
        self._observation_shape = observation_shape
        self._cur_update_cnt = 0
        self._update_interval = 10

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
            print(input_values.shape, output_values.shape)
            print('Start fitting GP, with data size: {}'.format(input_values.shape[0]))
            self._gp.fit(input_values, output_values)
            print('Finished fitting GP.')
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

        next_conservative_q_values = next_q_values_mean - next_q_values_std
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
