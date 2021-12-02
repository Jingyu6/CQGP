import numpy as np
from typing import Optional, Sequence, Any, Union, List, Tuple

from d3rlpy.algos.base import AlgoImplBase


class SklearnImplBase(AlgoImplBase):

    _observation_shape: Sequence[int]
    _action_size: int

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
    ):
        self._observation_shape = observation_shape
        self._action_size = action_size

    def predict_best_action(self, states: Union[np.ndarray, List[Any]]) -> np.ndarray:
        if isinstance(states, np.ndarray):
            assert states.ndim > 1, "Input must have batch dimension."
            return self._predict_best_action(states)
        else:
            # later use a decorator for conversion
            states = np.array(states)
            return self._predict_best_action(states)

    def _predict_best_action(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample_action(self, states: Union[np.ndarray, List[Any]]) -> np.ndarray:
        if isinstance(states, np.ndarray):
            assert states.ndim > 1, "Input must have batch dimension."
            return self._sample_action(states)
        else:
            # later use a decorator for conversion
            raise TypeError

    def _sample_action(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_value(
        self,
        states: Union[np.ndarray, List[Any]],
        actions: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if isinstance(states, np.ndarray) and isinstance(actions, np.ndarray):
            assert states.ndim > 1, "Input must have batch dimension."
            return self._predict_value(states, actions, with_std)
        else:
            # later use a decorator for conversion
            raise TypeError            

    def _predict_value(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        with_std: bool,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    """ From ImplBase """
    def save_model(self, fname: str) -> None:
        pass

    def load_model(self, fname: str) -> None:
        pass

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape

    @property
    def action_size(self) -> int:
        return self._action_size

    """ From AlgoImplBase """
    def save_policy(self, fname: str, as_onnx: bool) -> None:
        pass

    def copy_policy_from(self, impl: AlgoImplBase) -> None:
        pass

    def copy_policy_optim_from(self, impl: AlgoImplBase) -> None:
        pass

    def copy_q_function_from(self, impl: AlgoImplBase) -> None:
        pass

    def copy_q_function_optim_from(self, impl: AlgoImplBase) -> None:
        pass

    def reset_optimizer_states(self) -> None:
        pass
