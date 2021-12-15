from typing import Any, Dict, Optional, Sequence

from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.algos.base import AlgoBase

from .sklearn_impl.gpql_impl import GPQLImpl, GPMultiUpdateQLImpl


class GPQL(AlgoBase):

    _impl: Optional[GPQLImpl]
    _gamma: float

    def __init__(
        self,
        *,
		batch_size: int = 32,
		n_steps: int = 1,
		gamma: float = 0.99,
        impl: Optional[GPQLImpl] = None,
        **kwargs: Any,
    ):
        super().__init__(
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            # use only frames = 1 for now
            n_frames=1,
            # disable scalar functionality for now due to lack of support for np.ndarray
            scaler=None,
            reward_scaler=None,
            kwargs=kwargs,
        )
        self._impl = impl
        self._gamma = gamma

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = GPQLImpl(
            observation_shape=observation_shape,
        	action_size=action_size,
        	gamma=self._gamma,
        )
        self._impl.build()
    
    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update(batch)
        return { "loss": loss }

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE

class GPMultiUpdateQL(AlgoBase):

    _impl: Optional[GPMultiUpdateQLImpl]
    _gamma: float
    _max_buffer_size: int

    def __init__(
        self,
        *,
        batch_size: int = 32,
        n_steps: int = 1,
        gamma: float = 0.99,
        impl: Optional[GPMultiUpdateQLImpl] = None,
        max_buffer_size: int = 1000,
        **kwargs: Any,
    ):
        super().__init__(
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            # use only frames = 1 for now
            n_frames=1,
            # disable scalar functionality for now due to lack of support for np.ndarray
            scaler=None,
            reward_scaler=None,
            kwargs=kwargs,
        )
        self._impl = impl
        self._gamma = gamma
        self._max_buffer_size = max_buffer_size

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = GPMultiUpdateQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            gamma=self._gamma,
            max_buffer_size=self._max_buffer_size
        )
        self._impl.build()
    
    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update(batch)
        return { "loss": loss }

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE
