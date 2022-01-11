import gym
import numpy as np

from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast

from d3rlpy.metrics.scorer import AlgoProtocol

def evaluate_initial_state_value_estimate(env: gym.Env, n_trials: int = 10) -> Callable[..., float]:
    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        initial_state_value_estimate = []
        for _ in range(n_trials):
            observation = env.reset()

            q_values = []
            for action in range(env.action_space.n):
                q_estimate = algo.predict_value(
                    np.array([observation]), 
                    np.array([action])
                )
                q_values.append(q_estimate)

            initial_state_value_estimate.append(max(q_values))
        return float(np.mean(initial_state_value_estimate))

    return scorer
