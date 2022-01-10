# A wrapper file for d3rlpy.datasets

"""
	Some environments written in d3rlpy.datasets are deprecated or not included
	This file is a wrapper file that fixes some known deprecations
	It provides the same API
"""

import os
import gym # type: ignore
import numpy as np

from typing import Tuple, Optional, List, cast
from urllib import request

from d3rlpy.dataset import MDPDataset, Episode
from d3rlpy.datasets import (
	DATA_DIRECTORY, 
	PENDULUM_URL,
	PENDULUM_RANDOM_URL,
	get_dataset as d3rl_get_dataset
)

def get_pendulum_v1(dataset_type: str = "replay") -> Tuple[MDPDataset, gym.Env]:
	""" Adopted from d3rlpy.datasets.get_pendulum """
	if dataset_type == "replay":
		url = PENDULUM_URL
		file_name = "pendulum_replay.h5"
	elif dataset_type == "random":
		url = PENDULUM_RANDOM_URL
		file_name = "pendulum_random.h5"
	else:
		raise ValueError(f"Invalid dataset_type: {dataset_type}.")

	data_path = os.path.join(DATA_DIRECTORY, file_name)

	if not os.path.exists(data_path):
		os.makedirs(DATA_DIRECTORY, exist_ok=True)
		print(f"Donwloading pendulum.pkl into {data_path}...")
		request.urlretrieve(url, data_path)

	# load dataset
	dataset = MDPDataset.load(data_path)

	# environment
	env = gym.make("Pendulum-v1")

	return dataset, env

def get_acrobot_v1(num_of_episodes: Optional[int]) -> Tuple[List[Episode], gym.Env]:
	num_of_episodes = num_of_episodes or 10
	max_horizon = 1000

	env = gym.make("MountainCar-v0")
	obs_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n

	observations = np.zeros((0, obs_dim), dtype=np.float32)
	actions = np.zeros(0, dtype=np.int32)
	rewards = np.zeros(0, dtype=np.float32)
	terminals = np.zeros(0, dtype=np.float32)

	for _ in range(num_of_episodes):
		state = env.reset()
		for _ in range(max_horizon):
			action = np.random.choice(action_dim, 1)[0]
			next_state, reward, done, _ = env.step(action)

			observations = np.append(observations, [state], axis=0)
			actions = np.append(actions, action)
			rewards = np.append(rewards, reward)
			terminals = np.append(terminals, done)

			state = next_state
			if done:
				break

	dataset = MDPDataset(
		observations=observations,
		actions=actions,
		rewards=rewards,
		terminals=terminals,
		discrete_action=True,
	)

	return dataset.episodes, env


def _subsample_episodes(
	dataset: MDPDataset, 
	env: gym.Env, 
	subsampled_size: Optional[int] = None
) -> Tuple[List[Episode], gym.Env]:
	if not subsampled_size:
		return dataset.episodes, env
	
	sub_sampled_episodes = np.random.choice(
		np.array(dataset.episodes, dtype=object), subsampled_size)
	
	return cast(List[Episode], sub_sampled_episodes), env


def get_episodes(env_name: str, subsampled_size: Optional[int] = None) -> Tuple[List[Episode], gym.Env]:
	if env_name == "pendulum-replay":
		return _subsample_episodes(*get_pendulum_v1(dataset_type="replay"), subsampled_size)
	elif env_name == "pendulum-random":
		return _subsample_episodes(*get_pendulum_v1(dataset_type="random"), subsampled_size)
	elif env_name == 'acrobot-random':
		return get_acrobot_v1(subsampled_size)
	else:
		return _subsample_episodes(*d3rl_get_dataset(env_name), subsampled_size)
