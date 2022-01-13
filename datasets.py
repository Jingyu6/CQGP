# A wrapper file for d3rlpy.datasets

"""
	Some environments written in d3rlpy.datasets are deprecated or not included
	This file is a wrapper file that fixes some known deprecations
	It provides the same API
"""

import os
import gym # type: ignore
import argparse
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


def get_ppo_dataset(env_name: str, dataset_type: str = "replay") -> Tuple[MDPDataset, gym.Env]:	
	file_name = '{}_{}.h5'.format(env_name, dataset_type)
	data_path = os.path.join(DATA_DIRECTORY, file_name)

	if not os.path.exists(data_path):
		raise ValueError(f"Dataset {data_path} not found. Please call generate dataset first.")		

	# load dataset
	dataset = MDPDataset.load(data_path)

	# environment
	env = gym.make('Acrobot-v1' if env_name == 'acrobot' else 'MountainCar-v0')

	return dataset, env


def _subsample_episodes(
	dataset: MDPDataset, 
	env: gym.Env, 
	subsampled_size: Optional[int] = None
) -> Tuple[List[Episode], gym.Env]:
	if not subsampled_size:
		return dataset.episodes, env
	
	sub_sampled_indices = np.random.choice(len(dataset.episodes), subsampled_size)
	sub_sampled_episodes = []

	for i in sub_sampled_indices:
		sub_sampled_episodes.append(dataset.episodes[i])

	return cast(List[Episode], sub_sampled_episodes), env


def get_episodes(env_name: str, subsampled_size: Optional[int] = None) -> Tuple[List[Episode], gym.Env]:
	if env_name == "pendulum-replay":
		return _subsample_episodes(*get_pendulum_v1(dataset_type="replay"), subsampled_size)
	elif env_name == "pendulum-random":
		return _subsample_episodes(*get_pendulum_v1(dataset_type="random"), subsampled_size)
	elif 'acrobot' in env_name or 'mountaincar' in env_name:
		return _subsample_episodes(*get_ppo_dataset(*env_name.split('-')), subsampled_size)
	else:
		return _subsample_episodes(*d3rl_get_dataset(env_name), subsampled_size)


def generate_dataset_by_PPO(
	env_name: str, 
	num_of_episodes: int = 20,
	total_timesteps: int = 20000,
	dataset_name: Optional[str] = None
) -> None:
	from stable_baselines3 import PPO

	env = gym.make(env_name)

	obs_dim = env.observation_space.shape[0]

	model = PPO("MlpPolicy", env)
	model.learn(total_timesteps=total_timesteps)

	observations = np.zeros((0, obs_dim), dtype=np.float32)
	actions = np.zeros(0, dtype=np.int32)
	rewards = np.zeros(0, dtype=np.float32)
	terminals = np.zeros(0, dtype=np.float32)

	for _ in range(num_of_episodes):
		obs = env.reset()
		for i in range(120):
			action, _ = model.predict(obs, deterministic=True)
			next_obs, reward, done, _ = env.step(action)

			# manually set the terminal flat to be 1
			if i == 119:
				done = 1

			observations = np.append(observations, [obs], axis=0)
			actions = np.append(actions, action)
			rewards = np.append(rewards, reward)
			terminals = np.append(terminals, done)

			obs = next_obs
			if done:
				break

	dataset = MDPDataset(
		observations=observations,
		actions=actions,
		rewards=rewards,
		terminals=terminals,
		discrete_action=True,
	)

	print('Finished generating dataset, stats: {}'.format(dataset.compute_stats()))

	dataset_path = \
		'd3rlpy_data/{}_replay.h5'.format(env_name.split('_')[0].lower()) if not dataset_name else \
		'd3rlpy_data/{}.h5'.format(dataset_name)
	dataset.dump(dataset_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-e", "--env_name", type=str, default='Acrobot-v1', choices=['Acrobot-v1', 'MountainCar-v0', 'CartPole-v0'])
	parser.add_argument("-x", "--num_of_episodes", type=int, default=20)
	parser.add_argument("-i", "--total_timesteps", type=int, default=20000)
	parser.add_argument("-n", "--dataset_name", type=str)

	args = parser.parse_args()
	generate_dataset_by_PPO(**vars(args))
