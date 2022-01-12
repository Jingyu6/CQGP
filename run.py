import os
import shutil
import random
import argparse

import d3rlpy
import matplotlib.pyplot as plt
import numpy as np

from d3rlpy.algos import DQN, DiscreteCQL, DiscreteBC, DiscreteSAC, DiscreteBCQ
from d3rlpy.metrics import evaluate_on_environment

from algos.cqgp import CQGP
from datasets import get_episodes
from visualization.data_plot import plot_records_in_dir

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", type=str, default='cartpole-random', choices=['cartpole-replay', 'cartpole-random'])
parser.add_argument("-t", "--testing", action="store_true")
parser.add_argument("-n", "--num_of_trials", type=int, default=5)
parser.add_argument("-x", "--num_of_episodes", type=int, default=20)
parser.add_argument("-i", "--num_of_epochs", type=int, default=20)
parser.add_argument("-m", "--q_std_multiplier", type=int, default=20)

args = parser.parse_args()

print("Start experiments: ", args)

if args.testing:
	LOG_DIR = os.path.join('d3rlpy_logs/test', args.env)
	shutil.rmtree(LOG_DIR)
else:
	LOG_DIR = os.path.join('d3rlpy_logs/', args.env)

ALGOS = [CQGP, DQN, DiscreteCQL, DiscreteBC, DiscreteSAC, DiscreteBCQ]

def set_random_seed(s):
	d3rlpy.seed(t)
	random.seed(t)
	np.random.seed(t)

for t in range(args.num_of_trials):
	set_random_seed(t + 227 + 1998)

	# obtain dataset
	episodes, env = get_episodes(args.env, args.num_of_episodes)
	env.seed(t)

	transition_len = sum([len(episode) for episode in episodes])
	print('Total size: ', transition_len)

	for algo in ALGOS:
		# setup algorithm
		agent = algo(max_buffer_size=transition_len, q_std_multiplier=args.q_std_multiplier)
		agent.fit(
			episodes,
			eval_episodes=[None], # dummy
			n_epochs=args.num_of_epochs, 
			scorers={
				'rewards': evaluate_on_environment(env)
			},
			logdir=LOG_DIR
		)

# plot the results
plot_records_in_dir(log_dir=LOG_DIR, env_name=args.env, value_description='rewards')
