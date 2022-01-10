import os
import d3rlpy
import matplotlib.pyplot as plt
import numpy as np

from d3rlpy.algos import DQN, DiscreteCQL, DiscreteBC, DiscreteSAC, DiscreteBCQ
from d3rlpy.metrics import evaluate_on_environment

from algos.cqgp import CQGP
from datasets import get_episodes
from visualization.data_plot import plot_records_in_dir

ENV_NAME = 'cartpole-random'
TESTING = True
if TESTING:
	LOG_DIR = os.path.join('d3rlpy_logs/test', ENV_NAME)
else:
	LOG_DIR = os.path.join('d3rlpy_logs/', ENV_NAME)

N_TRIALS = 1 if TESTING else 5
N_SUBSAMPLED_EPISODES = 10
ALGOS = [CQGP, DQN, DiscreteCQL, DiscreteBC, DiscreteSAC, DiscreteBCQ]

N_EPOCHS = 30

# obtain dataset
episodes, env = get_episodes(ENV_NAME, N_SUBSAMPLED_EPISODES)

for t in range(N_TRIALS):
	d3rlpy.seed(t + 227 + 1998)
	env.seed(t + 227 + 1998)
	np.random.seed(t + 227 + 1998)

	transition_len = sum([len(episode) for episode in episodes])
	print('Total size: ', transition_len)

	for algo in ALGOS:
		# setup algorithm
		agent = algo(max_buffer_size=transition_len)
		agent.fit(
			episodes,
			eval_episodes=[None], # dummy
			n_epochs=N_EPOCHS, 
			scorers={'rewards': evaluate_on_environment(env)},
			logdir=LOG_DIR
		)

# plot the results
plot_records_in_dir(log_dir=LOG_DIR, env_name=ENV_NAME, value_description='rewards')