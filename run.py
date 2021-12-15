import os
import d3rlpy
import matplotlib.pyplot as plt
import numpy as np

from d3rlpy.algos import DQN, DiscreteCQL, DiscreteBC, DiscreteSAC, DiscreteBCQ
from d3rlpy.datasets import get_dataset
from d3rlpy.metrics import evaluate_on_environment

from visualization.data_plot import plot_records_in_dir
from algos.cqgp import CQGP

ENV_NAME = 'cartpole-random'
LOG_DIR = os.path.join('d3rlpy_logs/', ENV_NAME)

N_TRIALS = 5
ALGOS = [CQGP, DQN, DiscreteCQL, DiscreteBC, DiscreteSAC, DiscreteBCQ]

N_EPOCHS = 30

# obtain dataset
dataset, env = get_dataset(ENV_NAME)

for t in range(N_TRIALS):
	d3rlpy.seed(t + 227 + 1998)
	env.seed(t + 227 + 1998)
	np.random.seed(t + 227 + 1998)	

	sub_sampled_dataset = np.random.choice(np.array(dataset.episodes, dtype=object), 20)
	transition_len = sum([len(episode) for episode in sub_sampled_dataset])
	print('Total size: ', transition_len)

	for algo in ALGOS:
		# setup algorithm
		agent = algo(max_buffer_size=transition_len)
		agent.fit(
			# dataset.episodes, 
			sub_sampled_dataset,
			eval_episodes=[None], # dummy
			n_epochs=N_EPOCHS, 
			scorers={'rewards': evaluate_on_environment(env)},
			logdir=LOG_DIR
		)

# plot the results
plot_records_in_dir(log_dir=LOG_DIR, env_name=ENV_NAME, value_description='rewards')