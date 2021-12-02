import os
import d3rlpy
import matplotlib.pyplot as plt

from d3rlpy.algos import DQN, DiscreteCQL, DiscreteBC, DiscreteSAC, DiscreteBCQ
from d3rlpy.datasets import get_dataset
from d3rlpy.metrics import evaluate_on_environment

from visualization.data_plot import plot_records_in_dir

ENV_NAME = 'cartpole-random'
LOG_DIR = os.path.join('d3rlpy_logs', ENV_NAME)

ALGOS = [DQN, DiscreteCQL, DiscreteBC, DiscreteSAC, DiscreteBCQ, DiscreteSoftCQL]
N_EPOCHS = 15
N_TRIALS = 3

# obtain dataset
dataset, env = get_dataset(ENV_NAME)

for t in range(N_TRIALS):
	d3rlpy.seed(t + 227 + 1998)
	env.seed(t + 227 + 1998)

	for algo in ALGOS:
		# setup algorithm
		agent = algo()
		agent.fit(
			dataset.episodes, 
			eval_episodes=[None], # dummy
			n_epochs=N_EPOCHS, 
			scorers={'rewards': evaluate_on_environment(env)},
			logdir=LOG_DIR
		)

# plot the results
plot_records_in_dir(log_dir=LOG_DIR, env_name=ENV_NAME, value_description='rewards')