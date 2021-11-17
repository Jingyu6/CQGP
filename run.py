import d3rlpy
import matplotlib.pyplot as plt

from d3rlpy.algos import DQN, DiscreteCQL, DiscreteBC, DiscreteSAC, DiscreteBCQ
from d3rlpy.datasets import get_dataset
from d3rlpy.metrics import evaluate_on_environment
from sklearn.model_selection import train_test_split

from visualization.data_parser import D3rlpyCSVDataParser
from visualization.data_plot import plot_records_list, plot_records_in_dir

ENV_NAME = 'cartpole-random'
ALGOS = [DQN, DiscreteCQL, DiscreteBC, DiscreteSAC, DiscreteBCQ]
N_EPOCHS = 15
N_TRIALS = 3

# obtain dataset
dataset, env = get_dataset(ENV_NAME)
_, test_episodes = train_test_split(dataset, test_size=0.1) # this is not important

for t in range(N_TRIALS):
	d3rlpy.seed(t + 227 + 1998)
	env.seed(t + 227 + 1998)

	for algo in ALGOS:
		# setup algorithm
		agent = algo()
		agent.fit(
			dataset.episodes, 
			eval_episodes=test_episodes,
			n_epochs=N_EPOCHS, 
			scorers={'rewards': evaluate_on_environment(env)}
		)

# plot the results
plot_records_in_dir(log_dir='d3rlpy_logs', env_name=ENV_NAME, value_description='rewards')