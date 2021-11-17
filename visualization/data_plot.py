import gym # type: ignore
import matplotlib # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np

from pathlib import Path # type: ignore
from typing import List, Union, Literal, Dict, Any
from visualization.data_parser import Records, D3rlpyCSVDataParser

def plot_records_list(
		axes: matplotlib.axes.Axes,
		records_list: List[Records],
		env_name: str,
		value_description: str = 'loss',
		horizon_name: Union[Literal['epochs', 'steps']] = 'epochs',
		**kwargs: Any # arguments to the plot function
	) -> None:
	""" 
		Plot the graph of different algorithms,
		each algorithm contains multiple experiments,
		all experiments are from the same environment
	"""
	assert len(records_list) > 0, "Can not pass in empty records."
	
	# group them together
	algo_to_records: Dict[str, List[Records]] = {}
	for records in records_list:
		algo_name = records.algo_name
		if algo_name not in algo_to_records:
			algo_to_records[algo_name] = []
		algo_to_records[algo_name].append(records)

	# make sure all algorithms have the same number of experiments
	experiment_counts = set([len(data) for data in algo_to_records.values()])
	assert len(experiment_counts) == 1, \
		"All algorithms should have the same number of experiments"

	# truncate horizon (assuming monotonic increasing)
	min_horizon = min([len(records.get_data()[horizon_name]) for records in records_list])

	for algo_name in algo_to_records:
		algo_records_list = algo_to_records[algo_name]

		horizon = algo_records_list[0].get_data(min_horizon)[horizon_name]
		values = np.array([records.get_data(min_horizon)['values'] for records in algo_records_list])
		value_mean = np.mean(values, axis=0)
		value_std = np.std(values, axis=0)

		axes.plot(horizon, value_mean)
		axes.fill_between(horizon, value_mean - value_std, value_mean + value_std, alpha=0.2, interpolate=True)

	axes.set_title('{}: {} plots of {} over {} trials'.format(
		env_name, value_description, horizon_name, next(iter(experiment_counts))))
	axes.set_ylabel(value_description)
	axes.set_xlabel(horizon_name)
	axes.legend(list(algo_to_records.keys()))

def plot_records_in_dir(
		log_dir: str,
		env_name: str,
		value_description: str = 'loss',
		horizon_name: Union[Literal['epochs', 'steps']] = 'epochs',
		**kwargs: Any
	) -> None:
	
	log_dir_path = Path(log_dir)
	assert log_dir_path.is_dir(), "Invalid log dir."
	
	parser = D3rlpyCSVDataParser()
	records_list: List[Records] = []
	
	for sub_dir in log_dir_path.iterdir():
		records_list.append(parser.parse(str(sub_dir), value_description=value_description))

	plot_records_list(plt.gca(), records_list, env_name, value_description, horizon_name, **kwargs)
	plt.show()
