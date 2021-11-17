import pandas as pd # type: ignore

from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod


class Records:
	""" class representing data for plotting """
	def __init__(
		self, 		
		epochs: List[int], 
		steps: List[int], 
		values: List[int],
		algo_name: Optional[str],
		trial_name: Optional[str],
		value_description: str
	):
		self._algo_name = algo_name
		self._trial_name = trial_name
		self._value_description = value_description
		self._epochs = epochs
		self._steps = steps
		self._values = values

	def __repr__(self) -> str:
		return 'Data records of {} for algorithm {}, experiment {}, total length {}.'.format(
			self._value_description,
			self._algo_name,
			self._trial_name,
			self.__len__()
		)

	def __len__(self) -> int:
		return len(self._epochs)

	def __index__(self, idx) -> Dict[str, Any]:
		if idx < self.__len__():
			return dict(epoch=self._epochs[idx], step=self._steps[idx], value=self._values[idx])
		return {}

	def get_data(self, max_len: int = None) -> Dict[str, List[int]]:
		max_len = max_len or self.__len__()
		return dict(epochs=self._epochs[:max_len], steps=self._steps[:max_len], values=self._values[:max_len])

	@property
	def algo_name(self):
		return self._algo_name

	@property
	def trial_name(self):
		return self._trial_name

	@property
	def value_description(self):
		return self._value_description

class DataParser(ABC):
	""" base class dealing with parsing logics """
	def __init__(self):
		pass
	
	@abstractmethod
	def parse(self,
		data_source: Any,
		algo_name: str,
		trial_name: str,
		value_description: str
	) -> Records:
		pass

class CSVDataParser(DataParser):
	""" Generic CSV parser into format Records """
	def __init__(self):
		super(CSVDataParser, self).__init__()

	def parse(
		self, 
		csv_path: str, 
		algo_name: str = None,
		trial_name: str = None,
		value_description: str = 'value'
	) -> Records:
		print(csv_path)
		assert Path(csv_path).is_file(), 'Invalid csv file path.'
		data = pd.read_csv(csv_path, names=['epoch', 'step', value_description])
		parsed_data = Records(			
			data['epoch'].tolist(),
			data['step'].tolist(),
			data[value_description].tolist(),
			algo_name,
			trial_name,
			value_description
		)
		return parsed_data

class D3rlpyCSVDataParser(CSVDataParser):
	""" D3rlpy CSV parser that extracts algo name and experiment automatically """
	def __init__(self, parse_experiment_name: Callable[[str], Tuple[str, str, str]] = None):
		super(D3rlpyCSVDataParser, self).__init__()
		self._parse_experiment_name = parse_experiment_name

	def parse(
		self, 
		log_dir: str, 
		algo_name: str = None,
		trial_name: str = None,
		value_description: str = 'loss'
	) -> Records:
		# the log dir path should have a folder name like ${ALGO}_${EXPERIMENT}
		log_dir_path = Path(log_dir)
		base_name = log_dir_path.name
		assert log_dir_path.is_dir(), 'Invalid log directory.'
		assert '_' in base_name, 'The folder should have ALGONAME_EXPERIMENTNAME format.'
		delim_idx = base_name.find('_')
		if self._parse_experiment_name:
			algo_name, _, trial_name = self._parse_experiment_name(base_name)
		else:
			algo_name, trial_name = algo_name or base_name[:delim_idx], trial_name or base_name[delim_idx+1:]
		csv_path = log_dir_path / (value_description + '.csv')

		return super().parse(
			str(csv_path),
			algo_name,
			trial_name,
			value_description
		)
	
