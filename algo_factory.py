import d3rlpy

from d3rlpy.algos import DQN, DiscreteCQL, DiscreteSAC, DiscreteBCQ

class AlgoTemplateFactory:
	""" Algorithm template """

	# TODO: fill in templates with default values
	def __init__(self):
		pass

	@classmethod
	def get(cls, algo_name, **kwargs):
		assert algo_name in [
			'dqn', 
			'discrete_cql', 
			'discrete_sac', 
			'discrete_bc', 
			'discrete_bcq', 
			'discrete_soft_cql'
		]
		return getattr(cls, '_' + algo_name.lower())(**kwargs)

	@classmethod
	def _dqn(cls, **kwargs):
		arguments = dict()
		arguments.update(kwargs)
		return DQN(**arguments)

	@classmethod
	def _discrete_sac(cls, **kwargs):
		arguments = dict()
		arguments.update(kwargs)
		return DiscreteSAC(**arguments)

	@classmethod
	def _discrete_bcq(cls, **kwargs):
		arguments = dict()
		arguments.update(kwargs)
		return DiscreteBCQ(**arguments)

	@classmethod
	def _discrete_cql(cls, **kwargs):
		arguments = dict()
		arguments.update(kwargs)
		return DiscreteCQL(**arguments)
