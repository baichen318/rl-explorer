# Author: baichen318@gmail.com


import os
import abc


class Simulation(abc.ABC):
	def __init__(self, configs):
		super(Simulation, self).__init__()
		self.configs = configs
		self.macros = {}
		self.macros["gem5-research-root"] = None
		self.macros["rl-explorer-root"] = os.path.abspath(
			os.path.join(
				os.path.dirname(__file__),
				os.path.pardir
			)
		)
