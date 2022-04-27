# Author: baichen318@gmail.com


import os
import abc


class DesignSpace(abc.ABC):
	def __init__(self, size, dims):
		"""
			size: <int> total size of the design space
			dims: <int> dimension of a microarchitecture embedding
		"""
		self.size = size
		self.dims = dims

	@abc.abstractmethod
	def idx_to_vec(self, idx):
		"""
			transfer from an index to a vector
			idx: <int>
		"""
		raise NotImplementedError()

	@abc.abstractmethod
	def vec_to_idx(self, vec):
		"""
			transfer from a vector to an index
			vec: <list> microarchitecture encoding
		"""
		raise NotImplementedError()

	@abc.abstractmethod
	def generate_chisel_codes(self, batch):
		"""
			generate chisel codes w.r.t. code templates
			batch: <list> list of indexes
		"""
		raise NotImplementedError()


class Macros(abc.ABC):
	def __init__(self):
		self.macros = {}
		self.macros["chipyard-research-root"] = None
		self.macros["workstation-root"] = None

	@abc.abstractmethod
	def generate_core_cfg_impl(self, name, vec):
		"""
			core chisel codes template
			name: <str> name of the core
			vec: <list> microarchitecture encoding
		"""
		raise NotImplementedError()

	@abc.abstractmethod
	def generate_soc_cfg_impl(self):
		"""
			soc chisel codes template
		"""
		raise NotImplementedError()
