# Author: baichen318@gmail.com


import os
import abc
from typing import List


class DesignSpace(abc.ABC):
	def __init__(self, size: int, dims: int):
		"""
			size: total size of the design space
			dims: dimension of a microarchitecture embedding
		"""
		self.size = size
		self.dims = dims

	@abc.abstractmethod
	def idx_to_vec(self, idx: int):
		"""
			transfer from an index to a vector
			idx: index
		"""
		raise NotImplementedError()

	@abc.abstractmethod
	def vec_to_idx(self, vec: List[int]):
		"""
			transfer from a vector to an index
			vec: microarchitecture encoding
		"""
		raise NotImplementedError()

	@abc.abstractmethod
	def generate_chisel_codes(self, batch: List[int]):
		"""
			generate chisel codes w.r.t. code templates
			batch: list of indexes
		"""
		raise NotImplementedError()


class Macros(abc.ABC):
	def __init__(self):
		self.macros = {}
		self.macros["chipyard-research-root"] = None
		self.macros["workstation-root"] = None

	@abc.abstractmethod
	def generate_core_cfg_impl(self, name: str, vec: List[int]):
		"""
			core chisel codes template
			name: name of the core
			vec: microarchitecture encoding
		"""
		raise NotImplementedError()

	@abc.abstractmethod
	def generate_soc_cfg_impl(self):
		"""
			soc chisel codes template
		"""
		raise NotImplementedError()
