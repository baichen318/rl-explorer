# Author baichen318@gmail.com


import numpy as np


class Preference(object):
	def __init__(self, scale_factor, reward_space):
		"""
			scale_factor: <list>
		"""
		super(Preference, self).__init__()
		self.reward_space = reward_space
		# NOTICE: `scale_factor` compares inprovement contributions on PPA values
		# power model is not accurate compared to performance and area models
		# so we apply `scale_factor`
		self.scale_factor = scale_factor
	
	def init_preference(self):
		preference = np.ones(self.reward_space) * self.scale_factor
		return preference / np.sum(preference)

	def generate_preference(self, num_of_preference, fixed_preference=None):
		if fixed_preference is not None:
			preference = np.random.randn(
				num_of_preference - 1, self.reward_space
			) * self.scale_factor
			preference = np.abs(preference) / \
				np.linalg.norm(preference, ord=1, axis=1).reshape(
					num_of_preference - 1,
					1
				)
			return np.concatenate(([fixed_preference], preference))
		else:
			preference = np.random.randn(
				num_of_preference, self.reward_space
			) * self.scale_factor
			preference = np.abs(preference) / \
				np.linalg.norm(preference, ord=1, axis=1).reshape(
					num_of_preference,
					1
				)
			return preference

	def renew_preference(self, preference, dim):
		# ablation
		# _preference = np.random.randn(self.reward_space) * self.scale_factor
		_preference = np.ones(self.reward_space) * self.scale_factor
		_preference = np.abs(_preference) / np.linalg.norm(_preference, ord=1, axis=0)
		preference[dim] = _preference
		return preference
