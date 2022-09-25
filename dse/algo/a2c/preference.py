# Author baichen318@gmail.com


import numpy as np


class Preference(object):
	def __init__(self, scale_factor, reward_space, training):
		"""
			scale_factor: <list>
		"""
		super(Preference, self).__init__()
		self.reward_space = reward_space
		# NOTICE: `scale_factor` is valid when `training` = True
		self.scale_factor = scale_factor
		self.training = training
	
	def init_preference(self):
		preference = np.ones(self.reward_space) * self.scale_factor
		return preference / np.sum(preference)

	def generate_preference(self, num_of_preference, fixed_preference=None):
		if fixed_preference is not None:
			if self.training:
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
				return np.expand_dims(self.init_preference(), axis=0)
		else:
			if self.training:
				preference = np.random.randn(
					num_of_preference, self.reward_space
				) * self.scale_factor
				preference = np.abs(preference) / \
					np.linalg.norm(preference, ord=1, axis=1).reshape(
						num_of_preference,
						1
					)
				return preference
			else:
				return np.expand_dims(self.init_preference(), axis=0)

	def renew_preference(self, preference, dim):
		# ablation
		# _preference = np.random.randn(self.reward_space) * self.scale_factor
		_preference = np.ones(self.reward_space) * self.scale_factor
		_preference = np.abs(_preference) / np.linalg.norm(_preference, ord=1, axis=0)
		preference[dim] = _preference
		return preference
