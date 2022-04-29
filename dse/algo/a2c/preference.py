# Author baichen318@gmail.com


import numpy as np


class Preference(object):
	def __init__(self, reward_space):
		super(Preference, self).__init__()
		self.reward_space = reward_space
	
	def init_preference(self):
		preference = np.ones(self.reward_space)
		return preference / np.sum(preference)

	def generate_preference(self, num_of_preference, fixed_preference=None):
		preference = np.random.randn(num_of_preference - 1, self.reward_space)
		if fixed_preference is not None:
			preference = np.abs(preference) / \
				np.linalg.norm(preference, ord=1, axis=1).reshape(
					num_of_preference - 1,
					1
				)
			return np.concatenate(([fixed_preference], preference))
		else:
			preference = np.abs(preference) / \
				np.linalg.norm(preference, ord=1, axis=1).reshape(
					num_of_preference,
					1
				)
			return preference

	def renew_preference(self, preference, dim):
		_preference = np.random.randn(self.reward_space)
		_preference = np.abs(_preference) / np.linalg.norm(_preference, ord=1, axis=0)
		preference[dim] = _preference
		return preference