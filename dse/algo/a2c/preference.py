# Author baichen318@gmail.com


import numpy as np


class Preference(object):
	def __init__(self, reward_shape):
		super(Preference, self).__init__()
		self.reward_shape = reward_shape
	
	def init_preference(self):
		preference = np.ones(self.reward_shape)
		return preference / np.sum(preference)

	def generate_preference(self, num_of_preference, fixed_preference=None):
		preference = np.random.randn(num_of_preference - 1, self.reward_shape)
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
