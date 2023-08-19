# Author: baichen318@gmail.com


import random


with open("rocket-design.idx", 'w') as f:
	idxs = random.sample(range(1, 3600 + 1), k=300)
	for idx in idxs:
		f.write("{}\n".format(idx))
