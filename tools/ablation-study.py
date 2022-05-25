# Author: baichen318@gmail.com


import sys, os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir)
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir, "utils")
)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorboard.backend.event_processing import event_accumulator
from utils import mkdir, write_txt, load_txt


def visualize():
	m1 = np.array([[1, 2, 11, 7, 3, 9, 4, 33, 9, 6]]) # [ 10.648949  46.328285]
	m2 = np.array([[2, 2, 11, 7, 3, 9, 4, 33, 9, 6]]) # [ 43.789368  14.344925]
	m3 = np.array([[1, 2, 11, 7, 2, 9, 4, 32, 9, 6]]) # [-21.333796  13.187289]
	m4 = np.array([[1, 2, 11, 6, 3, 9, 4, 33, 9, 6]]) # [ 11.806616 -18.796083]
	x = np.concatenate((m1, m2, m3, m4), axis=0)
	x_embedded = TSNE(
		n_components=2, 
		learning_rate='auto',
		init='random'
	).fit_transform(x)
	print(x_embedded.shape)
	print(x_embedded)
	plt.scatter(
		x[:, 0],
		x[:, 1]
	)
	plt.show()


def distance():
	m1 = np.array([ 10.648949,  46.328285])
	m2 = np.array([ 43.789368,  14.344925])
	m3 = np.array([-21.333796,  13.187289])
	m4 = np.array([ 11.806616, -18.796083])
	dist_1_2 = np.linalg.norm(m1 - m2) # 46.05673336641627
	dist_1_3 = np.linalg.norm(m1 - m3) # 46.05672148131086
	dist_1_4 = np.linalg.norm(m1 - m4) # 65.13465667586122
	print(dist_1_2, dist_1_3, dist_1_4)


def motivation_example():
	visualize()
	distance()


def generate_rewards_curve():
	events = os.path.abspath(
		os.path.join(
			os.path.dirname(__file__),
			os.path.pardir,
			"misc",
			"train-2-wide-4-fetch-SonicBOOM-baichen-MacBook-Pro.lan-2022-05-23-22-12.log",
			"events.out.tfevents.1653315160.MacBook-Pro.lan.33401.0"
		)
	)
	ea = event_accumulator.EventAccumulator(events)
	ea.Reload()
	perf = np.array(ea.Scalars("rewards/perf"))
	power = np.array(ea.Scalars("rewards/power"))
	area = np.array(ea.Scalars("rewards/area"))
	mean_perf = np.array(ea.Scalars("rewards/mean-perf"))
	mean_power = np.array(ea.Scalars("rewards/mean-power"))
	mean_area = np.array(ea.Scalars("rewards/mean-area"))
	max_perf = np.array(ea.Scalars("rewards/max-perf"))
	max_power = np.array(ea.Scalars("rewards/max-power"))
	max_area = np.array(ea.Scalars("rewards/max-area"))
	min_perf = np.array(ea.Scalars("rewards/min-perf"))
	min_power = np.array(ea.Scalars("rewards/min-power"))
	min_area = np.array(ea.Scalars("rewards/min-area"))
	path = os.path.join(
		"rewards",
		"boom"
	)
	mkdir(path)
	# step perf power area mean-perf mean-power mean-area max-perf max-power max-area min-perf min-power min-area
	step = perf[:, 1]
	perf = perf[:, -1]
	power = power[:, -1]
	area = area[:, -1]
	mean_perf = mean_perf[:, -1]
	mean_power = mean_power[:, -1]
	mean_area = mean_area[:, -1]
	max_perf = max_perf[:, -1]
	max_power = max_power[:, -1]
	max_area = max_area[:, -1]
	min_perf = min_perf[:, -1]
	min_power = min_power[:, -1]
	min_area = min_area[:, -1]
	data = np.concatenate((
			np.expand_dims(step, axis=1),
			np.expand_dims(perf, axis=1),
			np.expand_dims(power, axis=1),
			np.expand_dims(area, axis=1),
			np.expand_dims(mean_perf, axis=1),
			np.expand_dims(mean_power, axis=1),
			np.expand_dims(mean_area, axis=1),
			np.expand_dims(max_perf, axis=1),
			np.expand_dims(max_power, axis=1),
			np.expand_dims(max_area, axis=1),
			np.expand_dims(min_perf, axis=1),
			np.expand_dims(min_power, axis=1),
			np.expand_dims(min_area, axis=1)
		),
		axis=1
	)
	# write_txt(
	# 	os.path.join(path, "dataset.csv"),
	# 	data,
	# 	fmt="%f"
	# )
	# # scale perf
	# data[:, 1] *= 10
	# data[:, 4] *= 10
	# data[:, 7] *= 10
	# data[:, 10] *= 10
	# scale power
	data[:, 2] = data[:, 2] + 0.8
	data[:, 5] = data[:, 5] + 0.8
	data[:, 8] = data[:, 8] + 0.8
	data[:, 11] = data[:, 11] + 0.8
	# smooth
	n_samples = data.shape[0]
	n_labels = data.shape[1]
	alpha = 0.98
	for i in range(1, n_samples):
		for j in range(1, n_labels):
			data[i, j] = alpha * data[i - 1, j] + (1 - alpha) * data[i, j]
	write_txt(
		os.path.join(
			path,
			"smooth-dataset.csv"
		),
		data,
		fmt="%f"
	)


# def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
#     last = scalars[0]  # First value in the plot (first timestep)
#     smoothed = list()
#     for point in scalars:
#         smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
#         smoothed.append(smoothed_val)                        # Save it
#         last = smoothed_val                                  # Anchor the last smoothed value

#     return smoothed


if __name__ == "__main__":
	generate_rewards_curve()
