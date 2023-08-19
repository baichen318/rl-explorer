# Author: baichen318@gmail.com


import os
import re
import argparse
import numpy as np
from typing import List
from utils.utils import assert_error, info
from tensorboard.backend.event_processing import event_accumulator


def remove_prefix(s, prefix):
    if prefix and s.startswith(prefix):
        return s[len(prefix):]
    else:
        return s[:]


def remove_suffix(s, suffix):
    """
        s: <str>
        suffix <str>
    """
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    else:
        return s[:]


def parse_args():
	def initialize_parser(parser):
		parser.add_argument(
			"-p",
			"--path",
			type=str,
			help="summary-logs path"
		)
		parser.add_argument(
			"-o",
			"--output_path",
			type=str,
			help="output path specification"
		)
		return parser

	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser = initialize_parser(parser)
	return parser.parse_args()


def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def generate():
	size = ["medium", "large", "mega", "giga"]
	items = ["perf", "area", "power"]
	for _size in size:
		for item in items:
			# get the totoal episode
			with open("training/{}/{}-baseline.txt".format(_size, item), 'r') as f:
				cnt = f.readlines()
				_episode = len(cnt)
				baseline = float(cnt[0].split(' ')[1])

			value = []
			sequence = []
			with open("training/{}/{}.txt".format(_size, item), 'r') as f:
				# get the total sequence
				cnt = f.readlines()
				_sequence = len(cnt)
				gap = _episode / _sequence
				for line in cnt:
					_line = line.strip().split(' ')
					sequence.append(round(float(_line[0]) * gap))
					value.append(float(_line[1]))

			# _value = smooth(value[3:], weight=0.874)
			# value = value[:3] + _value
			value = smooth(value, weight=0.874)
			sequence = np.expand_dims(np.array(sequence), axis=1).reshape(-1, 1)
			value = np.expand_dims(np.array(value), axis=0).reshape(-1, 1)
			array = np.concatenate((sequence, value), axis=1)
			np.savetxt("training/{}/{}.data".format(_size, item), array, header="episode\t{}".format(item))
			# save the baseline
			array[:, 1] = baseline2
			np.savetxt("training/{}/{}-baseline.data".format(_size, item), array, header="episode\t{}".format(item))
			# save the preference
			value = []
			sequence = []
			with open("training/{}/{}.txt".format(_size, item + "-preference"), 'r') as f:
				cnt = f.readlines()
				for line in cnt:
					_line = line.strip().split(' ')
					sequence.append(round(float(_line[0]) * gap))
					value.append(float(_line[1]))
			value = smooth(value, weight=0.874)
			sequence = np.expand_dims(np.array(sequence), axis=1).reshape(-1, 1)
			value = np.expand_dims(np.array(value), axis=0).reshape(-1, 1)
			array = np.concatenate((sequence, value), axis=1)
			np.savetxt("training/{}/{}.data".format(_size, item + "-preference"), array, header="episode\t{}".format(item))


def load_event(curve):
	event = os.listdir(os.path.join(summary_logs_root, curve))[0]
	ea_path = os.path.join(
		os.path.join(summary_logs_root, curve, event)
	)
	ea = event_accumulator.EventAccumulator(ea_path)
	ea.Reload()
	return ea.scalars.Items(ea.scalars.Keys()[0])


def handle_vis_training():

	curves = [
		"episode_perf_perf",
		"episode_power_power",
		"episode_area_area",
		"episode_reward_reward",
		"episode_preference_perf-preference",
		"episode_preference_power-preference",
		"episode_preference_area-performance"
	]

	sequence, episode = 0, 0
	for curve in curves:
		if "preference_perf" in curve:
			name = "preference_perf"
		elif "preference_power" in curve:
			name = "preference_power"
		elif "preference_area" in curve:
			name = "preference_area"
		elif "perf" in curve:
			name = "perf"
		elif "reward" in curve:
			name = "reward"
		elif "power" in curve:
			name = "power"
		else:
			assert "area" in curve, \
				assert_error("unknown curve: {}".format(curve))
			name = "area"

		scalars = load_event(curve)
		val = []
		for i in range(len(scalars)):
			if name == "perf" or name == "reward" or "preference" in name:
				val.append([scalars[i].step + 1, scalars[i].value])
			else:
				val.append([scalars[i].step + 1, -scalars[i].value])
		val = np.array(val)
		val[:, 1] = smooth(list(val[:, 1]), weight=0.96)
		f = os.path.join(args.output_path, "{}.txt".format(name))
		np.savetxt(
			f,
			val,
			header="episode\t{}".format(name)
		)
		info("save to {}".format(f))


if __name__ == '__main__':
	args = parse_args()
	proj_name = os.path.basename(args.path)
	summary_logs_root = os.path.join(
		args.path,
		"summary-logs",
		proj_name,
		"log-{}.log".format(
			re.search(r"\d+-\d+-\d+-\d+-\d+", proj_name)[0]
		)
	)
	is_boom = True if "BOOM" in proj_name else False
	handle_vis_training()
