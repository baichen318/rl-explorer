# Author: baichen318@gmail.com

import sys
import visdom
from subprocess import Popen, PIPE

class Visualizer(object):
    """docstring for Visualizer"""
    def __init__(self, configs):
        super(Visualizer, self).__init__()
        self.configs = configs
        self.port = self.configs["port"]
        self.vis = self.init_visdom()

    def init_visdom(self):

        vis = visdom.Visdom(
            server="http://localhost",
            port=self.port,
            env=self.configs["design"]
        )
        if not vis.check_connection():
            self.create_visdom_connections()
        return vis

    def create_visdom_connections(self):
        """
            if the program could not connect to Visdom server,
            this function will start a new server at port < self.port >
        """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print("[WARN]: Could not connect to Visdom server. \n Trying to start a server....")
        print("[WARN]: Command: %s" % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def plot_current_status(self, epoch, counter_ratio, losses, rewards):
        """
            epoch: <int> -- current epoch
            counter_ratio: <float> -- progress (percentage) in the current epoch, between 0 to 1
            losses <OrderedDict> -- training losses stored in the format of (name, float) pairs
            rewards: <OrderedDict> -- rewards
        """
        if not hasattr(self, "plot_loss"):
            self.plot_loss = {'X': [], 'Y': [], "legend": list(losses.keys())}
        if not hasattr(self, "plot_reward"):
            self.plot_reward = {'X': [], 'Y': [], "legend": list(rewards.keys())}
        self.plot_loss['X'].append(epoch + counter_ratio)
        self.plot_loss['Y'].append([losses[k] for k in self.plot_loss["legend"]])
        self.plot_reward['X'].append(epoch + counter_ratio)
        self.plot_reward['Y'].append([rewards[k] for k in self.plot_reward["legend"]])
        try:
            self.vis.line(
                X=np.stack(
                    [np.array(self.plot_loss['X'])] * len(self.plot_loss["legend"]), 1
                ),
                Y=np.array(self.plot_loss['Y']),
                opts={
                    "title": "A3C losses over time",
                    "legend": self.plot_loss["legend"],
                    "xlabel": "epoch",
                    "ylabel": "loss"},
                win=1
            )
            self.vis.line(
                X=np.stack(
                    [np.array(self.plot_reward['X'])] * len(self.plot_reward["legend"]), 1
                ),
                Y=np.array(self.plot_reward['Y']),
                opts={
                    "title": "Rewards over time",
                    "legend": self.plot_loss["legend"],
                    "xlabel": "epoch",
                    "ylabel": "reward"
                },
                win=2
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()
