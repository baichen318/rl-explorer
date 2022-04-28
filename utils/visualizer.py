# Author: baichen318@gmail.com

import sys
import visdom
import numpy as np
from subprocess import Popen, PIPE
from utils import warn


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class Visualizer(object):
    """ Visualizer"""
    def __init__(self, configs):
        super(Visualizer, self).__init__()
        self.configs = configs
        self.port = 9527
        self.vis = self.init_visdom()
        self.status = {}

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
            this function will start a new server at port `self.port`
        """
        cmd = sys.executable + \
            " -m visdom.server -p %d &>/dev/null &" % self.port
        warn("could not connect to Visdom server. trying to start a server....")
        warn("command: %s" % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def plot_perf_per_episode(self, perf_per_episode, episode):
        try:
            self.vis.line(
                X=np.array(episode),
                Y=np.array(perf_per_episode),
                opts={
                    "title": "IPC vs. Episode",
                    "xlabel": "Episode",
                    "ylabel": "IPC"
                },
                win=1
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_power_per_episode(self, power_per_episode, episode):
        try:
            self.vis.line(
                X=np.array(episode),
                Y=np.array(power_per_episode),
                opts={
                    "title": "Power vs. Episode",
                    "xlabel": "Episode",
                    "ylabel": "Power"
                },
                win=2
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_area_per_episode(self, area_per_episode, episode):
        try:
            self.vis.line(
                X=np.array(episode),
                Y=np.array(area_per_episode),
                opts={
                    "title": "Area vs. Episode",
                    "xlabel": "Episode",
                    "ylabel": "Area"
                },
                win=3
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_temperature_per_episode(self, temperature, episode):
        try:
            self.vis.line(
                X=np.array(episode),
                Y=np.array(temperature),
                opts={
                    "title": "Temperature vs. Episode",
                    "xlabel": "Episode",
                    "ylabel": "Temperature"
                },
                win=4
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_learning_rate_per_episode(self, learning_rate, episode):
        try:
            self.vis.line(
                X=np.array(episode),
                Y=np.array(learning_rate),
                opts={
                    "title": "Learning Rate vs. Episode",
                    "xlabel": "Episode",
                    "ylabel": "Learning Rate"
                },
                win=5
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_actor_loss_per_step(self, actor_loss_per_step, step):
        try:
            self.vis.line(
                X=np.array(step),
                Y=np.array(actor_loss_per_step),
                opts={
                    "title": "Actor Loss vs. Step",
                    "xlabel": "Step",
                    "ylabel": "Actor Loss"
                },
                win=6
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_critic_loss_per_step(self, critic_loss_per_step, step):
        try:
            self.vis.line(
                X=np.array(step),
                Y=np.array(critic_loss_per_step),
                opts={
                    "title": "Critic Loss vs. Step",
                    "xlabel": "Step",
                    "ylabel": "Critic Loss"
                },
                win=7
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_entropy_per_step(self, entropy_per_step, step):
        try:
            self.vis.line(
                X=np.array(step),
                Y=np.array(entropy_per_step),
                opts={
                    "title": "Entropy vs. Step",
                    "xlabel": "Step",
                    "ylabel": "Entropy"
                },
                win=8
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_loss_per_step(self, loss_per_step, step):
        try:
            self.vis.line(
                X=np.array(step),
                Y=np.array(loss_per_step),
                opts={
                    "title": "Loss vs. Step",
                    "xlabel": "Step",
                    "ylabel": "Loss"
                },
                win=9
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_current_status_per_episode(self, status):
        """
            status: <class "status">, refer it to "a2c.py"
        """
        self.plot_perf_per_episode(status.perf_per_episode, status.episode)
        self.plot_power_per_episode(status.power_per_episode, status.episode)
        self.plot_area_per_episode(status.area_per_episode, status.episode)
        self.plot_temperature_per_episode(status.temperature, status.episode)
        self.plot_learning_rate_per_episode(status.learning_rate, status.episode)
        self.plot_actor_loss_per_step(status.actor_loss_per_step, status.step)
        self.plot_critic_loss_per_step(status.critic_loss_per_step, status.step)
        self.plot_entropy_per_step(status.entropy_per_step, status.step)
        self.plot_loss_per_step(status.loss_per_step, status.step)

    def plot_current_status_per_step(self, status):
        """
            status: <class "status">, refer it to "a2c.py"
        """
        self.plot_actor_loss_per_step(status.actor_loss_per_step, status.step)
        self.plot_critic_loss_per_step(status.critic_loss_per_step, status.step)
        self.plot_entropy_per_step(status.entropy_per_step, status.step)
        self.plot_loss_per_step(status.loss_per_step, status.step)
