# Author: baichen318@gmail.com

import sys
import visdom
import numpy as np
from subprocess import Popen, PIPE

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

class Visualizer(object):
    """docstring for Visualizer"""
    def __init__(self, configs):
        super(Visualizer, self).__init__()
        self.configs = configs
        self.port = self.configs["port"]
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
            this function will start a new server at port < self.port >
        """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print("[WARN]: Could not connect to Visdom server. \n Trying to start a server....")
        print("[WARN]: Command: %s" % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def plot_current_status(self, epoch, counter_ratio, name, status, **kwargs):
        """
            epoch: <int>
            counter_ratio: <float>
            name: <str>
            status: <OrderedDict>
        """
        if not (name in self.status.keys()):
            self.status[name] = {
                'X': [],
                'Y': [],
                "legend": list(status.keys())
            }
        self.status[name]['X'].append(epoch + counter_ratio)
        self.status[name]['Y'].append(
            [status[k] for k in self.status[name]["legend"]]
        )
        try:
            self.vis.line(
                X=np.stack(
                    [np.array(self.status[name]['X'])] * len(self.status[name]["legend"]),
                    1
                ),
                Y=np.array(self.status[name]['Y']),
                opts={
                    "title": kwargs["title"],
                    "legend": self.status[name]["legend"],
                    "xlabel": kwargs["xlabel"],
                    "ylabel": kwargs["ylabel"]
                },
                win=kwargs["win"]
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()
