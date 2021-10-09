# Author: baichen318@gmail.com

import visdom


class Visualizer(object):
    """docstring for Visualizer"""
    def __init__(self, configs):
        super(Visualizer, self).__init__()
        self.configs = configs
        self.vis = self.init_visdom()

    def init_visdom(self):
        visdom.Visdom(
            server="http://localhost",
            port=20213,
            env=self.configs["design"]
        )
        if not self

