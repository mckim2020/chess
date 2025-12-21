import numpy as np
import torch


class SimulateChess():
    def __init__(self, config:dict, grid=None, search=None, train=None, play=None):
        self.config = config
        self.grid = grid
        self.search = search
        self.train = train
        self.play = play


    def set_random_seed(self):
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.Generator(self.config['device']).manual_seed(self.config['seed'])
        # torch.use_deterministic_algorithms(True)


    def set_device(self):
        self.config['model'].to(self.config['device'])