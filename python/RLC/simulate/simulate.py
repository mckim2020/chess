import numpy as np
import torch


class SimulateChess():
    def __init__(self, config:dict, grid=None, search=None, train=None, play=None):
        self.config = config
        self.grid = grid
        self.search = search
        self.train = train
        self.play = play


    def set_random_seed(self, device='cpu'):
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.Generator(device).manual_seed(self.config['seed'])
        # torch.use_deterministic_algorithms(True)