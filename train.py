import os
import sys
import argparse
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import math


sys.path.extend([os.path.abspath('./python')])
from RLC import Grid, SearchChess, TrainChess, SimulateChess
from RLC.models.models import MuZeroNet, ResMuZeroNet, MuZeroAttentionNet


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=100, help='Number of episodes to train')
    parser.add_argument('--state_dim', type=int, default=64, help='Dimension of states')
    parser.add_argument('--action_space', type=int, default=64, help='Dimension of states')
    parser.add_argument('--k_steps', type=int, default=5, help='Number of roll steps')
    parser.add_argument('--n_simulations', type=int, default=800, help='Number of MCTS simulations')
    parser.add_argument('--c_puct', type=float, default=1.5, help='PUCT exploration constant')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--noise', action='store_true', help="Add Dirichlet noise to root node priors")
    parser.add_argument('--noise_alpha', type=float, default=0.3, help='Dirichlet noise alpha parameter (concentration)')
    parser.add_argument('--noise_epsilon', type=float, default=0.25, help='Dirichlet noise epsilon parameter (mixing)')
    parser.add_argument('--verbose', action='store_true', help="verbose output")
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def main():
    args = set_args()

    config = {'n_episodes': args.n_episodes,
              'state_dim': args.state_dim,
              'action_space': args.action_space,
              'k_steps': args.k_steps,
              'n_simulations': args.n_simulations,
              'c_puct': args.c_puct,
              'batch_size': args.batch_size,
              'learning_rate': args.learning_rate,
              'model': MuZeroNet(), #ResMuZeroNet(), #MuZeroAttentionNet(),
              'noise': args.noise,
              'noise_alpha': args.noise_alpha,
              'noise_epsilon': args.noise_epsilon,
              'device': args.device,
              'seed': args.seed,
              'verbose': args.verbose}
    
    grid = Grid(config=config)
    search = SearchChess(config=config, grid=grid)
    train = TrainChess(config=config, grid=grid, search=search)
    sim = SimulateChess(config=config, grid=grid, search=search, train=train)

    model_path = (f'muzero_chess_model_'
                  f'n_episodes_{args.n_episodes}_'
                  f'k_steps_{args.k_steps}_'
                  f'n_simulations_{args.n_simulations}_'
                  f'c_puct_{args.c_puct}_'
                  f'batch_size_{args.batch_size}_'
                  f'learning_rate_{args.learning_rate}_'
                  f'noise_{args.noise}_'
                  f'noise_alpha_{args.noise_alpha}_'
                  f'noise_epsilon_{args.noise_epsilon}_'
                  f'.pth')

    sim.set_random_seed()
    sim.set_device()
    sim.train.train_muzero(model_path=model_path)

    return 0



if __name__ == "__main__":
    sys.exit( main() )