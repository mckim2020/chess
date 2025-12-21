import torch


class Grid():
    def __init__(self, config:dict):
        self.config = config


    def board_to_tensor(self, board):
        tensor = torch.zeros(1, 12, 8, 8)
        piece_map = {'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5, 'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11}
        for sq, pc in board.piece_map().items():
            tensor[0, piece_map[pc.symbol()], sq // 8, sq % 8] = 1.0
        return tensor


    def encode_action(self, move):
        return move.from_square * 64 + move.to_square