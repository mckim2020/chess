import chess
import numpy as np
from stockfish import Stockfish


class StockfishTeacher:
    def __init__(self, path_to_exe, depth=15):
        # On Mac, path might be "/usr/local/bin/stockfish" or similar
        self.engine = Stockfish(path=path_to_exe)
        self.engine.set_depth(depth)

    def get_teacher_data(self, board, action_space_size, grid_utils):
        self.engine.set_fen_position(board.fen())
        
        # 1. Get Best Move and convert to index (Target Policy)
        best_move_dict = self.engine.get_best_move_time(100) # 100ms per move
        move = chess.Move.from_uci(best_move_dict)
        
        # Create a "One-Hot" target policy
        # Or better: get top 5 moves and create a distribution
        target_pi = np.zeros(action_space_size)
        target_pi[grid_utils.encode_action(move)] = 1.0
        
        # 2. Get Eval (Target Value)
        evaluation = self.engine.get_evaluation()
        # Scale centipawns to [-1, 1] using tanh
        if evaluation['type'] == 'cp':
            val = np.tanh(evaluation['value'] / 300.0) 
        else: # It's a "mate in N"
            val = 1.0 if evaluation['value'] > 0 else -1.0
            
        return move, target_pi, val