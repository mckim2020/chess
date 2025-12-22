import os
import sys
import chess
import chess.pgn
import chess.engine
import math
import numpy as np
from stockfish import Stockfish


class StockfishTeacher:
    def __init__(self, config, grid, path_to_exe, depth=15):
        self.config = config
        self.grid = grid
        # On Mac, path might be "/usr/local/bin/stockfish" or similar
        self.engine = Stockfish(path=path_to_exe, parameters={"Threads": 12, "UCI_Elo": 3000})
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


    def run_forever(self):
        obs_list = []
        target_value_list = []
    
        for ep in range(self.config['n_episodes']):
            board = chess.Board()
            
            # --- DATA GENERATION PHASE (Teacher Lead) ---
            while not board.is_game_over():
                self.engine.set_fen_position(board.fen())
                
                # Get Teacher Policy (Best Move)
                best_move_uci = self.engine.get_best_move()
                if not best_move_uci: break
                best_move = chess.Move.from_uci(best_move_uci)
                
                # Get Teacher Value (Evaluation)
                eval_dict = self.engine.get_evaluation()
                raw_value = eval_dict['value']
                
                # Step 1: Convert to Side-to-Move POV
                # If it's Black's turn, a negative SF score is GOOD for Black.
                # We want 'pos' to mean 'good for the current player'
                stm_value = raw_value if board.turn == chess.WHITE else -raw_value
                
                # Step 2: Squash into 0.0 to 1.0 (Win Probability)
                if eval_dict['type'] == 'cp':
                    # Stockfish 16 uses this coefficient to anchor 100cp to 50% win chance
                    teacher_val = 1 / (1 + math.exp(-0.00368208 * stm_value))
                else:
                    # Handle Mate scores: Mate in X is a guaranteed win (1.0) or loss (0.0)
                    teacher_val = 1.0 if stm_value > 0 else 0.0

                obs = self.grid.board_to_tensor(board)

                obs_list.append(obs.numpy())
                target_value_list.append(teacher_val)

                board.push(best_move)

            obs_array = np.concatenate(obs_list)
            target_value_array = np.array(target_value_list).reshape(-1, 1)
    
            np.savez(f"./episodes/episode_{ep+1}.npz", observations=obs_array, target_values=target_value_array)
            print(f"Episode {ep+1}/{self.config['n_episodes']} completed.")


    def count_games_with_scan(self, pgn_path):
        count = 0
        with open(pgn_path) as pgn:
            # scan_headers yields a tuple of (offset, headers) for every game
            for _ in chess.pgn.read_headers(pgn):
                count += 1
        return count


    def count_games_fast(self, pgn_path):
        count = 0
        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('[Event '):
                    count += 1
        return count


    def process_pgn(self, pgn_path, output_dir='data'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        total = self.count_games_fast(pgn_path)
        print(f"Total games: {total}")

        with open(pgn_path) as pgn:
            save_idx = 0
            game_idx = 0
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None: break
                
                obs_list = []
                target_value_list = []
                board = game.board()
                
                for move in game.mainline_moves():
                    self.engine.set_fen_position(board.fen())
                    
                    # Get Teacher Value (Evaluation)
                    eval_dict = self.engine.get_evaluation()
                    raw_value = eval_dict['value']
                    
                    # Step 1: Convert to Side-to-Move POV
                    # If it's Black's turn, a negative SF score is GOOD for Black.
                    # We want 'pos' to mean 'good for the current player'
                    stm_value = raw_value if board.turn == chess.WHITE else -raw_value
                    
                    # Step 2: Squash into 0.0 to 1.0 (Win Probability)
                    if eval_dict['type'] == 'cp':
                        # Stockfish 16 uses this coefficient to anchor 100cp to 50% win chance
                        teacher_val = 1 / (1 + math.exp(-0.00368208 * stm_value))
                    else:
                        # Handle Mate scores: Mate in X is a guaranteed win (1.0) or loss (0.0)
                        teacher_val = 1.0 if stm_value > 0 else 0.0

                    obs = self.grid.board_to_tensor(board)

                    obs_list.append(obs.numpy())
                    target_value_list.append(teacher_val)

                    board.push(move)

                if len(obs_list) > int(1e4):
                    obs_array = np.concatenate(obs_list)
                    target_value_array = np.array(target_value_list).reshape(-1, 1)
                    np.savez_compressed(
                        f"{output_dir}/episode_{save_idx}_partial.npz",
                        obs_array=obs_array,
                        target_value_array=target_value_array
                    )
                    obs_list = []
                    target_value_list = []
                    print(f"Saved game {save_idx} with {len(obs_list)} positions.")
                    save_idx += 1

                print(f"Processed game {game_idx+1}/{total} of length {board.fullmove_number}.")
                game_idx += 1

        self.engine.quit()