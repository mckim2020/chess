import os
import chess
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from stockfish import Stockfish


class TrainChess():
    def __init__(self, config:dict, grid=None, search=None):
        self.config = config
        self.grid = grid
        self.search = search


    def train_muzero(self, model_path='muzero_chess_model.pth'):
        if os.path.exists(model_path):
            self.config['model'].load_state_dict(torch.load(model_path))
            self.config['model'].to(self.config['device'])
            print(f"Loaded model from {model_path}")
            return

        optimizer = optim.Adam(self.config['model'].parameters(), lr=self.config['learning_rate'])
        replay_buffer = deque(maxlen=100)
        
        for ep in range(self.config['n_episodes']):
            board = chess.Board()
            game_history = [] # Stores (obs, action, reward)
            
            # --- SELF-PLAY PHASE ---
            while not board.is_game_over() and len(game_history) < self.config['max_game_length']:
                obs = self.grid.board_to_tensor(board)
                state = self.config['model'].h(obs)
                
                # Plan move using Latent MCTS
                # 1. Get the move AND the search probabilities (pi) from MCTS
                # You'll need to modify your run_latent_mcts to return both
                # move = self.search.run_latent_mcts(state, board)
                # if move is None: break
                # action_idx = self.grid.encode_action(move)
                # game_history.append({'obs': obs, 'action': action_idx, 'reward': 0})
                # board.push(move)

                move, search_pi = self.search.run_latent_mcts(state, board, return_pi=True)
                action_idx = self.grid.encode_action(move)
                game_history.append({
                    'obs': obs, 
                    'action': action_idx, 
                    'target_pi': search_pi, # This is a vector of size ACTION_SPACE
                    'reward': 0
                })
                board.push(move)
            
            # Assign final game result as reward to the last state
            result = board.result()
            final_reward = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
            if game_history:
                game_history[-1]['reward'] = final_reward
            
            replay_buffer.append(game_history)
            
            # --- TRAINING PHASE (Unrolling) ---
            
            if len(replay_buffer) >= 1:
                optimizer.zero_grad()
                batch = random.sample(replay_buffer, min(len(replay_buffer), self.config['batch_size']))
                
                total_loss = 0
                for history in batch:
                    if len(history) < self.config['k_steps'] + 1: continue
                    
                    # Start at a random point in the game
                    start_idx = random.randint(0, len(history) - self.config['k_steps'] - 1)
                    
                    # Step 0: Initial representation
                    current_state = self.config['model'].h(history[start_idx]['obs'])
                    
                    # Unroll K steps into the future
                    for k in range(self.config['k_steps']):
                        target_action = history[start_idx + k]['action']
                        target_reward = history[start_idx + k]['reward']
                        target_pi = torch.tensor(history[start_idx + k]['target_pi'], device=self.config['device'])
                        
                        # Predict Dynamics (imagined next state and reward)
                        current_state, pred_reward = self.config['model'].g(current_state, torch.tensor([[target_action]], device=self.config['device']))
                        
                        # Predict Value/Policy from imagined state
                        pred_policy, pred_value = self.config['model'].f(current_state)
                        
                        # Loss = Reward Loss + Value Loss (Target is the game outcome)
                        loss_p = F.cross_entropy(pred_policy, target_pi.unsqueeze(0))
                        loss_r = nn.MSELoss()(pred_reward, torch.tensor([[float(target_reward)]], device=self.config['device']))
                        loss_v = nn.MSELoss()(pred_value, torch.tensor([[float(final_reward)]], device=self.config['device']))
                        
                        total_loss += loss_p + loss_v + loss_r
                
                if total_loss > 0:
                    total_loss.backward()
                    optimizer.step()
                    print(f"Episode {ep+1} | Loss: {total_loss.item():.4e} | Result: {result}")
        
            if (ep + 1) % self.config['save_every'] == 0:
                torch.save(self.config['model'].state_dict(), f'./models/model_ep_{ep+1}.pth')
                print(f"Model saved at episode {ep+1} as {f'./models/model_ep_{ep+1}.pth'}")

        print("Training completed.")
        torch.save(self.config['model'].state_dict(), model_path)
        print(f"Model saved as {model_path}")


    def train_with_teacher(self, stockfish_path, model_path='distilled_model.pth'):
        # 1. Initialize the Teacher
        # depth 12-15 is plenty for a teacher; higher is slower.
        teacher = Stockfish(path=stockfish_path, parameters={"Threads": 12, "UCI_Elo": 3000})
        teacher.set_depth(1)
        
        optimizer = optim.Adam(self.config['model'].parameters(), lr=self.config['learning_rate'])
        
        # Replay buffer stores high-quality teacher demonstrations
        replay_buffer = deque(maxlen=500) 
        
        for ep in range(self.config['n_episodes']):
            board = chess.Board()
            game_history = [] 
            
            # --- DATA GENERATION PHASE (Teacher Lead) ---
            while not board.is_game_over() and len(game_history) < self.config['max_game_length']:
                teacher.set_fen_position(board.fen())
                
                # Get Teacher Policy (Best Move)
                best_move_uci = teacher.get_best_move()
                if not best_move_uci: break
                best_move = chess.Move.from_uci(best_move_uci)
                
                # Create Target Pi (Probability Distribution)
                # We use a one-hot vector for the best move
                target_pi = np.zeros(self.config['action_space'])
                target_pi[self.grid.encode_action(best_move)] = 1.0
                
                # Get Teacher Value (Evaluation)
                eval_dict = teacher.get_evaluation()
                if eval_dict['type'] == 'cp':
                    # Map centipawns to [-1, 1] using tanh
                    teacher_val = math.tanh(eval_dict['value'] / 300.0)
                else:
                    # Handle mate scores
                    teacher_val = 1.0 if eval_dict['value'] > 0 else -1.0
                
                obs = self.grid.board_to_tensor(board)
                
                game_history.append({
                    'obs': obs,
                    'action': self.grid.encode_action(best_move),
                    'target_pi': target_pi,
                    'target_value': teacher_val
                })
                
                board.push(best_move)
                
            replay_buffer.append(game_history)
            
            # --- TRAINING PHASE (MuZero Unrolling) ---
            if len(replay_buffer) >= 1:
                self.config['model'].train()
                optimizer.zero_grad()
                
                batch = random.sample(replay_buffer, min(len(replay_buffer), self.config['batch_size']))
                total_loss = 0
                
                for history in batch:
                    # We need enough steps for the K-step unroll
                    if len(history) < self.config['k_steps'] + 1: continue
                    
                    start_idx = random.randint(0, len(history) - self.config['k_steps'] - 1)
                    
                    # Step 0: Initial representation (h)
                    # Ensure device consistency (MPS/CPU/CUDA)
                    current_state = self.config['model'].h(history[start_idx]['obs'].to(self.config['device']))
                    
                    # Unroll K steps into the "imagination"
                    for k in range(self.config['k_steps']):
                        step_data = history[start_idx + k]
                        target_action = torch.tensor([[step_data['action']]], device=self.config['device'])
                        target_pi = torch.tensor(step_data['target_pi'], dtype=torch.float, device=self.config['device'])
                        target_value = torch.tensor([[step_data['target_value']]], dtype=torch.float, device=self.config['device'])
                        
                        # 1. Dynamics g: Predict next state and reward (reward is 0 in distillation)
                        current_state, pred_reward = self.config['model'].g(current_state, target_action)
                        
                        # 2. Prediction f: Predict policy and value from imagined state
                        pred_policy, pred_value = self.config['model'].f(current_state)
                        
                        # 3. Calculate Losses
                        # Policy Loss: CrossEntropy between Model and Stockfish Best Move
                        loss_p = F.cross_entropy(pred_policy, target_pi.unsqueeze(0))
                        
                        # Value Loss: MSE between Model eval and Stockfish eval
                        loss_v = F.mse_loss(pred_value, target_value)
                        
                        total_loss += (loss_p + loss_v)
                        # total_loss += loss_v

                if total_loss > 0:
                    total_loss.backward()
                    optimizer.step()
                    print(f"Episode {ep+1} | Loss: {total_loss.item():.4e} | Game Length: {len(game_history)}")

            # Periodic Saving
            if (ep + 1) % self.config['save_every'] == 0:
                torch.save(self.config['model'].state_dict(), f"./models/distilled_model_{ep+1}.pth")

        print("Knowledge Distillation Complete.")
        torch.save(self.config['model'].state_dict(), model_path)
        print(f"Final distilled model saved as {model_path}")


    def train_hybrid_co_play(self, stockfish_path, model_path='hybrid_model.pth'):
        # 1. Setup Teacher
        # Using a moderate depth to keep the training loop fast
        teacher = Stockfish(path=stockfish_path, parameters={"Threads": 12, "UCI_Elo": 100})
        teacher.set_depth(1)
        
        optimizer = optim.Adam(self.config['model'].parameters(), lr=self.config['learning_rate'], weight_decay=1e-4)
        replay_buffer = deque(maxlen=500)
        
        for ep in range(self.config['n_episodes']):
            board = chess.Board()
            game_history = []
            
            # --- ROLE RANDOMIZATION ---
            # True: AI is White, Stockfish is Black
            # False: AI is Black, Stockfish is White
            ai_is_white = random.choice([True, False])
            
            # print(f"Starting Episode {ep+1}: AI is {'White' if ai_is_white else 'Black'}")
            
            while not board.is_game_over() and len(game_history) < self.config['max_game_length']:
                current_obs = self.grid.board_to_tensor(board)
                
                # --- DECISION PHASE ---
                if board.turn == ai_is_white:
                    # AI's Turn: Plan using MCTS
                    # We use the model's imagination to pick the best move
                    state = self.config['model'].h(current_obs.to(self.config['device']))
                    move, search_pi = self.search.run_latent_mcts(state, board, return_pi=True)
                else:
                    # Teacher's Turn: Stockfish plays
                    teacher.set_fen_position(board.fen())
                    move_uci = teacher.get_best_move()
                    if move_uci is None: break
                    move = chess.Move.from_uci(move_uci)
                    
                    # Teacher Target: One-hot distribution for the best move
                    search_pi = np.zeros(self.config['action_space'])
                    search_pi[self.grid.encode_action(move)] = 1.0

                # --- TEACHER EVALUATION ---
                # Get Stockfish's opinion on the board BEFORE the move is made
                teacher.set_fen_position(board.fen())
                eval_dict = teacher.get_evaluation()
                
                if eval_dict['type'] == 'cp':
                    # Scale centipawns. We use tanh to keep it in [-1, 1]
                    raw_val = math.tanh(eval_dict['value'] / 300.0)
                else:
                    # Handle Checkmate scores
                    raw_val = 1.0 if eval_dict['value'] > 0 else -1.0
                
                # Perspective Adjustment: 
                # If it's Black's turn, Stockfish returns values from Black's view.
                # We want the value target to be relative to the current player.
                target_value = raw_val if board.turn == chess.WHITE else -raw_val
                
                # --- RECORD HISTORY ---
                game_history.append({
                    'obs': current_obs,
                    'action': self.grid.encode_action(move),
                    'target_pi': search_pi,
                    'target_value': target_value
                })
                
                board.push(move)

            replay_buffer.append(game_history)
            
            # --- TRAINING PHASE (Unrolling) ---
            if len(replay_buffer) >= 1:
                self.config['model'].train()
                optimizer.zero_grad()
                
                # Sample a batch of game sequences
                batch = random.sample(replay_buffer, min(len(replay_buffer), self.config['batch_size']))
                total_loss = 0
                
                for history in batch:
                    if len(history) < self.config['k_steps'] + 1: continue
                    
                    # Random starting point for the K-step unroll
                    start_idx = random.randint(0, len(history) - self.config['k_steps'] - 1)
                    
                    # Initial Latent State
                    current_latent = self.config['model'].h(history[start_idx]['obs'].to(self.config['device']))
                    
                    for k in range(self.config['k_steps']):
                        step_data = history[start_idx + k]
                        
                        # Dynamics (g): Transition to next imagined state
                        act_tensor = torch.tensor([[step_data['action']]], device=self.config['device'])
                        current_latent, _ = self.config['model'].g(current_latent, act_tensor)
                        
                        # Prediction (f): Policy and Value
                        pred_p, pred_v = self.config['model'].f(current_latent)
                        
                        # Targets from Replay Buffer
                        t_pi = torch.tensor(step_data['target_pi'], dtype=torch.float, device=self.config['device'])
                        t_v = torch.tensor([[step_data['target_value']]], dtype=torch.float, device=self.config['device'])
                        
                        # Loss Calculation
                        # Cross-entropy for moves, MSE for position evaluation
                        loss_p = F.cross_entropy(pred_p, t_pi.unsqueeze(0))
                        loss_v = F.mse_loss(pred_v, t_v)
                        
                        # total_loss += (loss_p + loss_v)
                        total_loss += loss_v

                if total_loss > 0:
                    total_loss.backward()
                    optimizer.step()
                    # Determine the result from the AI's perspective
                    res = board.result()
                    if res == "1/2-1/2":
                        ai_outcome = "DRAW"
                    elif (res == "1-0" and ai_is_white) or (res == "0-1" and not ai_is_white):
                        ai_outcome = "AI WON"
                    else:
                        ai_outcome = "AI LOST"

                    print(f"Ep {ep+1:3d} | AI: {'White' if ai_is_white else 'Black':5s} | Result: {ai_outcome:7s} ({res}) | Length: {len(history)} | Loss: {total_loss.item():.4e}")

        # Save the final hybrid-trained model
        torch.save(self.config['model'].state_dict(), model_path)
        print(f"Hybrid training complete. Model saved to {model_path}")