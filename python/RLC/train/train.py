import os
import chess
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque


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
            while not board.is_game_over() and len(game_history) < 50:
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