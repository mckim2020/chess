import numpy as np
import torch
import random
import math
import chess


class SearchChess():
    def __init__(self, config:dict, grid=None):
        self.config = config
        self.grid = grid
        # Standard material values for evaluation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }


    # def evaluate_board(self, board):
        # """A simple material-based evaluation function."""
        # if board.is_checkmate():
        #     return -99999 if board.turn else 99999
        
        # score = 0
        # for piece_type, value in self.piece_values.items():
        #     score += len(board.pieces(piece_type, chess.WHITE)) * value
        #     score -= len(board.pieces(piece_type, chess.BLACK)) * value
        
        # # Return score relative to the side to move (Negamax style)
        # return score if board.turn == chess.WHITE else -score


    # def evaluate_board(self, board):
    #     if board.is_checkmate(): return -99999
    #     if board.is_draw(): return 0
        
    #     tensor = self.board_to_tensor(board)
    #     with torch.no_grad():
    #         # Output is already relative to the side to move!
    #         val = self.model(tensor).item() 
            
    #     return int(val * 1000)


    def evaluate_board(self, board):
        if board.is_checkmate(): return 0.0  # I am in checkmate, my win prob is 0
        if board.can_claim_draw(): return 0.5       # Draw is 50/50
        
        tensor = self.grid.board_to_tensor(board)
        # with torch.no_grad():
            # Model outputs -1.0 (Loss) to 1.0 (Win)
        nn_output = self.config['model'](tensor).item() 
            
        # Transform (-1 to 1) -> (0 to 1) to match the teacher's scale
        win_prob = (nn_output + 1) / 2
        return win_prob


    def order_moves(self, board, moves):
        """
        Stockfish's 'Secret Sauce': Searching better moves first 
        drastically increases pruning efficiency.
        """
        def score_move(move):
            score = 0
            # 1. Prioritize Captures (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += 10 * self.piece_values[victim.piece_type] - self.piece_values[attacker.piece_type]
            
            # 2. Prioritize Promotions
            if move.promotion:
                score += 900
            
            # 3. Give a small bonus for checks
            if board.gives_check(move):
                score += 50
                
            return score

        return sorted(moves, key=score_move, reverse=True)


    def quiescence_search(self, board, alpha, beta):
        """
        Prevents the 'Horizon Effect'. Only searches captures until 
        the position is 'quiet' so we don't miss a hanging piece.
        """
        stand_pat = self.evaluate_board(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        for move in self.order_moves(board, list(board.generate_legal_captures())):
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha


    def alpha_beta(self, board, depth, alpha, beta):
        """The core Negamax Alpha-Beta search."""
        if depth == 0:
            return self.quiescence_search(board, alpha, beta)

        if board.is_game_over():
            return self.evaluate_board(board)

        best_score = -float('inf')
        
        # Move Ordering is crucial for Alpha-Beta efficiency
        ordered_moves = self.order_moves(board, list(board.legal_moves))

        for move in ordered_moves:
            board.push(move)
            score = -self.alpha_beta(board, depth - 1, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta # Fail-high (Pruning)
            
            if score > alpha:
                alpha = score
                
        return alpha


    def select_move(self, board, max_depth=4):
        """
        Iterative Deepening: Searches depth 1, then 2, etc. 
        This is how engines manage time and improve move ordering.
        """
        best_move = None
        
        for depth in range(1, max_depth + 1):
            alpha = -float('inf')
            beta = float('inf')
            best_score = -float('inf')
            
            ordered_moves = self.order_moves(board, list(board.legal_moves))
            
            for move in ordered_moves:
                board.push(move)
                score = -self.alpha_beta(board, depth - 1, -beta, -alpha)
                board.pop()
                
                if score > best_score:
                    best_score = score
                    best_move = move
                
                alpha = max(alpha, score)
                
        return best_move


    def add_dirichlet_noise(self, node, board):
        """Adds noise to the priors of the root's children."""        
        actions = list(node.children.keys())
        if not actions:
            return

        # Generate noise from Dirichlet distribution
        noise = np.random.dirichlet([self.config.get('noise_alpha', 0.3)] * len(actions))
        
        for i, action in enumerate(actions):
            # Blend the model's prior with the noise
            node.children[action].prior = (1 - self.config.get('noise_epsilon', 0.25)) * node.children[action].prior + self.config.get('noise_epsilon', 0.25) * noise[i]


    def select_final_move(self, root, board, temperature=1.0):
        actions = []
        visit_counts = []
        
        for action_idx, child in root.children.items():
            actions.append(action_idx)
            visit_counts.append(child.visit_count)

        if temperature == 0: # Deterministic: pick the absolute best
            best_idx = np.argmax(visit_counts)
            chosen_action = actions[best_idx]
        else: # Stochastic: pick based on probability distribution
            visit_counts = np.array(visit_counts) ** (1 / temperature)
            probs = visit_counts / sum(visit_counts)
            chosen_action = np.random.choice(actions, p=probs)

        # Convert chosen_action back to chess.Move
        for move in board.legal_moves:
            if self.grid.encode_action(move) == chosen_action:
                return move


    def run_latent_mcts(self, initial_state, board, return_pi=False):
        self.config['model'].eval()
        
        # 1. Initialize Root with Representation h
        # Get initial policy and value from the Prediction function f
        with torch.no_grad():
            policy_logits, value = self.config['model'].f(initial_state)

            # --- MASKING ILLEGAL MOVES ---
            # We only care about the priors for moves that are legal in the real game
            mask = torch.full(policy_logits.shape, -float("inf"), device=self.config['device'])
            legal_indices = [self.grid.encode_action(m) for m in board.legal_moves]
            for idx in legal_indices:
                mask[0, idx] = policy_logits[0, idx]

            probs = torch.softmax(policy_logits, dim=1).flatten()

        root = MCTSNode(prior=1.0, hidden_state=initial_state)
        
        # Expand root with legal moves only
        for move in board.legal_moves:
            action_idx = self.grid.encode_action(move)
            root.children[action_idx] = MCTSNode(prior=probs[action_idx].item(), hidden_state=None)

        # Optionally add Dirichlet noise for exploration
        if self.config['noise']:
            self.add_dirichlet_noise(root, board)

        # 2. Run Simulations
        for _ in range(self.config['n_simulations']):
            node = root
            search_path = [node]

            # --- SELECT ---
            # Traverse until we find a node that hasn't been expanded
            while node.children and all(child.hidden_state is not None for child in node.children.values()):
                action, node = node.select_child(self.config['c_puct'])
                search_path.append(node)

            # --- EXPAND & EVALUATE ---
            # Pick a child to expand
            parent = search_path[-2] if len(search_path) > 1 else root
            # We need an action that hasn't been "imagined" yet
            unexpanded_actions = [a for a, c in parent.children.items() if c.hidden_state is None]
            
            if unexpanded_actions:
                action = random.choice(unexpanded_actions)
                child = parent.children[action]
                
                with torch.no_grad():
                    # Use Dynamics g to get next latent state and reward
                    next_state, reward = self.config['model'].g(parent.hidden_state, torch.tensor([[action]], device=self.config['device']))
                    # Use Prediction f to evaluate that state
                    policy_logits, value_tensor = self.config['model'].f(next_state)
                    
                    child.hidden_state = next_state
                    child.reward = reward.item()
                    
                    # Evaluation for backprop
                    leaf_value = value_tensor.item()
                    
                    # Set priors for child's potential moves (if we were to go deeper)
                    # For simplicity in this version, we don't pre-calculate legal moves here
                    # to save computation, but real MuZero does.
            else:
                leaf_value = node.value

            # --- BACKPROPAGATE ---
            # Update values from leaf to root
            for node in reversed(search_path):
                node.value_sum += leaf_value
                node.visit_count += 1
                # In MuZero, we also factor in the intermediate rewards
                leaf_value = node.reward + 0.99 * leaf_value 

        # 3. Calculate Target Policy (pi)
        # pi(a|s) = N(s,a)^(1/τ) / Σ N(s,b)^(1/τ)
        # During training, τ=1. During competitive play, τ is small.
        total_visits = sum(child.visit_count for child in root.children.values())
        pi = np.zeros(self.config['action_space']) # Ensure this matches your model output dim
        for action_idx, child in root.children.items():
            pi[action_idx] = child.visit_count / total_visits

        # 4. Final Move Selection
        # We use a temperature of 1.0 for training to ensure diverse experience
        # return self.select_final_move(root, board, temperature=1.0 if self.config['noise'] else 0.1)
        move = self.select_final_move(root, board, temperature=1.0 if self.config['noise'] else 0.1)

        if return_pi:
            return move, pi
        return move

        # best_action_idx = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        
        # # Convert index back to chess.Move
        # for move in board.legal_moves:
        #     if self.grid.encode_action(move) == best_action_idx:
        #         return move
        
        # return random.choice(list(board.legal_moves))


    # def run_latent_mcts(self, model, state, board, simulations=10):
    #     """
    #     Simplified MCTS: Plans using the model's 'imagination' (g and f).
    #     In a full version, this would build a tree. Here, we pick the best 
    #     immediate move predicted by f(g(s, a)).
    #     """
    #     best_move = None
    #     max_val = -float('inf')
        
    #     legal_moves = list(board.legal_moves)
    #     if not legal_moves: return None
        
    #     model.eval()
    #     with torch.no_grad():
    #         for move in legal_moves:
    #             action_tensor = torch.tensor([[self.grid.encode_action(move)]])
    #             # Use Dynamics to imagine next state
    #             next_state, _ = model.g(state, action_tensor)
    #             # Use Prediction to see how good that state is
    #             _, value = model.f(next_state)
                
    #             if value.item() > max_val:
    #                 max_val = value.item()
    #                 best_move = move
    #     return best_move


class MCTSNode:
    def __init__(self, prior, hidden_state):
        self.hidden_state = hidden_state  # The s output from g or h
        self.prior = prior                # The probability p from f
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}                # Map of action_idx -> MCTSNode
        self.reward = 0                   # Immediate reward r from g

    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def select_child(self, c_puct):
        """Standard PUCT selection formula."""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            # score = Q(s,a) + U(s,a)
            u_score = c_puct * child.prior * (math.sqrt(self.visit_count) / (1 + child.visit_count))
            score = child.value + u_score

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child