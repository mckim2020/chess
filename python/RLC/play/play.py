import chess
import chess.svg
import torch
from IPython.display import display, SVG, clear_output

class PlayChess():
    def __init__(self, config, grid, search):
        self.config = config
        self.grid = grid
        self.search = search


    def display_board(self, board):
        """Renders the board in the notebook/IDE output."""
        last_move = board.peek() if board.move_stack else None
        # We add the last move highlight to help you see the AI's response
        svg_data = chess.svg.board(board, size=400, lastmove=last_move)
        clear_output(wait=True)
        display(SVG(svg_data))
        # print(board)


    def get_user_move(self, board):
        """Prompts for input and validates the move."""
        while True:
            user_input = input("Your Move (e.g., e4, Nf3, or e2e4): ").strip()
            if user_input.lower() in ['quit', 'exit']:
                return None
            try:
                # Try SAN first (e4), then UCI (e2e4)
                try:
                    move = board.parse_san(user_input)
                except ValueError:
                    move = chess.Move.from_uci(user_input)
                
                if move in board.legal_moves:
                    return move
                else:
                    print("Illegal move! Try again.")
            except:
                print("Invalid format. Use SAN (e4) or UCI (e2e4).")


    def play(self):
        """Main game loop."""
        board = chess.Board()
        print("Game Started!")

        while not board.is_game_over():
            # 1. Show the board
            self.display_board(board)

            # 2. Human Turn
            move = self.get_user_move(board)
            if move is None: 
                print("Game closed.")
                return
            board.push(move)

            if board.is_game_over(): break

            # 3. AI Turn
            self.display_board(board)
            print("AI is thinking (MuZero MCTS)...")
            
            with torch.no_grad():
                # Prepare tensor and move to Mac GPU (MPS) if available
                obs = self.grid.board_to_tensor(board)
                latent_state = self.config['model'].h(obs)
                
                # Perform latent space search
                ai_move = self.search.run_latent_mcts(
                    latent_state, 
                    board
                    )
            
            if ai_move:
                board.push(ai_move)
            else:
                print("AI could not find a move. AI Resigns.")
                break

        # Final Result
        self.display_board(board)
        print(f"Game Over! Result: {board.result()}")