import torch
import chess


class Grid():
    def __init__(self, config:dict):
        self.config = config


    # def board_to_tensor(self, board):
    #     tensor = torch.zeros(1, 12, 8, 8, device=self.config['device'])
    #     piece_map = {'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5, 'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11}
    #     for sq, pc in board.piece_map().items():
    #         tensor[0, piece_map[pc.symbol()], sq // 8, sq % 8] = 1.0
    #     return tensor


    def board_to_tensor(self, board):
        # Now using 18 channels (12 pieces + 4 castling + 1 EP + 1 Turn)
        tensor = torch.zeros(1, 18, 8, 8, device=self.config['device'])
        
        side_to_move = board.turn
        
        # 1. Piece Channels (0-11)
        for sq, pc in board.piece_map().items():
            p_type = pc.piece_type - 1
            is_white = pc.color
            
            # Perspective mapping
            channel = p_type if is_white == side_to_move else p_type + 6
            row = sq // 8
            col = sq % 8
            if side_to_move == chess.BLACK:
                row = 7 - row
            
            tensor[0, channel, row, col] = 1.0

        # 2. Castling Channels (12-15)
        # We fill the entire 8x8 plane with 1.0 if the right exists
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[0, 12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[0, 13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[0, 14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[0, 15, :, :] = 1.0

        # 3. En Passant Channel (16)
        if board.ep_square:
            ep_row = board.ep_square // 8
            ep_col = board.ep_square % 8
            if side_to_move == chess.BLACK:
                ep_row = 7 - ep_row
            tensor[0, 16, ep_row, ep_col] = 1.0

        # 4. Turn Channel (Channel 17)
        # We fill the entire 8x8 grid with a single value
        if side_to_move == chess.WHITE:
            tensor[0, 17, :, :] = 1.0  # White's turn
        else:
            tensor[0, 17, :, :] = 0.0  # Black's turn (or -1.0 depending on preference)

        return tensor


    def encode_action(self, move):
        return move.from_square * 64 + move.to_square