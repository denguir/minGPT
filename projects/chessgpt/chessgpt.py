"""
Trains a GPT to play chess.
"""

import os
import sys
import random

try:
    import chess
    import chess.pgn as pgn
except ImportError:
    print("Please install chess module: pip install chess")


import torch
from functools import cached_property
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chessgpt'

    # data
    C.data = PGNDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class PGNDataset(Dataset):
    """
    Emits batches of boards
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 76 # number of chars needed to encode one board
        return C
    
    def __init__(self, config, games):
        self.config = config

        vocab = sorted([' ', '/', '0', '1', 'u', 'U', 'v', 'V', 'w', 'W', 'x', 'X', '#', 
                        'K', 'Q', 'R', 'B', 'N', 'P', 'k', 'q', 'r', 'b', 'n', 'p'])
        num_games, vocab_size = len(games), len(vocab)
        print('data contains %d games, encoded with %d unique chars.' % (num_games, vocab_size))

        self.stoi = { ch: i for i, ch in enumerate(vocab) }
        self.itos = { i: ch for i, ch in enumerate(vocab) }
        self.vocab_size = vocab_size
        self.games = games

    @cached_property
    def data(self):
        data = []
        for game in self.games:
            data += list(zip(game, game[1:]))
        return data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data)
                
    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = BoardEncoder.to_tensor(x, self.stoi)
        y = BoardEncoder.to_tensor(y, self.stoi)        
        return x, y
    

class BoardEncoder:

    @staticmethod
    def get_pieces(board):
        """We will keep the same encoding as in the FEN notation.
        But we will encode empty squares with a # character.
        """
        fen = board.fen()
        pieces = fen.split()[0]
        for piece in pieces:
            if piece.isdecimal():
                pieces = pieces.replace(piece, '#' * int(piece))
        return pieces

    @staticmethod
    def get_turn(board):
        """
        0: black
        1: white
        """
        return "0" if board.turn == chess.BLACK else "1"

    @staticmethod
    def get_castling_rights(board):
        """
        u: no castling rights for black
        U: no castling rights for white
        v: queenside castling rights for black
        V: queenside castling rights for white
        w: kingside castling rights for black
        W: kingside castling rights for white
        x: both castling rights for black
        X: both castling rights for white
        """

        castling_rights = ["U", "u"]

        if board.has_queenside_castling_rights(chess.WHITE) and board.has_kingside_castling_rights(chess.WHITE):
            castling_rights[0] = "X"
        elif board.has_queenside_castling_rights(chess.WHITE):
            castling_rights[0] = "V"
        elif board.has_kingside_castling_rights(chess.WHITE):
            castling_rights[0] = "W"

        if board.has_queenside_castling_rights(chess.BLACK) and board.has_kingside_castling_rights(chess.BLACK):
            castling_rights[1] = "x"
        elif board.has_queenside_castling_rights(chess.BLACK):
            castling_rights[1] = "v"
        elif board.has_kingside_castling_rights(chess.BLACK):
            castling_rights[1] = "w"

        return "".join(castling_rights)

    @staticmethod
    def encode_str(board):
        return BoardEncoder.get_pieces(board) + " " + \
                BoardEncoder.get_turn(board) + " " + \
                BoardEncoder.get_castling_rights(board)
    
    @staticmethod
    def encode_int(board, vocab_map):
        return [vocab_map[ch] for ch in BoardEncoder.encode_str(board)]
    
    @staticmethod
    def to_tensor(board, vocab_map):
        return torch.tensor(BoardEncoder.encode_int(board, vocab_map), dtype=torch.long)
    

def read_pgn(pgn_file):
    games = []
    with open(pgn_file) as f:
        while True:
            game = pgn.read_game(f)
            if game is None:
                break
            moves = []
            board = game.board()
            moves.append(board.fen())
            for move in game.mainline_moves():
                board.push(move)
                moves.append(board.fen())
            games.append(moves)
    return games

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    games = read_pgn('master_games.pgn')
    train_dataset = PGNDataset(config.data, games)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = "O God, O God!"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()