from game import Checkers
import sys
import numpy as np


def evaluate(new_game):
    return np.array(new_game.board_state).flatten().sum()


def min_max_alpha_beta(game, turn, max_depth, alpha=-sys.float_info.max, beta=sys.float_info.max):
    best_score_move = None
    game.available_moves()
    moves = game.moves

    if not moves:
        return 0, None

    for move in moves:
        new_game = Checkers()
        new_game.board_state = game.board_state
        new_game.turn = game.turn
        new_game.moves_queen_with_out_capture = game.moves_queen_with_out_capture
        new_game.move(move)
        winner = new_game.win
        if winner != 0:
            return winner*10000, move
        else:
            if max_depth <= 1:
                score = evaluate(new_game)
            else:
                score, _ = min_max_alpha_beta(new_game, -turn, max_depth-1, alpha, beta)
            if turn > 0:
                if score > alpha:
                    alpha = score
                    best_score_move = move
            else:
                if score < beta:
                    beta = score
                    best_score_move = move
            if alpha >= beta:
                break
    return alpha if turn > 0 else beta, best_score_move


def min_max_player(game, turn):
    return min_max_alpha_beta(game, turn, 4)[1]


