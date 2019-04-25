from game import Checkers
import random
import numpy as np
from min_max import min_max_player


def random_play(game):
    game.available_moves()
    moves = game.moves
    return random.choice(moves)


if __name__ == '__main__':
    game = Checkers()
    n = 0
    while True:
        n += 1
        game.reset()
        while not game.win:
            if game.turn == 1:
                game.move(random_play(game))
            else:
                new_game = Checkers()
                new_game.board_state = np.array(game.board_state)
                new_game.turn = game.turn
                new_game.moves_queen_with_out_capture = game.moves_queen_with_out_capture
                move = min_max_player(new_game, new_game.turn)
                game.move(move)
            game.print_board()
        if n % 100 == 0:
            print(n)
