from game import Checkers
import random


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
            game.move(random_play(game))
        if n % 100 == 0:
            print(n)
