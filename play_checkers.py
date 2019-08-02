from min_max import min_max_player
from ai import *
import tensorflow as tf
import min_max


if __name__ == '__main__':
    model = DQN(INPUT_SIZE, ACTION_SPACE_SIZE, CONV_LAYER_SIZE, HIDDEN_LAYER_SIZE, 'model')
    with tf.Session() as session:
        model.set_session(session)

        model.load('tf_dqn_weights1.npz')

        experience_replay_buffer = ReplayMemory(INPUT_SIZE)

        for i in range(50):
            GAME.available_moves()
            if GAME.win != 0:
                GAME.reset()
            move = random_play()
            action = encoding_move(move)
            GAME.move(move)
            if GAME.win == 0:
                new_game = Checkers()
                new_game.board_state = np.array(GAME.board_state)
                new_game.turn = GAME.turn
                new_game.moves_queen_with_out_capture = GAME.moves_queen_with_out_capture
                move = min_max.min_max_player(new_game, new_game.turn)
                GAME.move(move)
            reward = GAME.win
            experience_replay_buffer.add_experince(action, GAME.board_state, reward)

        n = 0
        while True:
            n += 1
            GAME.reset()
            while not GAME.win:
                if GAME.turn == 1:
                    GAME.available_moves()
                    states, _, _, _, _ = experience_replay_buffer.get_minibatch()
                    action = model.sample_action(states, eps=0.0001)
                    GAME.move(decoding_move(action))
                    # GAME.move(random_play(game))
                else:
                    new_game = Checkers()
                    new_game.board_state = np.array(GAME.board_state)
                    new_game.turn = GAME.turn
                    new_game.moves_queen_with_out_capture = GAME.moves_queen_with_out_capture
                    move = min_max_player(new_game, new_game.turn)
                    GAME.move(move)

                reward = GAME.win

                experience_replay_buffer.add_experince(action, GAME.board_state, reward)
                GAME.print_board()
            if n % 100 == 0:
                print(n)
