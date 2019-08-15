import tensorflow as tf
from game import Checkers
import numpy as np
import random
import min_max
from datetime import datetime
import sys
from ai import DQN, decoding_move, ReplayMemory, random_play, encoding_move, game

MIN_EXPERIENCES = 50


def play_one(total_t, experience_replay_buffer, model1, model2, epsilon):
    t0 = datetime.now()

    # Reset the environment
    game.reset()

    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0

    while not game.win:

        # Take action
        if game.turn == 1:
            game.available_moves()
            states, _, _, _, _ = experience_replay_buffer.get_minibatch()
            move = model1.sample_action(states, epsilon)
            game.move(decoding_move(move))
        if game.win == 0 and game.turn == -1:
            game.available_moves()
            states, _, _, _, _ = experience_replay_buffer.get_minibatch()
            move = model2.sample_action(states, epsilon)
            game.move(decoding_move(move))
        reward = game.win

        episode_reward += reward

        num_steps_in_episode += 1

        total_t += 1

    return total_t, episode_reward, (
                datetime.now() - t0), num_steps_in_episode, total_time_training / num_steps_in_episode, epsilon


if __name__ == '__main__':
    input_size = game.board_state.shape
    action_space_size = 32 * 32
    conv_layer_sizes = [(128, 2, 1), (128, 2, 1), (128, 2, 1)]
    hidden_layer_sizes = [256]

    gamma = 0.99
    batch_sz = 32
    num_episodes = 10000
    total_t = 0

    experience_replay_buffer = ReplayMemory(input_size)
    episode_rewards = np.zeros(num_episodes)

    epsilon = 0.001

    model1 = DQN(input_size, action_space_size, conv_layer_sizes, hidden_layer_sizes, 'model1')
    model2 = DQN(input_size, action_space_size, conv_layer_sizes, hidden_layer_sizes, 'model2')
    with tf.Session() as session:
        model1.set_session(session)
        model2.set_session(session)
        session.run(tf.global_variables_initializer())
        game.reset()

        wins = 0
        for i in range(MIN_EXPERIENCES):
            game.available_moves()
            if game.win != 0:
                game.reset()
            move = random_play(game)
            action = encoding_move(move)
            game.move(move)
            if game.win == 0:
                new_game = Checkers()
                new_game.board_state = np.array(game.board_state)
                new_game.turn = game.turn
                new_game.moves_queen_with_out_capture = game.moves_queen_with_out_capture
                move = min_max.min_max_player(new_game, new_game.turn)
                game.move(move)
            reward = game.win
            experience_replay_buffer.add_experince(action, game.board_state, reward)

        t0 = datetime.now()
        for i in range(num_episodes):
            total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one(
                total_t,
                experience_replay_buffer,
                model1,
                model2,
                epsilon
            )
            episode_rewards[i] = episode_reward

            last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
            print("Episode:", i,
                  "Duration:", duration,
                  "Num steps:", num_steps_in_episode,
                  "Reward:", episode_reward,
                  "Training time per step:", "%.3f" % time_per_step,
                  "Avg Reward (Last 100):", "%.3f" % last_100_avg,
                  "Epsilon:", "%.3f" % epsilon
                  )
            sys.stdout.flush()
        print("Total duration:", datetime.now() - t0)
