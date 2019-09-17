import tensorflow as tf
from game import Checkers
import numpy as np
import random
import min_max
from datetime import datetime
import sys


GAME = Checkers()
INPUT_SIZE = GAME.board_state.shape
ACTION_SPACE_SIZE= 32 * 32
CONV_LAYER_SIZE = [(32, 3, 3)]
HIDDEN_LAYER_SIZE = [64]
MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 100
TARGET_UPDATE_PERIOD = 100
AGENT_LENGTH_HISTORY = 10


class ReplayMemory:
    def __init__(self, input_size, agent_history_length=AGENT_LENGTH_HISTORY, batch_size=32, size=MAX_EXPERIENCES):
        self.size = size
        self.input_size = input_size
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.boards = np.empty((self.size, self.input_size[0], self.input_size[1]), dtype=np.float32)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        self.states = np.empty((self.batch_size, self.agent_history_length, self.input_size[0], self.input_size[1]), dtype=np.float32)
        self.new_states = np.empty((self.batch_size, self.agent_history_length, self.input_size[0], self.input_size[1]), dtype=np.float32)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experince(self, action, board, reward):
        if board.shape != self.input_size:
            raise ValueError('Dimension of size is wrong!')
        self.actions[self.current] = action
        self.boards[self.current] = board
        self.rewards[self.current] = reward
        if reward != 0:
            self.terminal_flags[self.current] = 1
        else:
            self.terminal_flags[self.current] = 0
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.boards[index - self.agent_history_length + 1:index + 1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)

        return self.states, self.actions[self.indices], self.rewards[
            self.indices], self.new_states, self.terminal_flags[self.indices]


class DQN:
    def __init__(self, input_size, action_space_size, conv_layer_sizes, hidden_layer_sizes, name):
        self.action_space_size = action_space_size
        self.name = name

        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32, shape=(None, AGENT_LENGTH_HISTORY, input_size[0], input_size[1]), name='input')

            self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

            Z = self.input / 2.
            for num_output_filters, filter_size, pool_size in conv_layer_sizes:
                Z = tf.layers.conv2d(Z, num_output_filters, (filter_size, pool_size), activation=tf.nn.relu)
                Z = tf.layers.batch_normalization(Z, axis=1)
                Z = tf.nn.leaky_relu(Z)

            for _ in range(10):
                input = Z
                Z = tf.layers.conv2d(Z, 32, (3, 3), padding='same')
                Z = tf.layers.batch_normalization(Z, axis=1)
                Z = tf.nn.relu(Z)
                Z = tf.keras.layers.add([input, Z])
                Z = tf.nn.leaky_relu(Z)

            ph = tf.layers.conv2d(Z, 2, (1, 1), activation=tf.nn.relu)
            ph = tf.layers.batch_normalization(ph)
            ph = tf.nn.leaky_relu(ph)
            ph = tf.layers.flatten(ph)
            ph = tf.layers.dense(ph, action_space_size, activation=tf.nn.relu)

            vh = tf.layers.conv2d(Z, 1, (1, 1), activation=tf.nn.relu)
            vh = tf.layers.batch_normalization(vh)
            vh = tf.nn.leaky_relu(vh)
            vh = tf.layers.flatten(vh)
            vh = tf.layers.dense(vh, 20, use_bias=False, activation=tf.nn.relu)
            vh = tf.nn.leaky_relu(vh)
            vh = tf.layers.dense(vh, 1, activation=tf.nn.tanh)

            # Z = tf.layers.flatten(Z)
            # for layer_size in hidden_layer_sizes:
            #     Z = tf.layers.dense(Z, layer_size)

            # self.predict_op = tf.layers.dense(Z, action_space_size)
            self.predict_op = ph

            selected_action_values = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions, action_space_size), reduction_indices=[1])

            cost = tf.reduce_mean(tf.losses.huber_loss(self.G, selected_action_values))
            self.train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)

            self.cost = cost

    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.name)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.name)]
        theirs = sorted(theirs, key=lambda v: v.name)

        ops = []
        for p, q in zip(mine, theirs):
            op = p.assign(q)
            ops.append(op)
        self.session.run(ops)

    def save(self, i):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.name)]
        params = self.session.run(params)
        np.savez('tf_dqn_weights'+str(i)+'.npz', *params)

    def load(self, path):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.name)]
        npz = np.load(path)
        ops = []
        for p, (_, v) in zip(params, npz.iteritems()):
            ops.append(p.assign(v))
        self.session.run(ops)

    def set_session(self, session):
        self.session = session

    def predict(self, states):
        move = self.session.run(self.predict_op, feed_dict={self.input: states})
        return move

    def update(self, states, actions, targets):
        c, _ = self.session.run([self.cost, self.train_op], feed_dict={self.input: states, self.G: targets, self.actions: actions})
        return c

    def sample_action(self, states, eps):
        if np.random.random() < eps:
            move = random_play()
            return encoding_move(move)
        else:
            p = self.predict(states)[0]
            GAME.available_moves()
            moves = GAME.moves
            list_move = []
            indices = []
            for mov in moves:
                list_move.append(p[encoding_move(mov)])
                indices.append(encoding_move(mov))
            index = np.argmax(list_move)
            move = indices[index]
            return move


def transform_board():
    board = GAME.board_state.flatten()

    even = -1
    for i in range(len(board)-1, -1, -1):
        if even == -1 and i % 2 == 1:
            board = np.delete(board, [i])
        if even == 1 and i % 2 == 0:
            board = np.delete(board, [i])
        if i % 8 == 0:
            even *= -1
    board = np.append(board, GAME.turn)
    board = np.append(board, GAME.must_capture*1.)
    board = np.append(board, GAME.moves_queen_with_out_capture)
    return board


def random_play():
    GAME.available_moves()
    moves = GAME.moves
    return random.choice(moves)


def encoding_move(move):
    enc_move = None
    new_move = [move[0], 0, move[2], 0]
    if move[0] % 2 == 0:
        new_move[1] = int((move[1] - 1) / 2)
    else:
        new_move[1] = int(move[1] / 2)

    if move[2] % 2 == 0:
        new_move[3] = int((move[3] - 1) / 2)
    else:
        new_move[3] = int(move[3] / 2)

    move32 = new_move[0] * 4 + new_move[1], new_move[2] * 4 + new_move[3]
    enc_move = move32[0] * 32 + move32[1]
    return enc_move


def decoding_move(move):
    x = int(move/32)
    y = move % 32
    Y_col = 0
    X_col = 0
    X_row = int(x / 4)
    Y_row = int(y / 4)

    if X_row % 2 == 0:
        X_col = 2 * int(x % 4) + 1
    else:
        X_col = 2 * int(x % 4)

    if Y_row % 2 == 0:
        Y_col = 2 * int(y % 4) + 1
    else:
        Y_col = 2 * int(y % 4)

    return X_row, X_col, Y_row, Y_col


def play(pmodel, vmodel, gamma):
    GAME.reset()
    board = transform_board()
    reward = 0

    while not GAME.win:
        GAME.available_moves()
        move = pmodel.sample_action(board, GAME.moves)
        prev_board = board
        if GAME.turn == 1:
            GAME.move(decoding_move(move))
            board = transform_board()
        else:
            new_game = Checkers()
            new_game.board_state = np.array(GAME.board_state)
            new_game.turn = GAME.turn
            new_game.moves_queen_with_out_capture = GAME.moves_queen_with_out_capture
            move = min_max.min_max_player(new_game, new_game.turn)
            GAME.move(move)
            reward = GAME.win
            if not reward:
                continue

        reward = GAME.win
        V_next = vmodel.predict(board)
        G = reward + gamma * np.max(V_next)
        advantage = G - vmodel.predict(prev_board)
        pmodel.partial_fit(prev_board, move, advantage)
        vmodel.partial_fit(prev_board, G)

    return reward


def learn(model, target_model, experience_replay_buffer, gamma):
    # Sample experiences
    states, actions, rewards, next_states, terminals = experience_replay_buffer.get_minibatch()

    # Calculate targets
    next_Qs = target_model.predict(next_states)
    next_Q = np.amax(next_Qs, axis=1)
    targets = rewards + np.invert(terminals).astype(np.float32) * gamma * next_Q

    # Update model
    loss = model.update(states, actions, targets)
    return loss


def play_one(total_t, experience_replay_buffer, model, target_model, gamma, epsilon, epsilon_change, epsilon_min):
    t0 = datetime.now()

    # Reset the environment
    GAME.reset()
    board = GAME.board_state
    loss = None

    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0

    while not GAME.win:

        # Update target network
        if total_t % TARGET_UPDATE_PERIOD == 0:
            target_model.copy_from(model)
            print("Copied model parameters to target network. total_t = %s, period = %s" % (
            total_t, TARGET_UPDATE_PERIOD))

        # Take action
        if GAME.turn == 1:
            GAME.available_moves()
            states, _, _, _, _ = experience_replay_buffer.get_minibatch()
            action = model.sample_action(states, epsilon)
            GAME.move(decoding_move(action))
        if GAME.win == 0 and GAME.turn == -1:
            new_game = Checkers()
            new_game.board_state = np.array(GAME.board_state)
            new_game.turn = GAME.turn
            new_game.moves_queen_with_out_capture = GAME.moves_queen_with_out_capture
            move = min_max.min_max_player(new_game, new_game.turn)
            GAME.move(move)
        reward = GAME.win
        board = GAME.board_state

        # Compute total reward
        episode_reward += reward

        # Save the latest experience
        experience_replay_buffer.add_experince(action, GAME.board_state, reward)

        # Train the model, keep track of time
        t0_2 = datetime.now()
        loss = learn(model, target_model, experience_replay_buffer, gamma)
        dt = datetime.now() - t0_2

        # More debugging info
        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1

        total_t += 1

        epsilon = max(epsilon - epsilon_change, epsilon_min)

    return total_t, episode_reward, (
                datetime.now() - t0), num_steps_in_episode, total_time_training / num_steps_in_episode, epsilon


if __name__ == '__main__':
    gamma = 0.99
    batch_sz = 32
    num_episodes = 10000
    total_t = 0

    experience_replay_buffer = ReplayMemory(INPUT_SIZE)
    episode_rewards = np.zeros(num_episodes)

    epsilon = 1.0
    epsilon_min = 0.001
    epsilon_change = (epsilon - epsilon_min) / num_episodes

    model = DQN(INPUT_SIZE, ACTION_SPACE_SIZE, CONV_LAYER_SIZE, HIDDEN_LAYER_SIZE, 'model')
    target_model = DQN(INPUT_SIZE, ACTION_SPACE_SIZE, CONV_LAYER_SIZE, HIDDEN_LAYER_SIZE, 'target_model')
    with tf.Session() as session:
        # model.load('tf_dqn_weights1.npz')
        model.set_session(session)
        target_model.set_session(session)
        session.run(tf.global_variables_initializer())
        GAME.reset()

        wins = 0
        for i in range(MIN_EXPERIENCES):
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

        t0 = datetime.now()
        for i in range(num_episodes):
            total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one(
                total_t,
                experience_replay_buffer,
                model,
                target_model,
                gamma,
                epsilon,
                epsilon_change,
                epsilon_min,
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
            if i % 1 == 0 and i != 0:
                model.save(i)
        print("Total duration:", datetime.now() - t0)

        model.save('last')
