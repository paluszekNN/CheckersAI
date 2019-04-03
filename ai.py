import tensorflow as tf
from game import Checkers
import numpy as np
import random


class HiddenLayer:
    def __init__(self, M1, M2, function=tf.nn.tanh, use_bias=True): # Use relu function if doesn't work
        self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
        self.function = function

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.function(a)


class PolicyModel:
    def __init__(self, D, K, hidden_layer_sizes):
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2
        layer = HiddenLayer(M1, K, tf.nn.softmax, use_bias=False) # Use tanh function if doesn't work
        self.layers.append(layer)

        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        p_a_given_s = Z
        self.predict_op = p_a_given_s

        selected_probs = tf.log(
            tf.reduce_sum(
                p_a_given_s * tf.one_hot(self.actions, K),
                reduction_indices=[1]
            )
        )

        cost = -tf.reduce_sum(self.advantages * selected_probs)

        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        self.session.run(
            self.train_op,
            feed_dict={
                self.X: X,
                self.actions: actions,
                self.advantages: advantages,
            }
        )

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def sample_action(self, X):
        p = self.predict(X)[0]
        return np.random.choice(len(p), p=p)


class ValueModel:
    def __init__(self, D, hidden_layer_sizes):
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        layer = HiddenLayer(M1, 1, lambda x: x)
        self.layers.append(layer)

        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None, ), name='Y')

        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = tf.reshape(Z, [-1])
        self.predict_op = Y_hat

        cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y)
        self.session.run(self.predict_op, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})


def transform_board(game):
    board = game.board_state.flatten()

    even = -1
    for i in range(len(board)-1, -1, -1):
        if even == -1 and i % 2 == 1:
            board = np.delete(board, [i])
        if even == 1 and i % 2 == 0:
            board = np.delete(board, [i])
        if i % 8 == 0:
            even *= -1
    board = np.append(board, game.turn)
    board = np.append(board, game.must_capture*1.)
    board = np.append(board, game.moves_queen_with_out_capture)
    return board


def random_play(game):
    game.available_moves()
    moves = game.moves
    return random.choice(moves)


def encoding_move(move):
    enc_move = None
    if move[0] % 2 == 0:
        move[1] = int((move[1] - 1) / 2)
    else:
        move[1] = int(move[1] / 2)
    
    if move[2] % 2 == 0:
        move[3] = int((move[3] - 1) / 2)
    else:
        move[3] = int(move[3] / 2)

    move32 = move[0] * 4 + move[1], move[2] * 4 + move[3]
    enc_move = move32[0] * move32[1]
    return enc_move

def decoding_move(move):
    x = int(move/32)
    y = move % 32
    Y_col = 0
    X_col = 0
    X_row = x % 8
    Y_row = y % 8

    if X_row % 2 == 0:
        X_col = 2 * int(x / 8) + 1
    else:
        X_col = 2 * int(x / 8)

    if Y_row % 2 == 0:
        Y_col = 2 * int(y / 8) + 1
    else:
        Y_col = 2 * int(y / 8)

    return X_row, X_col, Y_row, Y_col


def play(game, pmodel, vmodel, gamma, max_moves):
    game.reset()
    board = transform_board(game)
    reward = 0

    while not game.win:
        move = pmodel.sample_action(board)
        prev_board = board
        if game.turn == 1:
            game.move(decoding_move(move))
            board = transform_board(game)
        else:
            game.move(random_play(game))
            continue

        if game.win:
            reward = game.win
        V_next = vmodel.predict(board)
        G = reward + gamma * np.max(V_next)
        advantage = G - vmodel.predict(prev_board)
        pmodel.partial_fit(prev_board, move, advantage)
        vmodel.partial_fit(prev_board, G)

    if max_moves < game.count_movement:
        max_moves = game.count_movement
    return reward, max_moves


if __name__ == '__main__':
    max_moves = 0
    game = Checkers()
    D = len(transform_board(game))
    K = 32*32
    pmodel = PolicyModel(D, K, [32, 32, 32])
    vmodel = ValueModel(D, [32, 32, 32])
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    pmodel.set_session(session)
    vmodel.set_session(session)
    gamma = 0.99

    while True:
        totalreward, max_moves = play(game, pmodel, vmodel, gamma, max_moves)

        if totalreward == 1:
            break
    print(max_moves)
