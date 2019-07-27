from game import Checkers
import unittest
import numpy as np


class TestBoard(unittest.TestCase):

    def setUp(self):
        self.test_Game = Checkers()

    def test_move(self):
        self.test_Game.move((5, 0, 4, 1))
        print('move')
        expected_result = np.array([[0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, -1, 0, -1, 0, -1, 0],
                         [0, -1, 0, -1, 0, -1, 0, -1],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0, 1],
                         [1, 0, 1, 0, 1, 0, 1, 0]])

        np.testing.assert_array_equal(expected_result, self.test_Game.board_state)

        self.test_Game.reset()
        print("capture")
        self.test_Game.board_state = np.array([[0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, -1, 0, -1, 0, -1, 0],
                         [0, -1, 0, 0, 0, -1, 0, -1],
                         [0, 0, -1, 0, -1, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0, 1],
                         [1, 0, 1, 0, 1, 0, 1, 0]])

        self.test_Game.move((4, 1, 2, 3))

        expected_result = np.array([[0, -1, 0, -1, 0, -1, 0, -1],
                                    [-1, 0, -1, 0, -1, 0, -1, 0],
                                    [0, -1, 0, 1, 0, -1, 0, -1],
                                    [0, 0, 0, 0, -1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 1, 0, 1, 0],
                                    [0, 1, 0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1, 0, 1, 0]])

        np.testing.assert_array_equal(expected_result, self.test_Game.board_state)

        self.test_Game.reset()
        print("trying no capture")
        self.test_Game.board_state = np.array([[0, -1, 0, -1, 0, -1, 0, -1],
                                               [-1, 0, -1, 0, -1, 0, -1, 0],
                                               [0, -1, 0, 0, 0, -1, 0, -1],
                                               [0, 0, -1, 0, -1, 0, 0, 0],
                                               [0, 1, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 1, 0, 1, 0],
                                               [0, 1, 0, 1, 0, 1, 0, 1],
                                               [1, 0, 1, 0, 1, 0, 1, 0]])

        self.test_Game.move((4, 1, 0, 3))

        expected_result = np.array([[0, -1, 0, -1, 0, -1, 0, -1],
                                               [-1, 0, -1, 0, -1, 0, -1, 0],
                                               [0, -1, 0, 0, 0, -1, 0, -1],
                                               [0, 0, -1, 0, -1, 0, 0, 0],
                                               [0, 1, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 1, 0, 1, 0],
                                               [0, 1, 0, 1, 0, 1, 0, 1],
                                               [1, 0, 1, 0, 1, 0, 1, 0]])

        np.testing.assert_array_equal(expected_result, self.test_Game.board_state)

        self.test_Game.reset()
        print("capture from first col")
        self.test_Game.turn = -1
        self.test_Game.board_state = np.array([[0, -1, 0, -1, 0, -1, 0, -1],
                                               [-1, 0, -1, 0, -1, 0, -1, 0],
                                               [0, -1, 0, 0, 0, -1, 0, -1],
                                               [-1, 0, -1, 0, -1, 0, 0, 0],
                                               [0, 1, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 1, 0, 1, 0],
                                               [0, 1, 0, 1, 0, 1, 0, 1],
                                               [1, 0, 1, 0, 1, 0, 1, 0]])

        self.test_Game.move((3, 0, 5, 2))

        expected_result = np.array([[0, -1, 0, -1, 0, -1, 0, -1],
                                   [-1, 0, -1, 0, -1, 0, -1, 0],
                                   [0, -1, 0, 0, 0, -1, 0, -1],
                                   [0, 0, -1, 0, -1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, -1, 0, 1, 0, 1, 0],
                                   [0, 1, 0, 1, 0, 1, 0, 1],
                                   [1, 0, 1, 0, 1, 0, 1, 0]])

        np.testing.assert_array_equal(expected_result, self.test_Game.board_state)


if __name__ == '__main__':
    unittest.main()
