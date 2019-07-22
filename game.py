import numpy as np


class Checkers:
    def __init__(self):
        self.board_state = np.array([[0, -1, 0, -1, 0, -1, 0, -1],
                                     [-1, 0, -1, 0, -1, 0, -1, 0],
                                     [0, -1, 0, -1, 0, -1, 0, -1],
                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                     [1, 0, 1, 0, 1, 0, 1, 0],
                                     [0, 1, 0, 1, 0, 1, 0, 1],
                                     [1, 0, 1, 0, 1, 0, 1, 0]])
        self.turn = 1
        self.moves_queen_with_out_capture = 0 # 15 moves then draw
        self.list_moves = []
        self.list_captures = []
        self.must_capture = False
        self.count_movement = 0
        self.win = 0
        self.moves = []

    def convert_move(self, move):
        try:
            col_begin_pos = ord(move[0]) - ord('`')-1
            row_begin_pos = int(move[1])-1
            col_end_pos = ord(move[3]) - ord('`')-1
            row_end_pos = int(move[4])-1
            return row_begin_pos, col_begin_pos, row_end_pos, col_end_pos
        except ValueError:
            print("wrong move")
            return None

    def reset(self):
        self.board_state = np.array([[0, -1, 0, -1, 0, -1, 0, -1],
                                     [-1, 0, -1, 0, -1, 0, -1, 0],
                                     [0, -1, 0, -1, 0, -1, 0, -1],
                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                     [1, 0, 1, 0, 1, 0, 1, 0],
                                     [0, 1, 0, 1, 0, 1, 0, 1],
                                     [1, 0, 1, 0, 1, 0, 1, 0]])
        self.turn = 1
        self.moves_queen_with_out_capture = 0  # 15 moves then draw
        self.list_moves = []
        self.list_captures = []
        self.must_capture = False
        self.count_movement = 0
        self.win = 0

    def is_empty(self, field):
        return True if self.board_state[field[0], field[1]] == 0 else False

    def move(self, move):
        next = True
        self.available_moves()
        if move not in self.moves:
            self.win = -self.turn
        if move in self.list_moves:
            if self.board_state[move[0], move[1]] == self.turn*2:
                self.moves_queen_with_out_capture += 1
            else:
                self.moves_queen_with_out_capture = 0
            self.board_state[move[2], move[3]] = self.board_state[move[0], move[1]]
            self.board_state[move[0], move[1]] = 0
            self.next_turn()
        if move in self.list_captures:
            self.moves_queen_with_out_capture = 0
            self.board_state[move[2], move[3]] = self.board_state[move[0], move[1]]
            self.board_state[move[0], move[1]] = 0
            if move[0] > move[2]:
                if move[1] > move[3]:
                    self.board_state[move[0]-1, move[1]-1] = 0
                else:
                    self.board_state[move[0]-1, move[1]+1] = 0
            else:
                if move[1] > move[3]:
                    self.board_state[move[0]+1, move[1]-1] = 0
                else:
                    self.board_state[move[0]+1, move[1]+1] = 0
            self.must_capture = False
            self.available_moves()

            for capture in self.list_captures:
                if capture[0] == move[2] and capture[1] == move[3]:
                    next = False

            if next:
                self.must_capture = False
                self.next_turn()
        if next:
            self.who_win()

    def next_turn(self):
        for i in range(8):
            if self.board_state[0, i] == 1:
                self.board_state[0, i] = 2
            if self.board_state[7, i] == -1:
                self.board_state[7, i] = -2
        self.count_movement += 1
        self.turn *= -1

    def who_win(self, ai=1):
        self.available_moves()
        if self.moves_queen_with_out_capture == 15:
            self.win = -ai
        if not self.list_captures and not self.list_moves:
            self.win = -self.turn*2

    def available_moves(self):
        self.moves = []
        self.list_moves = []
        self.list_captures = []
        for row in range(8):
            for col in range(8):
                if row>0:
                    if self.board_state[row, col] == 1 and self.turn == 1:
                        if col>0 and self.is_empty((row-1, col-1)):
                            self.list_moves.append((row, col, row-1, col-1))
                        if col<7 and self.is_empty((row-1, col+1)):
                            self.list_moves.append((row, col, row-1, col+1))
                if row<7:
                    if self.board_state[row, col] == -1 and self.turn == -1:
                        if col>0 and self.is_empty((row+1, col-1)):
                            self.list_moves.append((row, col, row+1, col-1))
                        if col<7 and self.is_empty((row+1, col+1)):
                            self.list_moves.append((row, col, row+1, col+1))

                if self.board_state[row, col] == 1 and self.turn == 1:
                    if row>1:
                        if col>1 and self.is_empty((row-2, col-self.turn*2)) and \
                                (self.board_state[row-1, col-self.turn] == -self.turn or self.board_state[row-1, col-self.turn] == -self.turn*2):
                            self.must_capture = True
                            self.list_captures.append((row, col, row-2, col-self.turn*2))
                        if col<6 and self.is_empty((row-2, col+self.turn*2)) and \
                                (self.board_state[row-1, col+self.turn] == -self.turn or self.board_state[row-1, col+self.turn] == -self.turn*2):
                            self.must_capture = True
                            self.list_captures.append((row, col, row-2, col+self.turn*2))

                    if row<6:
                        if col>1 and self.is_empty((row+2, col-self.turn*2)) and \
                                (self.board_state[row+1, col-self.turn] == -self.turn or self.board_state[row+1, col-self.turn] == -self.turn*2):
                            self.must_capture = True
                            self.list_captures.append((row,col,row+2,col-self.turn*2))
                        if col<6 and self.is_empty((row+2, col+self.turn*2)) and \
                                (self.board_state[row+1, col+self.turn] == -self.turn or self.board_state[row+1, col+self.turn] == -self.turn*2):
                            self.must_capture = True
                            self.list_captures.append((row, col, row+2, col+self.turn*2))

                if self.board_state[row, col] == self.turn and self.turn == -1:
                    if row>1:
                        if col>1 and self.is_empty((row-2, col+self.turn*2)) and \
                                (self.board_state[row-1, col+self.turn] == -self.turn or self.board_state[row-1, col+self.turn] == -self.turn*2):
                            self.must_capture = True
                            self.list_captures.append((row, col, row-2, col+self.turn*2))
                        if col<6 and self.is_empty((row-2, col+self.turn*2)) and \
                                (self.board_state[row-1, col+self.turn] == -self.turn or self.board_state[row-1, col+self.turn] == -self.turn*2):
                            self.must_capture = True
                            self.list_captures.append((row, col, row-2, col+self.turn*2))

                    if row<6:
                        if col>1 and self.is_empty((row+2, col+self.turn*2)) and \
                                (self.board_state[row+1, col+self.turn] == -self.turn or self.board_state[row+1, col+self.turn] == -self.turn*2):
                            self.must_capture = True
                            self.list_captures.append((row,col,row+2,col+self.turn*2))
                        if col<6 and self.is_empty((row+2, col+self.turn*2)) and \
                                (self.board_state[row+1, col+self.turn] == -self.turn or self.board_state[row+1, col+self.turn] == -self.turn*2):
                            self.must_capture = True
                            self.list_captures.append((row, col, row+2, col+self.turn*2))

                if self.board_state[row, col] == self.turn*2:
                    if row>0:
                        if col > 0 and self.is_empty((row - 1, col - 1)):
                            self.list_moves.append((row, col, row - 1, col - 1))
                        if col < 7 and self.is_empty((row - 1, col + 1)):
                            self.list_moves.append((row, col, row - 1, col + 1))
                    if row<7:
                        if col > 0 and self.is_empty((row + 1, col - 1)):
                            self.list_moves.append((row, col, row + 1, col - 1))
                        if col < 7 and self.is_empty((row + 1, col + 1)):
                            self.list_moves.append((row, col, row + 1, col + 1))

                    if row>1:
                        if col>1 and self.is_empty((row-2, col-2)) and \
                                (self.board_state[row-1, col-1] == -self.turn or self.board_state[row-1, col-1] == -self.turn*2):
                            self.must_capture = True
                            self.list_captures.append((row,col,row-2,col-2))
                        if col<6 and self.is_empty((row-2, col+2)) and \
                                (self.board_state[row-1, col+1] == -self.turn or self.board_state[row-1, col+1] == -self.turn*2):
                            self.must_capture = True
                            self.list_captures.append((row,col,row-2,col+2))

                    if row<6:
                        if col>1 and self.is_empty((row+2, col-2)) and \
                                (self.board_state[row+1, col-1] == -self.turn or self.board_state[row+1, col-1] == -self.turn*2):
                            self.must_capture = True
                            self.list_captures.append((row,col,row+2,col-2))
                        if col<6 and self.is_empty((row+2, col+2)) and \
                                (self.board_state[row+1, col+1] == -self.turn or self.board_state[row+1, col+1] == -self.turn*2):
                            self.must_capture = True
                            self.list_captures.append((row,col,row+2,col+2))

        if self.must_capture:
            self.list_moves = []
        self.moves = self.list_captures + self.list_moves

    def print_board(self):
        print_state = ''
        for row in range(8):
            for col in range(8):
                if self.board_state[row, col] != -1:
                    print_state += '  ' + str(self.board_state[row, col])
                else:
                    print_state += ' ' + str(self.board_state[row, col])
            print_state += '\n'
        print(print_state)
