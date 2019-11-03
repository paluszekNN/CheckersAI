from tkinter import *
from PIL import Image, ImageTk
from game import Checkers
import min_max
import ai


game = Checkers()
p2 = None
row1 = None
column1 = None
row2 = None
column2 = None
is_clicked = False


def change(p):
    check = False
    global is_clicked, row1, column1, row2, column2, p2, game
    if is_clicked == False:
        p2 = p
        row1 = positions[p].grid_info()["row"]
        column1 = positions[p].grid_info()["column"]
        is_clicked = True
        positions[p].config(state=ACTIVE)
        positions[p].config(bg='blue')
    else:
        row2 = positions[p].grid_info()["row"]
        column2 = positions[p].grid_info()["column"]
        positions[p2].grid(row=row2, column=column2)
        positions[p].grid(row=row1, column=column1)
        positions[p].config(state=NORMAL)
        is_clicked = False
        move = convert_move((column1, row1, column2, row2))

        print(move)
        if chess.Move.from_uci(move) in game.legal_moves:
            game.push(chess.Move.from_uci(move))
        read_board()
        if not game.turn and not game.is_checkmate():
            # board_state = game.copy()
            move = chess_ai.best_move(state_ai, ai, game.turn)
            # move = chess_ai.min_max_NN.min_max_player(game, ai, state_ai)
            # game = board_state
            game.push(move)

        read_board()


def read_board():
    board = []
    p = 0
    for y in range(8):
        for x in range(8):
            if (x + y) % 2 == 0:
                if game.board_state[x, y] == -2:
                    board.append(tab[1])
                if game.board_state[x, y] == -1:
                    board.append(tab[2])
                if game.board_state[x, y] == 2:
                    board.append(tab[3])
                if game.board_state[x, y] == 1:
                    board.append(tab[4])
                if not game.board_state[x, y]:
                    board.append(tab[0])
            if (x + y) % 2 == 1:
                if game.board_state[x, y] == -2:
                    board.append(tab[6])
                if game.board_state[x, y] == -1:
                    board.append(tab[7])
                if game.board_state[x, y] == 2:
                    board.append(tab[8])
                if game.board_state[x, y] == 1:
                    board.append(tab[9])
                if not game.board_state[x, y]:
                    board.append(tab[5])
            positions[x + y * 8] = (Button(root, image=board[p], command=lambda p=p: change(p)))
            positions[p].grid(row=x, column=y)
            p += 1


root = Tk()
tab = []
tab.append(ImageTk.PhotoImage(Image.open("white_field.png")))
tab.append(ImageTk.PhotoImage(Image.open("white_field_black_king.png")))
tab.append(ImageTk.PhotoImage(Image.open("white_field_black_pawn.png")))
tab.append(ImageTk.PhotoImage(Image.open("white_field_white_king.png")))
tab.append(ImageTk.PhotoImage(Image.open("white_field_white_pawn.png")))
tab.append(ImageTk.PhotoImage(Image.open("black_field.png")))
tab.append(ImageTk.PhotoImage(Image.open("black_field_black_king.png")))
tab.append(ImageTk.PhotoImage(Image.open("black_field_black_pawn.png")))
tab.append(ImageTk.PhotoImage(Image.open("black_field_white_king.png")))
tab.append(ImageTk.PhotoImage(Image.open("black_field_white_pawn.png")))
positions = []
p = 0
board = []
for y in range(8):
    for x in range(8):
        if (x + y) % 2 == 0:
            if game.board_state[x, y] == -2:
                board.append(tab[1])
            if game.board_state[x, y] == -1:
                board.append(tab[2])
            if game.board_state[x, y] == 2:
                board.append(tab[3])
            if game.board_state[x, y] == 1:
                board.append(tab[4])
            if not game.board_state[x, y]:
                board.append(tab[0])
        if (x + y) % 2 == 1:
            if game.board_state[x, y] == -2:
                board.append(tab[6])
            if game.board_state[x, y] == -1:
                board.append(tab[7])
            if game.board_state[x, y] == 2:
                board.append(tab[8])
            if game.board_state[x, y] == 1:
                board.append(tab[9])
            if not game.board_state[x, y]:
                board.append(tab[5])
        positions.append(Button(root, image=board[p], command=lambda p=p: change(p)))
        positions[p].grid(row=x, column=y)
        p += 1

root.mainloop()

