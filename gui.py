from tkinter import *
from PIL import Image, ImageTk
import chess
import min_max_NN
import chess_ai
from state import Game


ai = chess_ai.AI()
state_ai = Game()
game = state_ai.board_state
p2 = None
row1 = None
column1 = None
row2 = None
column2 = None
is_clicked = False
btnQ, btnR, btnB, btnN = None, None, None, None


def convert_move(move):
    try:
        col_begin_pos = chr(96 + move[0] + 1)
        row_begin_pos = move[1] + 1
        col_end_pos = chr(96 + move[2] + 1)
        row_end_pos = move[3] + 1
        move = str(col_begin_pos) + str(row_begin_pos) + str(col_end_pos) + str(row_end_pos)
        return move
    except ValueError:
        print("wrong move")
        return None


def pawn(p, column1, column2, row1, row2):
    print('pawn')
    move = convert_move((column1, row1, column2, row2)) + p
    if chess.Move.from_uci(move) in game.legal_moves:
        game.push(chess.Move.from_uci(move))


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
        if game.piece_at(column1 + row1 * 8) == chess.Piece(1, chess.WHITE) and row2 == 7:
            move += lab_pawn.get()
        if game.piece_at(column1 + row1 * 8) == chess.Piece(1, chess.BLACK) and row2 == 0:
            move += lab_pawn

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
            if (x + y) % 2 == 1:
                if game.piece_at(x + y * 8) == chess.Piece(6, chess.WHITE):
                    board.append(tab[0])
                if game.piece_at(x + y * 8) == chess.Piece(5, chess.WHITE):
                    board.append(tab[1])
                if game.piece_at(x + y * 8) == chess.Piece(3, chess.WHITE):
                    board.append(tab[2])
                if game.piece_at(x + y * 8) == chess.Piece(2, chess.WHITE):
                    board.append(tab[3])
                if game.piece_at(x + y * 8) == chess.Piece(4, chess.WHITE):
                    board.append(tab[4])
                if game.piece_at(x + y * 8) == chess.Piece(1, chess.WHITE):
                    board.append(tab[5])
                if game.piece_at(x + y * 8) == chess.Piece(6, chess.BLACK):
                    board.append(tab[6])
                if game.piece_at(x + y * 8) == chess.Piece(5, chess.BLACK):
                    board.append(tab[7])
                if game.piece_at(x + y * 8) == chess.Piece(3, chess.BLACK):
                    board.append(tab[8])
                if game.piece_at(x + y * 8) == chess.Piece(2, chess.BLACK):
                    board.append(tab[9])
                if game.piece_at(x + y * 8) == chess.Piece(4, chess.BLACK):
                    board.append(tab[10])
                if game.piece_at(x + y * 8) == chess.Piece(1, chess.BLACK):
                    board.append(tab[11])
                if not game.piece_at(x + y * 8):
                    board.append(tab[24])
            if (x + y) % 2 == 0:
                if game.piece_at(x + y * 8) == chess.Piece(6, chess.WHITE):
                    board.append(tab[12])
                if game.piece_at(x + y * 8) == chess.Piece(5, chess.WHITE):
                    board.append(tab[13])
                if game.piece_at(x + y * 8) == chess.Piece(3, chess.WHITE):
                    board.append(tab[14])
                if game.piece_at(x + y * 8) == chess.Piece(2, chess.WHITE):
                    board.append(tab[15])
                if game.piece_at(x + y * 8) == chess.Piece(4, chess.WHITE):
                    board.append(tab[16])
                if game.piece_at(x + y * 8) == chess.Piece(1, chess.WHITE):
                    board.append(tab[17])
                if game.piece_at(x + y * 8) == chess.Piece(6, chess.BLACK):
                    board.append(tab[18])
                if game.piece_at(x + y * 8) == chess.Piece(5, chess.BLACK):
                    board.append(tab[19])
                if game.piece_at(x + y * 8) == chess.Piece(3, chess.BLACK):
                    board.append(tab[20])
                if game.piece_at(x + y * 8) == chess.Piece(2, chess.BLACK):
                    board.append(tab[21])
                if game.piece_at(x + y * 8) == chess.Piece(4, chess.BLACK):
                    board.append(tab[22])
                if game.piece_at(x + y * 8) == chess.Piece(1, chess.BLACK):
                    board.append(tab[23])
                if not game.piece_at(x + y * 8):
                    board.append(tab[25])
            positions[x + y * 8] = (Button(root, image=board[p], command=lambda p=p: change(p)))
            positions[p].grid(row=y, column=x)
            p += 1


def pieces():
    pieces = []
    photo = PhotoImage(file="images.png")
    ph_width = photo.width()
    ph_height = photo.height()
    w_puzzle = int(ph_width / 6)
    h_puzzle = int(ph_height / 2)
    im = Image.open("images.png")
    for y in range(2):
        for x in range(6):
            piece = im.crop((x * w_puzzle, y * h_puzzle, (x + 1) * w_puzzle, (y + 1) * h_puzzle))
            pieces.append(ImageTk.PhotoImage(piece))
    im_black = Image.open("images_black.png")
    for y in range(2):
        for x in range(6):
            piece = im_black.crop((x * w_puzzle, y * h_puzzle, (x + 1) * w_puzzle, (y + 1) * h_puzzle))
            pieces.append(ImageTk.PhotoImage(piece))
    return pieces


root = Tk()
tab = pieces()
tab.append(ImageTk.PhotoImage(Image.open("white_field.png")))
tab.append(ImageTk.PhotoImage(Image.open("black_field.png")))
positions = []
p = 0
board = []
for y in range(8):
    for x in range(8):
        if (x + y) % 2 == 1:
            if game.piece_at(x+y*8) == chess.Piece(6, chess.WHITE):
                board.append(tab[0])
            if game.piece_at(x+y*8) == chess.Piece(5, chess.WHITE):
                board.append(tab[1])
            if game.piece_at(x+y*8) == chess.Piece(3, chess.WHITE):
                board.append(tab[2])
            if game.piece_at(x+y*8) == chess.Piece(2, chess.WHITE):
                board.append(tab[3])
            if game.piece_at(x+y*8) == chess.Piece(4, chess.WHITE):
                board.append(tab[4])
            if game.piece_at(x+y*8) == chess.Piece(1, chess.WHITE):
                board.append(tab[5])
            if game.piece_at(x+y*8) == chess.Piece(6, chess.BLACK):
                board.append(tab[6])
            if game.piece_at(x+y*8) == chess.Piece(5, chess.BLACK):
                board.append(tab[7])
            if game.piece_at(x+y*8) == chess.Piece(3, chess.BLACK):
                board.append(tab[8])
            if game.piece_at(x+y*8) == chess.Piece(2, chess.BLACK):
                board.append(tab[9])
            if game.piece_at(x+y*8) == chess.Piece(4, chess.BLACK):
                board.append(tab[10])
            if game.piece_at(x+y*8) == chess.Piece(1, chess.BLACK):
                board.append(tab[11])
            if not game.piece_at(x+y*8):
                board.append(tab[24])
        if (x + y) % 2 == 0:
            if game.piece_at(x+y*8) == chess.Piece(6, chess.WHITE):
                board.append(tab[12])
            if game.piece_at(x+y*8) == chess.Piece(5, chess.WHITE):
                board.append(tab[13])
            if game.piece_at(x+y*8) == chess.Piece(3, chess.WHITE):
                board.append(tab[14])
            if game.piece_at(x+y*8) == chess.Piece(2, chess.WHITE):
                board.append(tab[15])
            if game.piece_at(x+y*8) == chess.Piece(4, chess.WHITE):
                board.append(tab[16])
            if game.piece_at(x+y*8) == chess.Piece(1, chess.WHITE):
                board.append(tab[17])
            if game.piece_at(x+y*8) == chess.Piece(6, chess.BLACK):
                board.append(tab[18])
            if game.piece_at(x+y*8) == chess.Piece(5, chess.BLACK):
                board.append(tab[19])
            if game.piece_at(x+y*8) == chess.Piece(3, chess.BLACK):
                board.append(tab[20])
            if game.piece_at(x+y*8) == chess.Piece(2, chess.BLACK):
                board.append(tab[21])
            if game.piece_at(x+y*8) == chess.Piece(4, chess.BLACK):
                board.append(tab[22])
            if game.piece_at(x+y*8) == chess.Piece(1, chess.BLACK):
                board.append(tab[23])
            if not game.piece_at(x+y*8):
                board.append(tab[25])
        positions.append(Button(root, image=board[p], command=lambda p=p: change(p)))
        positions[p].grid(row=y, column=x)
        p += 1

lab_pawn = Entry(root, width=10)
Label(root, text='change piece', width=9).grid(row=9, column=0)
lab_pawn.grid(row=10, column=0)
root.mainloop()

