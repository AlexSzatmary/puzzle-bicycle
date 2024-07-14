#!/usr/bin/env python
import numpy as np
from itertools import zip_longest

# format
# Board:
# 0, 1, 2, 3: number of bulbs
# X: black, unnumbered
# ".": empty
# Solution:
# o: bulb
# Marks:
# -, |, +: indicate light paths, + shows both directions


def load_pzprv3(pzprv3):
    """
    Loads PUZ-PRE v3 text and returns an Akari board
    """
    pzprv3 = (
        pzprv3.replace("\n", "/")
        .replace("//", "/")
        .replace("-", "X")
        .replace("#", "o")
        .split("/")
    )
    rows = int(pzprv3[3])
    cols = int(pzprv3[4])
    board = np.zeros((rows + 2, cols + 2), dtype="|S1")
    board[:, :] = "X"
    for i, row in enumerate(pzprv3[5 : 5 + rows]):
        board[i + 1, 1:-1] = list(row.replace(" ", ""))
    return board


def zero_pad(grid):
    grid2 = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2), dtype="|S1")
    grid2[:, :] = "X"
    grid2[1:-1, 1:-1] = grid
    return grid2


def print_board(board):
    for row in board.astype(str):
        print(''.join(list(row)))


def check_number(board):
    """
    Checks numbered spaces to see if they have the correct number of bulbs.

    Returns a list of tuples of coordinates of numbered spaces that touch the wrong
    number of bulbs.
    """
    wrong_bulbs = []
    for i in range(1, np.size(board, 0) - 1):
        for j in range(1, np.size(board, 1) - 1):
            if board[i, j] in b"0123":
                # breakpoint()
                if not int(board[i, j]) == (
                    (board[i - 1, j] == b"o")
                    + (board[i + 1, j] == b"o")
                    + (board[i, j - 1] == b"o")
                    + (board[i, j + 1] == b"o")
                ):
                    wrong_bulbs.append((i, j))
    return wrong_bulbs


def illuminate(board):
    """
    Takes board with bulbs. Returns a tuple with
    *a list of lists of tuples of coordinates of bulbs that shine on each other
    *a copy of board with light paths drawn
    """
    wrong_bulb_pairs = []
    board = board.copy()
    fill_chars = [b"|", b"-", b"|", b"-"]
    for i in range(1, np.size(board, 0) - 1):
        for j in range(1, np.size(board, 1) - 1):
            if board[i, j] == b"o":
                iters = [
                    zip_longest(range(i - 1, 0, -1), [], fillvalue=j),
                    zip_longest([], range(j - 1, 0, -1), fillvalue=i),
                    zip_longest(range(i + 1, np.size(board, 0) - 1), [], fillvalue=j),
                    zip_longest([], range(j + 1, np.size(board, 0) - 1), fillvalue=i),
                ]
                for (it, fill_char) in zip(iters, fill_chars):
                    for (i1, j1) in it:
                        if board[i1, j1] == b"o":
                            if i <= i1 and j <= j1:
                                wrong_bulb_pairs.append((i, j, i1, j1))
                            break
                        elif board[i1, j1] == fill_char:
                            # wrong bulb pair already detected
                            break
                        elif board[i1, j1] == b"-" or board[i1, j1] == b"|":
                            # this branch will only trigger if the char at this location
                            # is not the same as the fill_char
                            board[i1, j1] = b"+"
                        elif board[i1, j1] in b"01234X":
                            break
                        else:
                            board[i1, j1] = fill_char
    return (wrong_bulb_pairs, board)


def check_empty():
    pass
