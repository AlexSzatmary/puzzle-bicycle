#!/usr/bin/env python
import numpy as np

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


def illuminate():
    """
    Takes board with bulbs. Returns a tuple with
    *a list of lists of tuples of coordinates of bulbs that shine on each other
    *a copy of board with light paths drawn
    """
    pass


def check_empty():
    pass
