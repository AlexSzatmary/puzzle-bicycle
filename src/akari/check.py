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
    pzprv3 = pzprv3.replace("\n", "/").replace("//", "/").replace("-", "X").split("/")
    rows = int(pzprv3[3])
    cols = int(pzprv3[4])
    board = np.zeros((rows + 2, cols + 2), dtype="|S1")
    board[:, :] = "X"
    for i, row in enumerate(pzprv3[5:5 + rows]):
        board[i + 1, 1:-1] = list(row.replace(" ", ""))
    return board


def zero_pad(grid):
    grid2 = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2), dtype="|S1")
    grid2[:, :] = "X"
    grid2[1:-1, 1:-1] = grid
    return grid2
