#!/usr/bin/env python
import numpy as np

# format
# Board:
# 0, 1, 2, 3: number of bulbs
# X: black, unnumbered
# " " [space]: empty
# Solution:
# o: bulb
# Marks:
# -, |, +: indicate light paths, + shows both directions


def zero_pad(grid):
    grid2 = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2), dtype="|S1")
    grid2[:, :] = "X"
    grid2[1:-1, 1:-1] = grid
    return grid2
