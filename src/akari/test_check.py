#!/usr/bin/env python
import numpy as np
from check import zero_pad, load_pzprv3


pzprv3_1 = """
pzprv3/
lightup/
5/
5/
. . . . . /
. . 0 . . /
. 2 - 1 . /
. . 3 . . /
. . . . . /
"""

board_1 = """
XXXXXXX
X.....X
X..0..X
X.2X1.X
X..3..X
X.....X
XXXXXXX
"""[1:-1]
board_1 = np.array(list(map(list, board_1.split("\n"))), dtype="|S1")


def test_load_pzprv3():
    assert np.all(board_1 == load_pzprv3(pzprv3_1))


def test_zero_pad():
    a = np.zeros((3, 3), dtype="|S1")
    a[:, :] = " "
    b = zero_pad(a)
    c = np.zeros((5, 5), dtype="|S1")
    c[:, :] = "X"
    c[1:-1, 1:-1] = " "
    assert np.all(b == c)
