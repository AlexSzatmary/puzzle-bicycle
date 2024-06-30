#!/usr/bin/env python
import numpy as np
from check import zero_pad


def test_zero_pad():
    a = np.zeros((3, 3), dtype="|S1")
    a[:, :] = " "
    b = zero_pad(a)
    c = np.zeros((5, 5), dtype="|S1")
    c[:, :] = "X"
    c[1:-1, 1:-1] = " "
    assert np.all(b == c)
