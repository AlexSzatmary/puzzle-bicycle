import numpy as np
from puzzle import (
    check_all,
    check_lit_bulbs,
    check_number,
    check_unlit_cells,
    illuminate,
    load_pzprv3,
    print_board,
    save_pzprv3,
    stringify_board,
    zero_pad,
)

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
"""[1:]

pzprv3_1_sol = """
pzprv3/
lightup/
5/
5/
. # . . . /
. . 0 . # /
# 2 - 1 . /
. # 3 # . /
. . # . . /
"""[1:]

board_1_str = """
-------
-.....-
-..0..-
-.2-1.-
-..3..-
-.....-
-------
"""[1:-1]
board_1 = np.array(list(map(list, board_1_str.split("\n"))), dtype="str")

board_1_sol_str = """
-------
-.#...-
-..0.#-
-#2-1.-
-.#3#.-
-..#..-
-------
"""[1:-1]
board_1_sol = np.array(list(map(list, board_1_sol_str.split("\n"))), dtype="str")


board_1_sol_illuminated_str = """
-------
-x#__x-
-||0_#-
-#2-1|-
-x#3#x-
-xx#xx-
-------
"""[1:-1]
board_1_sol_illuminated = np.array(
    list(map(list, board_1_sol_illuminated_str.split("\n"))), dtype="str"
)


def test_load_pzprv3() -> None:
    assert np.all(board_1 == load_pzprv3(pzprv3_1))


def test_load_pzprv3_solved() -> None:
    assert np.all(board_1_sol == load_pzprv3(pzprv3_1_sol))


def test_save_pzprv3() -> None:
    # Some puzzles have trailing / and some don't. I don't add "/" at the end of my
    # lines in my save function so there isn't a perfect round trip between load and
    # save with this example.
    s1 = pzprv3_1_sol.replace(" /", "").replace("/", "")
    s2 = save_pzprv3(load_pzprv3(pzprv3_1_sol))
    assert s1 == s2


def test_stringify_board() -> None:
    print(stringify_board(board_1))
    print_board(board_1)
    assert np.all(stringify_board(board_1) == board_1_str)


def test_zero_pad() -> None:
    a = np.zeros((3, 3), dtype="str")
    a[:, :] = " "
    b = zero_pad(a)
    c = np.zeros((5, 5), dtype="str")
    c[:, :] = "-"
    c[1:-1, 1:-1] = " "
    assert np.all(b == c)


def test_check_number_1() -> None:
    assert not check_number(board_1_sol)
    board_1_sol_1 = board_1_sol.copy()
    board_1_sol_1[4, 2] = "."
    assert check_number(board_1_sol_1) == [(3, 2), (4, 3)]


def test_illuminate_1() -> None:
    (wrong_bulb_pairs, illuminated_board) = illuminate(board_1_sol.copy())
    assert not wrong_bulb_pairs
    assert np.all(illuminated_board == board_1_sol_illuminated)
    board_1_sol_wrong = board_1_sol.copy()
    board_1_sol_wrong[2, 1] = "#"
    board_1_sol_wrong[1, 5] = "#"
    (wrong_bulb_pairs_wrong, illuminated_board_wrong) = illuminate(
        board_1_sol_wrong.copy()
    )
    board_1_sol_illuminated_wrong = board_1_sol_illuminated.copy()
    board_1_sol_illuminated_wrong[2, 1] = "#"
    board_1_sol_illuminated_wrong[2, 2] = "x"
    board_1_sol_illuminated_wrong[1, 5] = "#"
    assert wrong_bulb_pairs_wrong == [(1, 2, 1, 5), (1, 5, 2, 5), (2, 1, 3, 1)]
    print(illuminated_board_wrong)
    print(board_1_sol_illuminated_wrong)
    print(illuminated_board_wrong == board_1_sol_illuminated_wrong)
    assert np.all(illuminated_board_wrong == board_1_sol_illuminated_wrong)


def test_check_unlit_cells_1() -> None:
    assert check_unlit_cells(board_1_sol)

    board_1_sol_wrong = board_1_sol.copy()
    board_1_sol_wrong[1, 2] = "."
    assert not check_unlit_cells(board_1_sol_wrong)


def test_check_unlit() -> None:
    assert check_lit_bulbs(board_1_sol)

    board_1_sol_wrong = board_1_sol.copy()
    board_1_sol_wrong[2, 1] = "#"
    assert not check_lit_bulbs(board_1_sol_wrong)


def test_check_all() -> None:
    assert check_all(board_1_sol)

    board_1_sol_wrong = board_1_sol.copy()
    board_1_sol_wrong[1, 2] = "."
    assert not check_all(board_1_sol_wrong)

    board_1_sol_wrong_2 = board_1_sol.copy()
    board_1_sol_wrong_2[2, 1] = "#"
    assert not check_all(board_1_sol_wrong)
