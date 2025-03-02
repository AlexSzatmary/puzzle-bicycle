import numpy as np
import pytest
from puzzle import (
    check_all,
    check_lit_bulbs,
    check_number,
    check_unlit_cells,
    count_free_near_number,
    count_missing_bulbs_near_number,
    fill_holes,
    illuminate,
    load_pzprv3,
    mark_dots_around_full_numbers,
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

board_test_fill_holes_pzprv3 = """
pzprv3
lightup
2
10
. - . . - . . - + . /
- - - 0 - - - - - - /
"""[1:-1]
board_test_fill_holes = load_pzprv3(board_test_fill_holes_pzprv3)

board_test_fill_holes_sol_pzprv3 = """
pzprv3
lightup
2
10
# - . . - . . - + # /
- - - 0 - - - - - - /
"""[1:-1]
board_test_fill_holes_sol = load_pzprv3(board_test_fill_holes_sol_pzprv3)


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


def test_fill_holes() -> None:
    print(stringify_board(fill_holes(board_test_fill_holes)))
    print_board(board_test_fill_holes_sol)
    assert np.all(fill_holes(board_test_fill_holes) == board_test_fill_holes_sol)


board_count_near_pzprv3 = """
pzprv3
lightup
6
19
. 0 . - . 1 . - . 2 . - . 3 . - . 2 # 
. + . - . + . - . + . - . # . - . # . 
- - - - - - - - - - - - - - - - - - - 
. . . - . . . - . + . - . . . - . # . 
. 0 + - + 1 . - . 2 . - # 3 . - # 4 . 
. . . - . . . - . . . - . + . - . . . 
"""[1:-1]  # noqa: W291


@pytest.fixture
def board_count_near() -> np.ndarray:
    return load_pzprv3(board_count_near_pzprv3)


def test_count_free_near_number(board_count_near: np.ndarray) -> None:
    assert count_free_near_number(board_count_near, 1, 2) == 2
    assert count_free_near_number(board_count_near, 1, 6) == 2
    assert count_free_near_number(board_count_near, 1, 10) == 2
    assert count_free_near_number(board_count_near, 1, 14) == 2
    assert count_free_near_number(board_count_near, 1, 18) == 1
    assert count_free_near_number(board_count_near, 5, 2) == 3
    assert count_free_near_number(board_count_near, 5, 6) == 3
    assert count_free_near_number(board_count_near, 5, 10) == 3
    assert count_free_near_number(board_count_near, 5, 14) == 2
    assert count_free_near_number(board_count_near, 5, 18) == 2


def test_count_missing_bulbs_near_number(board_count_near: np.ndarray) -> None:
    assert count_missing_bulbs_near_number(board_count_near, 1, 2) == 0
    assert count_missing_bulbs_near_number(board_count_near, 1, 6) == 1
    assert count_missing_bulbs_near_number(board_count_near, 1, 10) == 2
    assert count_missing_bulbs_near_number(board_count_near, 1, 14) == 2
    assert count_missing_bulbs_near_number(board_count_near, 1, 18) == 0
    assert count_missing_bulbs_near_number(board_count_near, 5, 2) == 0
    assert count_missing_bulbs_near_number(board_count_near, 5, 6) == 1
    assert count_missing_bulbs_near_number(board_count_near, 5, 10) == 2
    assert count_missing_bulbs_near_number(board_count_near, 5, 14) == 2
    assert count_missing_bulbs_near_number(board_count_near, 5, 18) == 2


board_mark_dots_around_full_numbers_pzprv3 = """
pzprv3
lightup
6
19
. 0 . - + 1 . - . 2 . - . 3 . - . 2 # 
. + . - . # . - . + . - . # . - . # . 
- - - - - - - - - - - - - - - - - - - 
. . . - . . . - . + . - . # . - . # . 
. 0 + - + 1 . - # 2 # - # 3 . - # 4 . 
. . . - . . . - . . . - . # . - . . . 
"""[1:-1]  # noqa: W291


board_mark_dots_around_full_numbers_sol_pzprv3 = """
pzprv3
lightup
6
19
+ 0 + - + 1 + - . 2 . - . 3 . - + 2 # 
. + . - . # . - . + . - . # . - . # . 
- - - - - - - - - - - - - - - - - - - 
. + . - . . . - . + . - . # . - . # . 
+ 0 + - + 1 . - # 2 # - # 3 + - # 4 . 
. + . - . . . - . + . - . # . - . . . 
"""[1:-1]  # noqa: W291


@pytest.fixture
def board_mark_dots_around_full_numbers() -> np.ndarray:
    return load_pzprv3(board_mark_dots_around_full_numbers_pzprv3)


@pytest.fixture
def board_mark_dots_around_full_numbers_sol() -> np.ndarray:
    return load_pzprv3(board_mark_dots_around_full_numbers_sol_pzprv3)


def test_mark_dots_around_full_numbers(
    board_mark_dots_around_full_numbers: np.ndarray,
    board_mark_dots_around_full_numbers_sol: np.ndarray,
) -> None:
    board_post = mark_dots_around_full_numbers(board_mark_dots_around_full_numbers)
    assert stringify_board(board_post) == stringify_board(
        board_mark_dots_around_full_numbers_sol
    )
