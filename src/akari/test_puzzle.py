from collections.abc import Generator
from inspect import cleandoc

import numpy as np
import pytest
from puzzle import (
    ThoughtProcess,
    boardify_string,
    check_all,
    check_lit_bulbs,
    check_number,
    check_unlit_cells,
    count_free_near_number,
    count_missing_bulbs_near_number,
    illuminate_all,
    load_pzprv3,
    print_board,
    save_pzprv3,
    stringify_board,
    transpose_board,
    zero_pad,
)


def all_orientations(board: np.ndarray) -> Generator[np.ndarray]:
    board = board.copy()
    yield board.copy()
    board = np.fliplr(board)
    yield board.copy()
    board = np.flipud(board)
    yield board.copy()
    board = np.fliplr(board)
    yield board.copy()
    board = transpose_board(board)
    yield board.copy()
    board = np.fliplr(board)
    yield board.copy()
    board = np.flipud(board)
    yield board.copy()
    board = np.fliplr(board)
    yield board


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
board_1 = boardify_string(board_1_str)

board_1_sol_str = """
-------
-.#...-
-..0.#-
-#2-1.-
-.#3#.-
-..#..-
-------
"""[1:-1]
board_1_sol = boardify_string(board_1_sol_str)


board_1_sol_illuminated_str = """
-------
-x#__x-
-||0_#-
-#2-1|-
-x#3#x-
-xx#xx-
-------
"""[1:-1]
board_1_sol_illuminated = boardify_string(board_1_sol_illuminated_str)

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
    for board in all_orientations(board_1_sol):
        assert not check_number(board)
    board_1_sol_1 = board_1_sol.copy()
    board_1_sol_1[4, 2] = "."
    assert check_number(board_1_sol_1) == [(3, 2), (4, 3)]


def test_illuminate_all_1() -> None:
    # Test for a good case
    for board, ref in zip(
        all_orientations(board_1_sol),
        all_orientations(board_1_sol_illuminated),
        strict=True,
    ):
        (wrong_bulb_pairs, illuminated_board) = illuminate_all(board)
        assert not wrong_bulb_pairs
        assert stringify_board(illuminated_board) == stringify_board(ref)

    # Test for detection of wrong pairs one time
    board_1_sol_wrong = board_1_sol.copy()
    board_1_sol_wrong[2, 1] = "#"
    board_1_sol_wrong[1, 5] = "#"
    (wrong_bulb_pairs_wrong, _) = illuminate_all(board_1_sol_wrong.copy())
    assert wrong_bulb_pairs_wrong == [(1, 2, 1, 5), (1, 5, 2, 5), (2, 1, 3, 1)]

    # Test that illumination happens for a known wrong case
    # and that 3 wrong pairs are detected
    board_1_sol_illuminated_wrong = board_1_sol_illuminated.copy()
    board_1_sol_illuminated_wrong[2, 1] = "#"
    board_1_sol_illuminated_wrong[2, 2] = "x"
    board_1_sol_illuminated_wrong[1, 5] = "#"
    for board, ref in zip(
        all_orientations(board_1_sol_wrong),
        all_orientations(board_1_sol_illuminated_wrong),
        strict=True,
    ):
        (wrong_bulb_pairs_wrong, illuminated_board_wrong) = illuminate_all(board)
        assert len(wrong_bulb_pairs_wrong) == 3
        assert stringify_board(illuminated_board_wrong) == stringify_board(ref)


def test_illuminate_one_1() -> None:
    board = boardify_string(
        cleandoc("""
            -------
            -.#...-
            -..0.#-
            -#2-1.-
            -.#3#.-
            -..#..-
            -------
            """)
    )
    tp_no_change = ThoughtProcess(board)
    tp_no_change.illuminate_one(1, 1)
    tp_no_change.illuminate_one(1, 3)

    assert not tp_no_change.lit_bulb_pairs
    assert stringify_board(tp_no_change.board) == stringify_board(board)

    tp_change = ThoughtProcess(board)
    tp_change.illuminate_one(1, 2)
    assert not tp_change.lit_bulb_pairs
    assert stringify_board(tp_change.board) == cleandoc("""
            -------
            -_#___-
            -.|0.#-
            -#2-1.-
            -.#3#.-
            -..#..-
            -------
            """)
    tp_change.illuminate_one(3, 1)
    assert not tp_change.lit_bulb_pairs
    assert stringify_board(tp_change.board) == cleandoc("""
            -------
            -x#___-
            -||0.#-
            -#2-1.-
            -|#3#.-
            -|.#..-
            -------
            """)

    # Test for detection of wrong pairs one time
    tp_wrong = ThoughtProcess(board)
    tp_wrong.illuminate_one(3, 1)
    tp_wrong.illuminate_one(1, 2)
    tp_wrong.board[2, 1] = "#"
    tp_wrong.board[1, 5] = "#"
    assert not tp_wrong.lit_bulb_pairs
    tp_wrong.illuminate_one(2, 1)
    print_board(tp_wrong.board)
    assert tp_wrong.lit_bulb_pairs == [(2, 1, 3, 1)]
    tp_wrong.illuminate_one(1, 5)
    assert stringify_board(tp_wrong.board) == cleandoc("""
            -------
            -x#__#-
            -#x0.#-
            -#2-1|-
            -|#3#|-
            -|.#.|-
            -------
            """)
    assert tp_wrong.lit_bulb_pairs == [(2, 1, 3, 1), (1, 5, 1, 2), (1, 5, 2, 5)]


def test_check_unlit_cells_1() -> None:
    for board in all_orientations(board_1_sol):
        assert check_unlit_cells(board)

    board_1_sol_wrong = board_1_sol.copy()
    board_1_sol_wrong[1, 2] = "."
    for board in all_orientations(board_1_sol_wrong):
        assert not check_unlit_cells(board)


def test_check_unlit() -> None:
    for board in all_orientations(board_1_sol):
        assert check_lit_bulbs(board)

    board_1_sol_wrong = board_1_sol.copy()
    board_1_sol_wrong[2, 1] = "#"
    for board in all_orientations(board_1_sol_wrong):
        assert not check_lit_bulbs(board)


def test_check_all() -> None:
    for board in all_orientations(board_1_sol):
        assert check_all(board)

    board_1_sol_wrong = board_1_sol.copy()
    board_1_sol_wrong[1, 2] = "."
    for board in all_orientations(board_1_sol_wrong):
        assert not check_all(board)

    board_1_sol_wrong_2 = board_1_sol.copy()
    board_1_sol_wrong_2[2, 1] = "#"
    for board in all_orientations(board_1_sol_wrong):
        assert not check_all(board)


def test_fill_holes() -> None:
    for board, ref in zip(
        all_orientations(board_test_fill_holes),
        all_orientations(board_test_fill_holes_sol),
        strict=True,
    ):
        tp = ThoughtProcess(board)
        tp.fill_holes(-1, -1, ".")
        assert stringify_board(tp.board == ref)


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
    for board, ref in zip(
        all_orientations(board_mark_dots_around_full_numbers),
        all_orientations(board_mark_dots_around_full_numbers_sol),
        strict=True,
    ):
        tp = ThoughtProcess(board)
        tp.mark_dots_around_full_numbers(-1, -1, ".")
        assert stringify_board(tp.board) == stringify_board(ref)


@pytest.fixture
def board_mark_bulbs_around_dotted_numbers() -> np.ndarray:
    return boardify_string(
        cleandoc(
            """
            ---------------------
            -.0.-+1.-.2.-.3.-+2#-
            -.+.-.#.-.+.-.#.-.#.-
            ---------------------
            -...-...-.+.-.#.-.#.-
            -.0+-+1.-#2#-#3.-#4.-
            -...-...-...-.#.-...-
            ---------------------
            """
        )
    )


@pytest.fixture
def board_mark_bulbs_around_dotted_numbers_sol() -> np.ndarray:
    return boardify_string(
        cleandoc(
            """
            ---------------------
            -.0.-+1.-#2#-#3#-+2#-
            -.+.-.#.-|+|-|#|-.#.-
            ---------------------
            -...-...-.+.-.#.-.#|-
            -.0+-+1.-#2#-#3.-#4#-
            -...-...-...-.#.-_#x-
            ---------------------
            """
        )
    )


def test_mark_bulbs_around_dotted_numbers(
    board_mark_bulbs_around_dotted_numbers: np.ndarray,
    board_mark_bulbs_around_dotted_numbers_sol: np.ndarray,
) -> None:
    print_board(board_mark_bulbs_around_dotted_numbers)
    print_board(board_mark_bulbs_around_dotted_numbers_sol)
    for board, ref in zip(
        all_orientations(board_mark_bulbs_around_dotted_numbers),
        all_orientations(board_mark_bulbs_around_dotted_numbers_sol),
        strict=True,
    ):
        tp = ThoughtProcess(board)
        tp.mark_bulbs_around_dotted_numbers(-1, -1, ".")
        assert stringify_board(tp.board) == stringify_board(ref)


@pytest.fixture
def board_mark_dots_at_corners() -> np.ndarray:
    return boardify_string(
        cleandoc(
            """
            ---------------------
            -2..-+1.-.2.-#3.-.2.-
            -...-...-.+.-...-...-
            ---------------------
            -...-...-.+.-+#.--+.-
            -.0+-+1.-#2#-#3.--1.-
            -...-...-...-+#.-...-
            ---------------------
            """
        )
    )


@pytest.fixture
def board_mark_dots_at_corners_sol() -> np.ndarray:
    return boardify_string(
        cleandoc(
            """
            ---------------------
            -2..-+1.-.2.-#3.-.2.-
            -...-..+-.+.-...-+.+-
            ---------------------
            -...-...-.+.-+#.--+.-
            -.0+-+1.-#2#-#3.--1.-
            -...-...-...-+#.-..+-
            ---------------------
            """
        )
    )


def test_mark_dots_at_corners(
    board_mark_dots_at_corners: np.ndarray,
    board_mark_dots_at_corners_sol: np.ndarray,
) -> None:
    for board, ref in zip(
        all_orientations(board_mark_dots_at_corners),
        all_orientations(board_mark_dots_at_corners_sol),
        strict=True,
    ):
        tp = ThoughtProcess(board)
        tp.mark_dots_at_corners(-1, -1, ".")
        assert stringify_board(tp.board) == stringify_board(ref)


@pytest.fixture
def board_mark_unique_bulbs_for_dot_cells() -> np.ndarray:
    return boardify_string(
        cleandoc(
            """
            -----
            -0---
            -+.--
            --.--
            -----
            -----
            -0---
            -+..-
            --.--
            -----
            """
        )
    )


@pytest.fixture
def board_mark_unique_bulbs_for_dot_cells_sol() -> np.ndarray:
    return boardify_string(
        cleandoc(
            """
            -----
            -0---
            -_#--
            --|--
            -----
            -----
            -0---
            -+..-
            --.--
            -----
            """
        )
    )


def test_mark_unique_bulbs_for_dot_cells(
    board_mark_unique_bulbs_for_dot_cells: np.ndarray,
    board_mark_unique_bulbs_for_dot_cells_sol: np.ndarray,
) -> None:
    for board, ref in zip(
        all_orientations(board_mark_unique_bulbs_for_dot_cells),
        all_orientations(board_mark_unique_bulbs_for_dot_cells_sol),
        strict=True,
    ):
        tp = ThoughtProcess(board)
        tp.mark_unique_bulbs_for_dot_cells(-1, -1, ".")
        assert stringify_board(tp.board) == stringify_board(ref)


def test_mark_unique_bulbs_for_dot_cells2() -> None:
    board2 = boardify_string(
        cleandoc(
            """
            -------
            -.....-
            -.-.-.-
            -..0..-
            -.-.-.-
            -.....-
            -------
            """
        )
    )

    board2_sol_str = cleandoc(
        """
            -------
            -x_#_x-
            -|-|-|-
            -#_0_#-
            -|-|-|-
            -x_#_x-
            -------
            """
    )

    tp2 = ThoughtProcess(board2)
    tp2.apply_methods(5)
    assert stringify_board(tp2.board) == board2_sol_str


@pytest.fixture
def board_analyze_diagonally_adjacent_numbers() -> np.ndarray:
    return boardify_string(
        cleandoc(
            """
            ------
            -1...-
            -.1..-
            -....-
            ------
            -....-
            -.3..-
            -..1.-
            -....-
            ------
            -....-
            -+2..-
            -..1.-
            -....-
            ------
            """
        )
    )


@pytest.fixture
def board_analyze_diagonally_adjacent_numbers_sol() -> np.ndarray:
    return boardify_string(
        cleandoc(
            """
            ------
            -1...-
            -.1+.-
            -.+..-
            ------
            -x#__-
            -#3..-
            -|.1+-
            -|.+.-
            ------
            -_#__-
            -+2..-
            -..1+-
            -..+.-
            ------
            """
        )
    )


def test_analyze_diagonally_adjacent_numbers(
    board_analyze_diagonally_adjacent_numbers: np.ndarray,
    board_analyze_diagonally_adjacent_numbers_sol: np.ndarray,
) -> None:
    for board, ref in zip(
        all_orientations(board_analyze_diagonally_adjacent_numbers),
        all_orientations(board_analyze_diagonally_adjacent_numbers_sol),
        strict=True,
    ):
        tp = ThoughtProcess(board)
        tp.analyze_diagonally_adjacent_numbers(-1, -1, ".")
        assert stringify_board(tp.board) == stringify_board(ref)


def test_find_wrong_numbers() -> None:
    for board in all_orientations(board_1_sol):
        tp = ThoughtProcess(board)
        for i, j in tp.all_interior_ij():
            tp.find_wrong_numbers(i, j)
        # tp.apply_methods(6)
        assert not tp.wrong_numbers
    board_1_sol_1 = board_1_sol.copy()
    board_1_sol_1[2, 2] = "#"
    board_1_sol_1[5, 3] = "+"
    board_1_sol_1[2, 4] = "#"
    print_board(board_1_sol_1)
    tp = ThoughtProcess(board_1_sol_1)
    for i, j in tp.all_interior_ij():
        tp.find_wrong_numbers(i, j)
    # tp.apply_methods(6)
    assert tp.wrong_numbers == {(2, 3), (3, 2), (3, 4), (4, 3)}


@pytest.fixture
def board_apply_methods() -> np.ndarray:
    return boardify_string(
        cleandoc(
            """
            ------
            -2.1.-
            -....-
            -1--.-
            -.1---
            -...--
            -.-..-
            -.1..-
            -...2-
            -..1.-
            -....-
            -....-
            ------
            """
        )
    )


@pytest.fixture
def boards_apply_methods_sol() -> list[np.ndarray]:
    board_strings = [
        """
            ------
            -2.1.-
            -....-
            -1--.-
            -.1---
            -...--
            -.-..-
            -.1..-
            -...2-
            -..1.-
            -....-
            -....-
            ------
            """,
        """
            ------
            -2#1+-
            -#x__-
            -1--.-
            -+1---
            -_#_--
            -.-..-
            -.1..-
            -...2-
            -..1.-
            -....-
            -....-
            ------
            """,
        """
            ------
            -2#1|-
            -#x_x-
            -1--#-
            -+1---
            -_#_--
            -.-..-
            -.1..-
            -...2-
            -..1.-
            -....-
            -....-
            ------
            """,
        """
            ------
            -2#1|-
            -#x_x-
            -1--#-
            -+1---
            -_#_--
            -.-..-
            -.1+.-
            -+..2-
            -..1.-
            -....-
            -....-
            ------
            """,
        """
            ------
            -2#1|-
            -#x_x-
            -1--#-
            -+1---
            -_#_--
            -.-.|-
            -.1_#-
            -+..2-
            -.+1.-
            -..+.-
            -....-
            ------
            """,
    ]
    return [boardify_string(cleandoc(s)) for s in board_strings]


def test_apply_methods(
    board_apply_methods: np.ndarray,
    boards_apply_methods_sol: np.ndarray,
) -> None:
    for level, board_ref in zip(
        range(1, len(boards_apply_methods_sol) + 1),
        boards_apply_methods_sol,
        strict=True,
    ):
        for board, ref in zip(
            all_orientations(board_apply_methods),
            all_orientations(board_ref),
            strict=True,
        ):
            tp = ThoughtProcess(board)
            tp.apply_methods(level)
            assert stringify_board(tp.board) == stringify_board(ref)


@pytest.fixture
def board_pairs_trace_shared_lanes_adjacent() -> list[tuple[np.ndarray, np.ndarray]]:
    board_pairs = [
        (
            """
            ----
            -..-
            -2.-
            -..-
            -..-
            -.2-
            -..-
            ----
            """,
            """
            ----
            -#_-
            -2.-
            -.+-
            -+.-
            -.2-
            -_#-
            ----
            """,
        ),
        (
            """
            ----
            -..-
            -2.-
            -.2-
            -..-
            ----
            """,
            """
            ----
            -..-
            -2.-
            -.2-
            -..-
            ----
            """,
        ),
        (
            """
            ------
            -.-..-
            -.1..-
            -.+..-
            -....-
            -..1+-
            -..-.-
            ------
            """,
            """
            ------
            -.-..-
            -.1..-
            -.+..-
            -....-
            -..1+-
            -..-.-
            ------
            """,
        ),
        (
            """
            ------
            ------
            -....-
            -.3..-
            -....-
            -....-
            -..3.-
            -....-
            ------
            ------
            """,
            """
            ------
            ------
            -x#_x-
            -#3.|-
            -|.+|-
            -|+.|-
            -|.3#-
            -x_#x-
            ------
            ------
            """,
        ),
    ]
    return [
        (boardify_string(cleandoc(pre)), boardify_string(cleandoc(post)))
        for (pre, post) in board_pairs
    ]


def test_trace_shared_lanes() -> None:
    null = boardify_string(
        cleandoc("""
        ----
        -..-
        -1.-
        -..-
        -.1-
        -..-
        ----
        """)
    )
    tp = ThoughtProcess(null)
    tp.shared_lanes_bot.mark_bulbs_and_dots_at_shared_lanes(-1, -1, ".")

    assert stringify_board(tp.board) == stringify_board(null)
    pre = boardify_string(
        cleandoc("""
            ----
            -..-
            -..-
            -2.-
            -..-
            -..-
            -.2-
            -..-
            -..-
            ----
            """)
    )
    post = """
            ----
            -|+-
            -#_-
            -2.-
            -.+-
            -+.-
            -.2-
            -_#-
            -+|-
            ----
            """
    tp = ThoughtProcess(pre)
    tp.shared_lanes_bot.mark_bulbs_and_dots_at_shared_lanes(-1, -1, ".")
    print_board(tp.board)
    assert stringify_board(tp.board) == cleandoc(post)


@pytest.fixture
def board_pairs_trace_shared_lanes_same_2() -> list[tuple[np.ndarray, np.ndarray]]:
    board_pairs = [
        (
            """
            -----
            -...-
            -.3.-
            -...-
            -.-.-
            -...-
            -.2.-
            -.-.-
            -----
            """,
            """
            -----
            -_#_-
            -.3.-
            -_#_-
            -+-+-
            -_#_-
            -.2.-
            -+-+-
            -----
            """,
        ),
        (
            """
            -----
            -...-
            -.3.-
            -...-
            --..-
            -...-
            -.3.-
            -...-
            -----
            """,
            """
            -----
            -x#_-
            -#3.-
            -|.+-
            --++-
            -|.+-
            -#3.-
            -x#_-
            -----
            """,
        ),
        (
            """
            -----
            -...-
            -.3.-
            -...-
            -.2.-
            -...-
            -----
            """,
            """
            -----
            -...-
            -.3.-
            -_#_-
            -.2.-
            -...-
            -----
            """,
        ),
        (  # this one looks like a "same_3" case but it's a special case of same_2
            """
            -----
            -...-
            -.3.-
            -...-
            -.3.-
            -...-
            -----
            """,
            """
            -----
            -_#_-
            -.3.-
            -_#_-
            -.3.-
            -_#_-
            -----
            """,
        ),
    ]
    return [
        (boardify_string(cleandoc(pre)), boardify_string(cleandoc(post)))
        for (pre, post) in board_pairs
    ]


def test_trace_shared_lanes_same_2(
    board_pairs_trace_shared_lanes_same_2: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    pre, post = board_pairs_trace_shared_lanes_same_2[0]
    tp = ThoughtProcess(pre)
    tp.shared_lanes_bot.mark_bulbs_and_dots_at_shared_lanes(-1, -1, ".")
    assert stringify_board(tp.board) == stringify_board(post)


@pytest.fixture
def board_pairs_trace_shared_lanes_same_3() -> list[tuple[np.ndarray, np.ndarray]]:
    board_pairs = [
        (
            """
            -----
            -...-
            -.3.-
            -...-
            -...-
            -...-
            -.2.-
            -...-
            -----
            """,
            """
            -----
            -_#_-
            -.3.-
            -+.+-
            -+++-
            -+.+-
            -.2.-
            -_#_-
            -----
            """,
        ),
        (
            """
            -----
            -...-
            -.3.-
            -...-
            -...-
            -...-
            -.1.-
            -.+.-
            -----
            """,
            """
            -----
            -_#_-
            -.3.-
            -+.+-
            -+++-
            -+.+-
            -.1.-
            -+++-
            -----
            """,
        ),
    ]
    return [
        (boardify_string(cleandoc(pre)), boardify_string(cleandoc(post)))
        for (pre, post) in board_pairs
    ]


def test_trace_shared_lanes_same_3(
    board_pairs_trace_shared_lanes_same_3: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    for pre, post in board_pairs_trace_shared_lanes_same_3:
        tp = ThoughtProcess(pre)
        tp.shared_lanes_bot.mark_bulbs_and_dots_at_shared_lanes(-1, -1, ".")
        print_board(tp.board)
        assert stringify_board(tp.board) == stringify_board(post)


@pytest.fixture
def board_pairs_trace_shared_lanes(
    board_pairs_trace_shared_lanes_adjacent: list[tuple[np.ndarray, np.ndarray]],
    board_pairs_trace_shared_lanes_same_2: list[tuple[np.ndarray, np.ndarray]],
    board_pairs_trace_shared_lanes_same_3: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    board_pairs = []
    board_pairs.extend(board_pairs_trace_shared_lanes_adjacent)
    board_pairs.extend(board_pairs_trace_shared_lanes_same_2)
    board_pairs.extend(board_pairs_trace_shared_lanes_same_3)
    return board_pairs


def test_trace_shared_lanes_all(
    board_pairs_trace_shared_lanes: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    for pre, post in board_pairs_trace_shared_lanes:
        for pre_rotated, post_rotated in zip(
            all_orientations(pre),
            all_orientations(post),
            strict=True,
        ):
            print_board(pre_rotated)
            tp = ThoughtProcess(pre_rotated)
            print(tp.shared_lanes_bot.shared_lanes)
            tp.shared_lanes_bot.mark_bulbs_and_dots_at_shared_lanes(-1, -1, ".")
            assert stringify_board(tp.board) == stringify_board(post_rotated)


def test_trace_shared_lanes_same_3_bug_1() -> None:
    # done to confirm fix of a specific bug that should not arise again
    pre = boardify_string(
        cleandoc("""
    --------
    -2.....-
    -.2..2.-
    -......-
    --------
    """)
    )
    post = boardify_string(
        cleandoc("""
    --------
    -2#____-
    -#2..2.-
    -|.....-
    --------
    """)
    )
    tp = ThoughtProcess(pre)
    tp.maybe_set_bulb(2, 1)
    tp.maybe_set_bulb(1, 2)
    tp.shared_lanes_bot.mark_bulbs_and_dots_at_shared_lanes(-1, -1, ".")
    assert stringify_board(tp.board) == stringify_board(post)


def test_trace_shared_lanes_diagonal() -> None:
    board_pairs_raw = [
        (
            """
            --------
            -......-
            -.3....-
            -......-
            -...3..-
            -......-
            -......-
            --------
            """,
            """
            --------
            -x#__x_-
            -#3..|.-
            -|.+.|+-
            -|..3#_-
            -x__#x_-
            -|.+||.-
            --------
            """,
        ),
        (
            """
            --------
            -......-
            -.3....-
            -......-
            -...2..-
            -......-
            -......-
            --------
            """,
            """
            --------
            -......-
            -.3....-
            -......-
            -...2..-
            -......-
            -......-
            --------
            """,
        ),
    ]
    board_pairs = [
        (boardify_string(cleandoc(pre)), boardify_string(cleandoc(post)))
        for (pre, post) in board_pairs_raw
    ]
    for pre, post in board_pairs:
        for pre_rotated, post_rotated in zip(
            all_orientations(pre),
            all_orientations(post),
            strict=True,
        ):
            print_board(pre_rotated)
            tp = ThoughtProcess(pre_rotated)
            print(tp.shared_lanes_bot.shared_lanes)
            tp.shared_lanes_bot.mark_bulbs_and_dots_at_shared_lanes(-1, -1, ".")
            assert stringify_board(tp.board) == stringify_board(post_rotated)


def test_find_unilluminatable_cells() -> None:
    board = boardify_string(
        cleandoc("""
            ----
            -0.-
            ----
            -2#-
            -#--
            -.+-
            -.0-
            ----
            -0.-
            -..-
            -..-
            ----
            """)
    )
    tp = ThoughtProcess(board)
    tp.apply_methods(6)
    print_board(tp.board)
    assert set(tp.unilluminatable_cells) == {(1, 2), (5, 2)}


def test_mark_dots_beyond_corners() -> None:
    pre = boardify_string(
        cleandoc("""
            -------
            -0+++.-
            --+00.-
            --+0-.-
            --....-
            -------
            """)
    )
    post = boardify_string(
        cleandoc("""
            -------
            -0+++.-
            --+00.-
            --+0-.-
            --...+-
            -------
            """)
    )
    for pre_rotated, post_rotated in zip(
        all_orientations(pre),
        all_orientations(post),
        strict=True,
    ):
        print_board(pre_rotated)
        tp = ThoughtProcess(pre_rotated)
        tp.mark_dots_beyond_corners(-1, -1, ".")
        print_board(tp.board)
        assert stringify_board(tp.board) == stringify_board(post_rotated)

    pre = boardify_string(
        cleandoc("""
            -------
            -0++.+-
            --+00.-
            --+0-.-
            --....-
            -------
            """)
    )
    post = boardify_string(
        cleandoc("""
            -------
            -0++.+-
            --+00.-
            --+0-.-
            --....-
            -------
            """)
    )
    for pre_rotated, post_rotated in zip(
        all_orientations(pre),
        all_orientations(post),
        strict=True,
    ):
        print_board(pre_rotated)
        tp = ThoughtProcess(pre_rotated)
        tp.mark_dots_beyond_corners(-1, -1, ".")
        print_board(tp.board)
        assert stringify_board(tp.board) == stringify_board(post_rotated)

    pre = boardify_string(
        cleandoc("""
            -------
            -.+++.-
            --+00.-
            --+0-.-
            --+...-
            -------
            """)
    )
    post = boardify_string(
        cleandoc("""
            -------
            -.+++.-
            --+00.-
            --+0-.-
            --+...-
            -------
            """)
    )
    for pre_rotated, post_rotated in zip(
        all_orientations(pre),
        all_orientations(post),
        strict=True,
    ):
        print_board(pre_rotated)
        tp = ThoughtProcess(pre_rotated)
        tp.mark_dots_beyond_corners(-1, -1, ".")
        print_board(tp.board)
        assert stringify_board(tp.board) == stringify_board(post_rotated)

    pre = boardify_string(
        cleandoc("""
            -----
            --0.-
            -0..-
            -...-
            -----
            """)
    )
    post = boardify_string(
        cleandoc("""
            -----
            --0|-
            -0x#-
            -_#x-
            -----
            """)
    )
    for pre_rotated, post_rotated in zip(
        all_orientations(pre),
        all_orientations(post),
        strict=True,
    ):
        print_board(pre_rotated)
        tp = ThoughtProcess(pre_rotated)
        tp.apply_methods(6)
        print_board(tp.board)
        assert stringify_board(tp.board) == stringify_board(post_rotated)
