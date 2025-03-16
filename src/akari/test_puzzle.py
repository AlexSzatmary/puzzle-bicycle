from collections.abc import Generator
from inspect import cleandoc

import numpy as np
import pytest
from puzzle import (
    analyze_diagonally_adjacent_numbers,
    apply_methods,
    boardify_string,
    check_all,
    check_lit_bulbs,
    check_number,
    check_unlit_cells,
    count_free_near_number,
    count_missing_bulbs_near_number,
    fill_holes,
    find_unilluminatable_cells,
    find_wrong_numbers,
    illuminate,
    load_pzprv3,
    mark_bulbs_around_dotted_numbers,
    mark_dots_around_full_numbers,
    mark_dots_at_corners,
    mark_unique_bulbs_for_dot_cells,
    print_board,
    save_pzprv3,
    stringify_board,
    trace_shared_lanes,
    transpose_board,
    zero_pad,
)


def all_orientations(board: np.ndarray) -> Generator[np.ndarray]:
    board = board.copy()
    yield board.copy()
    np.fliplr(board)
    yield board.copy()
    np.flipud(board)
    yield board.copy()
    np.fliplr(board)
    yield board.copy()
    board = transpose_board(board)
    yield board.copy()
    np.fliplr(board)
    yield board.copy()
    np.flipud(board)
    yield board.copy()
    np.fliplr(board)
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


def test_illuminate_1() -> None:
    # Test for a good case
    for board, ref in zip(
        all_orientations(board_1_sol),
        all_orientations(board_1_sol_illuminated),
        strict=True,
    ):
        (wrong_bulb_pairs, illuminated_board) = illuminate(board)
        assert not wrong_bulb_pairs
        assert stringify_board(illuminated_board) == stringify_board(ref)

    # Test for detection of wrong pairs one time
    board_1_sol_wrong = board_1_sol.copy()
    board_1_sol_wrong[2, 1] = "#"
    board_1_sol_wrong[1, 5] = "#"
    (wrong_bulb_pairs_wrong, _) = illuminate(board_1_sol_wrong.copy())
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
        (wrong_bulb_pairs_wrong, illuminated_board_wrong) = illuminate(board)
        assert len(wrong_bulb_pairs_wrong) == 3
        assert stringify_board(illuminated_board_wrong) == stringify_board(ref)


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
        assert stringify_board(fill_holes(board) == ref)


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
        board_post = mark_dots_around_full_numbers(board)
        assert stringify_board(board_post) == stringify_board(ref)


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
            -.+.-.#.-.+.-.#.-.#.-
            ---------------------
            -...-...-.+.-.#.-.#.-
            -.0+-+1.-#2#-#3.-#4#-
            -...-...-...-.#.-.#.-
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
        assert stringify_board(
            mark_bulbs_around_dotted_numbers(board)
        ) == stringify_board(ref)


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
        assert stringify_board(mark_dots_at_corners(board)) == stringify_board(ref)


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
        assert stringify_board(
            mark_unique_bulbs_for_dot_cells(board)
        ) == stringify_board(ref)


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
            -.#..-
            -#3..-
            -..1+-
            -..+.-
            ------
            -.#..-
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
        assert stringify_board(
            analyze_diagonally_adjacent_numbers(board)
        ) == stringify_board(ref)


def test_find_wrong_numbers() -> None:
    for board in all_orientations(board_1_sol):
        assert not find_wrong_numbers(board)
    board_1_sol_1 = board_1_sol.copy()
    board_1_sol_1[2, 2] = "#"
    board_1_sol_1[5, 3] = "+"
    board_1_sol_1[2, 4] = "#"
    print_board(board_1_sol_1)
    assert find_wrong_numbers(board_1_sol_1) == [(2, 3), (3, 2), (3, 4), (4, 3)]


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
    for i, board_ref in zip(
        range(1, len(boards_apply_methods_sol) + 1),
        boards_apply_methods_sol,
        strict=True,
    ):
        print(i)
        for board, ref in zip(
            all_orientations(board_apply_methods),
            all_orientations(board_ref),
            strict=True,
        ):
            assert stringify_board(apply_methods(board, i)) == stringify_board(ref)


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
            -#+-
            -2.-
            -.+-
            -+.-
            -.2-
            -+#-
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
    assert stringify_board(trace_shared_lanes(null)) == stringify_board(null)
    pre = boardify_string(
        cleandoc("""
            ----
            -..-
            -2.-
            -..-
            -..-
            -.2-
            -..-
            ----
            """)
    )
    post = """
            ----
            -#+-
            -2.-
            -.+-
            -+.-
            -.2-
            -+#-
            ----
            """
    assert stringify_board(trace_shared_lanes(pre)) == cleandoc(post)


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
            -+#+-
            -.3.-
            -+#+-
            -+-+-
            -+#+-
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
            -.2.-
            -...-
            -----
            """,
            """
            -----
            -...-
            -.3.-
            -...-
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
            -+#+-
            -.3.-
            -+#+-
            -.3.-
            -+#+-
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
    assert stringify_board(trace_shared_lanes(pre)) == stringify_board(post)


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
            -+#+-
            -.3.-
            -+.+-
            -+++-
            -+.+-
            -.2.-
            -+#+-
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
            -+#+-
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
    pre, post = board_pairs_trace_shared_lanes_same_3[0]
    assert stringify_board(trace_shared_lanes(pre)) == stringify_board(post)


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
    print(f"len(board_pairs_trace_shared_lanes) {len(board_pairs_trace_shared_lanes)}")
    j = 0
    for pre, post in board_pairs_trace_shared_lanes:
        i = 0
        for pre_rotated, post_rotated in zip(
            all_orientations(pre),
            all_orientations(post),
            strict=True,
        ):
            print(f"HI i{i}")
            if j == 0:
                print(pre_rotated)
            i += 1
            assert stringify_board(trace_shared_lanes(pre_rotated)) == stringify_board(
                post_rotated
            )
        j = 1


def test_find_unilluminatable_cells() -> None:
    board = boardify_string(
        cleandoc("""
            ----
            -0.-
            ----
            -2.-
            -.--
            -..-
            -.0-
            ----
            -0.-
            -..-
            -..-
            ----
            """)
    )
    board = apply_methods(board, 6)
    assert set(find_unilluminatable_cells(board)) == {(1, 2), (5, 2)}
