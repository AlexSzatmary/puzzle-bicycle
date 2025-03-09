from itertools import zip_longest

import numpy as np

# format
# Board:
# 0, 1, 2, 3, 4: number of bulbs
# -: black, unnumbered
# .: free, empty
# Solution:
# #: bulb
# Marks:
# _, |, x: indicate light paths, x shows both directions,
# +: "dotted", indicates no bulb but direction not indicated


def load_pzprv3(pzprv3: str) -> np.ndarray:
    """
    Loads PUZ-PRE v3 text and returns an Akari board
    """
    pzprv3_lines = pzprv3.replace(" ", "").replace("/", "").split("\n")
    rows = int(pzprv3_lines[2])
    cols = int(pzprv3_lines[3])
    board = np.zeros((rows + 2, cols + 2), dtype="str")
    board[:, :] = "-"
    for i, row in enumerate(pzprv3_lines[4 : 4 + rows]):
        board[i + 1, 1:-1] = list(row.replace(" ", ""))
    return board


def save_pzprv3(board: np.ndarray) -> str:
    lines = []
    lines.append("pzprv3")
    lines.append("lightup")
    lines.append(str(board.shape[0] - 2))
    lines.append(str(board.shape[1] - 2))
    for row in board[1:-1]:
        lines.append(" ".join(row[1:-1]))
    lines.append("")
    return "\n".join(lines)


def zero_pad(grid: np.ndarray) -> np.ndarray:
    grid2 = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2), dtype="str")
    grid2[:, :] = "-"
    grid2[1:-1, 1:-1] = grid
    return grid2


def stringify_board(board: np.ndarray) -> str:
    return "\n".join("".join(list(row)) for row in board.astype(str))


def boardify_string(s: str) -> np.ndarray:
    return np.array(list(map(list, s.split("\n"))), dtype="str")


def print_board(board: np.ndarray) -> None:
    print(stringify_board(board))


def transpose_board(board: np.ndarray) -> np.ndarray:
    """
    Returns the board transposed, accounting for swapping the _ and | symbols
    """
    board = board.T
    boardLR = board == "_"
    boardUD = board == "|"
    board[boardLR] = "|"
    board[boardUD] = "_"
    return board


def check_number(board: np.ndarray) -> list[tuple[int, int]]:
    """
    Checks numbered spaces to see if they have the correct number of bulbs.

    Returns a list of tuples of coordinates of numbered spaces that touch the wrong
    number of bulbs.
    """
    wrong_bulbs = []
    for i in range(1, np.size(board, 0) - 1):
        for j in range(1, np.size(board, 1) - 1):
            if board[i, j] in "0123":
                if not int(board[i, j]) == (
                    (board[i - 1, j] == "#")
                    + (board[i + 1, j] == "#")
                    + (board[i, j - 1] == "#")
                    + (board[i, j + 1] == "#")
                ):
                    wrong_bulbs.append((i, j))
    return wrong_bulbs


def illuminate(  # noqa: C901 This level of complexity is fine.
    board: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """
    Takes board with bulbs. Returns a tuple with
    *a list of lists of tuples of coordinates of bulbs that shine on each other
    *board with light paths drawn (same object as input board)
    """
    lit_bulb_pairs = []
    fill_chars = ["|", "_", "|", "_"]
    for i in range(1, np.size(board, 0) - 1):
        for j in range(1, np.size(board, 1) - 1):
            if board[i, j] == "#":
                iters = [
                    zip_longest(range(i - 1, 0, -1), [], fillvalue=j),
                    zip_longest([], range(j - 1, 0, -1), fillvalue=i),
                    zip_longest(range(i + 1, np.size(board, 0) - 1), [], fillvalue=j),
                    zip_longest([], range(j + 1, np.size(board, 1) - 1), fillvalue=i),
                ]
                for it, fill_char in zip(iters, fill_chars, strict=True):
                    for i1, j1 in it:
                        if board[i1, j1] == "#":
                            if i <= i1 and j <= j1:
                                lit_bulb_pairs.append((i, j, i1, j1))
                            break
                        elif board[i1, j1] == fill_char or board[i1, j1] == "x":
                            # row or column already filled
                            continue
                        elif board[i1, j1] == "_" or board[i1, j1] == "|":
                            # this branch will only trigger if the char at this location
                            # is not the same as the fill_char
                            board[i1, j1] = "x"
                        elif board[i1, j1] in "01234-":
                            break
                        else:
                            board[i1, j1] = fill_char
    return (lit_bulb_pairs, board)


def fill_holes(board: np.ndarray) -> np.ndarray:
    """
    Takes annotated and possibly illuminated board.
    Returns the board with holes filled in.

    A hole is a free cell that must be a bulb because no bulb can possibly reach it.

    Holes:      Not holes:
    -------     -------
    -.-+.+-     -..-..-
    -------     ----0--
     ^  ^        ^^  *

    Cases marked ^ are simple. Case * is not a hole because this method does not see
    the 0; after a + is marked above the 0, the * would then be a hole.
    """

    for i in range(1, np.size(board, 0) - 1):
        for j in range(1, np.size(board, 1) - 1):
            if board[i, j] == ".":
                is_hole = True  # presume a hole
                iters = [
                    zip_longest(range(i - 1, 0, -1), [], fillvalue=j),
                    zip_longest([], range(j - 1, 0, -1), fillvalue=i),
                    zip_longest(range(i + 1, np.size(board, 0) - 1), [], fillvalue=j),
                    zip_longest([], range(j + 1, np.size(board, 1) - 1), fillvalue=i),
                ]
                for it in iters:
                    for i1, j1 in it:
                        if board[i1, j1] == ".":
                            is_hole = False
                            break
                        elif board[i1, j1] in "01234-":
                            break
                    if not is_hole:
                        break
                if is_hole:
                    board[i, j] = "#"
    return board


def count_free_near_number(board: np.ndarray, i: int, j: int) -> int:
    dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    return sum(board[i + di, j + dj] == "." for (di, dj) in dirs)


def count_missing_bulbs_near_number(board: np.ndarray, i: int, j: int) -> int:
    dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    n_bulbs = sum(board[i + di, j + dj] == "#" for (di, dj) in dirs)
    return int(board[i, j]) - n_bulbs


def mark_dots_around_full_numbers(board: np.ndarray) -> np.ndarray:
    dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    for i in range(1, np.size(board, 0) - 1):
        for j in range(1, np.size(board, 1) - 1):
            if board[i, j] in "01234":
                if count_missing_bulbs_near_number(board, i, j) == 0:
                    for di, dj in dirs:
                        if board[i + di, j + dj] == ".":
                            board[i + di, j + dj] = "+"
    return board


def mark_bulbs_around_dotted_numbers(board: np.ndarray) -> np.ndarray:
    dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    for i in range(1, np.size(board, 0) - 1):
        for j in range(1, np.size(board, 1) - 1):
            if board[i, j] in "01234":
                if count_free_near_number(
                    board, i, j
                ) == count_missing_bulbs_near_number(board, i, j):
                    for di, dj in dirs:
                        if board[i + di, j + dj] == ".":
                            board[i + di, j + dj] = "#"
    return board


def mark_dots_at_corners(board: np.ndarray) -> np.ndarray:
    """
    Marks dots at free cells diagonal to numbers if a bulb in that cell would not work.

    Will fire for
    ---
    -1.
    -..
    to do
    ---
    -1.
    -.+

    But will not fire at all for
    ---
    -2.
    -..
    because that case should already be caught by mark_bulbs_around_dotted_numbers.
    """
    ortho_dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    diag_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for i in range(1, np.size(board, 0) - 1):
        for j in range(1, np.size(board, 1) - 1):
            if board[i, j] in "01234":
                n_free = sum(board[i + di, j + dj] == "." for (di, dj) in ortho_dirs)
                n_bulbs_already = sum(
                    board[i + di, j + dj] == "#" for (di, dj) in ortho_dirs
                )
                if n_free + n_bulbs_already == int(board[i, j]) + 1:
                    for di, dj in diag_dirs:
                        if (
                            board[i + di, j + dj] == "."
                            and board[i + di, j] == "."
                            and board[i, j + dj] == "."
                        ):
                            board[i + di, j + dj] = "+"
    return board


def mark_unique_bulbs_for_dot_cells(  # noqa: C901 This level of complexity is fine.
    board: np.ndarray,
) -> np.ndarray:
    """
    Takes annotated and possibly illuminated board. Marks cells that must be bulbs for
    a dotted cell to be illuminated. For example, if we have,
    -0--
    -+.-
    --.-
    ----
    we must obtain,
    -0--
    -+#-
    --.-
    ----
    Because otherwise the + below the 0 could not be illuminated.

    This function runs illuminate a lot and it's not obvious to me if that's an
    implementation decision. If it's inefficient to run illuminate a lot, the fix is
    to allow illuminate to take an argument for the single i, j for the new bulb.
    """
    board = illuminate(board)[1]
    # we have to illuminate a lot because this logic ignores bulbs
    for i in range(1, np.size(board, 0) - 1):
        for j in range(1, np.size(board, 1) - 1):
            if board[i, j] == "+":
                sees_free = False
                sees_multiple_free = False
                free_i = free_j = -1
                iters = [
                    zip_longest(range(i - 1, 0, -1), [], fillvalue=j),
                    zip_longest([], range(j - 1, 0, -1), fillvalue=i),
                    zip_longest(range(i + 1, np.size(board, 0) - 1), [], fillvalue=j),
                    zip_longest([], range(j + 1, np.size(board, 1) - 1), fillvalue=i),
                ]
                for it in iters:
                    for i1, j1 in it:
                        if board[i1, j1] == ".":
                            if sees_free:
                                sees_multiple_free = True
                                break
                            else:
                                sees_free = True
                                free_i = i1
                                free_j = j1
                        elif board[i1, j1] in "01234-":
                            break
                    if sees_multiple_free:
                        break
                if sees_free and not sees_multiple_free:
                    board[free_i, free_j] = "#"
                    board = illuminate(board)[1]
    return board


def analyze_diagonally_adjacent_numbers(board: np.ndarray) -> np.ndarray:
    """
    Adds dots and bulbs for certain diagonally adjacent numbers sharing 2 free spaces.

    Finds this kind of thing:
    ----
    -1..
    -.1.
    -...
    and sees that we add + as shown:
    ----
    -1..
    -.1+
    -.+.
    Likewise,
    -----
    -....
    -.3..
    -..1.
    -....
    becomes
    -----
    -.#..
    -#3..
    -..1+
    -..+.
    We also account for what happens if a bulb or dot is already known.
    """
    for iA in range(1, np.size(board, 0) - 1):
        for jA in range(1, np.size(board, 1) - 1):
            if board[iA, jA] in "01234" and board[iA + 1, jA] == ".":
                # only analyze numbered cells diagonally down and to the left or right
                iB = iA + 1
                for jB in [jA + 1, jA - 1]:
                    if board[iB, jB] in "01234" and board[iA, jB] == ".":
                        missing_A = count_missing_bulbs_near_number(board, iA, jA)
                        free_A = count_free_near_number(board, iA, jA)
                        missing_B = count_missing_bulbs_near_number(board, iB, jB)
                        free_B = count_free_near_number(board, iB, jB)
                        if missing_A == 1 and missing_B + 1 == free_B:
                            analyze_diagonally_adjacent_numbers_update_board(
                                board, iA, jA, iB, jB
                            )
                        elif missing_B == 1 and missing_A + 1 == free_A:
                            analyze_diagonally_adjacent_numbers_update_board(
                                board, iB, jB, iA, jA
                            )
    return board


def analyze_diagonally_adjacent_numbers_update_board(
    board: np.ndarray, iC: int, jC: int, iD: int, jD: int
) -> np.ndarray:
    # point C has 1 missing bulb and point D has one free space more
    di = iD - iC
    dj = jD - jC
    if board[iC - di, jC] == ".":
        board[iC - di, jC] = "+"
    if board[iC, jC - dj] == ".":
        board[iC, jC - dj] = "+"
    if board[iD + di, jD] == ".":
        board[iD + di, jD] = "#"
    if board[iD, jD + dj] == ".":
        board[iD, jD + dj] = "#"
    return board


def trace_shared_lanes(board: np.ndarray) -> np.ndarray:
    """
    If we have,
    -----
    -...-
    -.3.-
    -...-
    -...-
    -...-
    -.2.-
    -...-
    -----
    we can obtain,
    -----
    -.#.-
    -.3.-
    -...-
    -...-
    -...-
    -.2.-
    -.#.-
    -----
    because the 3 and 2 compete over 3 lanes, need 5 bulbs total, and can hide 2 bulbs
    from each other. Moreover, we can also get,
    -----
    -_#_-
    -.3.-
    -+.+-
    -+++-
    -+.+-
    -.2.-
    -_#_-
    -----
    because a bulb at any of the + locations would steal needed cells for the 3
    and 2.

    We can also consider sharing only 2 lanes. We can't show anything if only 1 lane
    is shared. This function can consider sharing like,
    ----
    -..-
    -2.-
    -..-
    -..-
    -.2-
    -..-
    ----
    that is, where the two numbers are off by one row or column. We can also account for
    two numbers in a line but that have one of their lanes blocked by a black square or
    a bulb.

    -----
    -...-
    -.3.-
    -...-
    -.2.-
    -...-
    -----
    is a fun special case because the 3 and 2 share the cell between them. If two cells
    are in a line and have one free cell between them, that cell is not considered in
    a competing lane. Thus, this rule does not fire; it would fire if the 2 were
    another 3.
    """
    board = _trace_shared_lanes_down(board)

    # We can just transpose the board back and forth to just write the logic to sweep
    # down.
    board = board.T
    board[board == "_"] = "S"
    board[board == "|"] = "_"
    board[board == "S"] = "|"
    board = _trace_shared_lanes_down(board)
    board = board.T
    board[board == "_"] = "S"
    board[board == "|"] = "_"
    board[board == "S"] = "|"
    return board


def _trace_shared_lanes_down(board: np.ndarray) -> np.ndarray:
    """
    This helper for trace_shared_lanes only sweeps down.
    """
    for iA in range(1, np.size(board, 0) - 1):
        for jA in range(1, np.size(board, 1) - 1):
            if board[iA, jA] in "123" and count_missing_bulbs_near_number(
                board, iA, jA
            ) + 1 == count_free_near_number(board, iA, jA):
                tracer_C = board[iA, jA - 1] == "."
                tracer_D = True
                tracer_E = board[iA, jA + 1] == "."
                for iB in range(iA + 1, np.size(board, 0)):
                    if tracer_C + tracer_D + tracer_E >= 2:
                        new_tracer_C = tracer_C and board[iB, jA - 1] in ".+_|x"
                        new_tracer_D = tracer_D and board[iB, jA] in ".+_|x"
                        new_tracer_E = tracer_E and board[iB, jA + 1] in ".+_|x"
                        _analyze_pairs_adjacent_columns(
                            board,
                            iA,
                            jA,
                            iB,
                            jA - 1,
                            tracer_C,
                            new_tracer_D,
                        )
                        _analyze_pairs_adjacent_columns(
                            board,
                            iA,
                            jA,
                            iB,
                            jA + 1,
                            tracer_E,
                            new_tracer_D,
                        )
                        tracer_C = new_tracer_C
                        tracer_D = new_tracer_D
                        tracer_E = new_tracer_E
                        # TODO write function for two numbers that share a column
    return board


def _analyze_pairs_adjacent_columns(
    board: np.ndarray,
    iA: int,
    jA: int,
    iB: int,
    jB: int,
    tracer_col_B: bool,  # noqa: FBT001
    new_tracer_col_A: bool,  # noqa: FBT001
) -> np.ndarray:
    dj = jB - jA
    if (
        tracer_col_B
        and new_tracer_col_A
        and board[iB, jB] in "123"
        and board[iB - 1, jB] == "."
        and board[iB, jA] == "."
    ):
        # points A and B are both numbers and share 2 lanes
        if count_missing_bulbs_near_number(board, iB, jB) + 1 == count_free_near_number(
            board, iB, jB
        ):
            if board[iA - 1, jA] == ".":
                board[iA - 1, jA] = "#"
            if board[iA, jA - dj] == ".":
                board[iA, jA - dj] = "#"
            if board[iB + 1, jB] == ".":
                board[iB + 1, jB] = "#"
            if board[iB, jB + dj] == ".":
                board[iB, jB + dj] = "#"
            # TODO mark out blanks in columns, going down from iA + 2, jA and up from
            # iB, jB - 2
    return board


def apply_methods(board: np.ndarray, level: int) -> np.ndarray:
    while True:
        old_board = board.copy()
        if level >= 1:
            board = illuminate(board)[1]
        if level >= 2:
            board = mark_dots_around_full_numbers(board)
            board = mark_bulbs_around_dotted_numbers(board)
        if level >= 3:
            board = illuminate(board)[1]
            board = fill_holes(board)
            board = mark_unique_bulbs_for_dot_cells(board)
        if level >= 4:
            board = mark_dots_at_corners(board)
        if level >= 5:
            board = analyze_diagonally_adjacent_numbers(board)
        if level >= 5:
            board = trace_shared_lanes(board)
        if np.all(board == old_board):
            break
    return board


def check_unlit_cells(board: np.ndarray) -> bool:
    """
    Returns True if a board has no unlit cells, False otherwise
    """
    (_, board) = illuminate(board.copy())
    return not np.any(np.logical_or(board == ".", board == "+")) == np.True_


def check_lit_bulbs(board: np.ndarray) -> bool:
    """
    Returns True if a board has no lit bulbs, False otherwise
    """
    (wrong_bulb_pairs, board) = illuminate(board.copy())
    return not bool(wrong_bulb_pairs)


def check_all(board: np.ndarray) -> bool:
    return (
        not check_number(board) and check_unlit_cells(board) and check_lit_bulbs(board)
    )


def find_wrong_numbers(board: np.ndarray) -> list[tuple[int, int]]:
    """
    Checks numbered spaces to see if they have the correct number of bulbs.

    Returns a list of tuples of coordinates of numbered spaces that touch the wrong
    number of bulbs.
    """
    wrong_numbers = []
    dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    for i in range(1, np.size(board, 0) - 1):
        for j in range(1, np.size(board, 1) - 1):
            if board[i, j] in "0123":
                n_free = sum(board[i + di, j + dj] == "." for (di, dj) in dirs)
                n_bulbs_already = sum(board[i + di, j + dj] == "#" for (di, dj) in dirs)
                if n_bulbs_already > int(board[i, j]) or n_free + n_bulbs_already < int(
                    board[i, j]
                ):
                    wrong_numbers.append((i, j))
    return wrong_numbers
