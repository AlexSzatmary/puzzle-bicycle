from collections import deque
from collections.abc import Callable
from itertools import zip_longest
from typing import cast

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

ORTHO_DIRS = [(1, 0), (0, -1), (-1, 0), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


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


def illuminate_all(
    board: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    tp = ThoughtProcess(board)
    tp.illuminate_all()
    board[:] = tp.board[:]
    return tp.lit_bulb_pairs, tp.board


def illuminate_one(
    board: np.ndarray, i: int, j: int
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    tp = ThoughtProcess(board)
    tp.illuminate_one(i, j)
    board[:] = tp.board[:]
    return tp.lit_bulb_pairs, tp.board


def board_maybe_set_bulb(board: np.ndarray, i: int, j: int) -> None:
    if board[i, j] == ".":
        board[i, j] = "#"
        illuminate_one(board, i, j)


def count_free_near_number(board: np.ndarray, i: int, j: int) -> int:
    return sum(board[i + di, j + dj] == "." for (di, dj) in ORTHO_DIRS)


def count_missing_bulbs_near_number(board: np.ndarray, i: int, j: int) -> int:
    n_bulbs = sum(board[i + di, j + dj] == "#" for (di, dj) in ORTHO_DIRS)
    return int(board[i, j]) - n_bulbs


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
            ) + 2 >= count_free_near_number(board, iA, jA):
                tracer_C = board[iA, jA - 1] == "."
                tracer_D = board[iA + 1, jA] == "."
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
                        _analyze_pairs_same_column(
                            board, iA, jA, iB, new_tracer_C, tracer_D, new_tracer_E
                        )
                        tracer_C = new_tracer_C
                        tracer_D = new_tracer_D
                        tracer_E = new_tracer_E
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
        count_missing_bulbs_near_number(board, iA, jA) + 1
        == count_free_near_number(board, iA, jA)
        and tracer_col_B
        and new_tracer_col_A
        and board[iB, jB] in "123"
        and board[iB - 1, jB] == "."
        and board[iB, jA] == "."
        and iB > iA + 1
    ):
        # points A and B are both numbers and share 2 lanes
        if count_missing_bulbs_near_number(board, iB, jB) + 1 == count_free_near_number(
            board, iB, jB
        ):
            board_maybe_set_bulb(board, iA - 1, jA)
            board_maybe_set_bulb(board, iA, jA - dj)
            board_maybe_set_bulb(board, iB + 1, jB)
            board_maybe_set_bulb(board, iB, jB + dj)
            _dot_adjacent_columns(board, iA, jA, iB, jB)
    return board


def _dot_adjacent_columns(
    board: np.ndarray, iA: int, jA: int, iB: int, jB: int
) -> np.ndarray:
    for i in range(iA + 2, np.size(board, 0) - 1):
        if board[i, jA] in "-01234":
            break
        elif i != iB and board[i, jA] == ".":
            board[i, jA] = "+"
    for i in range(iB - 2, 0, -1):
        if board[i, jB] in "-01234":
            break
        elif i != iA and board[i, jB] == ".":
            board[i, jB] = "+"
    return board


def _analyze_pairs_same_column(
    board: np.ndarray,
    iA: int,
    j: int,
    iB: int,
    new_tracer_C: bool,  # noqa: FBT001
    tracer_D: bool,  # noqa: FBT001
    new_tracer_E: bool,  # noqa: FBT001
) -> np.ndarray:
    new_tracer_C = new_tracer_C and board[iB, j - 1] == "."
    tracer_D = tracer_D and board[iB - 1, j] == "." and iB > iA + 2
    new_tracer_E = new_tracer_E and board[iB, j + 1] == "."
    if board[iB, j] in "123" and new_tracer_C + tracer_D + new_tracer_E >= 2:
        # points A and B are both numbers and share 2 or 3 lanes
        if new_tracer_C + tracer_D + new_tracer_E == 2:
            _analyze_pairs_same_column_2(
                board, iA, j, iB, new_tracer_C, tracer_D, new_tracer_E
            )
        elif new_tracer_C + tracer_D + new_tracer_E == 3:
            _analyze_pairs_same_column_3(
                board, iA, j, iB, new_tracer_C, tracer_D, new_tracer_E
            )
    return board


def _analyze_pairs_same_column_2(
    board: np.ndarray,
    iA: int,
    j: int,
    iB: int,
    new_tracer_C: bool,  # noqa: FBT001
    tracer_D: bool,  # noqa: FBT001
    new_tracer_E: bool,  # noqa: FBT001
) -> np.ndarray:
    if count_missing_bulbs_near_number(board, iA, j) + 1 == count_free_near_number(
        board, iA, j
    ) and count_missing_bulbs_near_number(board, iB, j) + 1 == count_free_near_number(
        board, iB, j
    ):
        board_maybe_set_bulb(board, iA - 1, j)
        if not new_tracer_C:
            board_maybe_set_bulb(board, iA, j - 1)
        if not tracer_D or iA + 2 == iB:
            board_maybe_set_bulb(board, iA + 1, j)
        if not new_tracer_E:
            board_maybe_set_bulb(board, iA, j + 1)

        board_maybe_set_bulb(board, iB + 1, j)
        if not new_tracer_C:
            board_maybe_set_bulb(board, iB, j - 1)
        if not tracer_D:
            board_maybe_set_bulb(board, iB - 1, j)
        if not new_tracer_E:
            board_maybe_set_bulb(board, iB, j + 1)
        _dot_columns_same(board, iA, j - 1, iB, new_tracer_C)
        _dot_columns_same(board, iA + 1, j, iB - 1, tracer_D)
        _dot_columns_same(board, iA, j + 1, iB, new_tracer_E)
    return board


def _analyze_pairs_same_column_3(
    board: np.ndarray,
    iA: int,
    j: int,
    iB: int,
    new_tracer_C: bool,  # noqa: FBT001
    tracer_D: bool,  # noqa: FBT001
    new_tracer_E: bool,  # noqa: FBT001
) -> np.ndarray:
    if (
        count_free_near_number(board, iA, j)
        + count_free_near_number(board, iB, j)
        - count_missing_bulbs_near_number(board, iA, j)
        - +count_missing_bulbs_near_number(board, iB, j)
        - 3
        == 0
    ):
        board_maybe_set_bulb(board, iA - 1, j)
        board_maybe_set_bulb(board, iB + 1, j)

        _dot_columns_same(board, iA, j - 1, iB, new_tracer_C)
        # _dot_columns_same(board, iA, j, iB, tracer_D)
        for i in range(iA + 2, iB - 1):
            if board[i, j] == ".":
                board[i, j] = "+"
        _dot_columns_same(board, iA, j + 1, iB, new_tracer_E)
    return board


def _dot_columns_same(
    board: np.ndarray,
    iA: int,
    j: int,
    iB: int,
    tracer: bool,  # noqa: FBT001
) -> np.ndarray:
    if tracer:
        for r in [
            range(iA - 1, 0, -1),
            range(iA + 1, iB),
            range(iB + 1, np.size(board, 0)),
        ]:
            for i in r:
                if board[i, j] in "-01234":
                    break
                elif board[i, j] == ".":
                    board[i, j] = "+"
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


def check_unlit_cells(board: np.ndarray) -> bool:
    """
    Returns True if a board has no unlit cells, False otherwise
    """
    (_, board) = illuminate_all(board.copy())
    return not np.any(np.logical_or(board == ".", board == "+")) == np.True_


def check_lit_bulbs(board: np.ndarray) -> bool:
    """
    Returns True if a board has no lit bulbs, False otherwise
    """
    (wrong_bulb_pairs, board) = illuminate_all(board.copy())
    return not bool(wrong_bulb_pairs)


def check_all(board: np.ndarray) -> bool:
    return (
        not check_number(board) and check_unlit_cells(board) and check_lit_bulbs(board)
    )


class ThoughtProcess:
    """
    The main state held is: board: np.ndarray

    new_mark: deque[tuple[int, int, str]], a queue of new dots "+" or bulbs "#", which
    will later trigger methods to run

    lit_bulb_pairs: list[tuple[int, int, int, int]], coordinates of pairs of bulbs that
    see each other

    unilluminatable_cells: list[tuple[int, int]], cells that cannot be lit

    wrong_numbers: list[tuple[int, int]], numbers that have infeasible numbers of bulbs
    (either too high or too many dots to ever have enough) The last 3 items are the
    error lists.
    """

    def __init__(self, board: np.ndarray) -> None:
        self.board = board.copy()
        self.new_mark = deque()
        self.lit_bulb_pairs = []
        self.unilluminatable_cells = []
        self.wrong_numbers = set()

    def maybe_set_bulb(self, i: int, j: int) -> None:
        """
        Confirms that board[i, j] is free; if so, board[i, j] = "#" and runs updates

        Runs illuminate and updates the queue.
        """
        if self.board[i, j] == ".":
            self.board[i, j] = "#"
            self.new_mark.append((i, j, "#"))
            self.illuminate_one(i, j)
            self.find_wrong_numbers(i, j)

    def maybe_set_dot(self, i: int, j: int) -> None:
        """
        Confirms that board[i, j] is free; if so, board[i, j] = "." and runs updates

        Updates the queue.
        """
        if self.board[i, j] == ".":
            self.board[i, j] = "+"
            self.new_mark.append((i, j, "+"))
            self.find_wrong_numbers(i, j)
            self.find_unilluminatable_cells(i, j)

    def all_interior_ij(self) -> list[tuple[int, int]]:
        return [
            (i, j)
            for i in range(1, self.board.shape[0] - 1)
            for j in range(1, self.board.shape[1] - 1)
        ]

    def line_of_sight(self, i: int, j: int) -> list[tuple[int, int]]:
        line_of_sight_cells = []
        iters = [
            zip_longest(range(i - 1, 0, -1), [], fillvalue=j),
            zip_longest([], range(j - 1, 0, -1), fillvalue=i),
            zip_longest(range(i + 1, np.size(self.board, 0) - 1), [], fillvalue=j),
            zip_longest([], range(j + 1, np.size(self.board, 1) - 1), fillvalue=i),
        ]
        for it in iters:
            for i, j in it:
                if self.board[i, j] in "01234-":
                    break
                line_of_sight_cells.append((i, j))
        return line_of_sight_cells

    def apply_methods(self, level: int) -> None:  # noqa: C901
        # The complexity here is fine
        """
        Applies the various logical methods as set by the level.

        If the queue is empty, it first, scans everything; otherwise, it just does the
        new_mark queue.
        """
        if level < 1:
            return
        if not self.new_mark:
            self.new_mark = deque((i, j, ".") for (i, j) in self.all_interior_ij())
            self.lit_bulb_pairs, self.board = illuminate_all(self.board)
            for i, j in self.all_interior_ij():
                self.find_wrong_numbers_at_cell(i, j)
                self.check_this_cell_unilluminatable(i, j)
        while self.new_mark:
            i, j, mark = self.new_mark.popleft()
            if level >= 2:
                self.mark_bulbs_around_dotted_numbers(i, j, mark)
                self.mark_dots_around_full_numbers(i, j, mark)
            if level >= 3:
                self.fill_holes(i, j, mark)
                self.mark_unique_bulbs_for_dot_cells(i, j, mark)
            if level >= 4:
                self.mark_dots_at_corners(i, j, mark)
            if level >= 5:
                self.analyze_diagonally_adjacent_numbers(i, j, mark)
            if level >= 6:
                ...
            #     self.trace_shared_lanes(i, j)
            if not self.check_unsolved() and not mark == ".":
                break
            if level >= 9 and not self.new_mark:
                # guess and check is orders of magnitude more expensive than other
                # methods and should only be called if all else has been tried.
                self.guess_and_check(level)

    def illuminate_one(self, i: int, j: int) -> None:
        """
        Illuminate the bulb at i, j
        """
        fill_chars = ["|", "_", "|", "_"]
        if self.board[i, j] == "#":
            iters = [
                zip_longest(range(i - 1, 0, -1), [], fillvalue=j),
                zip_longest([], range(j - 1, 0, -1), fillvalue=i),
                zip_longest(range(i + 1, np.size(self.board, 0) - 1), [], fillvalue=j),
                zip_longest([], range(j + 1, np.size(self.board, 1) - 1), fillvalue=i),
            ]
            for it, fill_char in zip(iters, fill_chars, strict=True):
                for i1, j1 in it:
                    if self.board[i1, j1] == "#":
                        if (i1, j1, i, j) not in self.lit_bulb_pairs:
                            self.lit_bulb_pairs.append((i, j, i1, j1))
                    elif self.board[i1, j1] == fill_char or self.board[i1, j1] == "x":
                        # row or column already filled
                        continue
                    elif self.board[i1, j1] == "_" or self.board[i1, j1] == "|":
                        # this branch will only trigger if the char at this location
                        # is not the same as the fill_char
                        self.board[i1, j1] = "x"
                    elif self.board[i1, j1] in "01234-":  # type: ignore
                        break
                    else:
                        self.maybe_set_dot(i1, j1)
                        self.board[i1, j1] = fill_char  # this assignment is cosmetic,
                        # making the light rays but not functionally changing state.

    def illuminate_all(self) -> None:
        """
        Illuminates the whole board
        """
        # numpy likes to give indices as np.int64, not int, which is gross.
        ijs_np_int64 = np.asarray(self.board == "#").nonzero()
        ijs = (map(int, ijs_np_int64[0]), map(int, ijs_np_int64[1]))
        for i, j in zip(*ijs, strict=True):
            self.illuminate_one(i, j)

    def mark_bulbs_around_dotted_numbers(self, i: int, j: int, mark: str) -> None:
        if mark == ".":
            cells_to_check = [(i, j)]
        elif mark in ".+_|x":
            cells_to_check = [(i + di1, j + dj1) for di1, dj1 in ORTHO_DIRS]
        else:
            return
        for i_number, j_number in cells_to_check:
            if self.board[i_number, j_number] in "01234":
                if count_free_near_number(
                    self.board, i_number, j_number
                ) == count_missing_bulbs_near_number(self.board, i_number, j_number):
                    for di, dj in ORTHO_DIRS:
                        self.maybe_set_bulb(i_number + di, j_number + dj)

    def mark_dots_around_full_numbers(self, i: int, j: int, mark: str) -> None:
        if mark == ".":
            cells_to_check = [(i, j)]
        elif mark in "#":
            cells_to_check = [(i + di1, j + dj1) for di1, dj1 in ORTHO_DIRS]
        else:
            return
        for i_number, j_number in cells_to_check:
            if self.board[i_number, j_number] in "01234":
                if count_missing_bulbs_near_number(self.board, i_number, j_number) == 0:
                    for di2, dj2 in ORTHO_DIRS:
                        self.maybe_set_dot(i_number + di2, j_number + dj2)

    def fill_holes(self, i0: int, j0: int, mark: str) -> None:
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
        if mark == ".":
            self._fill_holes_cell(i0, j0)
        elif mark == "+":
            for i, j in self.line_of_sight(i0, j0):
                self._fill_holes_cell(i, j)

    def _fill_holes_cell(self, i: int, j: int) -> None:
        if self.board[i, j] == ".":
            is_hole = True  # presume a hole
            # TODO use line of sight function here
            iters = [
                zip_longest(range(i - 1, 0, -1), [], fillvalue=j),
                zip_longest([], range(j - 1, 0, -1), fillvalue=i),
                zip_longest(range(i + 1, np.size(self.board, 0) - 1), [], fillvalue=j),
                zip_longest([], range(j + 1, np.size(self.board, 1) - 1), fillvalue=i),
            ]
            for it in iters:
                for i1, j1 in it:
                    if self.board[i1, j1] == ".":
                        is_hole = False
                        break
                    elif self.board[i1, j1] in "01234-":
                        break
                if not is_hole:
                    break
            if is_hole:
                self.maybe_set_bulb(i, j)

    def mark_unique_bulbs_for_dot_cells(self, i0: int, j0: int, mark: str) -> None:
        """
        Takes annotated and possibly illuminated board. Marks cells that must be bulbs
        for a dotted cell to be illuminated. For example, if we have,
        -0--
        -+.-
        --.-
        ----
        we must obtain,
        -0--
        -+#-
        --|-
        ----
        Because otherwise the + below the 0 could not be illuminated.
        """
        if mark == ".":
            self._mark_unique_bulbs_for_dot_cells_at_cell(i0, j0)
        elif mark == "+":
            for i, j in self.line_of_sight(i0, j0):
                self._mark_unique_bulbs_for_dot_cells_at_cell(i, j)

    def _mark_unique_bulbs_for_dot_cells_at_cell(self, i: int, j: int) -> None:
        if self.board[i, j] == "+":
            sees_free = False
            sees_multiple_free = False
            free_i = free_j = -1
            iters = [
                zip_longest(range(i - 1, 0, -1), [], fillvalue=j),
                zip_longest([], range(j - 1, 0, -1), fillvalue=i),
                zip_longest(range(i + 1, np.size(self.board, 0) - 1), [], fillvalue=j),
                zip_longest([], range(j + 1, np.size(self.board, 1) - 1), fillvalue=i),
            ]
            for it in iters:
                for i1, j1 in it:
                    if self.board[i1, j1] == ".":
                        if sees_free:
                            sees_multiple_free = True
                            break
                        else:
                            sees_free = True
                            free_i = i1
                            free_j = j1
                    elif self.board[i1, j1] in "01234-":
                        break
                if sees_multiple_free:
                    break
            if sees_free and not sees_multiple_free:
                self.maybe_set_bulb(free_i, free_j)

    def mark_dots_at_corners(self, i: int, j: int, mark: str) -> None:
        """
        Marks dots at free cells diagonal to numbers if a bulb in that cell would not
        work.

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
        if mark == ".":
            self._mark_dots_at_corners_at_cell(i, j)
        elif mark == "+":
            for di1, dj1 in ORTHO_DIRS:
                i_corner = i + di1
                j_corner = j + dj1
                self._mark_dots_at_corners_at_cell(i_corner, j_corner)

    def _mark_dots_at_corners_at_cell(self, i: int, j: int) -> None:
        if self.board[i, j] in "01234":
            n_free = sum(
                self.board[i + di2, j + dj2] == "." for (di2, dj2) in ORTHO_DIRS
            )
            n_bulbs_already = sum(
                self.board[i + di, j + dj] == "#" for (di, dj) in ORTHO_DIRS
            )
            if n_free + n_bulbs_already == int(self.board[i, j]) + 1:
                for di, dj in DIAG_DIRS:
                    if self.board[i + di, j] == "." and self.board[i, j + dj] == ".":
                        self.maybe_set_dot(i + di, j + dj)

    def analyze_diagonally_adjacent_numbers(self, i: int, j: int, mark: str) -> None:
        """
        Adds dots and bulbs for certain diagonally adjacent numbers sharing 2 free
        spaces.

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
        if mark == ".":
            cells_to_check = [(i, j)]
        else:
            cells_to_check = [(i + di1, j + dj1) for di1, dj1 in ORTHO_DIRS]
        for iA, jA in cells_to_check:
            if self.board[iA, jA] in "01234":
                for di2, dj2 in DIAG_DIRS:
                    iB = iA + di2
                    jB = jA + dj2
                    if (
                        self.board[iB, jB] in "01234"
                        and self.board[iA, jB] == "."
                        and self.board[iB, jA] == "."
                    ):
                        missing_A = count_missing_bulbs_near_number(self.board, iA, jA)
                        free_A = count_free_near_number(self.board, iA, jA)
                        missing_B = count_missing_bulbs_near_number(self.board, iB, jB)
                        free_B = count_free_near_number(self.board, iB, jB)
                        if missing_A == 1 and missing_B + 1 == free_B:
                            self._analyze_diagonally_adjacent_numbers_update_board(
                                iA, jA, iB, jB
                            )
                        elif missing_B == 1 and missing_A + 1 == free_A:
                            self._analyze_diagonally_adjacent_numbers_update_board(
                                iB, jB, iA, jA
                            )

    def _analyze_diagonally_adjacent_numbers_update_board(
        self, iC: int, jC: int, iD: int, jD: int
    ) -> None:
        # point C has 1 missing bulb and point D has one free space more
        di = iD - iC
        dj = jD - jC
        self.maybe_set_dot(iC - di, jC)
        self.maybe_set_dot(iC, jC - dj)
        self.maybe_set_bulb(iD + di, jD)
        self.maybe_set_bulb(iD, jD + dj)

    def transition_wrapper(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        old_board = self.board.copy()
        func(self.board)
        for i, j in zip(*(old_board != self.board).nonzero(), strict=True):
            self.new_mark.append((i, j, cast(str, self.board[i, j])))

    def trace_shared_lanes(self, i: int, j: int) -> None:
        self.transition_wrapper(trace_shared_lanes)

    def find_wrong_numbers(self, i: int, j: int) -> None:
        """
        Checks numbered spaces to see if they have a feasible number of bulbs.

        Returns a list of tuples of coordinates of numbered spaces that touch the wrong
        number of bulbs.
        """
        for di1, dj1 in ORTHO_DIRS:
            i_number = i + di1
            j_number = j + dj1
            self.find_wrong_numbers_at_cell(i_number, j_number)

    def find_wrong_numbers_at_cell(self, i_number: int, j_number: int) -> None:
        if self.board[i_number, j_number] in "01234":
            n_free = sum(
                self.board[i_number + di, j_number + dj] == "."
                for (di, dj) in ORTHO_DIRS
            )
            n_bulbs_already = sum(
                self.board[i_number + di, j_number + dj] == "#"
                for (di, dj) in ORTHO_DIRS
            )
            if n_bulbs_already > int(
                self.board[i_number, j_number]
            ) or n_free + n_bulbs_already < int(self.board[i_number, j_number]):
                self.wrong_numbers.add((i_number, j_number))

    def find_unilluminatable_cells(self, i: int, j: int) -> None:
        """
        Finds cells that cannot be illuminated and cannot be bulbs.

        Scans over line of sight from i, j for dotted cells, and checks those

        Examples are *:
        ----
        -0*-
        ----

        -----
        -2#_-
        -#-*0
        -----
        """
        self.check_this_cell_unilluminatable(i, j)
        iters1 = [
            zip_longest(range(i - 1, 0, -1), [], fillvalue=j),
            zip_longest([], range(j - 1, 0, -1), fillvalue=i),
            zip_longest(range(i + 1, np.size(self.board, 0) - 1), [], fillvalue=j),
            zip_longest([], range(j + 1, np.size(self.board, 1) - 1), fillvalue=i),
        ]
        for it1 in iters1:
            for i1, j1 in it1:
                if self.board[i1, j1] in "01234-":
                    break
                self.check_this_cell_unilluminatable(i1, j1)

    def check_this_cell_unilluminatable(self, i: int, j: int) -> None:
        if self.board[i, j] == "+":
            is_unilluminatable = True
            iters = [
                zip_longest(range(i - 1, 0, -1), [], fillvalue=j),
                zip_longest([], range(j - 1, 0, -1), fillvalue=i),
                zip_longest(range(i + 1, np.size(self.board, 0) - 1), [], fillvalue=j),
                zip_longest([], range(j + 1, np.size(self.board, 1) - 1), fillvalue=i),
            ]
            for it2 in iters:
                for i2, j2 in it2:
                    if self.board[i2, j2] in ".#":
                        is_unilluminatable = False
                        break
                    elif self.board[i2, j2] in "01234-":
                        break
                if not is_unilluminatable:
                    break
            if is_unilluminatable:
                self.unilluminatable_cells.append((i, j))

    def check_unsolved(self) -> bool:
        return not any(
            [self.wrong_numbers, self.lit_bulb_pairs, self.unilluminatable_cells]
        )

    def guess_and_check(self, level: int) -> None:
        """
        Guesses at every blank cell and uses apply_methods to eliminate impossible
        options.
        """
        level_to_use = min(level, 8)
        while True:
            old_board = self.board.copy()
            for i, j in zip(*np.asarray(self.board == ".").nonzero(), strict=True):
                if self.board[i, j] == ".":
                    try_tp_dot = ThoughtProcess(self.board)
                    try_tp_dot.maybe_set_dot(i, j)
                    try_tp_dot.apply_methods(level_to_use)
                    if not try_tp_dot.check_unsolved():
                        self.maybe_set_bulb(i, j)
                        self.apply_methods(level_to_use)
                        continue
                    # continue for this branch because we already know the cell
                    try_tp_bulb = ThoughtProcess(self.board)
                    try_tp_bulb.maybe_set_bulb(i, j)
                    try_tp_bulb.apply_methods(level_to_use)
                    if not try_tp_bulb.check_unsolved():
                        self.maybe_set_dot(i, j)
                        self.apply_methods(level_to_use)
            if np.all(self.board == old_board):
                break
