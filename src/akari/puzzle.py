import argparse
import csv
import math
import sys
import timeit
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import zip_longest
from typing import Literal, cast

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


def new_blank_board(rows: int, cols: int) -> np.ndarray:
    board = np.zeros((rows + 2, cols + 2), dtype="str")
    board[:, :] = "-"
    board[1:-1, 1:-1] = "."
    return board


def resize_board(
    board: np.ndarray,
    delta_top: int,
    delta_left: int,
    delta_bottom: int,
    delta_right: int,
) -> np.ndarray:
    new_board = new_blank_board(
        board.shape[0] + delta_top + delta_bottom - 2,
        board.shape[1] + delta_left + delta_right - 2,
    )
    new_board[
        1 + max(0, delta_top) : -1 - max(0, delta_bottom),
        1 + max(0, delta_left) : -1 - max(0, delta_right),
    ] = board[
        1 + max(0, -delta_top) : -1 - max(0, -delta_bottom),
        1 + max(0, -delta_left) : -1 - max(0, -delta_right),
    ]
    return new_board


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


def clear_board(board: np.ndarray) -> np.ndarray:
    """
    Removes annotations from board
    """
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if cast(str, board[i, j]) in "+_|x#":
                board[i, j] = "."
    return board


def intersect_boards(board1: np.ndarray, board2: np.ndarray) -> np.ndarray:
    """
    Intersection of 2 boards, that is, the cells that agree; other cells are free
    """
    board1 = board1.copy()
    for i in range(1, board1.shape[0] - 1):
        for j in range(1, board1.shape[1] - 1):
            if board1[i, j] == board2[i, j]:
                pass  # no change needed
            elif board1[i, j] in "+_|x" and board2[i, j] in "+_|x":
                board1[i, j] = "+"  # if both boards are basically dots but not the same
                # symbol, do a dot
            else:
                board1[i, j] = "."
    return board1


def subtract_boards(board1: np.ndarray, board2: np.ndarray) -> np.ndarray:
    """
    Subtraction of 2 boards, that is, board1 cells that disagree with board2
    """
    board1 = board1.copy()
    for i in range(1, board1.shape[0] - 1):
        for j in range(1, board1.shape[1] - 1):
            if board1[i, j] == board2[i, j]:
                board1[i, j] = "."
            elif board1[i, j] in "+_|x" and board2[i, j] in "+_|x":
                board1[i, j] = "."  # both basically dots
    return board1


def is_partially_correct_based_on_other_board(
    board_test: np.ndarray, board_reference: np.ndarray
) -> bool:
    """
    Returns true if board_test only has bulbs or dots where board_reference does
    """
    for i in range(1, board_test.shape[0] - 1):
        for j in range(1, board_test.shape[1] - 1):
            if board_test[i, j] == "#" and board_reference[i, j] != "#":
                return False
            elif board_test[i, j] in "+_|x" and board_reference[i, j] not in "+_|x":
                return False
    return True


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
    tp.illuminate(-1, -1, ".")
    board[:] = tp.board[:]
    return tp.lit_bulb_pairs, tp.board


def count_free_near_number(board: np.ndarray, i: int, j: int) -> int:
    return sum(board[i + di, j + dj] == "." for (di, dj) in ORTHO_DIRS)


def count_missing_bulbs_near_number(board: np.ndarray, i: int, j: int) -> int:
    n_bulbs = sum(board[i + di, j + dj] == "#" for (di, dj) in ORTHO_DIRS)
    return int(board[i, j]) - n_bulbs


COSTS = {
    "illuminate": 1.0,
    "mark_bulbs_around_dotted_numbers": 1.0,
    "mark_dots_around_full_numbers": 1.0,
    "fill_holes": 1.2,
    "mark_unique_bulbs_for_dot_cells": 1.2,
    "mark_dots_at_corners": 2.0,
    "analyze_diagonally_adjacent_numbers": 2.0,
    "mark_bulbs_and_dots_at_shared_lanes": 3.0,
    "mark_dots_beyond_corners": 2.8,
}

HINT_MESSAGES = {
    "illuminate": "Lit by bulbs",
    "mark_bulbs_around_dotted_numbers": "Number that has enough dots",
    "mark_dots_around_full_numbers": "Number that has enough bulbs",
    "fill_holes": "Fill hole",
    "mark_unique_bulbs_for_dot_cells": "Unique bulb position",
    "mark_dots_at_corners": "Mark dots at corners",
    "analyze_diagonally_adjacent_numbers": "Diagonally adjacent numbers",
    "mark_bulbs_and_dots_at_shared_lanes": "Shared lanes",
    "mark_dots_beyond_corners": "Cell cannot be bulb",
    "guess_and_check": "Guess and check",
    "check_unsolved": "Board contradictory",
}

LEVEL_NAMES = [
    "Solver Off",
    "Illuminate",
    "+Count At Numbers",
    "+Light Shadows",
    "+Mark Dots At Corners",
    "+Analyze Diagonal Numbers",
    "+Shared Lanes",
    "Brute Force",
]


class Step:
    """
    Records steps in simulation

    Members
    -------
    signal : tuple[int, int, str]
        the information triggering this step. In general, it is coordinates and a char.
        For init, the char is "." and each method should give a coordinate reasonably
        near whatever is being activated but it might point to a number rather than
        white cell.
    method : str
    cost : float
    outputs : list[int, int]
        coordinates being acted on

    Examples of inputs:
    illuminate: coordinates of bulb
    mark_bulbs_around_dotted_numbers: all coordinates around bulb containing dots
    """

    def __init__(
        self, signal: tuple[int, int, str], method: str, cost: float | None = None
    ) -> None:
        self.signal = signal
        self.method = method
        self.outputs = []
        if cost is not None:
            self.cost = cost  # cost must be provided for guess and check
        else:
            self.cost = COSTS[method]

    def __repr__(self) -> str:
        return f"Step({self.signal}, {self.method}, {self.outputs})"


@dataclass
class SharedLanesPair:
    """
    Data structure for tracking a pair of numbers that share lanes or touch.

    Members
    -------
    A : tuple[int, int]
    B : tuple[int, int]
        cell coordinates
    shared_pairs: list[tuple[int, int, int, int]]
        a list of pairs of cells adjacent to A and B. Coordinates are i1, j1, i2, j2.
        It is required that i1 <= i2 and j1 <= j2 and that a row or column be shared.
    nonshared_cells: list[tuple[int, int]]
        list of cells adjacent to A and B that are not "seen" by the other one.
    touching: Literal[False] | tuple[int, int]
        if A and B are not touching; otherwise, the coordinates of the cell between
        them.
    """

    A: tuple[int, int]
    B: tuple[int, int]
    shared_pairs: list[tuple[int, int, int, int]]
    nonshared_cells: list[tuple[int, int]]
    touching: Literal[False] | tuple[int, int]


class SharedLanesBot:
    """
    SharedLanesBot is a class for applying a related set of methods that are handled
    most simply and quickly if the board topology is pre-analyzed.

    The general idea is that two number cells have shared lanes if they have free cells
    that compete for multiple lanes. In this example,
    --------
    -.1X.X.-
    -.Y.Y1.-
    --------
    X and Y are the free cells that see each other. A lane is a line of sight. There are
    relevant lanes for X in the first row and Y in the second.

    If two numbers, A and B, have S > 1 shared lanes, they need at least A+B-S free
    cells that are not in those lanes. We can draw conclusions as follows.
    1. If only A+B-S non-shared free cells are available,
      a. they must all be filled with bulbs.
      b. all cells in the shared lanes must be dotted except for those adjacent to A and
         B.
    2. If only A+B-S+1 non-shared free cells are available, dot any cells that see 2
       non-shared cells. (This method is not yet implemented.)

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
    because the 3 and 2 compete over 3 lanes, need 5 bulbs total, and can hide 2
    bulbs from each other. Moreover, we can also get,
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
    that is, where the two numbers are off by one row or column. We can also
    account for two numbers in a line but that have one of their lanes blocked by a
    black square or a bulb.

    -----
    -...-
    -.3.-
    -...-
    -.2.-
    -...-
    -----
    is a fun special case because the 3 and 2 share the cell between them. If two
    cells are in a line and have one free cell between them, that cell is not
    considered in a shared lane and instead counts as two free cells. Thus, rule 1 does
    not fire; it would fire if the 2 were another 3.
    """

    def __init__(self, thought_process: "ThoughtProcess") -> None:
        self.thought_process = thought_process
        self.shared_lanes: defaultdict[tuple[int, int], list[SharedLanesPair]] = (
            defaultdict(list)
        )
        self.all_shared_lanes = []
        self._trace_shared_lanes(0)  # trace down
        self._trace_shared_lanes(1)  # trace right
        self._trace_diagonal()

    def _trace_shared_lanes(self, direction: int) -> None:
        # v is a vector in the direction that we're tracing and w is a vector
        # perpendicular to that
        # A supposed point B is traced out from cell A either straight down or right
        if direction == 0:
            vi = 1
            vj = 0
            wi = 0
            wj = 1
        else:
            vi = 0
            vj = 1
            wi = 1
            wj = 0
        board = self.thought_process.board
        for iA in range(1, np.size(board, 0) - 1):
            for jA in range(1, np.size(board, 1) - 1):
                if board[iA, jA] in "123":
                    # if direction == 0,
                    # tracer_C is in column jA - 1
                    # tracer_D is in column jA
                    # tracer_E is in column jA + 1
                    # other wise, the tracers are in rows iA - 1, iA, and iA + 1
                    if direction == 0:
                        cells_after_A = self.thought_process.it_down(iA, jA)
                    else:
                        cells_after_A = self.thought_process.it_right(iA, jA)
                    self._trace_shared_lanes_at_cell(
                        board, vi, vj, wi, wj, iA, jA, cells_after_A
                    )

    def _trace_shared_lanes_at_cell(
        self,
        board: np.ndarray,
        vi: int,
        vj: int,
        wi: int,
        wj: int,
        iA: int,
        jA: int,
        cells_after_A: Iterator[tuple[int, int]],
    ) -> None:
        tracer_C = board[iA - wi, jA - wj] == "."
        tracer_D = board[iA + vi, jA + vj] == "."
        tracer_E = board[iA + wi, jA + wj] == "."
        for iB, jB in cells_after_A:
            new_tracer_C = tracer_C and board[iB - wi, jB - wj] in ".+_|x"
            new_tracer_D = tracer_D and board[iB, jB] in ".+_|x"
            new_tracer_E = tracer_E and board[iB + wi, jB + wj] in ".+_|x"
            if (
                tracer_C
                and new_tracer_D
                and board[iB - wi, jB - wj] in "123"
                and board[iB - vi - wi, jB - vj - wj] == "."
                and board[iB, jB] == "."
                and iB + jB > iA + jA + 1
            ):
                self._add_adjacent_lane(
                    board, iA, jA, iB - wi, jB - wj, vi, vj, -wi, -wj
                )
            if (
                tracer_E
                and new_tracer_D
                and board[iB + wi, jB + wj] in "123"
                and board[iB - vi + wi, jB - vj + wj] == "."
                and board[iB, jB] == "."
                and iB + jB > iA + jA + 1
            ):
                self._add_adjacent_lane(board, iA, jA, iB + wi, jB + wj, vi, vj, wi, wj)
            if board[iB, jB] in "123":
                fit_C = new_tracer_C and board[iB - wi, jA - wj] == "."
                fit_D = tracer_D and board[iB - vi, jA + vj] == "."
                fit_E = new_tracer_E and board[iB + wi, jA + wj] == "."
                n_for_same = fit_C + fit_D + fit_E
                if n_for_same >= 2:
                    self._add_same_lane(
                        board,
                        iA,
                        jA,
                        iB,
                        jB,
                        vi,
                        vj,
                        wi,
                        wj,
                        fit_C,
                        fit_D,
                        fit_E,
                    )
            tracer_C = new_tracer_C
            tracer_D = new_tracer_D
            tracer_E = new_tracer_E
            if tracer_C + tracer_D + tracer_E < 2:
                break

    def _add_adjacent_lane(
        self,
        board: np.ndarray,
        iA: int,
        jA: int,
        iB: int,
        jB: int,
        vi: int,
        vj: int,
        di: int,
        dj: int,
    ) -> None:
        sl = SharedLanesPair(
            A=(iA, jA),
            B=(iB, jB),
            shared_pairs=[
                (iA + vi, jA + vj, iB - di, jB - dj),
                (iA + di, jA + dj, iB - vi, jB - vj),
            ],
            nonshared_cells=[
                (iA - vi, jA - vj),
                (iA - di, jA - dj),
                (iB + vi, jB + vj),
                (iB + di, jB + dj),
            ],
            touching=False,
        )
        self.all_shared_lanes.append(sl)
        for cell in sl.nonshared_cells:
            self.shared_lanes[cell].append(sl)

    def _add_same_lane(
        self,
        board: np.ndarray,
        iA: int,
        jA: int,
        iB: int,
        jB: int,
        vi: int,
        vj: int,
        wi: int,
        wj: int,
        fit_C: bool,  # noqa: FBT001
        fit_D: bool,  # noqa: FBT001
        fit_E: bool,  # noqa: FBT001
    ) -> None:
        sl = SharedLanesPair(
            A=(iA, jA),
            B=(iB, jB),
            shared_pairs=[],
            nonshared_cells=[(iA - vi, jA - vj), (iB + vi, jB + vj)],
            touching=False,
        )
        if fit_C:
            sl.shared_pairs.append((iA - wi, jA - wj, iB - wi, jB - wj))
        else:
            sl.nonshared_cells.append((iA - wi, jA - wj))
            sl.nonshared_cells.append((iB - wi, jB - wj))
        if fit_D:
            if iB + jB != iA + jA + 2:
                sl.shared_pairs.append((iA + vi, jA + vj, iB - vi, jB - vj))
            else:
                sl.touching = (iA + vi, jA + vj)
        else:
            sl.nonshared_cells.append((iA + vi, jA + vj))
            sl.nonshared_cells.append((iB - vi, jB - vj))
        if fit_E:
            sl.shared_pairs.append((iA + wi, jA + wj, iB + wi, jB + wj))
        else:
            sl.nonshared_cells.append((iA + wi, jA + wj))
            sl.nonshared_cells.append((iB + wi, jB + wj))
        self.all_shared_lanes.append(sl)
        for di, dj in ORTHO_DIRS:
            self.shared_lanes[iA + di, jA + dj].append(sl)
            self.shared_lanes[iB + di, jB + dj].append(sl)

    def _trace_diagonal(self) -> None:
        board = self.thought_process.board
        for iA in range(1, np.size(board, 0) - 3):
            for jA in range(1, np.size(board, 1) - 1):
                if board[iA, jA] in "123":
                    if jA > 2:
                        self._trace_diagonal_at_cell(iA, jA, -1)
                    if jA < np.size(board, 1) - 3:
                        self._trace_diagonal_at_cell(iA, jA, 1)

    def _trace_diagonal_at_cell(self, iA: int, jA: int, dj: int) -> None:
        board = self.thought_process.board
        if (
            board[iA + 2, jA + 2 * dj] in "123"
            and board[iA + 1, jA + dj] in ".+"
            and board[iA + 1, jA] == "."
            and board[iA, jA + dj] == "."
            and board[iA + 1, jA + 2 * dj] == "."
            and board[iA + 2, jA + dj] == "."
        ):
            sl = SharedLanesPair(
                A=(iA, jA),
                B=(iA + 2, jA + 2 * dj),
                shared_pairs=[
                    (iA + 1, min(jA, jA + 2 * dj), iA + 1, max(jA, jA + 2 * dj)),
                    (iA, jA + dj, iA + 2, jA + dj),
                ],
                nonshared_cells=[
                    (iA - 1, jA),
                    (iA, jA - dj),
                    (iA + 3, jA + 2 * dj),
                    (iA + 2, jA + 3 * dj),
                ],
                touching=False,
            )
            self.all_shared_lanes.append(sl)
            for cell in sl.nonshared_cells:
                self.shared_lanes[cell].append(sl)

    def mark_bulbs_and_dots_at_shared_lanes(  # noqa: C901 TODO split up function
        self, i: int, j: int, mark: str
    ) -> None:
        board = self.thought_process.board
        if mark == ".":
            shared_lanes_to_check = self.all_shared_lanes
        else:
            shared_lanes_to_check = self.shared_lanes[i, j]
        for sl in shared_lanes_to_check:
            active_lanes = []
            inactive_lanes = []
            for sp in sl.shared_pairs:
                i1, j1, i2, j2 = sp
                if board[i1, j1] == "." and board[i2, j2] == ".":
                    active_lanes.append(sp)
                else:
                    inactive_lanes.append(sp)
            balance = (
                count_missing_bulbs_near_number(board, sl.A[0], sl.A[1])
                + count_missing_bulbs_near_number(board, sl.B[0], sl.B[1])
                - (
                    count_free_near_number(board, sl.A[0], sl.A[1])
                    + count_free_near_number(board, sl.B[0], sl.B[1])
                    - len(active_lanes)
                )
            )
            if mark == ".":
                signal = (sl.A[0], sl.A[1], ".")
            else:
                signal = (i, j, mark)
            step = Step(signal, "mark_bulbs_and_dots_at_shared_lanes")
            if len(active_lanes) > 1 and balance == 0:
                for cell in sl.nonshared_cells:
                    self.thought_process.maybe_set_bulb(cell[0], cell[1], step)
                for sp in inactive_lanes:
                    i1, j1, i2, j2 = sp
                    self.thought_process.maybe_set_bulb(i1, j1, step)
                    self.thought_process.maybe_set_bulb(i2, j2, step)
                if sl.touching:
                    self.thought_process.maybe_set_bulb(
                        sl.touching[0], sl.touching[1], step
                    )
                self._dot_shared_lanes(board, sl, step)
            if sl.touching and balance == -1:
                self.thought_process.maybe_set_bulb(
                    sl.touching[0], sl.touching[1], step
                )

    def _dot_shared_lanes(
        self, board: np.ndarray, sl: SharedLanesPair, step: Step
    ) -> None:
        """
        Dots cells along shared lanes other than the shared cells

        Requires that i1 <= i2 and j1 <= j2
        """
        for i1, j1, i2, j2 in sl.shared_pairs:
            if j1 == j2:
                cells_after_1 = self.thought_process.it_down(i1, j1)
                cells_before_1 = self.thought_process.it_up(i1, j1)
            else:
                cells_after_1 = self.thought_process.it_right(i1, j1)
                cells_before_1 = self.thought_process.it_left(i1, j1)
            for ix, jx in cells_after_1:
                if board[ix, jx] in "-01234":
                    break
                elif (ix, jx) != (i2, j2):
                    self.thought_process.maybe_set_dot(ix, jx, step)
            for ix, jx in cells_before_1:
                if board[ix, jx] in "-01234":
                    break
                self.thought_process.maybe_set_dot(ix, jx, step)


def check_number(board: np.ndarray) -> list[tuple[int, int]]:
    """
    Checks numbered spaces to see if they have the correct number of bulbs.

    Returns a list of tuples of coordinates of numbered spaces that touch the wrong
    number of bulbs.
    """
    wrong_bulbs = []
    for i in range(1, np.size(board, 0) - 1):
        for j in range(1, np.size(board, 1) - 1):
            if board[i, j] in "01234":
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

    new_mark: list[deque[tuple[int, int, str]]], a list of queues of new dots "+"
        or bulbs "#", which will later trigger methods to run. Each queue is for a
        different level of method.

    lit_bulb_pairs: list[tuple[int, int, int, int]], coordinates of pairs of bulbs that
    see each other

    unilluminatable_cells: list[tuple[int, int]], cells that cannot be lit

    wrong_numbers: list[tuple[int, int]], numbers that have infeasible numbers of bulbs
    (either too high or too many dots to ever have enough) The last 3 items are the
    error lists.
    """

    def __init__(self, board: np.ndarray) -> None:
        self.board = board.copy()
        self.new_mark = [deque()]
        # I do not love this init for new_mark but we do not already know the level,
        # ThoughtProcess is almost always followed by apply_methods, and this makes all
        # tests pass.
        self.lit_bulb_pairs = []
        self.unilluminatable_cells = []
        self.wrong_numbers = set()
        if not hasattr(self, "shared_lanes_bot"):
            self.shared_lanes_bot = SharedLanesBot(self)
        self.solution_steps = []
        self.cost = 0.0

    def __copy__(self) -> "ThoughtProcess":
        cls = self.__class__
        new = cls.__new__(cls)
        new.shared_lanes_bot = self.shared_lanes_bot
        new.__init__(self.board)
        return new

    def maybe_set_bulb(self, i: int, j: int, step: Step) -> bool:
        """
        Confirms that board[i, j] is free; if so, board[i, j] = "#" and runs updates

        Runs illuminate and updates the queue.
        """
        if self.board[i, j] == ".":
            self.board[i, j] = "#"
            self.new_mark[0].append((i, j, "#"))
            self.find_wrong_numbers(i, j)
            self.update_solution_steps(i, j, "#", step)
            return True
        else:
            return False

    def maybe_set_dot(self, i: int, j: int, step: Step) -> bool:
        """
        Confirms that board[i, j] is free; if so, board[i, j] = "." and runs updates

        Updates the queue.
        """
        if self.board[i, j] == ".":
            self.board[i, j] = "+"
            self.new_mark[0].append((i, j, "+"))
            self.find_wrong_numbers(i, j)
            self.find_unilluminatable_cells(i, j)
            self.update_solution_steps(i, j, "+", step)
            return True
        else:
            return False

    def update_solution_steps(
        self, i: int, j: int, mark: str, step: Step | None
    ) -> None:
        if step is None:
            return
        elif len(self.solution_steps) and self.solution_steps[-1] is step:
            pass
        else:
            self.solution_steps.append(step)
        step.outputs.append((i, j, mark))
        self.cost += step.cost

    def all_interior_ij(self) -> list[tuple[int, int]]:
        return [
            (i, j)
            for i in range(1, self.board.shape[0] - 1)
            for j in range(1, self.board.shape[1] - 1)
        ]

    def it_up(self, i: int, j: int) -> Iterator[tuple[int, int]]:
        """
        Iterator of cells above (i, j)
        """
        return zip_longest(range(i - 1, 0, -1), [], fillvalue=j)

    def it_left(self, i: int, j: int) -> Iterator[tuple[int, int]]:
        """
        Iterator of cells to the left of (i, j)
        """
        return zip_longest([], range(j - 1, 0, -1), fillvalue=i)

    def it_down(self, i: int, j: int) -> Iterator[tuple[int, int]]:
        """
        Iterator of cells below (i, j)
        """
        return zip_longest(range(i + 1, np.size(self.board, 0) - 1), [], fillvalue=j)

    def it_right(self, i: int, j: int) -> Iterator[tuple[int, int]]:
        """
        Iterator of cells to the right of (i, j)
        """
        return zip_longest([], range(j + 1, np.size(self.board, 1) - 1), fillvalue=i)

    def line_of_sight_iters(
        self, i: int, j: int
    ) -> Iterator[Iterator[tuple[int, int]]]:
        """
        Returns an iterator of it_up, it_left, it_down, and it_right

        This is useful because we often want to break partway through searching in a
        given direction, but then look in the next direction.
        """
        return (
            it(i, j) for it in [self.it_up, self.it_left, self.it_down, self.it_right]
        )

    def line_of_sight(self, i: int, j: int) -> list[tuple[int, int]]:
        """
        Returns an iterator of all cells in line of sight to a cell at i, j
        """
        line_of_sight_cells = []
        for it in self.line_of_sight_iters(i, j):
            for i, j in it:
                if self.board[i, j] in "01234-":
                    break
                line_of_sight_cells.append((i, j))
        return line_of_sight_cells

    def apply_methods(  # noqa: C901
        self,
        max_level: int,
        *,
        calculate_difficulty: bool = False,
        find_hint: bool = False,
        budget: float = math.inf,
        max_cost: float = math.inf,
    ) -> None:
        # The complexity here is fine
        """
        Applies the various logical methods as set by the level.

        If the queue is empty, it first, scans everything; otherwise, it just does the
        new_mark queue.
        """
        starting_solution_steps = len(self.solution_steps)
        if max_level < 1:
            return
        if len(self.new_mark) < max_level:
            # new_mark is initially just a list of one deque; this fills in enough
            # deques for all levels being handled. The indexing is funny because we're
            # counting for deques for levels 2 through max_level
            for _i in range(2, max_level + 1):
                self.new_mark.append(deque())
        if not any(self.new_mark):
            self.new_mark[0].append((-1, -1, "."))
            for i, j in self.all_interior_ij():
                self.find_wrong_numbers_at_cell(i, j)
                self.check_this_cell_unilluminatable(i, j)
        while any(self.new_mark):
            mark_level, queue = next(
                (i, q) for i, q in enumerate(self.new_mark, start=1) if q
            )
            i, j, mark = queue.popleft()
            if mark_level == 1:
                self.illuminate(i, j, mark)
            elif mark_level == 2:
                self.mark_bulbs_around_dotted_numbers(i, j, mark)
                self.mark_dots_around_full_numbers(i, j, mark)
            elif mark_level == 3:
                self.fill_holes(i, j, mark)
                self.mark_unique_bulbs_for_dot_cells(i, j, mark)
            elif mark_level == 4:
                self.mark_dots_at_corners(i, j, mark)
            elif mark_level == 5:
                self.analyze_diagonally_adjacent_numbers(i, j, mark)
            elif mark_level == 6:
                self.shared_lanes_bot.mark_bulbs_and_dots_at_shared_lanes(i, j, mark)
                self.mark_dots_beyond_corners(i, j, mark)
            elif max_level == 9 and not any(self.new_mark):
                # guess and check is orders of magnitude more expensive than other
                # methods and should only be called if all else has been tried.
                if calculate_difficulty or find_hint:
                    self.guess_and_check_thrifty(max_level, max_cost=max_cost)
                else:
                    self.guess_and_check(max_level)
            if find_hint and len(self.solution_steps) > starting_solution_steps:
                self.hint = self.solution_steps[starting_solution_steps]
                return
            if (
                not self.new_mark[0]  # force illuminate before ending
                and not self.check_unsolved()
            ) or self.cost > budget:
                break
            if mark_level < max_level:
                self.new_mark[mark_level].append((i, j, mark))
                # You would think that self.new_mark[mark_level] should have a + 1 to
                # push the mark onto the next level. However, the first queue is at
                # level 1, not 0.

    def illuminate(self, i: int, j: int, mark: str) -> None:
        """
        Illuminate the bulb at i, j
        """
        if mark == ".":
            for i, j in self.all_interior_ij():
                self._illuminate_one(i, j, mark)
        else:
            self._illuminate_one(i, j, mark)

    def _illuminate_one(self, i: int, j: int, mark: str) -> None:
        """
        helper for illuminate
        """
        step = Step((i, j, mark), "illuminate")
        fill_chars = ["|", "_", "|", "_"]
        if self.board[i, j] == "#":
            for it, fill_char in zip(
                self.line_of_sight_iters(i, j), fill_chars, strict=True
            ):
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
                        self.maybe_set_dot(i1, j1, step=step)
                        self.board[i1, j1] = fill_char  # this assignment is cosmetic,
                        # making the light rays but not functionally changing state.

    def mark_bulbs_around_dotted_numbers(self, i: int, j: int, mark: str) -> None:
        if mark == ".":
            cells_to_check = self.all_interior_ij()
        elif mark in ".+_|x":
            cells_to_check = [(i + di1, j + dj1) for di1, dj1 in ORTHO_DIRS]
        else:
            return
        for i_number, j_number in cells_to_check:
            if self.board[i_number, j_number] in "01234":
                if count_free_near_number(
                    self.board, i_number, j_number
                ) == count_missing_bulbs_near_number(self.board, i_number, j_number):
                    if mark == ".":
                        signal = (i_number, j_number, mark)
                    else:
                        signal = i, j, mark
                    step = Step(signal, "mark_bulbs_around_dotted_numbers")
                    for di, dj in ORTHO_DIRS:
                        self.maybe_set_bulb(
                            i_number + di,
                            j_number + dj,
                            step=step,
                        )

    def mark_dots_around_full_numbers(self, i: int, j: int, mark: str) -> None:
        if mark == ".":
            cells_to_check = self.all_interior_ij()
        elif mark in "#":
            cells_to_check = [(i + di1, j + dj1) for di1, dj1 in ORTHO_DIRS]
        else:
            return
        for i_number, j_number in cells_to_check:
            if self.board[i_number, j_number] in "01234":
                if count_missing_bulbs_near_number(self.board, i_number, j_number) == 0:
                    if mark == ".":
                        signal = (i_number, j_number, mark)
                    else:
                        signal = i, j, mark
                    step = Step(signal, "mark_dots_around_full_numbers")
                    for di2, dj2 in ORTHO_DIRS:
                        self.maybe_set_dot(i_number + di2, j_number + dj2, step=step)

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
            for i, j in self.all_interior_ij():
                self._fill_holes_cell(i, j, (i, j, "."))
        elif mark == "+":
            signal = (i0, j0, mark)
            self._fill_holes_cell(i0, j0, signal)
            for i, j in self.line_of_sight(i0, j0):
                self._fill_holes_cell(i, j, signal)

    def _fill_holes_cell(self, i: int, j: int, signal: tuple[int, int, str]) -> None:
        step = Step(signal, "fill_holes")
        if self.board[i, j] == ".":
            is_hole = True  # presume a hole
            for it in self.line_of_sight_iters(i, j):
                for i1, j1 in it:
                    if self.board[i1, j1] == ".":
                        is_hole = False
                        break
                    elif self.board[i1, j1] in "01234-":
                        break
                if not is_hole:
                    break
            if is_hole:
                self.maybe_set_bulb(i, j, step)

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
            for i, j in self.all_interior_ij():
                self._mark_unique_bulbs_for_dot_cells_at_cell(i, j, (i, j, "."))
        elif mark == "+":
            signal = (i0, j0, mark)
            self._mark_unique_bulbs_for_dot_cells_at_cell(i0, j0, signal)
            for i, j in self.line_of_sight(i0, j0):
                self._mark_unique_bulbs_for_dot_cells_at_cell(i, j, signal)

    def _mark_unique_bulbs_for_dot_cells_at_cell(
        self, i: int, j: int, signal: tuple[int, int, str]
    ) -> None:
        if self.board[i, j] == "+":
            step = Step(signal, "mark_unique_bulbs_for_dot_cells")
            sees_free = False
            sees_multiple_free = False
            sees_bulb = False
            free_i = free_j = -1
            for it in self.line_of_sight_iters(i, j):
                for i1, j1 in it:
                    if self.board[i1, j1] == "#":
                        sees_bulb = True
                        break
                    elif self.board[i1, j1] == ".":
                        if sees_free:
                            sees_multiple_free = True
                            break
                        else:
                            sees_free = True
                            free_i = i1
                            free_j = j1
                    elif self.board[i1, j1] in "01234-":
                        break
                if sees_multiple_free or sees_bulb:
                    break
            if not sees_bulb and sees_free and not sees_multiple_free:
                self.maybe_set_bulb(free_i, free_j, step)

    def mark_dots_at_corners(self, i: int, j: int, mark: str) -> None:
        """
        Marks dots at free cells diagonal to numbers if a bulb in that cell would make
        it impossible for the number to get enough bulbs.

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
            for i, j in self.all_interior_ij():
                self._mark_dots_at_corners_at_cell(i, j, mark)
        elif mark == "+":
            for di1, dj1 in ORTHO_DIRS:
                i_corner = i + di1
                j_corner = j + dj1
                self._mark_dots_at_corners_at_cell(i_corner, j_corner, mark)

    def _mark_dots_at_corners_at_cell(self, i: int, j: int, mark: str) -> None:
        if self.board[i, j] in "01234":
            n_free = sum(
                self.board[i + di2, j + dj2] == "." for (di2, dj2) in ORTHO_DIRS
            )
            n_bulbs_already = sum(
                self.board[i + di, j + dj] == "#" for (di, dj) in ORTHO_DIRS
            )
            if n_free + n_bulbs_already == int(self.board[i, j]) + 1:
                step = Step((i, j, mark), "mark_dots_at_corners")
                for di, dj in DIAG_DIRS:
                    if self.board[i + di, j] == "." and self.board[i, j + dj] == ".":
                        self.maybe_set_dot(i + di, j + dj, step)

    def mark_dots_beyond_corners(self, i0: int, j0: int, mark: str) -> None:
        """
        Marks cells that must not be bulbs for a dotted cell to be illuminated. For
        example, if we have,
        -----
        -++.-
        -+..-
        -..A-
        -----
        then A cannot be a bulb because the top left + would become unilluminatable.

        This is like a mix of mark_unique_bulbs_for_dot_cells and mark_dots_at_corners.
        """
        if mark == ".":
            for i0, j0 in self.all_interior_ij():
                self._mark_dots_beyond_corners_at_cell(i0, j0, (i0, j0, mark))
        elif mark == "+":
            self._mark_dots_beyond_corners_at_cell(i0, j0, (i0, j0, mark))
            for i, j in self.line_of_sight(i0, j0):
                self._mark_dots_beyond_corners_at_cell(i, j, (i0, j0, mark))

    def _mark_dots_beyond_corners_at_cell(
        self, i: int, j: int, mark: tuple[int, int, str]
    ) -> None:
        if self.board[i, j] == "+":
            cells = []
            for it in self.line_of_sight_iters(i, j):
                seen_in_this_direction = False
                for i1, j1 in it:
                    if self.board[i1, j1] == ".":
                        if seen_in_this_direction or len(cells) > 1:
                            return
                        else:
                            cells.append((i1, j1))
                            seen_in_this_direction = True
                    elif self.board[i1, j1] in "01234-":
                        break
            self._mark_dots_beyond_corners_process_cells(i, j, mark, cells)

    def _mark_dots_beyond_corners_process_cells(
        self, i: int, j: int, mark: tuple[int, int, str], cells: list[tuple[int, int]]
    ) -> None:
        if len(cells) == 2:
            (iA, jA), (iB, jB) = cells
            if iA == iB or jA == jB:
                return
            if iB == i:
                (iA, jA), (iB, jB) = (iB, jB), (iA, jA)
            if self._mark_dots_beyond_corners_check_line_of_sight(i, j, iA, jA, iB, jB):
                self.maybe_set_dot(iB, jA, Step(mark, "mark_dots_beyond_corners"))

    def _mark_dots_beyond_corners_check_line_of_sight(
        self, i: int, j: int, iA: int, jA: int, iB: int, jB: int
    ) -> bool:
        if iB > i:
            it = self.it_down(iA, jA)
        else:
            it = self.it_up(iA, jA)
        for ix, jx in it:
            if self.board[ix, jx] in "01234-":
                return False
            elif ix == iB:
                break
        if jA > j:
            it = self.it_right(iB, jB)
        else:
            it = self.it_left(iB, jB)
        for ix, jx in it:
            if self.board[ix, jx] in "01234-":
                return False
            elif jx == jA:
                break
        return True

    def analyze_diagonally_adjacent_numbers(self, i: int, j: int, mark: str) -> None:
        """
        Adds dots and bulbs for certain diagonally adjacent numbers sharing 2 free
        spaces.

        When one number or the other "knows" that one of the 2 free spaces must
        have a bulb and the other must not, we can fill bulbs or dots around the other
        number. More generally, if one number needs the two free cells to have at least
        1 bulb, and the other needs the two free cells to have at least 1 dot, we can
        add a dot or bulb to each non-shared space adjacent to those numbers.

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

        ------
        -....-
        -+2..-
        -..1.-
        -....-
        becomes
        ------
        -_#__-
        -+2..-
        -..1+-
        -..+.-

        These examples do not show what happens if dots or bulbs are already placed but
        the method accounts for those possibilities, also.
        """
        if mark == ".":
            cells_to_check = self.all_interior_ij()
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
                        if mark == ".":
                            signal = (iA, jA, ".")
                        else:
                            signal = (i, j, mark)
                        if missing_A == 1 and missing_B + 1 == free_B:
                            self._analyze_diagonally_adjacent_numbers_update_board(
                                iA, jA, iB, jB, signal
                            )
                        elif missing_B == 1 and missing_A + 1 == free_A:
                            self._analyze_diagonally_adjacent_numbers_update_board(
                                iB, jB, iA, jA, signal
                            )

    def _analyze_diagonally_adjacent_numbers_update_board(
        self, iC: int, jC: int, iD: int, jD: int, signal: tuple[int, int, str]
    ) -> None:
        # point C has 1 missing bulb and point D has one free space more
        di = iD - iC
        dj = jD - jC
        step = Step(signal, "analyze_diagonally_adjacent_numbers")
        self.maybe_set_dot(iC - di, jC, step)
        self.maybe_set_dot(iC, jC - dj, step)
        self.maybe_set_bulb(iD + di, jD, step)
        self.maybe_set_bulb(iD, jD + dj, step)

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
        for it in self.line_of_sight_iters(i, j):
            for i1, j1 in it:
                if self.board[i1, j1] in "01234-":
                    break
                self.check_this_cell_unilluminatable(i1, j1)

    def check_this_cell_unilluminatable(self, i: int, j: int) -> None:
        if self.board[i, j] == "+":
            is_unilluminatable = True
            for it in self.line_of_sight_iters(i, j):
                for i2, j2 in it:
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
        gacbc = 1.0  # guess_and_check base cost
        level_to_use = min(level, 8)
        for i, j in zip(
            *np.asarray(self.board == ".", dtype=int).nonzero(), strict=True
        ):
            i = int(i)
            j = int(j)
            if self.board[i, j] == ".":
                try_tp_dot = self.__copy__()
                try_tp_dot.maybe_set_dot(
                    i, j, Step((i, j, "?"), "guess_and_check_guess", cost=gacbc)
                )
                try_tp_dot.apply_methods(level_to_use)
                if not try_tp_dot.check_unsolved():
                    cost = sum(step.cost for step in try_tp_dot.solution_steps)
                    self.maybe_set_bulb(
                        i, j, Step((i, j, "?"), "guess_and_check", cost=cost)
                    )
                    # return
                try_tp_bulb = self.__copy__()
                try_tp_bulb.maybe_set_bulb(
                    i, j, Step((i, j, "?"), "guess_and_check_guess", cost=gacbc)
                )
                try_tp_bulb.apply_methods(level_to_use)
                if not try_tp_bulb.check_unsolved():
                    cost = sum(step.cost for step in try_tp_bulb.solution_steps)
                    self.maybe_set_dot(
                        i, j, Step((i, j, "?"), "guess_and_check", cost=cost)
                    )
                    # return
                dot_marks = {o for s in try_tp_dot.solution_steps for o in s.outputs}
                bulb_marks = {o for s in try_tp_bulb.solution_steps for o in s.outputs}
                invariant = dot_marks.intersection(bulb_marks)
                if invariant:
                    cost = sum(step.cost for step in try_tp_dot.solution_steps) + sum(
                        step.cost for step in try_tp_bulb.solution_steps
                    )
                    for i, j, mark in invariant:
                        step = Step((i, j, "?"), "invariant", cost=cost)
                        if mark == "#":
                            self.maybe_set_bulb(i, j, step)
                    for i, j, mark in invariant:
                        step = Step((i, j, "?"), "invariant", cost=cost)
                        if mark == "+":
                            self.maybe_set_dot(i, j, step)
                    return
                if (
                    not try_tp_dot.check_unsolved()
                    or not try_tp_bulb.check_unsolved()
                    or invariant
                ):
                    return

    def guess_and_check_thrifty(self, level: int, max_cost: float = math.inf) -> None:
        """
        Guesses at every blank cell and uses apply_methods to eliminate impossible
        options. Sticks with only the lowest-cost option.
        """
        gacbc = 1.0  # guess_and_check base cost
        level_to_use = min(level, 8)

        cheapest_choice = None
        lowest_cost = max_cost  # the starting value for the lowest cost should be the
        # maximum allowed cost

        for i, j in zip(
            *np.asarray(self.board == ".", dtype=int).nonzero(), strict=True
        ):
            i = int(i)
            j = int(j)
            if self.board[i, j] == ".":
                try_tp_dot = self.__copy__()
                try_tp_dot.maybe_set_dot(
                    i, j, Step((i, j, "?"), "guess_and_check_guess", cost=gacbc)
                )
                try_tp_dot.apply_methods(level_to_use, budget=lowest_cost)
                if not try_tp_dot.check_unsolved():
                    cost = sum(step.cost for step in try_tp_dot.solution_steps)
                    if cost < lowest_cost:
                        lowest_cost = cost
                        cheapest_choice = (
                            "#",
                            (i, j, Step((i, j, "?"), "guess_and_check", cost=cost)),
                        )
                    continue
                    # continue for this branch because we already know the cell
                try_tp_bulb = self.__copy__()
                try_tp_bulb.maybe_set_bulb(
                    i, j, Step((i, j, "?"), "guess_and_check_guess", cost=gacbc)
                )
                try_tp_bulb.apply_methods(level_to_use, budget=lowest_cost)
                if not try_tp_bulb.check_unsolved():
                    cost = sum(step.cost for step in try_tp_bulb.solution_steps)
                    if cost < lowest_cost:
                        lowest_cost = cost
                        cheapest_choice = (
                            "+",
                            (i, j, Step((i, j, "?"), "guess_and_check", cost=cost)),
                        )
        if cheapest_choice is not None:
            if cheapest_choice[0] == "#":
                self.maybe_set_bulb(*cheapest_choice[1])
            elif cheapest_choice[0] == "+":
                self.maybe_set_dot(*cheapest_choice[1])


def do_best_to_get_a_non_wrong_solution(board: np.ndarray) -> np.ndarray:
    """
    If the board has a solution with the current state, return that solution
    Otherwise, clearing the board and return that solved if possible
    Otherwise, return a cleared board

    This function is flaky if the cleared board does not have a solution
    """
    tp = ThoughtProcess(board)
    tp.apply_methods(9)
    if tp.check_unsolved():
        return tp.board

    board = clear_board(board.copy())
    tp = ThoughtProcess(board)
    tp.apply_methods(9)
    if tp.check_unsolved():
        return tp.board

    return board


def batch_calculate_difficulty(filenames: list[str], csvfilename: str) -> None:
    boards = []
    for filename in filenames:
        with open(filename) as hin:
            text = hin.read()
        if text:
            board = load_pzprv3(text)
            boards.append(board.copy())
        else:
            msg = f"file empty: {filename}"
            raise ValueError(msg)
    results = []
    for filename, board in zip(filenames, boards, strict=True):
        start = timeit.default_timer()
        tp = ThoughtProcess(board)
        tp.apply_methods(9, calculate_difficulty=True)
        stop = timeit.default_timer()
        if not tp.check_unsolved():
            cost = "No solution"
            difficulty = cost
        elif not check_all(tp.board):
            cost = "Probably multiple solutions"
            difficulty = cost
        else:
            cost = sum(step.cost for step in tp.solution_steps)
            difficulty = max(s.cost for s in tp.solution_steps)
        result = (filename, cost, difficulty, stop - start)
        print(result)
        results.append(result)
    if csvfilename:
        with open(csvfilename, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "cost", "difficulty", "time"])
            writer.writerows(results)


def main(argv: list | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    ap = argparse.ArgumentParser(
        description="Currently just analyzes batches of puzzle files"
    )
    ap.add_argument("filenames", nargs="+", help="files to analyze")
    ap.add_argument("--output", "-o", help="Output csv file", default="")
    args = ap.parse_args(argv)
    print(args.filenames, args.output)
    batch_calculate_difficulty(args.filenames, args.output)


if __name__ == "__main__":
    sys.exit(main())
