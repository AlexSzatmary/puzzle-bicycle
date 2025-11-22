# This code should eventually get broken up into multiple files, especially when more
# genres are supported but for now all the backend is going in here.
import math
import sys
import textwrap
from copy import copy
from dataclasses import dataclass
from inspect import cleandoc
from typing import Any, cast

import numpy as np

SHADED = "#"
UNSHADED = "+"
UNKNOWN = "."
ORTHO_DIRS = [(1, 0), (0, -1), (-1, 0), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


COSTS = {
    "search": 1.0,
}


# ***** General
def uniq(a: list) -> list:
    return list(set(a))


class Puzzle:
    def __init__(
        self, board: np.ndarray, constraint_classes: list[type["Constraint"]]
    ) -> None:
        self.board = board
        self.constraints = [c(self) for c in constraint_classes]

    def __copy__(self) -> "Puzzle":
        cls = self.__class__
        new = cls.__new__(cls)
        new.board = self.board.copy()
        new.constraints = []
        for c in self.constraints:
            new_c = copy(c)
            new_c.puzzle = new
            new.constraints.append(new_c)
        return new

    def print_board(self) -> None:
        print(self.stringify_board())

    def apply_constraints(
        self, new_moves: list["State"]
    ) -> tuple["ProofConstraintContradiction", list["State"]]:
        hot = []
        for c in self.constraints:
            p, new_hot = c.check(new_moves)
            if p:
                return p, []
            else:
                hot.extend(new_hot)
        return ProofConstraintContradiction([], "all constraints pass"), hot

    def apply_state(self, state: "State") -> None:
        # This should work for anything where state is tracked only on the board
        var, val = state
        self.board[var] = val

    def guesses_to_make(self) -> list["State"]:
        guesses = []
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i, j] == UNKNOWN:
                    guesses.append(((i, j), SHADED))
                    guesses.append(((i, j), UNSHADED))
        return guesses

    def any_unknown(self) -> bool:
        return bool(np.any(self.board == UNKNOWN))


# ***** Constraint class
Variable = tuple[int, int]  # this could be expanded later
Value = str
State = tuple[Variable, Value]


@dataclass
class Step:
    consequents: list[State]
    reason: "Proof"
    cost: float

    def __repr__(self) -> str:
        return cleandoc(f"""
        Step(
          consequents={self.consequents},
          reason={self.reason},
          cost={self.cost}
        """)


@dataclass
class Implication:
    antecedent: State
    consequents: list[State]
    reason: "Proof"
    cost: float


@dataclass
class Proof:
    """
    A proof is anything that explains how we made a step or implication. It should
    list relevant immediately existing state.
    """

    antecedents: list[State]
    reason: Any


@dataclass
class ProofStr(Proof):
    reason: str


@dataclass
class ProofConstraintContradiction(Proof):
    """
    reason: a label based on the constraint, stating in present tense what is incorrect,
        e.g., "The unshaded cells are divided."

    wrong_moves: This reveals particular moves that are wrong. Some move must be listed
        as wrong, multiple can be, it is okay to miss some moves. For example, for a
        dynasty puzzle, every new move that is adjacent to a shaded cell should be
        listed (even if it was placed first in the list of new moves); but it's rough to
        list every single move that is responsible for breaking connectedness.

    If there is no actual contradiction, still return something of this type but with
    wrong_moves = [].
    """

    reason: str

    def __len__(self) -> int:
        return len(self.antecedents)


@dataclass
class ProofWrongGuess(Proof):
    reason: ProofConstraintContradiction


@dataclass
class ProofStepList(Proof):
    reason: list[Step]


class Constraint:
    def __init__(self, puzzle: Puzzle) -> None:
        self.puzzle = puzzle

    def __copy__(self) -> "Constraint":
        cls = self.__class__
        new = cls.__new__(cls)
        return new

    def check(
        self, new_moves: list[State]
    ) -> tuple[ProofConstraintContradiction, list[State]]:
        """
        A check takes a list of moves. It applies those moves and

        * if there's a contradiction, returns a proof of why.
        * if there's no contradiction, return a list of good next moves to try that are
          likely to violate the constraint.
        """
        raise NotImplementedError


def swap_state_shaded(state: State) -> State:
    return (
        state[0],
        UNSHADED if state[1] == SHADED else SHADED,
    )


def search_base(
    puzzle: Puzzle, *, depth: int = 2, budget: float = 10.0
) -> tuple[Puzzle, Proof | None, list[Step]]:
    steps = []
    # hot = set() # TODO add hot set checking for base
    # flag = False  # flag will be set if a solution is set or there is a stall
    # while not flag:  # TODO give some condition based on find solution or stall
    allowed_budget = 2.0
    guesses = puzzle.guesses_to_make()
    new_steps = []
    while allowed_budget < budget:
        for guess in guesses:
            new_steps, _implications = search(
                copy(puzzle),
                guess,
                hot=set(),
                depth=depth - 1,
                budget=allowed_budget,
            )
            for step in new_steps:
                for consequent in step.consequents:
                    puzzle.apply_state(consequent)
                    puzzle.print_just_board(indent=(5 - depth) * 2)
                proof, _new_hot = puzzle.apply_constraints(step.consequents)
                if proof:
                    return (puzzle, proof, steps)
                # checked.difference_update(new_hot)
                # hot.update(new_hot)  # make sure we don't get an infinite loop
                steps.append(step)
            if new_steps:
                break  # reset variables and budget
        if new_steps:
            allowed_budget = 2.0
        else:
            allowed_budget += 1
        print(f"allowed_budget {allowed_budget}")
        guesses = puzzle.guesses_to_make()
        new_steps = []

    return puzzle, None, steps


def search(
    puzzle: Puzzle,
    this_move: State,
    *,
    hot: set[State],
    depth: int = 2,
    budget: float = math.inf,
    # implications: dict | None = None, # Not actually doing implications yet
) -> tuple[list[Step], list[Implication]]:
    """ """
    steps = []
    steps.append(Step([this_move], ProofStr([], "assumed"), COSTS["search"]))
    cost = steps[-1].cost
    var, val = this_move
    puzzle.board[var] = val
    proof, new_hot = puzzle.apply_constraints([this_move])
    if proof:
        return _search_handle_contradiction(steps, this_move, proof)
    hot.update(new_hot)

    checked = set()
    while cost < budget and hot:
        guess = hot.pop()
        if guess in checked:
            continue
        new_steps, _implications = search(
            copy(puzzle),
            guess,
            hot=hot.copy(),
            depth=depth - 1,
            budget=budget - cost,
        )
        for step in new_steps:
            for consequent in step.consequents:
                puzzle.apply_state(consequent)
                puzzle.print_just_board(indent=(5 - depth) * 2)
                print()
            proof, new_hot = puzzle.apply_constraints(step.consequents)
            if proof:
                return _search_handle_contradiction(steps, this_move, proof)
            checked.difference_update(new_hot)
            hot.update(new_hot)  # make sure we don't get an infinite loop
            steps.append(step)
            # TODO add costs as a thing for proofs?

    # change board state with new moves

    # when constraints are applied to a new move, they either fail and give a proof or
    # succeed and put State on the hot list

    # apply implications until contradiction or we run out. Implications use up budget.
    # check constraints aggressively as we do that
    # if False, return proof
    # postpone invariants. if invariant detected, return proof

    # At the depth = 0 level,
    # * budget can be huge but the next levels get budget doled
    #   out in a stingy way
    # * new_move should be null or something.
    # Or maybe we do need a different function for depth = 0.

    # this should really be a uniform cost search type of thing; depth is there just in
    # case and shallowish depth (5?) should be plenty

    # TODO this should only return steps if a contradiction is found, an implication is
    # found; otherwise, return invariants
    return ([], [])  # add second return when doing invariants


# TODO redo cost to measure from depth of new proof graph


def _search_handle_contradiction(
    steps: list[Step], this_move: State, proof: ProofConstraintContradiction
) -> tuple[list[Step], list[Implication]]:
    steps.append(Step([swap_state_shaded(this_move)], proof, 1.0))
    return (
        [
            Step(
                [swap_state_shaded(this_move)],
                ProofStepList([this_move], steps),
                1.0,
            )
        ],
        [],
    )


# ***** Dynasty constraints
class NoShadedAdjacent(Constraint):
    def __init__(self, puzzle: Puzzle) -> None:
        super().__init__(puzzle)

    def check(
        self, new_moves: list[State]
    ) -> tuple[ProofConstraintContradiction, list[State]]:
        shaded_adjacent = set()
        hot = set()
        for (i, j), value in new_moves:
            if value != SHADED:
                continue
            if i > 0 and self.puzzle.board[i - 1, j] == SHADED:
                shaded_adjacent.add((i, j))
                shaded_adjacent.add((i - 1, j))
                hot.add((i - 1, j))
            if (
                i < self.puzzle.board.shape[0] - 1
                and self.puzzle.board[i + 1, j] == SHADED
            ):
                shaded_adjacent.add((i, j))
                shaded_adjacent.add((i + 1, j))
                hot.add((i + 1, j))
            if j > 0 and self.puzzle.board[i, j - 1] == SHADED:
                shaded_adjacent.add((i, j))
                shaded_adjacent.add((i, j - 1))
                hot.add((i, j - 1))
            if (
                j < self.puzzle.board.shape[1] - 1
                and self.puzzle.board[i, j + 1] == SHADED
            ):
                shaded_adjacent.add((i, j))
                shaded_adjacent.add((i, j + 1))
                hot.add((i, j + 1))
        if shaded_adjacent:
            return (
                ProofConstraintContradiction(
                    [(var, SHADED) for var in shaded_adjacent],
                    "Some shaded cells are adjacent.",
                ),
                [],
            )
        else:
            return (
                ProofConstraintContradiction([], ""),
                [(var, SHADED) for var in hot],
            )


class AllUnshadedOrthogonallyConnected(Constraint):
    def __init__(self, puzzle: Puzzle) -> None:
        super().__init__(puzzle)

    def check(
        self, new_moves: list[State]
    ) -> tuple[ProofConstraintContradiction, list[State]]:
        nodes_in_loops, hot_var = self._find_loops_at_ij(new_moves)
        if nodes_in_loops:
            return (
                ProofConstraintContradiction(
                    [(var, SHADED) for var in nodes_in_loops],
                    "The unshaded cells are divided.",
                ),
                [],
            )
        else:
            return (
                ProofConstraintContradiction([], ""),
                [(var, SHADED) for var in hot_var],
            )

    def _find_loops_at_ij(
        self, new_moves: list[State]
    ) -> tuple[list[Variable], list[Variable]]:
        loop_cells = []
        moves_in_loops = []
        hot_var = []
        nrows = self.puzzle.board.shape[0]
        ncols = self.puzzle.board.shape[1]
        for (i, j), value in new_moves:
            if value != SHADED:
                continue
            # branches = []  # see definition in _trace_graph
            at_edge = i == 0 or j == 0 or i == nrows - 1 or j == ncols - 1

            if (i, j) in loop_cells:
                moves_in_loops.append((i, j))
                continue

            new_loop_cells, new_hot = self._trace_graph(
                [(i, j, i, j, [(i, j)])],
                # [(i + di, j + dj, i, j, [(i, j)]) for (di, dj) in DIAG_DIRS],
                start_at_edge=at_edge,
            )

            if new_loop_cells:
                moves_in_loops.append((i, j))
                loop_cells.extend(new_loop_cells)
            hot_var.extend(new_hot)

            # # The following if statements are repetitive but I am pretty sure that
            # # unrolling them rather than looping is easier to get right (due to the
            # # need to bounds check) and will have a worthwhile perf advantage
            # if i > 0 and j > 0:
            #     if self.puzzle.board[i - 1, j - 1] == SHADED:
            #         branches.append((i - 1, j - 1, i, j, [(i, j)]))
            #     elif self.puzzle.board[i - 1, j - 1] == UNKNOWN:
            #         hot_var.append((i - 1, j - 1))
            # if i < nrows - 1 and j > 0:
            #     if self.puzzle.board[i + 1, j - 1] == SHADED:
            #         branches.append((i + 1, j - 1, i, j, [(i, j)]))
            #     elif self.puzzle.board[i + 1, j - 1] == UNKNOWN:
            #         hot_var.append((i + 1, j - 1))
            # if i > 0 and j < ncols - 1:
            #     if self.puzzle.board[i - 1, j + 1] == SHADED:
            #         branches.append((i - 1, j + 1, i, j, [(i, j)]))
            #     elif self.puzzle.board[i - 1, j + 1] == UNKNOWN:
            #         hot_var.append((i - 1, j + 1))
            # if i < nrows - 1 and j < ncols - 1:
            #     if self.puzzle.board[i + 1, j + 1] == SHADED:
            #         branches.append((i + 1, j + 1, i, j, [(i, j)]))
            #     elif self.puzzle.board[i + 1, j + 1] == UNKNOWN:
            #         hot_var.append((i + 1, j + 1))
            # if len(branches) > 1 or (len(branches) > 0 and at_edge):
            #     new_loop_cells, new_hot = self._trace_graph(
            #         branches, start_at_edge=at_edge
            #     )
            #     if new_loop_cells:
            #         moves_in_loops.append((i, j))
            #         loop_cells.extend(new_loop_cells)
            #     hot_var.extend(new_hot)
        return moves_in_loops, list(set(hot_var))

    def _trace_graph(  # noqa: C901 There is a lot of necessarily repetitive code
        self,
        branches: list[tuple[int, int, int, int, list[tuple[int, int]]]],
        *,
        start_at_edge: bool,
    ) -> tuple[list[Variable], list[Variable]]:
        """
        Each "branch" is a tuple of the i and j for the next cell to check, the i and j
        for the cell that led to this cell being checked, and the list of cells that the
        branch has already passed through.
        """
        loop_cells = []
        cells_on_edge_path = []
        seen = {(branch[0], branch[1]) for branch in branches}
        seen.add((branches[0][2], branches[0][2]))  # also add original move
        nrows = self.puzzle.board.shape[0]
        ncols = self.puzzle.board.shape[1]
        hot_var = []
        edge_contacts = int(start_at_edge)
        true_at_start = True
        # To be on a loop, the new_move must touch two branches (that it might connect)
        # or one branch and the perimeter of the board
        while (
            true_at_start
            or (not start_at_edge and len(branches) > 1)
            or (start_at_edge and branches)
        ):
            true_at_start = False
            new_branches = []
            for i0, j0, from_i, from_j, path in branches:
                for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    i = i0 + di
                    j = j0 + dj
                    if i == from_i and j == from_j:  # don't double-check cells
                        continue
                    if i < 0 or j < 0 or i > nrows - 1 or j > ncols - 1:
                        continue
                    if self.puzzle.board[i, j] == UNKNOWN:
                        hot_var.append((i, j))
                        continue
                    elif self.puzzle.board[i, j] == UNSHADED:
                        continue
                    if i == 0 or j == 0 or i == nrows - 1 or j == ncols - 1:
                        edge_contacts += 1
                        cells_on_edge_path.extend(path)
                        continue
                    if (i, j) in seen:
                        loop_cells.extend(path)
                        continue
                    new_branches.append((i, j, i0, j0, [*path, (i, j)]))
                    # for di2, dj2 in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    #     i2 = i + di2
                    #     j2 = j = dj2
                    #     print("    (i2, j2)", (i2, j2))
                    #     if self.puzzle.board[i2, j2] == SHADED:
                    #         new_branches.append((i2, j2, i, j, [*path, (i2, j2)]))
                    #     elif self.puzzle.board[i2, j2] == UNKNOWN:
                    #         hot_var.append((i2, j2))
            branches = new_branches
        if edge_contacts > 1:
            loop_cells.extend(cells_on_edge_path)

        return loop_cells, hot_var


class EqualNumbersRowColumn(Constraint):
    def __init__(self, puzzle: Puzzle) -> None:
        super().__init__(puzzle)
        nrows = puzzle.board.shape[0]
        ncols = puzzle.board.shape[1]
        # empty list is only used for numbers that do not share a row or column so this
        # is the rare case in which initializing a list of lists with * sense.
        null = np.array([], dtype=int)
        self._equal_by_cols: np.ndarray = [[null] * ncols for _i in range(nrows)]
        self._equal_by_rows: np.ndarray = [[null] * ncols for _i in range(nrows)]
        for i in range(nrows):
            for j in range(ncols):
                if not self._equal_by_cols[i][j].size:
                    sames = (puzzle.numbers[i:, j] == puzzle.numbers[i, j]).nonzero()[
                        0
                    ] + i
                    if sames.size > 1:
                        for same in sames:
                            self._equal_by_cols[same][j] = sames
                if not self._equal_by_rows[i][j].size:
                    sames = (puzzle.numbers[i, j:] == puzzle.numbers[i, j]).nonzero()[
                        0
                    ] + j
                    if sames.size > 1:
                        for same in sames:
                            self._equal_by_rows[i][same] = sames

    def __copy__(self) -> "Constraint":
        new = super().__copy__()
        new._equal_by_cols = self._equal_by_cols
        new._equal_by_rows = self._equal_by_rows
        return new

    def check(
        self, new_moves: list[State]
    ) -> tuple[ProofConstraintContradiction, list[State]]:
        equal_numbers, hot_var = self._find_unshaded_equal_numbers(new_moves)
        if equal_numbers:
            return (
                ProofConstraintContradiction(
                    [(var, UNSHADED) for var in equal_numbers],
                    "There are equal numbers in a row or column.",
                ),
                [],
            )
        else:
            return (
                ProofConstraintContradiction([], ""),
                [(var, SHADED) for var in hot_var],
            )

    def _find_unshaded_equal_numbers(  # noqa: C901 it's fine, a lot of logic is doubled for rows and cols
        self, new_moves: list[State]
    ) -> tuple[list[Variable], list[Variable]]:
        board: np.ndarray = self.puzzle.board
        equal_numbers = []
        hot_var = []
        for (i, j), value in new_moves:
            if value != UNSHADED:
                continue
            if self._equal_by_cols[i][j].size:
                if np.sum(board[self._equal_by_cols[i][j], j] == UNSHADED) > 1:
                    for k in self._equal_by_cols[i][j]:
                        if board[k, j] == UNSHADED:
                            equal_numbers.append((int(k), j))
                else:
                    for k in self._equal_by_cols[i][j]:
                        if board[k, j] == UNKNOWN:
                            hot_var.append((int(k), j))
            if self._equal_by_rows[i][j].size:
                if np.sum(board[i, self._equal_by_rows[i][j]] == UNSHADED) > 1:
                    for k in self._equal_by_rows[i][j]:
                        if board[i, k] == UNSHADED:
                            equal_numbers.append((i, int(k)))
                else:
                    for k in self._equal_by_cols[i][j]:
                        if board[i, k] == UNKNOWN:
                            hot_var.append((i, int(k)))
        return uniq(equal_numbers), uniq(hot_var)


# class Trees:
#     def __init__(self) -> None:
#         self.trees_of_cells = {}
#         self.cells_in_trees = {}

#     def merge_trees(self, tree_A: tuple[int, int], tree_B: tuple[int, int]) -> bool:
#         root_A = self.trees_of_cells[tree_A]
#         root_B = self.trees_of_cells[tree_B]
#         if root_A == root_B:
#             return False
#         elif root_A > root_B:
#             root_A, root_B = root_B, root_A
#         for cell in self.trees_of_cells[root_B]:
#             self.trees_of_cells[cell] = root_A
#         self.trees_of_cells[root_A].extend(self.trees_of_cells.pop(root_B))
#         return True

#     def add_unit_tree(self, i: int, j: int) -> None:
#         self.trees_of_cells[i, j] = (i, j)
#         self.cells_in_trees[i, j] = (i, j)


# class OrthogonallyConnectedConstraintManager:
#     def __init__(self, puzzle: Puzzle) -> None:
#         self.trees = Trees()
#         board = puzzle.board
#         self.nrows = board.shape[0]
#         self.ncols = board.shape[1]
#         for i in range(self.nrows):
#             for j in range(self.ncols):
#                 if board[i, j] == SHADED:
#                     board[i, j] = (0, 0)

#     def _add_shaded(self, i: int, j: int) -> tuple[list, list]:

#         best_tree = (i, j)
#         if i == 0 or j == 0 or i == self.rows - 1 or j == self.cols - 1:
#             best_tree = (0, 0)
#         if (i - 1, j - 1) in self.trees:
#             if best_tree < self.trees[i - 1, j - 1]:
#                 self.trees[i - 1, j - 1] = best_tree
#         return ([], [])


# ***** Hitori
class HitoriPuzzle(Puzzle):
    def __init__(self, numbers: np.ndarray, board: np.ndarray) -> None:
        self.numbers = numbers
        super().__init__(
            board,
            [
                NoShadedAdjacent,
                EqualNumbersRowColumn,
                AllUnshadedOrthogonallyConnected,
            ],
        )

    def __copy__(self) -> "HitoriPuzzle":
        new = cast("HitoriPuzzle", super().__copy__())
        new.numbers = self.numbers
        return new

    def stringify_board(self) -> str:
        return "\n".join(
            "".join(
                SHADED if value == SHADED else number
                for (number, value) in zip(number_row, value_row, strict=False)
            )
            for (number_row, value_row) in zip(self.numbers, self.board, strict=False)
        )

    def print_just_board(self, indent: int = 0) -> None:
        print(
            textwrap.indent(
                "\n".join("".join(value_row) for value_row in self.board), " " * indent
            )
        )
        print()


def hitori_puzzle_from_strings(board_str: str, numbers_str: str) -> HitoriPuzzle:
    return HitoriPuzzle(
        np.array(
            [line.split() for line in cleandoc(numbers_str).split("\n")], dtype="str"
        ),
        np.array(
            [line.split() for line in cleandoc(board_str).split("\n") if line],
            dtype="str",
        ),
    )


# **** Board manipulation utilities
def load_pzprv3(pzprv3: str) -> HitoriPuzzle:
    """
    Loads PUZ-PRE v3 text and returns a Hitori board
    """
    pzprv3_lines = pzprv3.split("\n")
    rows = int(pzprv3_lines[2])
    cols = int(pzprv3_lines[3])
    numbers = np.zeros((rows, cols), dtype="str")
    numbers[:, :] = "-"
    for i, row in enumerate(pzprv3_lines[4 : 4 + rows]):
        numbers[i, :] = row.split()
    board = np.zeros((rows, cols), dtype="str")
    board[:, :] = "-"
    for i, row in enumerate(pzprv3_lines[4 + rows : 4 + rows * 2]):
        board[i, :] = row.split()
    return HitoriPuzzle(numbers, board)


def step_list_repr(step_list: list[Step]) -> str:
    return "\n".join(repr(step) for step in step_list)


def verbose_output(puzzle: Puzzle) -> None:
    sb = search_base(puzzle)
    print("*** PROOF FOR WHY CONTRADICTORY ***")
    print(sb[1])
    print()
    print("*** STEPS ***")
    print(sb[2])
    print()
    print("*** FINAL ***")
    print(sb[0].board)
    sb[0].print_board()
    print(f"Solved: {not bool(sb[1]) and not puzzle.any_unknown()}")
    print()


def main(argv: list | None = None) -> None:
    if argv is None:
        argv = sys.argv
    for file in argv[1:]:
        print(file)
        with open(file) as hin:
            hitori_puzzle = load_pzprv3(hin.read())
            hitori_puzzle.print_board()
            puzzle = hitori_puzzle
            verbose_output(puzzle)
    # puzzle.apply_state(((0, 0), SHADED))
    # print(puzzle.board)
    # puzzle.print_board()
    # print(puzzle.constraints)
    # # print(step_list_repr(search(copy(puzzle), ((0, 0), SHADED), hot=set())[0]))
    # # print(step_list_repr(search(copy(puzzle), ((0, 1), SHADED), hot=set())[0]))
    # # print(step_list_repr(search(copy(puzzle), ((1, 0), SHADED), hot=set())[0]))
    # # print(step_list_repr(search(copy(puzzle), ((1, 1), SHADED), hot=set())[0]))
    # print(search(copy(puzzle), ((0, 0), SHADED), hot=set())[0])
    # print(search(copy(puzzle), ((0, 1), SHADED), hot=set())[0])
    # print(search(copy(puzzle), ((1, 0), SHADED), hot=set())[0])
    # print(search(copy(puzzle), ((1, 1), SHADED), hot=set())[0])
    # print("***")
    # print(search(copy(puzzle), ((1, 1), SHADED), hot=set())[0])
    # puzzle.print_board()
    # print(search_base(puzzle))
    # puzzle.apply_state(((1, 1), SHADED))
    # puzzle = hitori_puzzle_from_strings(
    #     """
    #     . #
    #     . .
    #     """,
    #     """
    #     1 1
    #     1 2
    #     """,
    # )
    # sb = search_base(puzzle)
    # print(sb)
    # print("*** FINAL ***")
    # print(sb[0].board)
    # print()

    # puzzle = hitori_puzzle_from_strings(
    #     """
    #     . . .
    #     . . .
    #     . . .
    #     """,
    #     """
    #     1 1 3
    #     1 3 1
    #     2 2 1
    #     """,
    # )
    # print(puzzle.board)
    # sb = search_base(puzzle)
    # print(sb)
    # print("*** FINAL ***")
    # print(sb[0].board)
    # print()

    # puzzle = hitori_puzzle_from_strings(
    #     """
    #     . . . .
    #     # . . .
    #     . . . .
    #     . . # .
    #     """,
    #     """
    #     1 1 3
    #     1 2 4
    #     5 6 7
    #     """,
    # )
    # print(puzzle.board)
    # sb = search_base(puzzle)
    # print(sb)
    # print("*** FINAL ***")
    # print(sb[0].board)
    # print()

    # puzzle = hitori_puzzle_from_strings(
    #     """
    #     . .
    #     . .
    #     """,
    #     """
    #     1 1
    #     1 2
    #     """,
    # )


if __name__ == "__main__":
    sys.exit(main())
