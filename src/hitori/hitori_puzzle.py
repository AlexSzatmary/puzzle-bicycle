# This code should eventually get broken up into multiple files, especially when more
# genres are supported but for now all the backend is going in here.
import math
import sys
import textwrap
from copy import copy
from dataclasses import dataclass
from inspect import cleandoc
from typing import Any, cast

import black
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

    def is_known(self, var: "Variable") -> bool:
        return self.board[var] != UNKNOWN

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
        return (
            f"Step(consequents={self.consequents}, reason={self.reason},"
            f"cost={self.cost})"
        )
        # return cleandoc(f"""
        # Step(
        #   consequents={self.consequents},
        #   reason={self.reason},
        #   cost={self.cost})
        # """)


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
    _saved_cost: float | None = None

    @property
    def cost(self) -> float:
        if self._saved_cost is None:
            self._calculate_cost()
        assert self._saved_cost is not None
        return self._saved_cost

    def _calculate_cost(self) -> None:
        # The cost of a ProofStepList is the total cost of the steps actually needed to
        # support the final step in the list. Steps that were proven before this list
        # are free, so we do not count any state that is an antecedent to one of the
        # steps in the list, but not a consequent to any of them.
        #
        # This is intended to measure how much stuff you need in your head that you
        # didn't already write on the paper; it might not be a helpful measure if there
        # are many layers of recursion.

        needed_steps = []
        needed_antecedents = set(self.reason[-1].reason.antecedents)
        for step in self.reason[-2::-1]:
            if any(c in needed_antecedents for c in step.consequents):
                needed_antecedents.update(step.reason.antecedents)
                needed_steps.append(step)
        self._saved_cost = sum(step.cost for step in needed_steps)
        if isinstance(self.reason[-1].reason, ProofConstraintContradiction):
            self._saved_cost += self.reason[-1].cost


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
        A check takes a list of moves.

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


def search_base(  # noqa: C901 TODO simplify
    puzzle: Puzzle, *, max_depth: int = 5, budget: float = 16.0
) -> tuple[Puzzle, Proof | None, list[Step]]:
    steps = []
    # hot = set() # TODO add hot set checking for base
    allowed_depth = 1
    allowed_budget = 2.0 * allowed_depth
    guesses = puzzle.guesses_to_make()
    new_steps = []
    while allowed_depth <= max_depth and puzzle.any_unknown():
        for guess in guesses:
            if puzzle.is_known(guess[0]):
                continue
            new_steps, _implications = search(
                copy(puzzle),
                guess,
                hot=set(),
                depth=1,
                max_depth=allowed_depth,
                budget=allowed_budget,
            )
            for step in new_steps:
                if step.cost > budget:
                    new_steps = []
                    break
                for consequent in step.consequents:
                    puzzle.apply_state(consequent)
                proof, _new_hot = puzzle.apply_constraints(step.consequents)
                if proof:
                    return (puzzle, proof, steps)
                steps.append(step)
                puzzle.print_just_board(indent=2 * 0)  # this function is at 0 depth
            print()
            if new_steps:
                break  # reset variables and budget
        if new_steps:
            allowed_depth = 1
            allowed_budget = 2.0 * allowed_depth
        elif allowed_budget < budget:
            allowed_budget += 1.0
        else:
            allowed_depth += 1
            allowed_budget = 2.0 * allowed_depth
        print(f"allowed_budget {allowed_budget} allowed_depth {allowed_depth}")
        guesses = puzzle.guesses_to_make()
        new_steps = []

    return puzzle, None, steps


def search(
    puzzle: Puzzle,
    this_move: State,
    *,
    hot: set[State],
    depth: int,
    max_depth: int = 2,
    budget: float = math.inf,
    # implications: dict | None = None, # Not actually doing implications yet
) -> tuple[list[Step], list[Implication]]:
    """ """
    steps = []
    steps.append(Step([this_move], ProofStr([], "assumed"), COSTS["search"]))
    cost = steps[-1].cost
    puzzle.apply_state(this_move)
    puzzle.print_just_board(indent=2 * depth)
    print()
    proof, new_hot = puzzle.apply_constraints([this_move])
    if proof:
        return _search_handle_contradiction(steps, this_move, proof)
    if depth >= max_depth:
        return [], []
    hot.update(new_hot)

    checked = set()
    while hot:
        guess = hot.pop()
        if guess in checked or puzzle.is_known(guess[0]):
            continue
        new_steps, _implications = search(
            copy(puzzle),
            guess,
            hot=hot.copy(),
            depth=depth + 1,
            max_depth=max_depth,
            budget=budget - cost,
        )
        for step in new_steps:
            if step.cost > budget:
                continue
            for consequent in step.consequents:
                puzzle.apply_state(consequent)
            proof, new_hot = puzzle.apply_constraints(step.consequents)
            if proof:
                return _search_handle_contradiction(steps, this_move, proof)
            checked.difference_update(new_hot)
            checked.update(step.consequents)
            hot.update(new_hot)  # make sure we don't get an infinite loop
            steps.append(step)
            # TODO add costs as a thing for proofs?

    # TODO this should only return steps if a contradiction is found, an implication is
    # found; otherwise, return invariants
    return [], []  # add second return when doing invariants


def _search_handle_contradiction(
    steps: list[Step], this_move: State, proof: ProofConstraintContradiction
) -> tuple[list[Step], list[Implication]]:
    steps.append(Step([swap_state_shaded(this_move)], proof, 1.0))
    psl = ProofStepList([this_move], steps)
    return ([Step([swap_state_shaded(this_move)], psl, psl.cost)], [])


# ***** Dynasty constraints
class NoShadedAdjacent(Constraint):
    def __init__(self, puzzle: Puzzle) -> None:
        super().__init__(puzzle)

    def check(  # noqa: C901 repetitive complexity, nbd
        self, new_moves: list[State]
    ) -> tuple[ProofConstraintContradiction, list[State]]:
        shaded_adjacent = set()
        hot = set()
        for (i, j), value in new_moves:
            if value != SHADED:
                continue
            if i > 0:
                if self.puzzle.board[i - 1, j] == SHADED:
                    shaded_adjacent.add((i, j))
                    shaded_adjacent.add((i - 1, j))
                else:
                    hot.add((i - 1, j))
            if i < self.puzzle.board.shape[0] - 1:
                if self.puzzle.board[i + 1, j] == SHADED:
                    shaded_adjacent.add((i, j))
                    shaded_adjacent.add((i + 1, j))
                else:
                    hot.add((i + 1, j))
            if j > 0:
                if self.puzzle.board[i, j - 1] == SHADED:
                    shaded_adjacent.add((i, j))
                    shaded_adjacent.add((i, j - 1))
                else:
                    hot.add((i, j - 1))
            if j < self.puzzle.board.shape[1] - 1:
                if self.puzzle.board[i, j + 1] == SHADED:
                    shaded_adjacent.add((i, j))
                    shaded_adjacent.add((i, j + 1))
                else:
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
        all_moves_in_loops = []
        for (i, j), value in new_moves:
            if value != SHADED:
                continue
            at_edge = i == 0 or j == 0 or i == nrows - 1 or j == ncols - 1

            if (i, j) in loop_cells:
                moves_in_loops.append((i, j))
                continue

            new_loop_cells, new_hot = self._trace_graph(
                [(i, j, i, j, [(i, j)])],
                start_at_edge=at_edge,
            )

            if new_loop_cells:
                all_moves_in_loops.extend(new_loop_cells)
                moves_in_loops.append((i, j))
                loop_cells.extend(new_loop_cells)
            hot_var.extend(new_hot)

        return uniq(all_moves_in_loops), list(set(hot_var))

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
        nrows = self.puzzle.board.shape[0]
        ncols = self.puzzle.board.shape[1]
        hot_var = []
        edge_contacts = int(start_at_edge)
        true_at_start = True
        # To be on a loop, the new_move must touch two branches (that it might connect)
        # or one branch and the perimeter of the board. edge_contacts is the total
        # number of contacts with the perimeter but bool(edge_contacts) is needed here
        # because for this condition, the number of contacts does not matter as long as
        # it is more than 0.
        while true_at_start or (bool(edge_contacts) + len(branches) > 1):
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
                        cells_on_edge_path.append((i, j))
                        continue
                    if (i, j) in seen:
                        loop_cells.extend(path)
                        continue
                    new_branches.append((i, j, i0, j0, [*path, (i, j)]))
                    seen.add((i, j))
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
                [(var, UNSHADED) for var in hot_var],
            )

    def _find_unshaded_equal_numbers(  # noqa: C901 it's fine, a lot doubled logic
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
                    for k in self._equal_by_rows[i][j]:
                        if board[i, k] == UNKNOWN:
                            hot_var.append((i, int(k)))
        return uniq(equal_numbers), uniq(hot_var)


# ***** Hitori
class HitoriPuzzle(Puzzle):
    def __init__(
        self, board: np.ndarray, numbers: np.ndarray, number_field_size: int
    ) -> None:
        self.numbers = numbers
        self.number_field_size = number_field_size
        super().__init__(
            board,
            [
                NoShadedAdjacent,
                EqualNumbersRowColumn,
                AllUnshadedOrthogonallyConnected,
            ],
        )
        empty = ".".rjust(self.number_field_size)
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i, j] == UNKNOWN and (
                    self.numbers[i, j] == empty or self.numbers[i, j] == "."
                ):
                    self.board[i, j] = UNSHADED
                    # have to reformat . numbers as spaces so the solution does not look
                    # incomplete
                    self.numbers[i, j] = " "

    def __copy__(self) -> "HitoriPuzzle":
        new = cast("HitoriPuzzle", super().__copy__())
        new.numbers = self.numbers
        new.number_field_size = self.number_field_size
        return new

    def stringify_board(self) -> str:
        return "\n".join(
            "".join(
                SHADED.rjust(self.number_field_size)
                if value == SHADED
                else number.rjust(self.number_field_size)
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
    max_num_str_len = max(
        len(n) for L in cleandoc(numbers_str).split("\n") for n in L.split()
    )
    if max_num_str_len > 1:
        max_num_str_len += 1
    numbers = np.array(
        [line.split() for line in cleandoc(numbers_str).split("\n")], dtype="str"
    )
    return HitoriPuzzle(
        np.array(
            [line.split() for line in cleandoc(board_str).split("\n") if line],
            dtype="str",
        ),
        numbers,
        max_num_str_len,
    )


# **** Board manipulation utilities
def load_pzprv3(pzprv3: str) -> HitoriPuzzle:
    """
    Loads PUZ-PRE v3 text and returns a Hitori board
    """
    pzprv3_lines = pzprv3.split("\n")
    rows = int(pzprv3_lines[2])
    cols = int(pzprv3_lines[3])

    # The size of the number strings is a suprising hassle. If a Hitori puzzle has
    # numbers with multiple digits, we need to capture that.
    max_num_str_len = max(len(n) for L in pzprv3_lines[4 : 4 + rows] for n in L.split())
    if max_num_str_len > 1:  # if it's a multiple digit puzzle, put spaces between cells
        max_num_str_len += 1

    numbers = np.zeros((rows, cols), dtype=f"<U{max_num_str_len}")
    numbers[:, :] = "-"
    for i, row in enumerate(pzprv3_lines[4 : 4 + rows]):
        numbers[i, :] = row.split()
    board = np.zeros((rows, cols), dtype="str")
    board[:, :] = "-"
    for i, row in enumerate(pzprv3_lines[4 + rows : 4 + rows * 2]):
        board[i, :] = row.split()
    return HitoriPuzzle(board, numbers, max_num_str_len)


def step_list_repr(step_list: list[Step]) -> str:
    return "\n".join(repr(step) for step in step_list)


def verbose_output(puzzle: Puzzle) -> None:
    sb = search_base(puzzle)
    print("*** PROOF FOR WHY CONTRADICTORY ***")
    print(sb[1])
    print()
    print("*** STEPS ***")
    # print(sb[2])
    print(
        black.format_str(
            step_list_repr(sb[2]),
            mode=black.Mode(
                line_length=88,
            ),
        )
    )
    print()
    print("*** FINAL ***")
    sb[0].print_just_board()
    print()
    sb[0].print_board()
    print(f"Solved: {not bool(sb[1]) and not puzzle.any_unknown()}")
    print(f"Difficulty: {max(step.cost for step in sb[2])}")
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


if __name__ == "__main__":
    sys.exit(main())
