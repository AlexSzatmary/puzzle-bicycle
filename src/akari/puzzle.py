from itertools import zip_longest

import numpy as np

# format
# Board:
# 0, 1, 2, 3, 4: number of bulbs
# -: black, unnumbered
# .: empty
# Solution:
# #: bulb
# Marks:
# _, |, x: indicate light paths, x shows both directions,
# + indicates no bulb but direction not indicated


def load_pzprv3(pzprv3: str) -> np.ndarray:
    """
    Loads PUZ-PRE v3 text and returns an Akari board
    """
    pzprv3_lines = pzprv3.replace(" ", "").replace("/", "").split("\n")
    # print(pzprv3)
    # print()
    # print(pzprv3_lines)
    rows = int(pzprv3_lines[2])
    cols = int(pzprv3_lines[3])
    board = np.zeros((rows + 2, cols + 2), dtype="str")
    board[:, :] = "-"
    for i, row in enumerate(pzprv3_lines[4 : 4 + rows]):
        board[i + 1, 1:-1] = list(row.replace(" ", ""))
    return board


def save_pzprv3(board: np.ndarray):
    lines = []
    lines.append("pzprv3")
    lines.append("lightup")
    lines.append(str(board.shape[0] - 2))
    lines.append(str(board.shape[1] - 2))
    for row in board[1:-1]:
        lines.append(" ".join(row[1:-1]))
    lines.append("")
    return "\n".join(lines)



def zero_pad(grid):
    grid2 = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2), dtype="str")
    grid2[:, :] = "-"
    grid2[1:-1, 1:-1] = grid
    return grid2


def print_board(board):
    for row in board.astype(str):
        print("".join(list(row)))


def check_number(board):
    """
    Checks numbered spaces to see if they have the correct number of bulbs.

    Returns a list of tuples of coordinates of numbered spaces that touch the wrong
    number of bulbs.
    """
    wrong_bulbs = []
    for i in range(1, np.size(board, 0) - 1):
        for j in range(1, np.size(board, 1) - 1):
            if board[i, j] in "0123":
                # breakpoint()
                if not int(board[i, j]) == (
                    (board[i - 1, j] == "#")
                    + (board[i + 1, j] == "#")
                    + (board[i, j - 1] == "#")
                    + (board[i, j + 1] == "#")
                ):
                    wrong_bulbs.append((i, j))
    return wrong_bulbs


def illuminate(board):
    """
    Takes board with bulbs. Returns a tuple with
    *a list of lists of tuples of coordinates of bulbs that shine on each other
    *a copy of board with light paths drawn
    """
    lit_bulb_pairs = []
    board = board.copy()
    fill_chars = ["|", "_", "|", "_"]
    for i in range(1, np.size(board, 0) - 1):
        for j in range(1, np.size(board, 1) - 1):
            if board[i, j] == "#":
                iters = [
                    zip_longest(range(i - 1, 0, -1), [], fillvalue=j),
                    zip_longest([], range(j - 1, 0, -1), fillvalue=i),
                    zip_longest(range(i + 1, np.size(board, 0) - 1), [], fillvalue=j),
                    zip_longest([], range(j + 1, np.size(board, 0) - 1), fillvalue=i),
                ]
                for it, fill_char in zip(iters, fill_chars):
                    for i1, j1 in it:
                        if board[i1, j1] == "#":
                            if i <= i1 and j <= j1:
                                lit_bulb_pairs.append((i, j, i1, j1))
                            break
                        elif board[i1, j1] == fill_char:
                            # wrong bulb pair already detected
                            break
                        elif board[i1, j1] == "_" or board[i1, j1] == "|":
                            # this branch will only trigger if the char at this location
                            # is not the same as the fill_char
                            board[i1, j1] = "x"
                        elif board[i1, j1] in "01234-":
                            break
                        else:
                            board[i1, j1] = fill_char
    return (lit_bulb_pairs, board)


def check_unlit_cells(board):
    """
    Returns True if a board has no unlit cells, False otherwise
    """
    (wrong_bulb_pairs, board) = illuminate(board)
    return not np.any(board == ".") == np.True_


def check_lit_bulbs(board):
    """
    Returns True if a board has no lit bulbs, False otherwise
    """
    (wrong_bulb_pairs, board) = illuminate(board)
    return not bool(wrong_bulb_pairs)


def check_all(board):
    return (
        not check_number(board) and check_unlit_cells(board) and check_lit_bulbs(board)
    )
