import sys

import puzzle


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv
    boards = []
    for name in sys.argv[1:]:
        with open(name) as hin:
            text = hin.read()
        if text:
            board = puzzle.load_pzprv3(text)
            boards.append(board.copy())
            board_solved = puzzle.apply_methods(board.copy(), 9)
            print()
            print(name)
            puzzle.print_board(board_solved)
            print(f"solved: {puzzle.check_all(board_solved)}")
    for _i in range(10):
        for board in boards:
            puzzle.apply_methods(board.copy(), 9)


if __name__ == "__main__":
    sys.exit(main())
