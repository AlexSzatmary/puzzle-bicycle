import sys

import puzzle


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv
    for name in sys.argv[1:]:
        with open(name) as hin:
            text = hin.read()
        if text:
            board = puzzle.load_pzprv3(text)
            board = puzzle.apply_methods(board, 9)
            print()
            print(name)
            puzzle.print_board(board)
            print(f"solved: {puzzle.check_all(board)}")


if __name__ == "__main__":
    sys.exit(main())
