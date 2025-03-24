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
            tp = puzzle.ThoughtProcess(board)
            tp.apply_methods(9)
            print()
            print(name)
            puzzle.print_board(tp.board)
            print(f"solved: {puzzle.check_all(tp.board)}")
    # for _i in range(10):
    #     print(_i)
    #     for board in boards:
    #         tp = puzzle.ThoughtProcess(board)
    #         tp.apply_methods(9)


if __name__ == "__main__":
    sys.exit(main())
