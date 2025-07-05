import sys

import puzzle

N_RUNS = 10


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv
    names = []
    boards = []
    for name in sys.argv[1:]:
        with open(name) as hin:
            text = hin.read()
        if text:
            names.append(name)
            board = puzzle.load_pzprv3(text)
            boards.append(board.copy())
            tp = puzzle.ThoughtProcess(board)
            tp.apply_methods(9)
            print()
            print(name)
            puzzle.print_board(tp.board)
            print(f"solved: {puzzle.check_all(tp.board)}")
    for _i in range(N_RUNS - 1):
        print(_i)
        for name, board in zip(names, boards, strict=True):
            print(name)
            tp = puzzle.ThoughtProcess(board)
            tp.apply_methods(9)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
