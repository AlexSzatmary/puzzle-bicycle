import os
from typing import Any

import numpy as np
import puzzle
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

AUTO_ILLUMINATE = True
AUTO_APPLY_METHODS = 9


def fake_next_state(state: str) -> str:
    s = "._-|+O#01234x."
    return s[s.find(state) + 1]


class Cell(QWidget):
    clicked = Signal()

    def __init__(
        self,
        main_window: QMainWindow,
        i: int,
        j: int,
        state: str,
        *args: Any,  # noqa: ANN401
        correct: bool = True,
        **kwargs: dict,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.i = i
        self.j = j
        self.main_window = main_window
        self.setFixedSize(QSize(20, 20))
        self.state_user = state
        self.state_auto = state
        self.correct = correct

    def paintEvent(self, event: QPaintEvent) -> None:
        if self.state_auto in "-01234":
            self.paint_black(event)
        else:
            self.paint_white(event)

    def paint_white(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.fillRect(0, 0, 20, 20, QBrush(Qt.white))
        p.drawRect(0, 0, 20, 20)

        if not self.main_window.puzzle_complete:
            if not self.correct:
                brush = QBrush()
                brush.setStyle(Qt.DiagCrossPattern)
                brush.setColor(Qt.black)
                p.fillRect(0, 0, 20, 20, brush)

            match self.state_auto:
                case "_":
                    self.draw_horizontal(event)
                case "|":
                    self.draw_vertical(event)
                case "x":
                    self.draw_cross(event)
                case "#":
                    self.draw_circle(event)
                case "+":
                    self.draw_dot(event)
            self.superimpose_user_state(event)
        elif self.state_auto == "#":
            self.draw_circle(event)

    def paint_black(self, event: QPaintEvent) -> None:
        match self.state_auto:
            case "-":
                self.draw_black_square(event)
            case num if "0" <= num <= "4":
                self.draw_black_square(event)
                self.draw_num(event, num)

    def superimpose_user_state(self, event: QPaintEvent) -> None:
        if self.state_auto in "_|x":
            if self.state_user == "+":
                self.draw_dot(event)
        elif self.state_auto != self.state_user:
            self.draw_dotted_border(event)

    def draw_horizontal(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(1)
        pen.setColor(Qt.gray)
        p.setPen(pen)
        brush = QBrush()
        brush.setColor(Qt.gray)
        brush.setStyle(Qt.SolidPattern)
        p.setBrush(brush)
        p.drawRect(2, 9, 16, 2)

    def draw_vertical(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(1)
        pen.setColor(Qt.gray)
        p.setPen(pen)
        brush = QBrush()
        brush.setColor(Qt.gray)
        brush.setStyle(Qt.SolidPattern)
        p.setBrush(brush)
        p.drawRect(9, 2, 2, 16)

    def draw_cross(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(1)
        pen.setColor(Qt.gray)
        p.setPen(pen)
        brush = QBrush()
        brush.setColor(Qt.gray)
        brush.setStyle(Qt.SolidPattern)
        p.setBrush(brush)
        p.drawRect(2, 9, 16, 2)
        p.drawRect(9, 2, 2, 16)

    def draw_circle(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(Qt.gray)
        p.setPen(pen)
        if self.state_user == "#" or self.main_window.puzzle_complete:
            p.drawEllipse(2, 2, 16, 16)
        else:
            p.drawEllipse(4, 4, 12, 12)

    def draw_dot(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(Qt.gray)
        p.setPen(pen)
        # p.drawEllipse(9, 9, 2, 2)
        brush = QBrush()
        brush.setColor(Qt.gray)
        brush.setStyle(Qt.SolidPattern)
        p.setBrush(brush)
        p.drawEllipse(8, 8, 4, 4)

    def draw_black_square(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        brush = QBrush()
        if self.correct:
            brush.setStyle(Qt.SolidPattern)
        else:
            brush.setStyle(Qt.Dense3Pattern)
        p.setBrush(brush)
        p.fillRect(0, 0, 20, 20, brush)

    def draw_num(self, event: QPaintEvent, num: str) -> None:
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(Qt.white)
        p.setPen(pen)
        p.drawText(
            2,
            2,
            16,
            16,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            str(num),
        )

    def draw_dotted_border(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(1)
        pen.setColor(Qt.gray)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawRect(2, 2, 16, 16)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        mwb = self.main_window.board
        match self.state_user:
            case "#":
                self.state_user = "."
                mwb[self.i + 1, self.j + 1] = self.state_user
            case ".":
                self.state_user = "+"
                mwb[self.i + 1, self.j + 1] = self.state_user
            case "+":
                self.state_user = "#"
                mwb[self.i + 1, self.j + 1] = self.state_user
            case _:
                return super().mouseReleaseEvent(event)
        self.clicked.emit()
        self.update()
        self.main_window.apply_methods()
        return super().mouseReleaseEvent(event)


pzprv3_1 = """
pzprv3/
lightup/
5/
5/
. . . . . /
. . 0 . . /
. 2 - 1 . /
. . 3 . . /
. . . . . /
"""[1:]


class MainWindow(QMainWindow):
    def __init__(self, *args: Any, **kwargs: dict) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)
        self.w = QWidget()
        self.hb = QHBoxLayout()
        self.vbl = QVBoxLayout()
        self.vbr = QVBoxLayout()
        self.w.setLayout(self.hb)
        self.hb.addLayout(self.vbl)
        self.hb.addLayout(self.vbr)
        self.setCentralWidget(self.w)
        self.auto_illuminate = True
        self.auto_apply_methods = 9
        self.full_auto = True
        self.puzzle_complete = False

        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_pressed)
        open_action.setShortcut(QKeySequence("Ctrl+o"))
        file_menu.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_pressed)
        save_action.setShortcut(QKeySequence("Ctrl+s"))
        file_menu.addAction(save_action)

        settings_menu = menu.addMenu("&Settings")
        auto_illuminate_action = QAction("Illuminate", self)
        auto_illuminate_action.triggered.connect(self.auto_illuminate_toggled)
        auto_illuminate_action.setShortcut(QKeySequence("Ctrl+i"))
        auto_illuminate_action.setCheckable(True)
        auto_illuminate_action.setChecked(self.auto_illuminate)
        settings_menu.addAction(auto_illuminate_action)

        auto_apply_methods = QAction("Full auto", self)
        auto_apply_methods.triggered.connect(self.full_auto_toggled)
        auto_apply_methods.setShortcut(QKeySequence("Ctrl+9"))
        auto_apply_methods.setCheckable(True)
        auto_apply_methods.setChecked(self.auto_apply_methods == 9)
        settings_menu.addAction(auto_apply_methods)

        # TODO
        # self.clear = QPushButton()
        # self.clear.setText("Clear")
        # self.vb.addWidget(self.clear)

        # self.check = QPushButton()
        # self.check.setText("Check")
        # self.vb.addWidget(self.check)

        # self.solve = QPushButton()
        # self.solve.setText("Solve")
        # self.vb.addWidget(self.solve)

        # self.settings = QPushButton()
        # self.settings.setText("Settings")
        # self.vb.addWidget(self.settings)

        # self.board = np.zeros((7, 7), dtype=str)
        # self.board[:] = "-"
        # self.board[1:-1, 1:-1] = "."
        self.board = puzzle.load_pzprv3(pzprv3_1)
        self.board_auto = self.board.copy()
        self.initialize_grid()
        self.apply_methods()

        self.puzzle_status = QLabel()
        self.puzzle_status.setText("")
        self.puzzle_status.setVisible(False)
        self.vbr.addWidget(self.puzzle_status)

        self.show()

    def open_pressed(self) -> None:
        qfd = QFileDialog()
        filename, _ = qfd.getOpenFileName(
            self, "Open pzprv3", os.path.dirname(__file__), "(*.txt)"
        )
        with open(filename) as hin:
            text = hin.read()
        if text:
            self.board = puzzle.load_pzprv3(text)
            self.board_auto = puzzle.illuminate(self.board)[1]
            clearLayout(self.grid)
            self.initialize_grid()
            self.apply_methods()

    def auto_illuminate_toggled(self) -> None:
        self.auto_illuminate = not self.auto_illuminate
        if self.auto_illuminate:
            self.board_auto = self.board.copy()
            self.apply_methods()
        else:
            self.board_auto = self.board.copy()
            clearLayout(self.grid)
            self.initialize_grid()
            self.apply_methods()

    def full_auto_toggled(self) -> None:
        self.full_auto = not self.full_auto
        if self.full_auto:
            self.board_auto = self.board.copy()
            self.apply_methods()
        else:
            self.board_auto = self.board.copy()
            clearLayout(self.grid)
            self.initialize_grid()
            self.apply_methods()

    def initialize_grid(self) -> None:
        self.grid = QGridLayout()
        for i in range(self.board_auto.shape[0] - 2):
            for j in range(self.board_auto.shape[0] - 2):
                c = Cell(self, i, j, self.board_auto[i + 1, j + 1])
                self.grid.addWidget(c, i, j)
                c.update()
        self.grid.setSpacing(0)
        self.grid.addItem(
            QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding),
            self.grid.rowCount(),
            0,
        )
        self.grid.addItem(
            QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum),
            0,
            self.grid.columnCount(),
        )
        self.show()
        self.vbr.insertLayout(0, self.grid)

    def save_pressed(self) -> None:
        qfd = QFileDialog()
        qfd.open()
        filename, _ = qfd.getSaveFileName(
            self, "Save pzprv3", os.path.dirname(__file__), "(*.txt)"
        )
        if filename:
            pzprv3 = puzzle.save_pzprv3(self.board)
            with open(filename, "w") as hout:
                hout.write(pzprv3)

    def apply_methods(self) -> None:
        if self.full_auto:
            new_board_auto = puzzle.apply_methods(self.board.copy(), 9)
        elif self.auto_illuminate:
            new_board_auto = puzzle.illuminate(self.board.copy())[1]
        else:
            new_board_auto = self.board.copy()

        old_puzzle_complete = self.puzzle_complete
        self.puzzle_complete = puzzle.check_all(new_board_auto)
        if self.puzzle_complete != old_puzzle_complete:
            if self.puzzle_complete:
                self.puzzle_status.setText("Puzzle solved")
                self.puzzle_status.setVisible(True)
            else:
                self.puzzle_status.setText("")
                self.puzzle_status.setVisible(False)

        for i in range(self.board_auto.shape[0] - 2):
            for j in range(self.board_auto.shape[1] - 2):
                ci = self.grid.itemAtPosition(i, j)
                assert ci is not None
                c = ci.widget()
                if (
                    new_board_auto[i + 1, j + 1] != self.board_auto[i + 1, j + 1]
                    or not c.correct
                    or old_puzzle_complete != self.puzzle_complete
                ):
                    c.state_auto = new_board_auto[i + 1, j + 1]
                    c.correct = True  # presume correct then indicate if not below
                    c.update()

        self.indicate_contradictions(new_board_auto)
        self.board_auto = new_board_auto

    def indicate_contradictions(self, new_board_auto: np.ndarray) -> None:
        for i, j in puzzle.find_wrong_numbers(new_board_auto):
            ci = self.grid.itemAtPosition(i - 1, j - 1)
            assert ci is not None
            c = ci.widget()
            c.correct = False
            c.update()

        for i1, j1, i2, j2 in puzzle.illuminate(new_board_auto)[0]:
            if i1 == i2:
                ijs = [(i1, j) for j in range(j1, j2 + 1)]
            else:
                ijs = [(i, j1) for i in range(i1, i2 + 1)]
            for i, j in ijs:
                ci = self.grid.itemAtPosition(i - 1, j - 1)
                assert ci is not None
                c = ci.widget()
                c.correct = False
                c.update()


# Taken from https://stackoverflow.com/a/9383780/400793 by ekhumoro
def clearLayout(layout: QLayout) -> None:
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                clearLayout(item.layout())


if __name__ == "__main__":
    app = QApplication([])
    app.setApplicationName("Puzzle Bicycle")
    window = MainWindow()
    app.exec()
