import os
# from itertools import cycle

import numpy as np
import puzzle
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QBrush, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)


def fake_next_state(state: str) -> str:
    s = "._-|+O#01234x."
    return s[s.find(state) + 1]


class Cell(QWidget):
    clicked = Signal()

    def __init__(self, main_window, i, j, state, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.i = i
        self.j = j
        self.main_window = main_window
        self.setFixedSize(QSize(20, 20))
        self.state = state

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(0, 0, 20, 20, QBrush(Qt.white))
        p.drawRect(0, 0, 20, 20)

        match self.state:
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
            case "-":
                self.draw_black_square(event)
            case num if "0" <= num <= "4":
                self.draw_black_square(event)
                self.draw_num(event, num)

    def draw_horizontal(self, event) -> None:
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

    def draw_vertical(self, event) -> None:
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

    def draw_cross(self, event) -> None:
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

    def draw_circle(self, event) -> None:
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(Qt.gray)
        p.setPen(pen)
        p.drawEllipse(2, 2, 16, 16)

    def draw_dot(self, event) -> None:
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(Qt.gray)
        p.setPen(pen)
        p.drawEllipse(9, 9, 2, 2)

    def draw_black_square(self, event) -> None:
        p = QPainter(self)
        p.fillRect(0, 0, 20, 20, QBrush(Qt.black))

    def draw_num(self, event, num: str) -> None:
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

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        match self.main_window.board[self.i + 1, self.j + 1]:
            case "#":
                self.state = "."
                self.main_window.board[self.i + 1, self.j + 1] = self.state
            case ".":
                self.state = "+"
                self.main_window.board[self.i + 1, self.j + 1] = self.state
            case "+":
                self.state = "#"
                self.main_window.board[self.i + 1, self.j + 1] = self.state
        self.clicked.emit()
        self.update()
        return super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.w = QWidget()
        self.hb = QHBoxLayout()
        self.vb = QVBoxLayout()
        self.grid = QGridLayout()
        self.w.setLayout(self.hb)
        self.hb.addLayout(self.vb)
        self.hb.addLayout(self.grid)
        self.setCentralWidget(self.w)

        self.open = QPushButton()
        self.open.setText("Open")
        self.vb.addWidget(self.open)
        self.open.pressed.connect(self.open_pressed)

        self.save = QPushButton()
        self.save.setText("Save")
        self.vb.addWidget(self.save)
        self.save.pressed.connect(self.save_pressed)

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

        # symbols = cycle(".-|+O#01234*.")
        # for i in range(5):
        #     for j in range(5):
        #         c = Cell(0, 0, self)
        #         c.state = next(symbols)
        #         self.grid.addWidget(c, i, j)
        self.board = np.zeros((7, 7), dtype=str)
        self.board[:] = "-"
        self.board[1:-1, 1:-1] = "."
        for i in range(5):
            for j in range(5):
                c = Cell(self, i, j, self.board[i + 1, j + 1])
                self.grid.addWidget(c, i, j)
        self.grid.setSpacing(0)
        self.grid.addItem(
            QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding),
            5,
            0,
        )
        self.show()

    def open_pressed(self) -> None:
        qfd = QFileDialog()
        qfd.open()
        filename, _ = qfd.getOpenFileName(
            self, "Open pzprv3", os.path.dirname(__file__), "(*.txt)"
        )
        with open(filename) as hin:
            text = hin.read()
        self.board = puzzle.load_pzprv3(text)
        clearLayout(self.grid)
        self.grid = QGridLayout()
        for i in range(self.board.shape[0] - 2):
            for j in range(self.board.shape[0] - 2):
                c = Cell(self, i, j, self.board[i + 1, j + 1])
                self.grid.addWidget(c, i, j)
                c.update()
        self.grid.setSpacing(0)
        self.grid.addItem(
            QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding),
            self.grid.rowCount(),
            0,
        )
        self.show()
        self.hb.addLayout(self.grid)

    def save_pressed(self) -> None:
        qfd = QFileDialog()
        qfd.open()
        filename, _ = qfd.getSaveFileName(
            self, "Save pzprv3", os.path.dirname(__file__), "(*.txt)"
        )
        pzprv3 = puzzle.save_pzprv3(self.board)
        with open(filename, "w") as hout:
            hout.write(pzprv3)


# Taken from https://stackoverflow.com/a/9383780/400793 by ekhumoro
def clearLayout(layout):
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
