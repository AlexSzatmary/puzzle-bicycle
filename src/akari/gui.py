from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSpacerItem,
    QSizePolicy,
)
from PySide6.QtCore import Qt
from itertools import cycle


def fake_next_state(state):
    s = ".-|+O#01234*."
    return s[s.find(state) + 1]


class Cell(QWidget):
    clicked = Signal()

    def __init__(self, x, y, main_window, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setFixedSize(QSize(20, 20))
        self.state = "."

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(0, 0, 20, 20, QBrush(Qt.white))
        p.drawRect(0, 0, 20, 20)

        match self.state:
            case "-":
                self.draw_horizontal(event)
            case "|":
                self.draw_vertical(event)
            case "+":
                self.draw_cross(event)
            case "O":
                self.draw_circle(event)
            case "*":
                self.draw_dot(event)
            case "#":
                self.draw_black_square(event)
            case num if "0" <= num <= "4":
                self.draw_black_square(event)
                self.draw_num(event, num)

    def draw_horizontal(self, event) -> None:
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(1)
        pen.setColor(Qt.gray)  # pyright: ignore[reportAttributeAccessIssue]
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
        pen.setColor(Qt.gray)  # pyright: ignore[reportAttributeAccessIssue]
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
        pen.setColor(Qt.gray)  # pyright: ignore[reportAttributeAccessIssue]
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
        pen.setColor(Qt.gray)  # pyright: ignore[reportAttributeAccessIssue]
        p.setPen(pen)
        # brush = QBrush()
        # brush.setColor(Qt.gray)
        # brush.setStyle(Qt.SolidPattern)
        # p.setBrush(brush)
        p.drawEllipse(2, 2, 16, 16)

    def draw_dot(self, event) -> None:
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(Qt.gray)  # pyright: ignore[reportAttributeAccessIssue]
        p.setPen(pen)
        # brush = QBrush()
        # brush.setColor(Qt.gray)
        # brush.setStyle(Qt.SolidPattern)
        # p.setBrush(brush)
        p.drawEllipse(9, 9, 2, 2)

    def draw_black_square(self, event) -> None:
        p = QPainter(self)
        p.fillRect(0, 0, 20, 20, QBrush(Qt.black))

    def draw_num(self, event, num: str) -> None:
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(Qt.white)  # pyright: ignore[reportAttributeAccessIssue]
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
        print("hi", self.state)
        self.state = fake_next_state(self.state)
        print("bye", self.state)
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

        symbols = cycle(".-|+O#01234*.")
        for i in range(5):
            for j in range(5):
                c = Cell(0, 0, self)
                c.state = next(symbols)
                self.grid.addWidget(c, i, j)
        self.grid.setSpacing(0)
        self.grid.addItem(
            QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding),
            5,
            0,
        )
        self.show()


if __name__ == "__main__":
    app = QApplication([])
    app.setApplicationName("Puzzle Bicycle")
    window = MainWindow()
    app.exec()
