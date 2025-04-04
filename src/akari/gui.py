import os
from typing import Any, cast

import numpy as np
import puzzle
from PySide6.QtCore import (
    Property,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
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

AUTO_APPLY_METHODS_LEVEL = 1


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
        self._white_out_step = 0

    def paintEvent(self, event: QPaintEvent) -> None:
        if self.state_auto in "-01234":
            self.paint_black(event)
        else:
            self.paint_white(event)

    def paint_white(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.fillRect(0, 0, 20, 20, QBrush(Qt.white))
        p.drawRect(0, 0, 20, 20)

        if not self.correct:
            brush = QBrush()
            brush.setStyle(Qt.DiagCrossPattern)
            brush.setColor(Qt.black)
            p.fillRect(0, 0, 20, 20, brush)

        p.end()

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

        if not self.main_window.puzzle_complete:
            self.superimpose_user_state(event)
        elif self.state_auto != "#":
            white_out_patterns = [
                Qt.Dense6Pattern,
                Qt.Dense5Pattern,
                Qt.Dense4Pattern,
                Qt.Dense3Pattern,
                Qt.Dense2Pattern,
                Qt.Dense1Pattern,
                Qt.SolidPattern,
            ]
            brush = QBrush()
            brush.setStyle(white_out_patterns[cast(int, self.white_out_step)])
            brush.setColor(Qt.white)
            p = QPainter(self)
            p.fillRect(1, 1, 18, 18, brush)
            p.end()

    def white_out_step_getter(self) -> int:
        return self._white_out_step

    def white_out_step_setter(self, i: int) -> None:
        self._white_out_step = i
        self.update()

    white_out_step = Property(
        int, white_out_step_getter, white_out_step_setter, freset=None, doc=""
    )

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
        p.end()

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
        p.end()

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
        p.end()

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
        p.end()

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
        p.end()

    def draw_black_square(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        brush = QBrush()
        if self.correct:
            brush.setStyle(Qt.SolidPattern)
        else:
            brush.setStyle(Qt.Dense3Pattern)
        p.setBrush(brush)
        p.fillRect(0, 0, 20, 20, brush)
        p.end()

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
        p.end()

    def draw_dotted_border(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(1)
        pen.setColor(Qt.gray)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawRect(2, 2, 16, 16)
        p.end()

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
        self.auto_apply_methods_level = AUTO_APPLY_METHODS_LEVEL
        self.puzzle_complete = False
        self.puzzle_file_name = os.path.join(os.path.dirname(__file__), "default.txt")

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

        self.methods_group = QActionGroup(self)
        for level in range(10):
            if 6 < level < 9:
                continue  # these levels are not yet implemented
            if level == 0:
                name = "No automatic solving"
            elif level == 1:
                name = "Just illuminate"
            elif level == 9:
                name = "Full automatic"
            else:
                name = f"Level {level}"
            action = QAction(name, self.methods_group)
            action.triggered.connect(self.auto_level_checked)
            action.setShortcut(QKeySequence(f"Ctrl+{level}"))
            action.setCheckable(True)
            action.setChecked(level == AUTO_APPLY_METHODS_LEVEL)
            settings_menu.addAction(action)
        self.board = puzzle.load_pzprv3(pzprv3_1)
        self.board_auto = self.board.copy()
        self.initialize_grid()

        self.puzzle_status = QLabel()
        self.puzzle_status.setText("")
        self.puzzle_status.setVisible(False)
        labelhb = QHBoxLayout()
        labelhb.addItem(
            QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum),
        )
        labelhb.addWidget(self.puzzle_status)
        labelhb.addItem(
            QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum),
        )
        self.labelhb = labelhb
        self.vbr.addLayout(labelhb)

        self.apply_methods()
        self.adjustSize()
        self.show()
        self.resize(self.sizeHint())

    def open_pressed(self) -> None:
        qfd = QFileDialog()
        filename, _ = qfd.getOpenFileName(
            self, "Open pzprv3", os.path.dirname(self.puzzle_file_name), "(*.txt)"
        )
        with open(filename) as hin:
            text = hin.read()
        if text:
            self.puzzle_file_name = filename
            self.board = puzzle.load_pzprv3(text)
            self.board_auto = puzzle.illuminate_all(self.board)[1]
            self.puzzle_complete = False
            self.puzzle_status.setText("")
            self.puzzle_status.setVisible(False)
            clearLayout(self.grid)
            self.initialize_grid()
            self.apply_methods()
            QTimer.singleShot(0, self.adjustSize)

    def auto_level_checked(self) -> None:
        self.auto_apply_methods_level = next(
            i
            for (i, action) in enumerate(self.methods_group.actions())
            if action.isChecked()
        )
        if self.auto_apply_methods_level + 1 == len(self.methods_group.actions()):
            self.auto_apply_methods_level = 9
        self.update_all(self.board)
        self.board_auto = self.board
        self.apply_methods()

    def initialize_grid(self) -> None:
        self.grid = QGridLayout()
        for i in range(self.board_auto.shape[0] - 2):
            for j in range(self.board_auto.shape[1] - 2):
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
        self.vbr.insertLayout(0, self.grid)

    def save_pressed(self) -> None:
        qfd = QFileDialog()
        qfd.open()
        filename, _ = qfd.getSaveFileName(
            self, "Save pzprv3", self.puzzle_file_name, "(*.txt)"
        )
        if filename:
            pzprv3 = puzzle.save_pzprv3(self.board)
            with open(filename, "w") as hout:
                hout.write(pzprv3)

    def apply_methods(self) -> None:
        thought_process = puzzle.ThoughtProcess(self.board.copy())
        thought_process.apply_methods(self.auto_apply_methods_level)
        new_board_auto = thought_process.board

        old_puzzle_complete = self.puzzle_complete
        self.puzzle_complete = puzzle.check_all(new_board_auto)
        if self.puzzle_complete != old_puzzle_complete:
            if self.puzzle_complete:
                self.puzzle_status.setText("Puzzle solved")
                self.puzzle_status.setVisible(True)
            else:
                self.puzzle_status.setText("")
                self.puzzle_status.setVisible(False)
                QTimer.singleShot(0, self.adjustSize)

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

        self.indicate_contradictions(thought_process)
        self.board_auto = new_board_auto
        if self.puzzle_complete and not old_puzzle_complete:
            self.animate_puzzle_complete()

    def update_all(self, board: np.ndarray) -> None:
        for i in range(self.board_auto.shape[0] - 2):
            for j in range(self.board_auto.shape[1] - 2):
                ci = self.grid.itemAtPosition(i, j)
                assert ci is not None
                c = ci.widget()
                c.state_auto = board[i + 1, j + 1]
                c.correct = True  # presume correct then indicate if not elsewhere
                c.update()

    def indicate_contradictions(self, thought_process: puzzle.ThoughtProcess) -> None:
        for i, j in thought_process.wrong_numbers:
            ci = self.grid.itemAtPosition(i - 1, j - 1)
            assert ci is not None
            c = ci.widget()
            c.correct = False
            c.update()

        for i1, j1, i2, j2 in thought_process.lit_bulb_pairs:
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

        for i, j in thought_process.unilluminatable_cells:
            ci = self.grid.itemAtPosition(i - 1, j - 1)
            assert ci is not None
            c = ci.widget()
            c.correct = False
            c.update()

    def animate_puzzle_complete(self) -> None:
        pag = QParallelAnimationGroup()
        for i in range(self.board_auto.shape[0] - 2):
            for j in range(self.board_auto.shape[1] - 2):
                ci = self.grid.itemAtPosition(i, j)
                assert ci is not None
                c = ci.widget()
                if self.board_auto[i + 1, j + 1] in "_|x+":
                    c.anim = QPropertyAnimation(c, b"white_out_step")
                    c.anim.setStartValue(0)
                    c.anim.setEndValue(6)
                    c.anim.setDuration(500)
                    pag.addAnimation(c.anim)
        pag.start()
        self.pag = pag


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
