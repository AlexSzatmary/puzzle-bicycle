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
    QIntValidator,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QPushButton,
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
        """
        Sends events based on mode
        """
        match self.main_window.edit_mode:
            case "play":
                self.click_play_mode(event)
            case "block":
                self.click_block_mode(event)
            case "number":
                self.click_number_mode(event)

    def set_state_user_and_mwb(self, value: str) -> None:
        self.state_user = value
        self.main_window.board[self.i + 1, self.j + 1] = self.state_user

    def click_play_mode(self, event: QMouseEvent) -> None:
        """
        Toggles white cells between free, dot, and bulb, for solving puzzles
        """
        match self.state_user:
            case "#":
                self.set_state_user_and_mwb(".")
            case ".":
                self.set_state_user_and_mwb("+")
            case "+":
                self.set_state_user_and_mwb("#")
            case _:
                return super().mouseReleaseEvent(event)
        self.clicked.emit()
        self.update()
        self.main_window.apply_methods()
        return super().mouseReleaseEvent(event)

    def click_block_mode(self, event: QMouseEvent) -> None:
        """
        Toggles white and black cells for editing puzzles

        Numbered cells lose numbers, white cells lose marks
        """
        if self.state_user in "-01234":
            self.set_state_user_and_mwb(".")
        else:
            self.set_state_user_and_mwb("-")
        self.clicked.emit()
        self.update()
        self.main_window.apply_methods()
        return super().mouseReleaseEvent(event)

    def click_number_mode(self, event: QMouseEvent) -> None:
        """
        Edits block numbers, and toggles white cells for test solves
        """
        black = "-01234"
        if (i := black.find(self.state_user)) != -1:
            self.set_state_user_and_mwb(black[(i + 1) % len(black)])
            self.clicked.emit()
            self.update()
            self.main_window.apply_methods()
            return super().mouseReleaseEvent(event)
        else:
            # have identical behavior to play mode for white cells
            return self.click_play_mode(event)


class NewPuzzleDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("New puzzle")
        layout = QGridLayout()
        layout.addWidget(QLabel("Rows"), 0, 0)
        self.n_rows = QLineEdit()
        self.n_rows.setValidator(QIntValidator(1, 99))
        self.n_rows.setMaxLength(2)
        layout.addWidget(self.n_rows, 0, 1)
        self.n_cols = QLineEdit()
        self.n_cols.setValidator(QIntValidator(1, 99))
        self.n_rows.setMaxLength(2)
        layout.addWidget(QLabel("Columns"), 1, 0)
        layout.addWidget(self.n_cols, 1, 1)
        self.buttonbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)
        layout.addWidget(self.buttonbox, 2, 0, 1, 2)
        self.setLayout(layout)


class ResizeDialog(QDialog):
    def __init__(self, parent: "MainWindow") -> None:
        super().__init__(parent)
        self.setWindowTitle("Resize")
        self.p = parent
        layout = QVBoxLayout()
        dpad = QGridLayout()

        b = QPushButton("+")
        b.clicked.connect(self.add_top)
        dpad.addWidget(b, 0, 2)
        b = QPushButton("+")
        b.clicked.connect(self.add_left)
        dpad.addWidget(b, 2, 0)
        b = QPushButton("+")
        b.clicked.connect(self.add_bottom)
        dpad.addWidget(b, 4, 2)
        b = QPushButton("+")
        b.clicked.connect(self.add_right)
        dpad.addWidget(b, 2, 4)
        b = QPushButton("-")
        b.clicked.connect(self.rem_top)
        dpad.addWidget(b, 1, 2)
        b = QPushButton("-")
        b.clicked.connect(self.rem_left)
        dpad.addWidget(b, 2, 1)
        b = QPushButton("-")
        b.clicked.connect(self.rem_bottom)
        dpad.addWidget(b, 3, 2)
        b = QPushButton("-")
        b.clicked.connect(self.rem_right)
        dpad.addWidget(b, 2, 3)
        layout.addLayout(dpad)
        self.buttonbox = QDialogButtonBox(QDialogButtonBox.Close)
        self.buttonbox.clicked.connect(self.close)
        layout.addWidget(self.buttonbox)
        self.setLayout(layout)
        self.p.show()

    def add_top(self) -> None:
        self.p.resize_board(1, 0, 0, 0)

    def add_left(self) -> None:
        self.p.resize_board(0, 1, 0, 0)

    def add_bottom(self) -> None:
        self.p.resize_board(0, 0, 1, 0)

    def add_right(self) -> None:
        self.p.resize_board(0, 0, 0, 1)

    def rem_top(self) -> None:
        self.p.resize_board(-1, 0, 0, 0)

    def rem_left(self) -> None:
        self.p.resize_board(0, -1, 0, 0)

    def rem_bottom(self) -> None:
        self.p.resize_board(0, 0, -1, 0)

    def rem_right(self) -> None:
        self.p.resize_board(0, 0, 0, -1)


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
        new_action = QAction("New", self)
        new_action.triggered.connect(self.new_pressed)
        new_action.setShortcut(QKeySequence("Ctrl+N"))
        file_menu.addAction(new_action)
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_pressed)
        open_action.setShortcut(QKeySequence("Ctrl+o"))
        file_menu.addAction(open_action)
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_pressed)
        save_action.setShortcut(QKeySequence("Ctrl+s"))
        file_menu.addAction(save_action)

        edit_menu = menu.addMenu("&Edit")
        resize_action = QAction("Resize", self)
        resize_action.triggered.connect(self.resize_pressed)
        resize_action.setShortcut(QKeySequence("Ctrl+U"))
        edit_menu.addAction(resize_action)
        self.clear_board_action = QAction("Clear board")
        self.clear_board_action.triggered.connect(self.clear_board)
        self.clear_board_action.setShortcut(QKeySequence("Ctrl+K"))
        # self.clear_board_action.setCheckable(True)
        # self.clear_board_action.setChecked(False)
        edit_menu.addAction(self.clear_board_action)

        settings_menu = menu.addMenu("&Settings")

        edit_menu.addSeparator()
        self.edit_mode_group = QActionGroup(self)
        self.edit_mode = "play"
        action = QAction("Play", self.edit_mode_group)
        action.triggered.connect(self.edit_mode_changed)
        action.setShortcut(QKeySequence("Ctrl+E"))
        action.setCheckable(True)
        action.setChecked(True)
        edit_menu.addAction(action)
        action = QAction("Edit blocks", self.edit_mode_group)
        action.triggered.connect(self.edit_mode_changed)
        action.setShortcut(QKeySequence("Ctrl+R"))
        action.setCheckable(True)
        action.setChecked(False)
        edit_menu.addAction(action)
        action = QAction("Edit numbers", self.edit_mode_group)
        action.triggered.connect(self.edit_mode_changed)
        action.setShortcut(QKeySequence("Ctrl+T"))
        action.setCheckable(True)
        action.setChecked(False)
        edit_menu.addAction(action)

        settings_menu.addSeparator()
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

        settings_menu.addSeparator()
        self.contradiction_checker_enabled = False
        self.contradiction_action = QAction("Check for contradictions")
        self.contradiction_action.triggered.connect(self.set_contradiction_checker_mode)
        self.contradiction_action.setShortcut(QKeySequence("Ctrl+D"))
        self.contradiction_action.setCheckable(True)
        self.contradiction_action.setChecked(False)
        settings_menu.addAction(self.contradiction_action)

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

    def new_pressed(self) -> None:
        dlg = NewPuzzleDialog()
        if dlg.exec():
            n_rows = int(dlg.n_rows.text())
            n_cols = int(dlg.n_cols.text())
            if n_rows < 1 or n_cols < 1:
                return
            self.board = puzzle.new_blank_board(n_rows, n_cols)
            self.board_auto = self.board.copy()
            self.puzzle_complete = False
            self.puzzle_status.setText("")
            self.puzzle_status.setVisible(False)
            clearLayout(self.grid)
            self.initialize_grid()
            self.apply_methods()
            QTimer.singleShot(0, self.adjustSize)

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

    def clear_board(self) -> None:
        self.board = puzzle.clear_board(self.board)
        self.update_all(self.board)
        for i in range(self.board_auto.shape[0] - 2):
            for j in range(self.board_auto.shape[1] - 2):
                ci = self.grid.itemAtPosition(i, j)
                assert ci is not None
                c = ci.widget()
                c.state_user = "."
        self.board_auto = self.board
        self.apply_methods()

    def resize_pressed(self) -> None:
        dlg = ResizeDialog(self)
        if dlg.show():
            print("resize closed")

    def resize_board(
        self, delta_top: int, delta_left: int, delta_bottom: int, delta_right: int
    ) -> None:
        self.board = puzzle.resize_board(
            self.board, delta_top, delta_left, delta_bottom, delta_right
        )
        self.board_auto = self.board.copy()
        self.puzzle_complete = False
        self.puzzle_status.setText("")
        self.puzzle_status.setVisible(False)
        clearLayout(self.grid)
        self.initialize_grid()
        self.apply_methods()
        QTimer.singleShot(0, self.adjustSize)

    def set_contradiction_checker_mode(self) -> None:
        """
        Toggles contradiction checker mode
        """
        self.contradiction_checker_enabled = self.contradiction_action.isChecked()
        self.update_all(self.board)
        self.board_auto = self.board
        self.apply_methods()

    def edit_mode_changed(self) -> None:
        """
        Toggles mode between Play, Block, and Number
        """
        modes = ["play", "block", "number"]
        self.edit_mode = modes[
            next(
                i
                for (i, action) in enumerate(self.edit_mode_group.actions())
                if action.isChecked()
            )
        ]
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

        # if the contradiction checker is on, run a full solve and, if a contradiction
        # is detected, hijack the render
        if self.contradiction_checker_enabled:
            thought_process = self.handle_contradiction(thought_process)

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

    def handle_contradiction(
        self,
        thought_process: puzzle.ThoughtProcess,
    ) -> puzzle.ThoughtProcess:
        thought_process_contradiction = puzzle.ThoughtProcess(self.board.copy())
        thought_process_contradiction.apply_methods(9)
        if not thought_process_contradiction.check_unsolved():
            thought_process = thought_process_contradiction
        return thought_process

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
