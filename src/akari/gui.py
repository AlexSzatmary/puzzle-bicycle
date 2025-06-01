import os
import sys
from collections.abc import Generator
from inspect import cleandoc
from typing import Any, cast

import numpy as np
import puzzle
from PySide6.QtCore import (
    Property,
    QEventLoop,
    QParallelAnimationGroup,
    QPoint,
    QPropertyAnimation,
    QRect,
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
    QPixmap,
    QRegion,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

AUTO_APPLY_METHODS_LEVEL = 1
ALWAYS_HINT = False


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
        self.state_hint = False

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
        elif self.state_hint:
            self.draw_bezel(event)

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

    def draw_bezel(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(1)
        pen.setColor(Qt.lightGray)
        p.setPen(pen)
        p.drawRect(2, 2, 16, 16)
        pen.setColor(Qt.darkGray)
        p.setPen(pen)
        p.drawRect(1, 1, 18, 18)
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
        self.state_hint = False
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
. 1 - 2 . /
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
        if os.path.exists(os.path.join(os.path.dirname(__file__), "Sample Puzzles")):
            self.puzzle_file_name = os.path.join(
                os.path.join(os.path.dirname(__file__), "Sample Puzzles"),
                "default.txt",
            )
        else:
            self.puzzle_file_name = os.path.join(
                os.path.dirname(__file__), "default.txt"
            )
        self.current_hint = None

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
        calc_diff_multi_action = QAction(
            "Calculate Difficulty for Multiple Files", self
        )
        calc_diff_multi_action.triggered.connect(self.calc_diff_multi_pressed)
        file_menu.addAction(calc_diff_multi_action)

        edit_menu = menu.addMenu("&Edit")
        resize_action = QAction("Resize", self)
        resize_action.triggered.connect(self.resize_pressed)
        resize_action.setShortcut(QKeySequence("Ctrl+U"))
        edit_menu.addAction(resize_action)
        self.check_board_action = QAction("Check Board")
        self.check_board_action.triggered.connect(self.check_board)
        self.check_board_action.setShortcut(QKeySequence("Ctrl+I"))
        edit_menu.addAction(self.check_board_action)
        self.clear_board_action = QAction("Clear Board")
        self.clear_board_action.triggered.connect(self.clear_board)
        self.clear_board_action.setShortcut(QKeySequence("Ctrl+K"))
        edit_menu.addAction(self.clear_board_action)
        calculate_difficulty_action = QAction("Calculate Difficulty", self)
        calculate_difficulty_action.triggered.connect(self.calculate_difficulty_pressed)
        edit_menu.addAction(calculate_difficulty_action)

        edit_menu.addSeparator()
        self.edit_mode_group = QActionGroup(self)
        self.edit_mode_radios = []
        self.edit_mode = "play"
        action = QAction("Play", self.edit_mode_group)
        action.triggered.connect(self.edit_mode_changed)
        action.setShortcut(QKeySequence("Ctrl+E"))
        action.setCheckable(True)
        action.setChecked(True)
        radio = QRadioButton("Play")
        action.toggled.connect(radio.setChecked)
        radio.toggled.connect(action.setChecked)
        radio.clicked.connect(self.edit_mode_changed)
        self.edit_mode_radios.append(radio)
        radio.setChecked(True)
        edit_menu.addAction(action)
        action = QAction("Edit Blocks", self.edit_mode_group)
        action.triggered.connect(self.edit_mode_changed)
        action.setShortcut(QKeySequence("Ctrl+R"))
        action.setCheckable(True)
        action.setChecked(False)
        radio = QRadioButton("Edit Blocks")
        action.toggled.connect(radio.setChecked)
        radio.toggled.connect(action.setChecked)
        radio.clicked.connect(self.edit_mode_changed)
        self.edit_mode_radios.append(radio)
        edit_menu.addAction(action)
        action = QAction("Edit Numbers", self.edit_mode_group)
        action.triggered.connect(self.edit_mode_changed)
        action.setShortcut(QKeySequence("Ctrl+T"))
        action.setCheckable(True)
        action.setChecked(False)
        radio = QRadioButton("Edit Numbers")
        action.toggled.connect(radio.setChecked)
        radio.toggled.connect(action.setChecked)
        radio.clicked.connect(self.edit_mode_changed)
        self.edit_mode_radios.append(radio)
        edit_menu.addAction(action)

        solver_menu = menu.addMenu("&Solver")
        solver_menu.addSeparator()
        self.methods_group = QActionGroup(self)
        self.methods_radios = []
        for level in range(10):
            if 6 < level < 9:
                continue  # these levels are not yet implemented
            if level == 0:
                name = "No Automatic Solving"
            elif level == 1:
                name = "Just Illuminate"
            elif level == 9:
                name = "Full Automatic"
            else:
                name = f"Level {level}"
            action = QAction(name, self.methods_group)
            action.triggered.connect(self.auto_level_checked)
            action.setShortcut(QKeySequence(f"Ctrl+{level}"))
            action.setCheckable(True)
            action.setChecked(level == AUTO_APPLY_METHODS_LEVEL)
            radio = QRadioButton(name)
            radio.setChecked(level == AUTO_APPLY_METHODS_LEVEL)
            action.toggled.connect(radio.setChecked)
            radio.toggled.connect(action.setChecked)
            radio.clicked.connect(self.auto_level_checked)
            self.methods_radios.append(radio)
            solver_menu.addAction(action)

        solver_menu.addSeparator()
        self.contradiction_checker_enabled = False
        self.contradiction_action = QAction("Check For Contradictions")
        self.contradiction_action.triggered.connect(self.set_contradiction_checker_mode)
        self.contradiction_action.setShortcut(QKeySequence("Ctrl+D"))
        self.contradiction_action.setCheckable(True)
        self.contradiction_action.setChecked(False)
        solver_menu.addAction(self.contradiction_action)
        self.hint_action = QAction("Hint")
        self.hint_action.triggered.connect(self.request_hint)
        self.hint_action.setShortcut(QKeySequence("Ctrl+A"))
        solver_menu.addAction(self.hint_action)

        view_menu = menu.addMenu("&View")
        self.show_controls_in_window_action = QAction("Show Controls In Window")
        self.show_controls_in_window_action.setCheckable(True)
        self.show_controls_in_window_action.setChecked(True)
        self.show_controls_in_window_action.toggled.connect(self.show_controls)
        view_menu.addAction(self.show_controls_in_window_action)

        self.board = puzzle.load_pzprv3(pzprv3_1)
        self.board_reference = puzzle.do_best_to_get_a_non_wrong_solution(self.board)

        self.puzzle_status = QLabel()
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
        self.vbr.addItem(QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.gb_solver = QGroupBox("Solver")
        self.vbl.addWidget(self.gb_solver)
        vbsolver = QVBoxLayout()
        for radio in self.methods_radios:
            vbsolver.addWidget(radio)
        self.contradiction_checker_checkbox = QCheckBox("Check For Contradictions")
        self.contradiction_checker_checkbox.toggled.connect(
            self.contradiction_action.setChecked
        )
        self.contradiction_action.toggled.connect(
            self.contradiction_checker_checkbox.setChecked
        )
        self.contradiction_checker_checkbox.clicked.connect(
            self.set_contradiction_checker_mode
        )
        vbsolver.addWidget(self.contradiction_checker_checkbox)
        self.gb_solver.setLayout(vbsolver)
        hint_button = QPushButton("Hint")
        hint_button.clicked.connect(self.hint_action.triggered)
        vbsolver.addWidget(hint_button)

        self.gb_edit = QGroupBox("Edit")
        self.vbl.addWidget(self.gb_edit)
        vbedit = QVBoxLayout()
        check_board_button = QPushButton("Check Board")
        check_board_button.clicked.connect(self.check_board_action.triggered)
        vbedit.addWidget(check_board_button)
        clear_board_button = QPushButton("Clear Board")
        clear_board_button.clicked.connect(self.clear_board_action.triggered)
        vbedit.addWidget(clear_board_button)
        for radio in self.edit_mode_radios:
            vbedit.addWidget(radio)
        self.gb_edit.setLayout(vbedit)
        self.vbl.addItem(QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding))
        resize_button = QPushButton("Resize")
        resize_button.clicked.connect(resize_action.triggered)
        vbedit.addWidget(resize_button)
        calculate_difficulty_button = QPushButton("Calculate Difficulty")
        calculate_difficulty_button.clicked.connect(
            calculate_difficulty_action.triggered
        )
        vbedit.addWidget(calculate_difficulty_button)

        self.refresh_GUI()
        self.show()
        self.resize(self.sizeHint())

    def show_controls(self) -> None:
        self.gb_solver.setVisible(self.show_controls_in_window_action.isChecked())
        self.gb_edit.setVisible(self.show_controls_in_window_action.isChecked())
        self.refresh_GUI()

    def new_pressed(self) -> None:
        dlg = NewPuzzleDialog()
        if dlg.exec():
            n_rows = int(dlg.n_rows.text())
            n_cols = int(dlg.n_cols.text())
            if n_rows < 1 or n_cols < 1:
                return
            self.board = puzzle.new_blank_board(n_rows, n_cols)
            self.refresh_GUI()

    def open_pressed(self) -> None:
        qfd = QFileDialog()
        filename, _ = qfd.getOpenFileName(
            self, "Open pzprv3", os.path.dirname(self.puzzle_file_name), "(*.txt)"
        )
        if not filename:
            return
        with open(filename) as hin:
            text = hin.read()
        if text:
            self.puzzle_file_name = filename
            self.board = puzzle.load_pzprv3(text)
            self.board_reference = puzzle.do_best_to_get_a_non_wrong_solution(
                self.board
            )
            self.most_recent_good_board = puzzle.intersect_boards(
                self.board, self.board_reference
            )
            # the default good board is cleared; if the puzzle is loaded in a valid
            # state, then that state gets set in apply_methods
            self.refresh_GUI()

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

    def calc_diff_multi_pressed(self) -> None:
        qfd = QFileDialog()
        filenames, _ = qfd.getOpenFileNames(
            self,
            "Open multiple files to calculate difficulty",
            os.path.dirname(self.puzzle_file_name),
            "(*.txt)",
        )
        output, _ = QFileDialog.getSaveFileName(self, "Save pzprv3", "", "(*.csv)")
        if not filenames or not output:
            return
        puzzle.batch_calculate_difficulty(filenames, output)

    def refresh_GUI(self) -> None:
        self.board_auto = puzzle.illuminate_all(self.board)[1]
        self.puzzle_complete = False
        self.puzzle_status.setText("")
        self.puzzle_status.setVisible(False)
        if hasattr(self, "grid"):
            clearLayout(self.grid)
            initializing = False
        else:
            initializing = True
        self.initialize_grid()
        self.apply_methods()
        if not initializing:
            QTimer.singleShot(0, self.adjustSize)

    def refresh_board(self) -> None:
        self.update_all_cells(self.board)
        self.board_auto = self.board
        self.apply_methods()

    def i_j_cell(self) -> Generator[tuple[int, int, Cell]]:
        """
        An iterator for looping over all cells
        """
        for i in range(self.board_auto.shape[0] - 2):
            for j in range(self.board_auto.shape[1] - 2):
                ci = self.grid.itemAtPosition(i, j)
                assert ci is not None
                c = ci.widget()
                yield i, j, cast(Cell, c)

    def auto_level_checked(self) -> None:
        """
        Changes solver level and updates board accordingly
        """
        self.auto_apply_methods_level = next(
            i
            for (i, action) in enumerate(self.methods_group.actions())
            if action.isChecked()
        )
        if self.auto_apply_methods_level + 1 == len(self.methods_group.actions()):
            self.auto_apply_methods_level = 9
        self.refresh_board()

    def clear_board(self) -> None:
        """
        Zeros out player state
        """
        self.board = puzzle.clear_board(self.board)
        for _i, _j, c in self.i_j_cell():
            if c.state_user in "#+_|x":
                c.state_user = "."
        self.refresh_board()

    def check_board(self) -> None:
        thought_process_correct = puzzle.ThoughtProcess(self.board)
        thought_process_correct.apply_methods(9)
        if not thought_process_correct.check_unsolved():
            thought_process_solved = puzzle.ThoughtProcess(
                puzzle.clear_board(self.board.copy())
            )
            thought_process_solved.apply_methods(9)
            if not thought_process_solved.check_unsolved():
                msg = QMessageBox(
                    QMessageBox.Icon.Warning,
                    "title",
                    "No solution exists to this puzzle",
                    buttons=QMessageBox.StandardButton.Ok,
                )
                msg.exec()
                return
            msg = QMessageBox(
                QMessageBox.Icon.NoIcon,
                "title",
                "Your solution has mistakes",
                buttons=QMessageBox.StandardButton.Cancel,
            )
            erase_all_button = msg.addButton(
                "Erase all mistakes but keep all other work", QMessageBox.ActionRole
            )
            first_mistake_button = msg.addButton(
                "Go to before first mistake", QMessageBox.ActionRole
            )
            msg.setDefaultButton(first_mistake_button)
            msg.exec()
            if msg.clickedButton() == first_mistake_button:
                self.board = self.most_recent_good_board.copy()
                for i, j, c in self.i_j_cell():
                    c.state_user = self.board[i + 1, j + 1]
                self.refresh_board()
            elif msg.clickedButton() == erase_all_button:
                # intersect with the true solution
                self.board = puzzle.intersect_boards(
                    self.board.copy(), thought_process_solved.board
                )
                for i, j, c in self.i_j_cell():
                    c.state_user = self.board[i + 1, j + 1]
                self.refresh_board()
        else:
            msg = QMessageBox(
                QMessageBox.Icon.NoIcon,
                "title",
                "No mistakes",
                buttons=QMessageBox.StandardButton.Ok,
            )
            msg.exec()

    def calculate_difficulty_pressed(self) -> None:
        thought_process_from_here = puzzle.ThoughtProcess(self.board)
        thought_process_from_here.apply_methods(9, calculate_difficulty=True)
        if not thought_process_from_here.check_unsolved():
            from_here_message = "This puzzle state is contradictory"
        elif not puzzle.check_all(thought_process_from_here.board):
            from_here_message = "This puzzle state probably leads to multiple solutions"
        else:
            cost_from_here = sum(
                step.cost for step in thought_process_from_here.solution_steps
            )
            difficulty_from_here = max(
                s.cost for s in thought_process_from_here.solution_steps
            )
            from_here_message = (
                f"From this state, the puzzle has solve cost: {cost_from_here}"
                f"\n and peak difficulty: {difficulty_from_here}"
            )
        thought_process_from_blank_board = puzzle.ThoughtProcess(
            puzzle.clear_board(self.board.copy())
        )
        thought_process_from_blank_board.apply_methods(9, calculate_difficulty=True)
        if not thought_process_from_blank_board.check_unsolved():
            from_blank_board_message = "This puzzle has no solution"
        elif not puzzle.check_all(thought_process_from_blank_board.board):
            from_blank_board_message = "This puzzle probably has multiple solutions"
        else:
            cost_from_blank_board = sum(
                step.cost for step in thought_process_from_blank_board.solution_steps
            )
            difficulty_from_blank_board = max(
                s.cost for s in thought_process_from_blank_board.solution_steps
            )
            from_blank_board_message = (
                f"From the start, the puzzle has solve cost: {cost_from_blank_board}"
                f"\n and peak difficulty: {difficulty_from_blank_board}"
            )
        msg = QMessageBox(
            QMessageBox.Icon.NoIcon,
            "Difficulty",
            from_here_message + "\n" + from_blank_board_message,
            buttons=QMessageBox.StandardButton.Ok,
        )
        msg.exec()
        return

    def request_hint(self) -> None:
        """
        Find a hint if possible, mark hint squares, and display message
        """
        thought_process_hint = puzzle.ThoughtProcess(self.board.copy())
        thought_process_hint.apply_methods(self.auto_apply_methods_level)
        thought_process_hint.apply_methods(9, find_hint=True)
        if not thought_process_hint.check_unsolved():
            # if the puzzle is contradictory, show where by making a fake Step
            thought_process_hint.hint = puzzle.Step(
                (-1, -1, "!"), "check_unsolved", cost=0
            )
            thought_process_correct = puzzle.ThoughtProcess(
                puzzle.clear_board(self.board.copy())
            )
            thought_process_correct.apply_methods(9)
            wrong_board = puzzle.subtract_boards(
                self.board, thought_process_correct.board
            )

            thought_process_hint.hint.outputs = [
                (i, j)
                for (i, j) in thought_process_hint.all_interior_ij()
                if wrong_board[i, j] != "."
            ]
            thought_process_hint.hint.wrong_values = {
                loc: wrong_board[loc] for loc in thought_process_hint.hint.outputs
            }
        if hasattr(thought_process_hint, "hint"):
            self.current_hint = thought_process_hint.hint
            print(f"hint: {self.current_hint}, cost: {self.current_hint.cost}")
            for i, j, c in self.i_j_cell():
                c.state_hint = (i + 1, j + 1) in self.current_hint.outputs
                c.update()
            self.puzzle_status.setText(puzzle.HINT_MESSAGES[self.current_hint.method])
            self.puzzle_status.setVisible(True)
        else:
            self.puzzle_status.setText("No hint available")
            self.puzzle_status.setVisible(True)
            self.current_hint = None

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
        self.refresh_GUI()

    def set_contradiction_checker_mode(self) -> None:
        """
        Toggles contradiction checker mode
        """
        self.contradiction_checker_enabled = self.contradiction_action.isChecked()
        self.refresh_board()

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
        if self.edit_mode == "play":
            self.board_reference = puzzle.do_best_to_get_a_non_wrong_solution(
                self.board
            )
            self.most_recent_good_board = puzzle.intersect_boards(
                self.board, self.board_reference
            )

    def initialize_grid(self) -> None:
        self.grid = QGridLayout()
        for i in range(self.board_auto.shape[0] - 2):
            for j in range(self.board_auto.shape[1] - 2):
                c = Cell(self, i, j, self.board_auto[i + 1, j + 1])
                self.grid.addWidget(c, i, j)
                c.correct = True
                c.update()
        self.grid.setSpacing(0)
        self.grid.addItem(
            QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Minimum),
            self.grid.rowCount(),
            0,
        )
        self.grid.addItem(
            QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum),
            0,
            self.grid.columnCount(),
        )
        self.vbr.insertLayout(0, self.grid)

    def apply_methods(self) -> None:  # noqa: C901 TODO refactor gui.apply_methods
        thought_process = puzzle.ThoughtProcess(self.board.copy())
        thought_process.apply_methods(
            self.auto_apply_methods_level,  # calculate_difficulty=True
        )
        print()
        print("\n".join(map(str, thought_process.solution_steps)))
        if thought_process.solution_steps:
            print(f"cost: {sum(step.cost for step in thought_process.solution_steps)}")
            print(f"difficulty: {max(s.cost for s in thought_process.solution_steps)}")
        if ALWAYS_HINT:
            self.request_hint()
        # In play mode, we do not need to run a full solve with every click
        if (
            self.edit_mode == "play"
            and puzzle.is_partially_correct_based_on_other_board(
                thought_process.board, self.board_reference
            )
        ):
            self.most_recent_good_board = self.board.copy()
        elif self.contradiction_checker_enabled:
            # if  we're in edit mode, or if we're in play and know that the solution is
            # wrong, we must run a full solve
            thought_process_correct = puzzle.ThoughtProcess(self.board.copy())
            thought_process_correct.apply_methods(9)
            if not thought_process_correct.check_unsolved():
                # if a contradiction is detected, hijack the render
                thought_process = thought_process_correct
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

        if self.current_hint is not None:
            # if there was a contradiction, see if it's cleared or corrected
            if (
                self.current_hint.method == "check_unsolved"
                and all(
                    self.board[loc] != self.current_hint.wrong_values[loc]
                    for loc in self.current_hint.outputs
                )
            ) or (
                self.current_hint.method != "check_unsolved"
                and all(self.board[i, j] != "." for (i, j) in self.current_hint.outputs)
            ):
                # otherwise, clear the hint if any action is taken in the hint squares
                self.current_hint = None
                self.puzzle_status.setText("")
                self.puzzle_status.setVisible(False)

        for i, j, c in self.i_j_cell():
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
        if self.puzzle_complete:
            self.animate_puzzle_complete()

    def update_all_cells(self, board: np.ndarray) -> None:
        for i, j, c in self.i_j_cell():
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
                ijs = [(i1, j) for j in range(min(j1, j2), max(j1, j2) + 1)]
            else:
                ijs = [(i, j1) for i in range(min(i1, i2), max(i1, i2) + 1)]
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
        for i, j, c in self.i_j_cell():
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


# Taken from https://stackoverflow.com/a/43003223/400793 by dvntehn00bz
def delay(millisecondsWait: int) -> None:
    loop = QEventLoop()
    t = QTimer()
    t.timeout.connect(loop.quit)
    t.start(millisecondsWait)
    loop.exec()


def take_window_screenshot(window: MainWindow | QDialog, filename: str) -> None:
    scale = 1
    pixmap = QPixmap(window.size() * scale)
    pixmap.setDevicePixelRatio(scale)
    window.render(pixmap)
    pixmap.save(filename, "PNG", -1)


def take_grid_screenshot(window: MainWindow, filename: str) -> None:
    grid = window.grid
    top_left_item = grid.itemAtPosition(0, 0)
    assert top_left_item is not None
    top_left = top_left_item.widget()
    bottom_right_item = grid.itemAtPosition(grid.rowCount() - 2, grid.columnCount() - 2)
    assert bottom_right_item is not None
    bottom_right = bottom_right_item.widget()
    white_cells = QRect(
        top_left.pos(),
        bottom_right.pos() + QPoint(bottom_right.width(), bottom_right.height()),
    )
    scale = 1
    pixmap = QPixmap(white_cells.size() * scale)
    pixmap.setDevicePixelRatio(scale)
    window.render(pixmap, QPoint(0, 0), QRegion(white_cells))
    pixmap.save(filename, "PNG", -1)


def take_method_screenshot(
    window: MainWindow, filename: str, board_str: str, auto_level: int
) -> None:
    """
    This function is unusual in zero padding a board_str, but that's not bad.
    """
    board_backup = window.board
    window.board = puzzle.zero_pad(puzzle.boardify_string(cleandoc(board_str)))
    window.methods_group.actions()[auto_level].setChecked(True)
    window.refresh_GUI()
    window.auto_level_checked()
    window.refresh_board()
    delay(100)
    take_grid_screenshot(window, filename)
    puzzle.print_board(window.board)
    delay(100)
    window.board = board_backup
    window.refresh_GUI()
    window.refresh_board()


def take_all_doc_screenshots(window: MainWindow) -> None:
    take_window_screenshot(window, os.path.normpath("../../pic/on-open.png"))

    window.board[4, 2] = "#"
    window.board[5, 3] = "#"
    window.refresh_board()
    window.refresh_GUI()
    delay(100)
    take_window_screenshot(window, os.path.normpath("../../pic/Just-Illuminate.png"))
    delay(100)
    window.board[4, 2] = "."
    window.board[5, 3] = "."
    window.refresh_GUI()
    window.refresh_board()

    window.methods_group.actions()[2].setChecked(True)
    window.auto_level_checked()
    delay(100)
    take_window_screenshot(window, os.path.normpath("../../pic/Level-2.png"))
    delay(100)
    window.methods_group.actions()[3].setChecked(True)
    window.auto_level_checked()
    take_window_screenshot(window, os.path.normpath("../../pic/Level-3.png"))

    # Causes a lot of this message to no clear ill effect:
    # QPropertyAnimation::updateState (white_out_step): Changing state of an animation
    # without target
    window.methods_group.actions()[0].setChecked(True)
    window.auto_level_checked()
    window.clear_board()
    window.show_controls_in_window_action.setChecked(False)
    window.show_controls()
    window.refresh_GUI()
    delay(100)
    puzzle.print_board(window.board)
    puzzle.print_board(window.board_auto)
    take_window_screenshot(window, os.path.normpath("../../pic/Minimal-UI.png"))

    resize_dlg = ResizeDialog(window)
    resize_dlg.show()
    take_window_screenshot(resize_dlg, os.path.normpath("../../pic/Resize.png"))
    resize_dlg.close()

    new_puzzle_dlg = NewPuzzleDialog()
    new_puzzle_dlg.show()
    take_window_screenshot(new_puzzle_dlg, os.path.normpath("../../pic/New-Puzzle.png"))
    new_puzzle_dlg.close()

    take_method_screenshot(
        window,
        os.path.normpath("../../pic/illuminate.png"),
        """
        ...#.
        ..0.#
        .2-1.
        ..3..
        .....
        """,
        1,
    )

    take_method_screenshot(
        window,
        os.path.normpath("../../pic/mark_bulbs_around_dotted_numbers.png"),
        """
        .....
        ..0..
        .2--.
        ..-..
        .....
        """,
        2,
    )

    take_method_screenshot(
        window,
        os.path.normpath("../../pic/mark_dots_around_full_numbers.png"),
        """
        .....
        ..-..
        .--1.
        ..3..
        .....
        """,
        2,
    )

    take_method_screenshot(
        window,
        os.path.normpath("../../pic/mark_unique_bulbs_for_dot_cells.png"),
        """
        0--.
        +.-.
        -...
        """,
        3,
    )

    take_method_screenshot(
        window,
        os.path.normpath("../../pic/fill_holes.png"),
        """
        .-..
        -2#.
        .#..
        ..-.
        -...
        """,
        3,
    )

    take_method_screenshot(
        window,
        os.path.normpath("../../pic/mark_dots_at_corners.png"),
        """
        -1..-.2.
        ....-...
        -----...
        ....----
        ..3..---
        .....---
        """,
        4,
    )

    take_method_screenshot(
        window,
        os.path.normpath("../../pic/analyze_diagonally_adjacent_numbers.png"),
        """
        1..-....-....
        .1.-.1..-+2..
        ...-..3.-..1.
        ----....-....
        """,
        5,
    )

    take_method_screenshot(
        window,
        os.path.normpath("../../pic/mark_bulbs_and_dots_at_shared_lanes-1.png"),
        """
        .......
        .3...2.
        .......
        """,
        6,
    )

    take_method_screenshot(
        window,
        os.path.normpath("../../pic/mark_bulbs_and_dots_at_shared_lanes-2.png"),
        """
        .2....
        ....2.
        """,
        6,
    )

    take_method_screenshot(
        window,
        os.path.normpath("../../pic/mark_bulbs_and_dots_at_shared_lanes-3.png"),
        """
        .......
        ..3.2..
        .......
        """,
        6,
    )

    take_method_screenshot(
        window,
        os.path.normpath("../../pic/mark_dots_beyond_corners.png"),
        """
        ++.
        +..
        ...
        """,
        6,
    )

    print("DONE WITH take_all_doc_screenshots")


def main(argv: list | None = None) -> None:
    if argv is None:
        argv = sys.argv
    app = QApplication([])
    app.setApplicationName("Puzzle Bicycle")
    window = MainWindow()
    if argv[-1] == "--screenshots":
        take_all_doc_screenshots(window)
    app.exec()


if __name__ == "__main__":
    sys.exit(main())
