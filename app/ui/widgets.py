from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QSizePolicy, QFrame
from PyQt6.QtCore import Qt

BTN_STYLE = """
    QPushButton {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 3px;
        color: #666;
        font-size: 14px;
        padding: 0px;
        min-width: 22px;
        max-width: 22px;
        min-height: 20px;
        max-height: 20px;
    }
    QPushButton:hover   { background: #242424; border-color: #444; color: #ccc; }
    QPushButton:pressed { background: #111; color: #fff; }
"""


class SliderRow(QWidget):
    def __init__(self, label, min_val, max_val, step, initial, display_fn=None):
        super().__init__()
        self.step       = step
        self.display_fn = display_fn or (lambda v: str(v))
        self.min_int    = round(min_val / step)
        self.max_int    = round(max_val / step)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 4)
        root.setSpacing(4)

        # Top row: label + value
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(0)

        self.lbl_name = QLabel(label)
        self.lbl_name.setStyleSheet("color: #555; font-size: 11px;")

        self.lbl_val = QLabel(self.display_fn(initial))
        self.lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_val.setStyleSheet("color: #c0c0c0; font-size: 12px;")

        top.addWidget(self.lbl_name)
        top.addStretch()
        top.addWidget(self.lbl_val)
        root.addLayout(top)

        # Bottom row: ‹  slider  ›
        self.btn_dec = QPushButton("‹")
        self.btn_dec.setStyleSheet(BTN_STYLE)
        self.btn_dec.setAutoRepeat(True)
        self.btn_dec.setAutoRepeatDelay(400)
        self.btn_dec.setAutoRepeatInterval(80)
        self.btn_dec.clicked.connect(self._decrement)

        self.btn_inc = QPushButton("›")
        self.btn_inc.setStyleSheet(BTN_STYLE)
        self.btn_inc.setAutoRepeat(True)
        self.btn_inc.setAutoRepeatDelay(400)
        self.btn_inc.setAutoRepeatInterval(80)
        self.btn_inc.clicked.connect(self._increment)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(self.min_int, self.max_int)
        self.slider.setValue(round(initial / step))
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(5)
        bottom.addWidget(self.btn_dec)
        bottom.addWidget(self.slider)
        bottom.addWidget(self.btn_inc)
        root.addLayout(bottom)

        self.slider.valueChanged.connect(self._on_change)

    def _on_change(self, int_val):
        self.lbl_val.setText(self.display_fn(int_val * self.step))

    def _decrement(self):
        self.slider.setValue(max(self.min_int, self.slider.value() - 1))

    def _increment(self):
        self.slider.setValue(min(self.max_int, self.slider.value() + 1))

    def value(self):
        return self.slider.value() * self.step

    def set_value(self, v):
        self.slider.setValue(round(v / self.step))


# ── Accordion section ─────────────────────────────────────────────────────────

class AccordionSection(QWidget):
    """
    A collapsible panel with a clickable header.
    Pass a list of QWidgets as `contents` — they'll be shown/hidden together.
    Call accordion.open(section) from outside to enforce single-open behaviour.
    """

    def __init__(self, title: str, contents: list, accordion=None):
        super().__init__()
        self._accordion = accordion
        self._open = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header button ─────────────────────────────────────────────────
        self.header = QPushButton()
        self.header.setCheckable(False)
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._title = title
        self._update_header()
        self.header.setStyleSheet("""
            QPushButton {
                background: #161616;
                border: none;
                border-top: 1px solid #1e1e1e;
                border-bottom: 1px solid #1e1e1e;
                color: #666;
                font-size: 10px;
                letter-spacing: 1px;
                text-align: left;
                padding: 9px 16px;
            }
            QPushButton:hover { background: #1a1a1a; color: #999; }
        """)
        self.header.clicked.connect(self._toggle)
        root.addWidget(self.header)

        # ── Body ──────────────────────────────────────────────────────────
        self.body = QWidget()
        self.body.setStyleSheet("background: transparent;")
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(16, 10, 16, 14)
        body_layout.setSpacing(4)

        for widget in contents:
            body_layout.addWidget(widget)

        self.body.setVisible(False)
        root.addWidget(self.body)

    def _toggle(self):
        if self._accordion:
            # Let the accordion handle open/close so others collapse
            self._accordion.open(self) if not self._open else self._accordion.close_all()
        else:
            self._set_open(not self._open)

    def _set_open(self, state: bool):
        self._open = state
        self.body.setVisible(state)
        self._update_header()

    def _update_header(self):
        arrow = "▾" if self._open else "▸"
        self.header.setText(f"  {arrow}   {self._title}")


class Accordion(QWidget):
    """
    Container that holds AccordionSections and enforces single-open behaviour.
    Usage:
        acc = Accordion()
        acc.add_section("HEAD", [widget1, widget2, ...])
        acc.add_section("LEFT SHOULDER", [...])
    """

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background: transparent;")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._sections: list[AccordionSection] = []

    def add_section(self, title: str, contents: list) -> AccordionSection:
        section = AccordionSection(title, contents, accordion=self)
        self._sections.append(section)
        self._layout.addWidget(section)
        return section

    def open(self, target: AccordionSection):
        for s in self._sections:
            s._set_open(s is target)

    def close_all(self):
        for s in self._sections:
            s._set_open(False)