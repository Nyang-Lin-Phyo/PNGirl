APP_STYLE = """
QMainWindow, QWidget {
    background-color: #111111;
    color: #d0d0d0;
    font-family: "Inter", "Segoe UI", sans-serif;
    font-size: 13px;
}

/* ── Camera picker ── */
QListWidget {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 4px;
    outline: none;
}
QListWidget::item {
    padding: 10px 14px;
    border-radius: 4px;
    color: #888;
}
QListWidget::item:selected {
    background: #1e1e1e;
    color: #e8e8e8;
    border: 1px solid #333;
}
QListWidget::item:hover {
    background: #1a1a1a;
    color: #bbb;
}

/* ── Buttons ── */
QPushButton {
    background: #1c1c1c;
    border: 1px solid #2e2e2e;
    border-radius: 5px;
    padding: 7px 18px;
    color: #c0c0c0;
}
QPushButton:hover {
    background: #222;
    border-color: #444;
    color: #e8e8e8;
}
QPushButton:pressed {
    background: #181818;
}
QPushButton#primary {
    background: #1a1a1a;
    border-color: #505050;
    color: #e8e8e8;
}
QPushButton#primary:hover {
    border-color: #888;
}

/* ── Sliders ── */
QSlider::groove:horizontal {
    height: 2px;
    background: #2a2a2a;
    border-radius: 1px;
}
QSlider::handle:horizontal {
    background: #888;
    border: none;
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover {
    background: #bbb;
}
QSlider::sub-page:horizontal {
    background: #555;
    border-radius: 1px;
}

/* ── Dividers ── */
QFrame[frameShape="4"],
QFrame[frameShape="5"] {
    color: #222;
}

/* ── Scroll bar ── */
QScrollBar:vertical {
    background: #111;
    width: 6px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #2a2a2a;
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #3a3a3a;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: none;
}
"""