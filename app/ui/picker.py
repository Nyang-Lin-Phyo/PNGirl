import cv2
import json
import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

from app.settings import SETTINGS_FILE


class CameraPickerPage(QWidget):
    camera_selected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 32, 32, 32)
        root.setSpacing(20)

        title = QLabel("Select camera")
        title.setStyleSheet("color: #e0e0e0; font-size: 18px; font-weight: 500;")
        root.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(line)

        self.list_widget = QListWidget()
        self.list_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.list_widget.itemDoubleClicked.connect(self._confirm)
        root.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()

        self.btn_confirm = QPushButton("Open")
        self.btn_confirm.setObjectName("primary")
        self.btn_confirm.setFixedWidth(90)
        self.btn_confirm.clicked.connect(self._confirm)

        btn_row.addWidget(self.btn_confirm)
        root.addLayout(btn_row)

        self._cameras = []
        self._scan()

    def _scan(self):
        self.list_widget.clear()
        self._cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._cameras.append(i)
                has_saved = (
                    os.path.exists(SETTINGS_FILE) and
                    str(i) in json.load(open(SETTINGS_FILE))
                )
                suffix = "  ·  saved settings" if has_saved else ""
                self.list_widget.addItem(
                    QListWidgetItem(f"Camera {i}    {w}x{h}{suffix}")
                )
            cap.release()
        if self._cameras:
            self.list_widget.setCurrentRow(0)

    def _confirm(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.camera_selected.emit(self._cameras[row])