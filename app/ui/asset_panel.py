"""
Left sidebar that shows all PNGs in the assets/ directory as thumbnails.
Supports mouse drag-out: press on a thumbnail and drag into the video feed.
Emits drag_started(png_path, grab_offset_x, grab_offset_y) when a drag begins.
"""

import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QPushButton, QHBoxLayout

ASSETS_DIR = "assets"
THUMB_SIZE = 80
THUMB_PAD = 8


class ThumbnailWidget(QWidget):
    """Single PNG thumbnail — handles press + drag-start."""

    drag_started = pyqtSignal(str, int, int)  # path, grab_offset_x, grab_offset_y

    def __init__(self, png_path: str):
        super().__init__()
        self.png_path = png_path
        self._drag_start = None
        self.setFixedSize(THUMB_SIZE + THUMB_PAD * 2, THUMB_SIZE + THUMB_PAD * 2 + 18)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(THUMB_PAD, THUMB_PAD, THUMB_PAD, 4)
        layout.setSpacing(4)

        # Image
        self.img_label = QLabel()
        self.img_label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("""
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 4px;
        """)

        pix = QPixmap(png_path)
        if not pix.isNull():
            pix = pix.scaled(
                THUMB_SIZE,
                THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.img_label.setPixmap(pix)
        layout.addWidget(self.img_label)

        # Filename label
        name = os.path.splitext(os.path.basename(png_path))[0]
        name_lbl = QLabel(name)
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet("color: #555; font-size: 10px;")
        name_lbl.setWordWrap(False)
        name_lbl.setMaximumWidth(THUMB_SIZE + THUMB_PAD * 2)
        layout.addWidget(name_lbl)

        self.setStyleSheet("""
            ThumbnailWidget {
                background: transparent;
                border-radius: 6px;
            }
            ThumbnailWidget:hover {
                background: #1a1a1a;
            }
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self._drag_start is not None
            and (event.pos() - self._drag_start).manhattanLength() > 8
        ):
            # Compute where in the thumbnail they grabbed
            # pos() is relative to this widget; img_label starts at THUMB_PAD, THUMB_PAD
            grab_x = event.pos().x() - THUMB_PAD
            grab_y = event.pos().y() - THUMB_PAD
            self._drag_start = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.drag_started.emit(self.png_path, grab_x, grab_y)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)


class AssetPanel(QWidget):

    drag_started = pyqtSignal(str, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._paths = []
        self._index = 0
        self._thumbnails = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        self.setStyleSheet("""
            background: rgba(10,10,10,180);
            border-radius: 12px;
        """)

        title = QLabel("ASSETS")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        row = QHBoxLayout()

        self.prev_btn = QPushButton("◀")
        self.next_btn = QPushButton("▶")

        self.prev_btn.clicked.connect(self._prev)
        self.next_btn.clicked.connect(self._next)

        row.addWidget(self.prev_btn)

        self.thumb_container = QVBoxLayout()
        row.addLayout(self.thumb_container)

        row.addWidget(self.next_btn)

        root.addLayout(row)

        self.refresh()

    def _prev(self):
        if not self._paths:
            return

        self._index = (self._index - 1) % len(self._paths)
        self._show_current()

    def _next(self):
        if not self._paths:
            return

        self._index = (self._index + 1) % len(self._paths)
        self._show_current()

    def _show_current(self):
        while self.thumb_container.count():
            item = self.thumb_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._thumbnails.clear()

        if not self._paths:
            return

        thumb = ThumbnailWidget(self._paths[self._index])
        thumb.drag_started.connect(self.drag_started)
        self._thumbnails.append(thumb)
        self.thumb_container.addStretch()
        self.thumb_container.addWidget(thumb)
        self.thumb_container.addStretch()

    def refresh(self):
        self._paths = []

        if not os.path.isdir(ASSETS_DIR):
            return

        self._paths = sorted(
            [
                os.path.join(ASSETS_DIR, f)
                for f in os.listdir(ASSETS_DIR)
                if f.lower().endswith(".png")
            ]
        )

        self._index = 0

        self._show_current()
