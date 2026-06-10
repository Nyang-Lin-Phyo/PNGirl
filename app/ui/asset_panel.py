"""
Left sidebar that shows all PNGs in the assets/ directory as thumbnails.
Supports mouse drag-out: press on a thumbnail and drag into the video feed.
Emits drag_started(png_path, grab_offset_x, grab_offset_y) when a drag begins.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea,
    QGridLayout, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QSize
from PyQt6.QtGui import QPixmap, QDrag, QCursor
from PyQt6.QtCore import QMimeData

ASSETS_DIR   = "assets"
THUMB_SIZE   = 80
THUMB_PAD    = 8


class ThumbnailWidget(QWidget):
    """Single PNG thumbnail — handles press + drag-start."""

    drag_started = pyqtSignal(str, int, int)   # path, grab_offset_x, grab_offset_y

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
            pix = pix.scaled(THUMB_SIZE, THUMB_SIZE,
                             Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
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
        if (self._drag_start is not None and
                (event.pos() - self._drag_start).manhattanLength() > 8):
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
    """
    Left sidebar.
    drag_started(png_path, grab_offset_x, grab_offset_y) fires when a drag begins.
    """

    drag_started = pyqtSignal(str, int, int)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(200)
        self.setStyleSheet("background: #0e0e0e; border-right: 1px solid #1e1e1e;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QLabel("ASSETS")
        header.setContentsMargins(16, 14, 16, 10)
        header.setStyleSheet(
            "color: #3a3a3a; font-size: 10px; letter-spacing: 1px;"
            "border-bottom: 1px solid #1e1e1e;")
        root.addWidget(header)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border: none; background: transparent;")

        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet("background: transparent;")
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid.setSpacing(4)

        scroll.setWidget(self._grid_widget)
        root.addWidget(scroll)

        self._thumbnails: list[ThumbnailWidget] = []
        self.refresh()

    def refresh(self):
        """Scan assets/ and rebuild the thumbnail grid."""
        # Clear existing
        for thumb in self._thumbnails:
            thumb.setParent(None)
        self._thumbnails.clear()

        if not os.path.isdir(ASSETS_DIR):
            return

        paths = sorted([
            os.path.join(ASSETS_DIR, f)
            for f in os.listdir(ASSETS_DIR)
            if f.lower().endswith(".png")
        ])

        cols = 2
        for i, path in enumerate(paths):
            thumb = ThumbnailWidget(path)
            thumb.drag_started.connect(self.drag_started)
            self._thumbnails.append(thumb)
            self._grid.addWidget(thumb, i // cols, i % cols)

        # Push items to top
        self._grid.setRowStretch(len(paths) // cols + 1, 1)