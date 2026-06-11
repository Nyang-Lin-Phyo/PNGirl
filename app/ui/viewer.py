import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QRect, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from app.ui.widgets import SliderRow, Accordion
from app.ui.asset_panel import AssetPanel

ASSET_PANEL_W = 180
ASSET_PANEL_H = 220


class VideoLabel(QLabel):
    mouse_drag_move    = pyqtSignal(int, int)
    mouse_drag_release = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self._dragging   = False
        self._frame_size = (1, 1)
        self.setMouseTracking(True)

    def set_frame_size(self, w, h):
        self._frame_size = (w, h)

    def _label_to_frame(self, lx, ly):
        lw, lh = self.width(), self.height()
        fw, fh = self._frame_size
        scale  = min(lw / fw, lh / fh)
        off_x  = (lw - fw * scale) / 2
        off_y  = (lh - fh * scale) / 2
        return int((lx - off_x) / scale), int((ly - off_y) / scale)

    def start_drag(self):
        self._dragging = True
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self.mouse_drag_move.emit(*self._label_to_frame(event.pos().x(), event.pos().y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.mouse_drag_release.emit(*self._label_to_frame(event.pos().x(), event.pos().y()))
        super().mouseReleaseEvent(event)


class ViewerPage(QWidget):
    def __init__(self, on_save, on_back):
        super().__init__()
        self._on_save     = on_save
        self._on_back     = on_back
        self._drag_state  = None
        self._anchor_pos  = {}
        self._grab_offset = (0, 0)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Video feed fills the full centre ──────────────────────────────
        self.video_label = VideoLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: #0a0a0a;")
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.mouse_drag_move.connect(self._on_mouse_drag_move)
        self.video_label.mouse_drag_release.connect(self._on_mouse_drag_release)
        outer.addWidget(self.video_label, stretch=1)

        # ── Right settings panel ──────────────────────────────────────────
        panel = QWidget()
        panel.setFixedWidth(250)
        panel.setStyleSheet("background: #111; border-left: 1px solid #1e1e1e;")

        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(0)

        header_widget = QWidget()
        header_widget.setStyleSheet("background: #111;")
        hl = QVBoxLayout(header_widget)
        hl.setContentsMargins(20, 20, 20, 14)
        name_lbl = QLabel("PNGirl")
        name_lbl.setStyleSheet("color: #e0e0e0; font-size: 15px; font-weight: 500;")
        hl.addWidget(name_lbl)
        pv.addWidget(header_widget)

        div = QFrame(); div.setFrameShape(QFrame.Shape.HLine)
        pv.addWidget(div)

        self.accordion = Accordion()

        self.sl_offset = SliderRow("Offset",  0.1, 3.0,  0.05, 0.9,  lambda v: f"{v:.2f}")
        self.sl_xnudge = SliderRow("X nudge", -200, 200, 5,    0,    lambda v: f"{int(v)}px")
        self.sl_ynudge = SliderRow("Y nudge", -200, 200, 5,    110,  lambda v: f"{int(v)}px")
        self.sl_scale  = SliderRow("Scale",   0.1, 4.0,  0.05, 1.0,  lambda v: f"{v:.2f}")
        self.accordion.add_section("HEAD", [self.sl_offset, self.sl_xnudge, self.sl_ynudge, self.sl_scale])

        self.sl_l_ynudge = SliderRow("Y nudge", -200, 200, 5, 60,  lambda v: f"{int(v)}px")
        self.sl_l_xnudge = SliderRow("X nudge", -200, 200, 5, 0,   lambda v: f"{int(v)}px")
        self.sl_l_scale  = SliderRow("Scale",   0.1, 4.0, 0.05, 1.0, lambda v: f"{v:.2f}")
        self.accordion.add_section("LEFT SHOULDER", [self.sl_l_ynudge, self.sl_l_xnudge, self.sl_l_scale])

        self.sl_r_ynudge = SliderRow("Y nudge", -200, 200, 5, 60,  lambda v: f"{int(v)}px")
        self.sl_r_xnudge = SliderRow("X nudge", -200, 200, 5, 0,   lambda v: f"{int(v)}px")
        self.sl_r_scale  = SliderRow("Scale",   0.1, 4.0, 0.05, 1.0, lambda v: f"{v:.2f}")
        self.accordion.add_section("RIGHT SHOULDER", [self.sl_r_ynudge, self.sl_r_xnudge, self.sl_r_scale])

        self.sl_snap = SliderRow("Snap radius", 20, 300, 5, 80, lambda v: f"{int(v)}px")
        self.accordion.add_section("SNAPPING", [self.sl_snap])

        pv.addWidget(self.accordion)
        pv.addStretch()

        self.angles_lbl = QLabel("r—   p—   y—")
        self.angles_lbl.setContentsMargins(20, 0, 20, 0)
        self.angles_lbl.setStyleSheet(
            "color: #2a2a2a; font-size: 11px;"
            "font-family: 'Consolas', 'Courier New', monospace;")
        pv.addWidget(self.angles_lbl)
        pv.addSpacing(10)

        footer = QWidget()
        footer.setStyleSheet("background: #111; border-top: 1px solid #1e1e1e;")
        foot_row = QHBoxLayout(footer)
        foot_row.setContentsMargins(16, 10, 16, 10)
        foot_row.setSpacing(8)
        self.btn_back = QPushButton("Back")
        self.btn_save = QPushButton("Save")
        self.btn_save.setObjectName("primary")
        self.btn_back.clicked.connect(self._on_back)
        self.btn_save.clicked.connect(self._handle_save)
        foot_row.addWidget(self.btn_back)
        foot_row.addWidget(self.btn_save)
        pv.addWidget(footer)

        outer.addWidget(panel)

        # ── Asset panel floats over video label ───────────────────────────
        # Must be created AFTER video_label is added to layout so geometry is valid.
        # Parent is self so it renders above the video label.
        self.asset_panel = AssetPanel(self)
        self.asset_panel.drag_started.connect(self._on_asset_drag_started)
        self.asset_panel.raise_()

        for sl in self._all_sliders():
            sl.slider.valueChanged.connect(self._slider_changed)

    # ── Overlay geometry ──────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_asset_panel()

    def showEvent(self, event):
        super().showEvent(event)
        self._reposition_asset_panel()

    def _reposition_asset_panel(self):
        """Pin the asset panel to the top-left of the video label area."""
        vl = self.video_label
        # video_label geometry is in ViewerPage coords
        x = vl.x()
        y = vl.y()
        h = vl.height()
        panel_x = x + 20
        panel_y = y + (h - ASSET_PANEL_H) // 2

        self.asset_panel.setGeometry(
            QRect(
                panel_x,
                panel_y,
                ASSET_PANEL_W,
                ASSET_PANEL_H
            )
        )

    # ── Public API ────────────────────────────────────────────────────────

    def set_drag_state(self, drag_state):
        self._drag_state = drag_state

    def update_anchor_positions(self, positions: dict):
        self._anchor_pos = positions

    def update_frame(self, frame, angles):
        if angles:
            r, p, y = angles
            self.angles_lbl.setText(f"r{r:+.0f}   p{p:+.0f}   y{y:+.0f}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self.video_label.set_frame_size(w, h)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        pix  = pix.scaled(self.video_label.size(),
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def load_settings(self, s):
        self.sl_offset.set_value(s["head_offset_mult"])
        self.sl_xnudge.set_value(s["head_x_nudge"])
        self.sl_ynudge.set_value(s["head_y_nudge"])
        self.sl_scale.set_value(s["head_scale_mult"])
        self.sl_l_ynudge.set_value(s["left_shoulder_y_nudge"])
        self.sl_l_xnudge.set_value(s["left_shoulder_x_nudge"])
        self.sl_l_scale.set_value(s["left_shoulder_scale_mult"])
        self.sl_r_ynudge.set_value(s["right_shoulder_y_nudge"])
        self.sl_r_xnudge.set_value(s["right_shoulder_x_nudge"])
        self.sl_r_scale.set_value(s["right_shoulder_scale_mult"])
        self.sl_snap.set_value(s.get("snap_threshold", 80))

    def read_settings(self):
        return {
            "head_offset_mult":          self.sl_offset.value(),
            "head_x_nudge":              int(self.sl_xnudge.value()),
            "head_y_nudge":              int(self.sl_ynudge.value()),
            "head_scale_mult":           self.sl_scale.value(),
            "left_shoulder_y_nudge":     int(self.sl_l_ynudge.value()),
            "left_shoulder_x_nudge":     int(self.sl_l_xnudge.value()),
            "left_shoulder_scale_mult":  self.sl_l_scale.value(),
            "right_shoulder_y_nudge":    int(self.sl_r_ynudge.value()),
            "right_shoulder_x_nudge":    int(self.sl_r_xnudge.value()),
            "right_shoulder_scale_mult": self.sl_r_scale.value(),
            "snap_threshold":            int(self.sl_snap.value()),
        }

    # ── Asset panel drag ──────────────────────────────────────────────────

    def _on_asset_drag_started(self, png_path: str, grab_x: int, grab_y: int):
        if self._drag_state is None:
            return
        self._grab_offset = (grab_x, grab_y)
        cx = self.video_label.width() // 2
        cy = self.video_label.height() // 2
        self._drag_state.start_mouse_drag(png_path, cx, cy)
        self.video_label.start_drag()

    def _on_mouse_drag_move(self, fx: int, fy: int):
        if self._drag_state:
            self._drag_state.move_mouse_drag(fx, fy)

    def _on_mouse_drag_release(self, fx: int, fy: int):
        if self._drag_state is None:
            return
        self._drag_state.move_mouse_drag(fx, fy)
        self._drag_state.release_mouse_drag(self._anchor_pos, int(self.sl_snap.value()))

    # ── Internals ─────────────────────────────────────────────────────────

    def _all_sliders(self):
        return (
            self.sl_offset, self.sl_xnudge, self.sl_ynudge, self.sl_scale,
            self.sl_l_ynudge, self.sl_l_xnudge, self.sl_l_scale,
            self.sl_r_ynudge, self.sl_r_xnudge, self.sl_r_scale,
            self.sl_snap,
        )

    def _slider_changed(self, _):
        self._on_save(self.read_settings(), autosave=False)

    def _handle_save(self):
        self._on_save(self.read_settings(), autosave=True)
        self.btn_save.setText("Saved!")
        QTimer.singleShot(1500, lambda: self.btn_save.setText("Save"))