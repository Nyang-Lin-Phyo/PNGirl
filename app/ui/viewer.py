import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

from app.ui.widgets import SliderRow, Accordion


class ViewerPage(QWidget):
    def __init__(self, on_save, on_back):
        super().__init__()
        self._on_save = on_save
        self._on_back = on_back

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Left: video feed ─────────────────────────────────────────────
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: #0a0a0a;")
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        outer.addWidget(self.video_label, stretch=1)

        # ── Right: panel ─────────────────────────────────────────────────
        panel = QWidget()
        panel.setFixedWidth(250)
        panel.setStyleSheet("background: #111; border-left: 1px solid #1e1e1e;")

        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(0)

        # App name header
        header_widget = QWidget()
        header_widget.setStyleSheet("background: #111;")
        hl = QVBoxLayout(header_widget)
        hl.setContentsMargins(20, 20, 20, 14)
        hl.setSpacing(4)

        name_lbl = QLabel("PNGirl")
        name_lbl.setStyleSheet("color: #e0e0e0; font-size: 15px; font-weight: 500;")
        hl.addWidget(name_lbl)

        pv.addWidget(header_widget)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        pv.addWidget(divider)

        # ── Accordion ─────────────────────────────────────────────────────
        self.accordion = Accordion()

        # Head sliders
        self.sl_offset = SliderRow("Offset",  0.1, 3.0,  0.05, 0.9,  lambda v: f"{v:.2f}")
        self.sl_xnudge = SliderRow("X nudge", -200, 200, 5,    0,    lambda v: f"{int(v)}px")
        self.sl_ynudge = SliderRow("Y nudge", -200, 200, 5,    110,  lambda v: f"{int(v)}px")
        self.sl_scale  = SliderRow("Scale",   0.1, 4.0,  0.05, 1.0,  lambda v: f"{v:.2f}")
        self.accordion.add_section("HEAD", [
            self.sl_offset, self.sl_xnudge, self.sl_ynudge, self.sl_scale
        ])

        # Left shoulder sliders
        self.sl_l_ynudge = SliderRow("Y nudge", -200, 200, 5,   60,  lambda v: f"{int(v)}px")
        self.sl_l_xnudge = SliderRow("X nudge", -200, 200, 5,   0,   lambda v: f"{int(v)}px")
        self.sl_l_scale  = SliderRow("Scale",   0.1, 4.0,  0.05, 1.0, lambda v: f"{v:.2f}")
        self.accordion.add_section("LEFT SHOULDER", [
            self.sl_l_ynudge, self.sl_l_xnudge, self.sl_l_scale
        ])

        # Right shoulder sliders
        self.sl_r_ynudge = SliderRow("Y nudge", -200, 200, 5,   60,  lambda v: f"{int(v)}px")
        self.sl_r_xnudge = SliderRow("X nudge", -200, 200, 5,   0,   lambda v: f"{int(v)}px")
        self.sl_r_scale  = SliderRow("Scale",   0.1, 4.0,  0.05, 1.0, lambda v: f"{v:.2f}")
        self.accordion.add_section("RIGHT SHOULDER", [
            self.sl_r_ynudge, self.sl_r_xnudge, self.sl_r_scale
        ])

        pv.addWidget(self.accordion)
        pv.addStretch()

        # Angles readout
        self.angles_lbl = QLabel("r—   p—   y—")
        self.angles_lbl.setContentsMargins(20, 0, 20, 0)
        self.angles_lbl.setStyleSheet(
            "color: #2a2a2a; font-size: 11px;"
            "font-family: 'Consolas', 'Courier New', monospace;")
        pv.addWidget(self.angles_lbl)
        pv.addSpacing(10)

        # ── Footer ────────────────────────────────────────────────────────
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

        # Wire all sliders → live update
        for sl in self._all_sliders():
            sl.slider.valueChanged.connect(self._slider_changed)

    # ── Public API ────────────────────────────────────────────────────────

    def update_frame(self, frame, angles):
        if angles:
            r, p, y = angles
            self.angles_lbl.setText(f"r{r:+.0f}   p{p:+.0f}   y{y:+.0f}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
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
        }

    # ── Internals ─────────────────────────────────────────────────────────

    def _all_sliders(self):
        return (
            self.sl_offset, self.sl_xnudge, self.sl_ynudge, self.sl_scale,
            self.sl_l_ynudge, self.sl_l_xnudge, self.sl_l_scale,
            self.sl_r_ynudge, self.sl_r_xnudge, self.sl_r_scale,
        )

    def _slider_changed(self, _):
        self._on_save(self.read_settings(), autosave=False)

    def _handle_save(self):
        self._on_save(self.read_settings(), autosave=True)
        self.btn_save.setText("Saved!")
        QTimer.singleShot(1500, lambda: self.btn_save.setText("Save"))