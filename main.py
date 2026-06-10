import sys
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import json
import os

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QListWidget, QListWidgetItem,
    QStackedWidget, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette

# ── MediaPipe setup ──────────────────────────────────────────────────────────

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_facial_transformation_matrixes=True,
)

SETTINGS_FILE = "camera_settings.json"
RELEVANT_LANDMARKS = {0, 7, 8, 11, 12}
SMOOTH = 0.4
ANCHOR_SMOOTH = 0.4

# ── Vision helpers ───────────────────────────────────────────────────────────

def create_work_canvas(img, scale=2.0):
    h, w = img.shape[:2]
    canvas_w, canvas_h = int(w * scale), int(h * scale)
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    x, y = (canvas_w - w) // 2, (canvas_h - h) // 2
    canvas[y:y+h, x:x+w] = img
    return canvas


def get_head_pose(face_results, w, h, smoothed_angles):
    if not face_results.facial_transformation_matrixes:
        return smoothed_angles
    matrix = np.array(face_results.facial_transformation_matrixes[0])
    R = matrix[:3, :3]
    yaw   = math.degrees(math.atan2(R[0, 2], R[2, 2]))
    pitch = math.degrees(math.atan2(-R[1, 2], np.sqrt(R[0, 2]**2 + R[2, 2]**2)))
    roll  = math.degrees(math.atan2(R[1, 0], R[1, 1]))
    pitch, roll = -pitch, -roll
    if smoothed_angles is None:
        return (roll, pitch, yaw)
    sr, sp, sy = smoothed_angles
    return (
        sr + SMOOTH * (roll  - sr),
        sp + SMOOTH * (pitch - sp),
        sy + SMOOTH * (yaw   - sy),
    )


def get_head_anchor(landmarks, w, h, offset_mult, x_nudge, y_nudge, roll_deg):
    nose      = landmarks[0]
    left_ear  = landmarks[7]
    right_ear = landmarks[8]
    nose_y     = int(nose.y * h)
    lx, ly     = int(left_ear.x  * w), int(left_ear.y * h)
    rx, ry     = int(right_ear.x * w), int(right_ear.y * h)
    ear_mid_x  = (lx + rx) // 2
    ear_mid_y  = (ly + ry) // 2
    head_height   = abs(nose_y - ear_mid_y)
    offset_dist   = int(head_height * offset_mult) - y_nudge
    tilt_angle    = math.radians(roll_deg)
    ax = int(ear_mid_x + offset_dist * math.sin(tilt_angle)) + x_nudge
    ay = int(ear_mid_y - offset_dist * math.cos(tilt_angle))
    ear_width = int(math.dist((lx, ly), (rx, ry)))
    return ax, ay, ear_width


def rotate_png(img, roll, pitch, yaw):
    img = create_work_canvas(img)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), -roll, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    pitch_factor = np.clip(pitch / 90.0, -0.6, 0.6)
    src  = np.float32([[0,0],[w,0],[w,h],[0,h]])
    sh   = pitch_factor * (h * 0.3)
    dst  = np.float32([[w*0.1,sh],[w*0.9,sh],[w*0.9,h-sh],[w*0.1,h-sh]])
    rotated = cv2.warpPerspective(rotated, cv2.getPerspectiveTransform(src, dst), (w, h),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    yaw_factor = np.clip(yaw / 90.0, -0.6, 0.6)
    sy   = yaw_factor * (w * 0.3)
    dst2 = np.float32([[sy,h*0.1],[w-sy,h*0.1],[w-sy,h*0.9],[sy,h*0.9]])
    rotated = cv2.warpPerspective(rotated, cv2.getPerspectiveTransform(src, dst2), (w, h),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated


def scale_png(img, ear_width, scale_mult):
    ph, pw = img.shape[:2]
    target_w = int(ear_width * scale_mult)
    if target_w < 1:
        return img
    target_h = int(ph * (target_w / pw))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def composite(frame, png, cx, cy):
    fh, fw = frame.shape[:2]
    ph, pw = png.shape[:2]
    x1, y1 = cx - pw // 2, cy - ph // 2
    x2, y2 = x1 + pw, y1 + ph
    fx1, fy1 = max(x1, 0), max(y1, 0)
    fx2, fy2 = min(x2, fw), min(y2, fh)
    if fx1 >= fx2 or fy1 >= fy2:
        return
    px1, py1 = fx1 - x1, fy1 - y1
    px2, py2 = px1 + (fx2 - fx1), py1 + (fy2 - fy1)
    png_crop = png[py1:py2, px1:px2]
    alpha = png_crop[:, :, 3:4] / 255.0
    roi = frame[fy1:fy2, fx1:fx2]
    frame[fy1:fy2, fx1:fx2] = (png_crop[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)


# ── Settings ─────────────────────────────────────────────────────────────────

def load_settings(cam_index):
    defaults = {"head_offset_mult": 0.9, "head_x_nudge": 0,
                "head_y_nudge": 110, "head_scale_mult": 1.0}
    if not os.path.exists(SETTINGS_FILE):
        return defaults
    with open(SETTINGS_FILE) as f:
        data = json.load(f)
    return data.get(str(cam_index), defaults)


def save_settings(cam_index, settings):
    data = {}
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            data = json.load(f)
    data[str(cam_index)] = settings
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Camera worker thread ──────────────────────────────────────────────────────

class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, object)   # frame, angles

    def __init__(self, cam_index, get_settings_fn):
        super().__init__()
        self.cam_index      = cam_index
        self.get_settings   = get_settings_fn
        self.overlay_img    = cv2.imread("assets/testCat.png", cv2.IMREAD_UNCHANGED)
        self._running       = True
        self.smoothed_angles = None
        self.smoothed_anchor = None

    def stop(self):
        self._running = False
        self.wait()

    def run(self):
        cap = cv2.VideoCapture(self.cam_index)

        with PoseLandmarker.create_from_options(pose_options) as pose_lm, \
             FaceLandmarker.create_from_options(face_options) as face_lm:

            while self._running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame  = cv2.flip(frame, 1)
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts     = int(time.time() * 1000)

                pose_results = pose_lm.detect_for_video(mp_img, ts)
                face_results = face_lm.detect_for_video(mp_img, ts)

                self.smoothed_angles = get_head_pose(
                    face_results, *frame.shape[1::-1], self.smoothed_angles)

                angles = self.smoothed_angles
                s = self.get_settings()

                if pose_results.pose_landmarks:
                    h, w, _ = frame.shape
                    landmarks = pose_results.pose_landmarks[0]

                    roll  = angles[0] if angles else 0.0
                    pitch = angles[1] if angles else 0.0
                    yaw   = angles[2] if angles else 0.0

                    ax, ay, ear_width = get_head_anchor(
                        landmarks, w, h,
                        s["head_offset_mult"], s["head_x_nudge"],
                        s["head_y_nudge"], roll)

                    DEADZONE = 3
                    if self.smoothed_anchor is None:
                        self.smoothed_anchor = (ax, ay)
                    else:
                        sx, sy = self.smoothed_anchor
                        dx = ax - sx if abs(ax - sx) >= DEADZONE else 0
                        dy = ay - sy if abs(ay - sy) >= DEADZONE else 0
                        self.smoothed_anchor = (sx + ANCHOR_SMOOTH * dx,
                                                sy + ANCHOR_SMOOTH * dy)

                    ax = int(self.smoothed_anchor[0])
                    ay = int(self.smoothed_anchor[1])

                    scaled  = scale_png(self.overlay_img, ear_width, s["head_scale_mult"])
                    rotated = rotate_png(scaled, roll, pitch, yaw)
                    composite(frame, rotated, ax, ay)

                    # Minimal crosshair anchor
                    r = 5
                    cv2.line(frame, (ax-r, ay), (ax+r, ay), (255,255,255), 1, cv2.LINE_AA)
                    cv2.line(frame, (ax, ay-r), (ax, ay+r), (255,255,255), 1, cv2.LINE_AA)
                    cv2.circle(frame, (ax, ay), r+3, (180,180,180), 1, cv2.LINE_AA)

                self.frame_ready.emit(frame, angles)

        cap.release()


# ── Stylesheet ────────────────────────────────────────────────────────────────

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

/* ── Divider ── */
QFrame[frameShape="4"],
QFrame[frameShape="5"] {
    color: #222;
}

/* ── Labels ── */
QLabel#section {
    color: #444;
    font-size: 10px;
    letter-spacing: 1px;
}
QLabel#value {
    color: #e0e0e0;
    font-size: 13px;
}
QLabel#angles {
    color: #383838;
    font-size: 11px;
    font-family: "Consolas", "Courier New", monospace;
}
"""


# ── Reusable slider row widget ────────────────────────────────────────────────

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

        # Top row: label left, value right
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
        btn_style = """
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

        self.btn_dec = QPushButton("‹")
        self.btn_dec.setStyleSheet(btn_style)
        self.btn_dec.setAutoRepeat(True)
        self.btn_dec.setAutoRepeatDelay(400)
        self.btn_dec.setAutoRepeatInterval(80)
        self.btn_dec.clicked.connect(self._decrement)

        self.btn_inc = QPushButton("›")
        self.btn_inc.setStyleSheet(btn_style)
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


# ── Camera picker screen ──────────────────────────────────────────────────────

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

        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(line)

        self.list_widget = QListWidget()
        self.list_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
                has_saved = (os.path.exists(SETTINGS_FILE) and
                             str(i) in json.load(open(SETTINGS_FILE)))
                suffix = "  ·  saved settings" if has_saved else ""
                item = QListWidgetItem(f"Camera {i}    {w}x{h}{suffix}")
                self.list_widget.addItem(item)
            cap.release()
        if self._cameras:
            self.list_widget.setCurrentRow(0)

    def _confirm(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.camera_selected.emit(self._cameras[row])


# ── Main app window ───────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PNGirl")
        self.setMinimumSize(900, 560)
        self.cam_index = None
        self.settings  = {}
        self.worker    = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Page 0: camera picker
        self.picker = CameraPickerPage()
        self.picker.camera_selected.connect(self._start_camera)
        self.stack.addWidget(self.picker)

        # Page 1: viewer (built on demand)
        self.viewer_page = None

    # ── Camera start ──

    def _start_camera(self, cam_index):
        self.cam_index = cam_index
        self.settings  = load_settings(cam_index)

        if self.viewer_page is None:
            self._build_viewer()
            self.stack.addWidget(self.viewer_page)

        self._sync_sliders()
        self.stack.setCurrentWidget(self.viewer_page)

        self.worker = CameraWorker(cam_index, lambda: self.settings)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.start()

    def _build_viewer(self):
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Left: video feed ──
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: #0a0a0a;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        outer.addWidget(self.video_label, stretch=1)

        # ── Right: panel ──
        panel = QWidget()
        panel.setFixedWidth(240)
        panel.setStyleSheet("background: #111; border-left: 1px solid #1e1e1e;")
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(20, 24, 20, 20)
        pv.setSpacing(6)

        # App name
        name_lbl = QLabel("PNGirl")
        name_lbl.setStyleSheet("color: #e0e0e0; font-size: 15px; font-weight: 500;")
        pv.addWidget(name_lbl)

        line1 = QFrame(); line1.setFrameShape(QFrame.Shape.HLine)
        pv.addWidget(line1)
        pv.addSpacing(8)

        # Section label
        adj_lbl = QLabel("ADJUSTMENT")
        adj_lbl.setObjectName("section")
        pv.addWidget(adj_lbl)
        pv.addSpacing(6)

        # Sliders
        self.sl_offset = SliderRow("Offset",  0.1, 3.0,  0.05, self.settings.get("head_offset_mult", 0.9),
                                   lambda v: f"{v:.2f}")
        self.sl_xnudge = SliderRow("X nudge", -200, 200, 5, self.settings.get("head_x_nudge", 0),
                                   lambda v: f"{int(v)}px")
        self.sl_ynudge = SliderRow("Y nudge", -200, 200, 5, self.settings.get("head_y_nudge", 110),
                                   lambda v: f"{int(v)}px")
        self.sl_scale  = SliderRow("Scale",   0.1, 4.0,  0.05, self.settings.get("head_scale_mult", 1.0),
                                   lambda v: f"{v:.2f}")

        for sl in (self.sl_offset, self.sl_xnudge, self.sl_ynudge, self.sl_scale):
            pv.addWidget(sl)
            sl.slider.valueChanged.connect(self._on_slider_change)

        pv.addSpacing(12)
        line2 = QFrame(); line2.setFrameShape(QFrame.Shape.HLine)
        pv.addWidget(line2)
        pv.addSpacing(8)

        # Angles readout
        self.angles_lbl = QLabel("r—  p—  y—")
        self.angles_lbl.setObjectName("angles")
        pv.addWidget(self.angles_lbl)

        pv.addStretch()

        line3 = QFrame(); line3.setFrameShape(QFrame.Shape.HLine)
        pv.addWidget(line3)
        pv.addSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_save = QPushButton("Save")
        self.btn_save.setObjectName("primary")
        self.btn_save.clicked.connect(self._save)

        self.btn_back = QPushButton("Back")
        self.btn_back.clicked.connect(self._stop_camera)

        btn_row.addWidget(self.btn_back)
        btn_row.addWidget(self.btn_save)
        pv.addLayout(btn_row)

        outer.addWidget(panel)
        self.viewer_page = page

    def _sync_sliders(self):
        if self.viewer_page is None:
            return
        self.sl_offset.set_value(self.settings["head_offset_mult"])
        self.sl_xnudge.set_value(self.settings["head_x_nudge"])
        self.sl_ynudge.set_value(self.settings["head_y_nudge"])
        self.sl_scale.set_value(self.settings["head_scale_mult"])

    def _on_slider_change(self, _):
        self.settings["head_offset_mult"] = self.sl_offset.value()
        self.settings["head_x_nudge"]     = int(self.sl_xnudge.value())
        self.settings["head_y_nudge"]     = int(self.sl_ynudge.value())
        self.settings["head_scale_mult"]  = self.sl_scale.value()

    def _on_frame(self, frame, angles):
        # Update angles label
        if angles:
            r, p, y = angles
            self.angles_lbl.setText(f"r{r:+.0f}   p{p:+.0f}   y{y:+.0f}")

        # Convert BGR frame → QPixmap and display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)

        lbl = self.video_label
        pix = pix.scaled(lbl.size(), Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
        lbl.setPixmap(pix)

    def _save(self):
        if self.cam_index is not None:
            save_settings(self.cam_index, self.settings)
            self.btn_save.setText("Saved!")
            QTimer.singleShot(1500, lambda: self.btn_save.setText("Save"))

    def _stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.stack.setCurrentWidget(self.picker)

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())