import cv2
import mediapipe as mp
import time
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from app.vision import (
    get_head_pose, get_head_anchor, get_shoulder_anchor,
    smooth_anchor, rotate_png, scale_png, composite, draw_crosshair,
)
from app.hand_tracker import HandTracker

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

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

DRAG_SCALE = 0.5   # PNG is drawn at this fraction of its snapped size while dragging


class CameraWorker(QThread):
    frame_ready    = pyqtSignal(np.ndarray, object)          # frame, angles
    hand_state     = pyqtSignal(list)                         # list of hand dicts
    anchor_updated = pyqtSignal(dict)                         # anchor_key -> (px, py)

    def __init__(self, cam_index, get_settings_fn, drag_state):
        super().__init__()
        self.cam_index    = cam_index
        self.get_settings = get_settings_fn
        self.drag_state   = drag_state
        self._running     = True

        self._png_cache: dict[str, np.ndarray] = {}

        self.smoothed_angles       = None
        self.smoothed_head_anchor  = None
        self.smoothed_left_anchor  = None
        self.smoothed_right_anchor = None

    def stop(self):
        self._running = False
        self.wait()

    def _load_png(self, path: str):
        if path not in self._png_cache:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            self._png_cache[path] = img
        return self._png_cache[path]

    def run(self):
        cap          = cv2.VideoCapture(self.cam_index)
        hand_tracker = HandTracker()

        with PoseLandmarker.create_from_options(pose_options) as pose_lm, \
             FaceLandmarker.create_from_options(face_options) as face_lm:

            while self._running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame  = cv2.flip(frame, 1)
                h, w   = frame.shape[:2]
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts     = int(time.time() * 1000)

                pose_results = pose_lm.detect_for_video(mp_img, ts)
                face_results = face_lm.detect_for_video(mp_img, ts)
                hands        = hand_tracker.detect(mp_img, ts, w, h)

                self.smoothed_angles = get_head_pose(
                    face_results, w, h, self.smoothed_angles)

                angles = self.smoothed_angles
                s      = self.get_settings()

                # Current anchor pixel positions (needed for snapping + crosshairs)
                anchor_positions = {}

                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks[0]

                    roll  = angles[0] if angles else 0.0
                    pitch = angles[1] if angles else 0.0
                    yaw   = angles[2] if angles else 0.0

                    # ── Compute all anchor positions ───────────────────────
                    ax, ay, ear_width = get_head_anchor(
                        landmarks, w, h,
                        s["head_offset_mult"], s["head_x_nudge"],
                        s["head_y_nudge"], roll)
                    self.smoothed_head_anchor = smooth_anchor(
                        self.smoothed_head_anchor, (ax, ay))
                    hx = int(self.smoothed_head_anchor[0])
                    hy = int(self.smoothed_head_anchor[1])
                    anchor_positions["head"] = (hx, hy)

                    lx_raw, ly_raw = get_shoulder_anchor(
                        landmarks, w, h, "left",
                        s["left_shoulder_y_nudge"], s["left_shoulder_x_nudge"])
                    self.smoothed_left_anchor = smooth_anchor(
                        self.smoothed_left_anchor, (lx_raw, ly_raw))
                    lx = int(self.smoothed_left_anchor[0])
                    ly = int(self.smoothed_left_anchor[1])
                    anchor_positions["left_shoulder"] = (lx, ly)

                    rx_raw, ry_raw = get_shoulder_anchor(
                        landmarks, w, h, "right",
                        s["right_shoulder_y_nudge"], s["right_shoulder_x_nudge"])
                    self.smoothed_right_anchor = smooth_anchor(
                        self.smoothed_right_anchor, (rx_raw, ry_raw))
                    rx = int(self.smoothed_right_anchor[0])
                    ry = int(self.smoothed_right_anchor[1])
                    anchor_positions["right_shoulder"] = (rx, ry)

                    # ── Composite snapped PNGs ─────────────────────────────
                    drags, slots, falling = self.drag_state.snapshot()
                    dragged_anchors = {d.png_path for d in drags}

                    snap_configs = {
                        "head":           (hx, hy, ear_width, s["head_scale_mult"],
                                           roll, pitch, yaw, True),
                        "left_shoulder":  (lx, ly, ear_width, s["left_shoulder_scale_mult"],
                                           0, 0, 0, False),
                        "right_shoulder": (rx, ry, ear_width, s["right_shoulder_scale_mult"],
                                           0, 0, 0, False),
                    }

                    for key, (cx_, cy_, ew, sm, r, p, y_, do_rotate) in snap_configs.items():
                        draw_crosshair(frame, cx_, cy_)
                        png_path = slots.get(key)
                        if png_path is None:
                            continue
                        img = self._load_png(png_path)
                        if img is None:
                            continue
                        scaled = scale_png(img, ew, sm)
                        if do_rotate:
                            scaled = rotate_png(scaled, r, p, y_)
                        composite(frame, scaled, cx_, cy_)

                    # ── Composite falling PNGs ────────────────────────────
                    for f in falling:
                        img = self._load_png(f.png_path)
                        if img is None:
                            continue
                        thumb = scale_png(img, 80, 1.0)
                        composite(frame, thumb, int(f.cx), int(f.cy))

                    # ── Composite dragged PNGs ────────────────────────────
                    for drag in drags:
                        img = self._load_png(drag.png_path)
                        if img is None:
                            continue
                        # Scale relative to ear_width but smaller
                        drag_w = max(40, int(ear_width * DRAG_SCALE))
                        ph, pw = img.shape[:2]
                        drag_h = int(ph * (drag_w / pw))
                        thumb  = cv2.resize(img, (drag_w, drag_h),
                                            interpolation=cv2.INTER_AREA)
                        composite(frame, thumb, drag.cx, drag.cy)
                        draw_crosshair(frame, drag.cx, drag.cy)

                    # ── Emit anchor positions so UI can do snapping ───────
                    self.anchor_updated.emit(anchor_positions)

                # ── Advance falling physics ───────────────────────────────
                self.drag_state.tick_falling(h)

                # ── Emit hand state so main thread can handle pinch ───────
                self.hand_state.emit(hands)
                self.frame_ready.emit(frame, angles)

        hand_tracker.close()
        cap.release()