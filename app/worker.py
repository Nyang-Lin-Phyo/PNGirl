import cv2
import mediapipe as mp
import time
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from app.vision import (
    get_head_pose, get_head_anchor, get_shoulder_anchor,
    smooth_anchor, rotate_png, scale_png, composite, draw_crosshair,
)

BaseOptions       = mp.tasks.BaseOptions
PoseLandmarker    = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
FaceLandmarker    = mp.tasks.vision.FaceLandmarker
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


class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, object)  # frame, angles

    def __init__(self, cam_index, get_settings_fn):
        super().__init__()
        self.cam_index    = cam_index
        self.get_settings = get_settings_fn
        self.overlay_img  = cv2.imread("assets/testCat.png", cv2.IMREAD_UNCHANGED)
        self._running     = True

        self.smoothed_angles         = None
        self.smoothed_head_anchor    = None
        self.smoothed_left_anchor    = None
        self.smoothed_right_anchor   = None

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

                    # ── Head ──────────────────────────────────────────────
                    ax, ay, ear_width = get_head_anchor(
                        landmarks, w, h,
                        s["head_offset_mult"], s["head_x_nudge"],
                        s["head_y_nudge"], roll)

                    self.smoothed_head_anchor = smooth_anchor(
                        self.smoothed_head_anchor, (ax, ay))
                    hx = int(self.smoothed_head_anchor[0])
                    hy = int(self.smoothed_head_anchor[1])

                    scaled  = scale_png(self.overlay_img, ear_width, s["head_scale_mult"])
                    rotated = rotate_png(scaled, roll, pitch, yaw)
                    composite(frame, rotated, hx, hy)
                    draw_crosshair(frame, hx, hy)

                    # ── Left shoulder ─────────────────────────────────────
                    lx_raw, ly_raw = get_shoulder_anchor(
                        landmarks, w, h, "left",
                        s["left_shoulder_y_nudge"], s["left_shoulder_x_nudge"])

                    self.smoothed_left_anchor = smooth_anchor(
                        self.smoothed_left_anchor, (lx_raw, ly_raw))
                    lx = int(self.smoothed_left_anchor[0])
                    ly = int(self.smoothed_left_anchor[1])

                    l_scaled = scale_png(self.overlay_img, ear_width,
                                         s["left_shoulder_scale_mult"])
                    composite(frame, l_scaled, lx, ly)
                    draw_crosshair(frame, lx, ly)

                    # ── Right shoulder ────────────────────────────────────
                    rx_raw, ry_raw = get_shoulder_anchor(
                        landmarks, w, h, "right",
                        s["right_shoulder_y_nudge"], s["right_shoulder_x_nudge"])

                    self.smoothed_right_anchor = smooth_anchor(
                        self.smoothed_right_anchor, (rx_raw, ry_raw))
                    rx = int(self.smoothed_right_anchor[0])
                    ry = int(self.smoothed_right_anchor[1])

                    r_scaled = scale_png(self.overlay_img, ear_width,
                                          s["right_shoulder_scale_mult"])
                    composite(frame, r_scaled, rx, ry)
                    draw_crosshair(frame, rx, ry)

                self.frame_ready.emit(frame, angles)

        cap.release()