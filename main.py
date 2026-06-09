import cv2
import mediapipe as mp
import time
import math
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

head_offset_mult = 0.9
head_x_nudge = 0
head_y_nudge = 110
head_scale_mult = 1.0  # 1.0 = ear-to-ear width

overlay_img = cv2.imread("assets/testCat.png", cv2.IMREAD_UNCHANGED)
overlay_img = cv2.flip(overlay_img, 0)

def get_head_anchor(landmarks, w, h):
    nose      = landmarks[0]
    left_ear  = landmarks[7]
    right_ear = landmarks[8]

    nose_y      = int(nose.y * h)
    left_ear_x  = int(left_ear.x * w)
    left_ear_y  = int(left_ear.y * h)
    right_ear_x = int(right_ear.x * w)
    right_ear_y = int(right_ear.y * h)
    ear_mid_x   = (left_ear_x + right_ear_x) // 2
    ear_mid_y   = (left_ear_y + right_ear_y) // 2

    head_height = abs(nose_y - ear_mid_y)
    offset_dist = int(head_height * head_offset_mult) - head_y_nudge

    tilt_angle = math.atan2(
        right_ear_y - left_ear_y,
        right_ear_x - left_ear_x
    )

    ax = int(ear_mid_x + offset_dist * math.sin(tilt_angle)) + head_x_nudge
    ay = int(ear_mid_y - offset_dist * math.cos(tilt_angle))

    ear_width = int(math.dist((left_ear_x, left_ear_y), (right_ear_x, right_ear_y)))

    return ax, ay, math.degrees(tilt_angle), ear_width

def rotate_png(img, angle_deg):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def scale_png(img, ear_width):
    ph, pw = img.shape[:2]
    target_w = int(ear_width * head_scale_mult)
    if target_w < 1:
        return img
    target_h = int(ph * (target_w / pw))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

def composite(frame, png, cx, cy):
    fh, fw = frame.shape[:2]
    ph, pw = png.shape[:2]

    x1 = cx - pw // 2
    y1 = cy - ph // 2
    x2 = x1 + pw
    y2 = y1 + ph

    fx1 = max(x1, 0)
    fy1 = max(y1, 0)
    fx2 = min(x2, fw)
    fy2 = min(y2, fh)

    if fx1 >= fx2 or fy1 >= fy2:
        return

    px1 = fx1 - x1
    py1 = fy1 - y1
    px2 = px1 + (fx2 - fx1)
    py2 = py1 + (fy2 - fy1)

    png_crop = png[py1:py2, px1:px2]
    alpha = png_crop[:, :, 3:4] / 255.0
    rgb   = png_crop[:, :, :3]

    roi = frame[fy1:fy2, fx1:fx2]
    frame[fy1:fy2, fx1:fx2] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)

def draw_panel(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (310, 152), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        "HEAD ANCHOR TWEAKS",
        f"[W/S]  offset mult : {head_offset_mult:.2f}",
        f"[A/D]  x nudge     : {head_x_nudge}px",
        f"[I/K]  y nudge     : {head_y_nudge}px",
        f"[Z/X]  scale mult  : {head_scale_mult:.2f}",
        "[Q] quit",
    ]
    for i, line in enumerate(lines):
        color = (100, 255, 100) if i == 0 else (220, 220, 220)
        cv2.putText(frame, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if results.pose_landmarks:
            h, w, _ = frame.shape
            landmarks = results.pose_landmarks[0]

            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 200, 0), -1)

            ax, ay, tilt_deg, ear_width = get_head_anchor(landmarks, w, h)

            rotated = rotate_png(overlay_img, tilt_deg)
            scaled  = scale_png(rotated, ear_width)
            composite(frame, scaled, ax, ay)

            cv2.circle(frame, (ax, ay), 6, (0, 0, 255), -1)
            cv2.circle(frame, (ax, ay), 6, (255, 255, 255), 2)

        draw_panel(frame)
        cv2.imshow("PNGirl", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            head_offset_mult = round(head_offset_mult + 0.1, 2)
        elif key == ord('s'):
            head_offset_mult = round(head_offset_mult - 0.1, 2)
        elif key == ord('a'):
            head_x_nudge -= 5
        elif key == ord('d'):
            head_x_nudge += 5
        elif key == ord('i'):
            head_y_nudge += 5
        elif key == ord('k'):
            head_y_nudge -= 5
        elif key == ord('z'):
            head_scale_mult = round(head_scale_mult + 0.1, 2)
        elif key == ord('x'):
            head_scale_mult = round(head_scale_mult - 0.1, 2)

cap.release()
cv2.destroyAllWindows()