import cv2
import mediapipe as mp
import time
import math
import numpy as np
import json
import os

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
smoothed_anchor = None
ANCHOR_SMOOTH = 0.4

overlay_img = cv2.imread("assets/testCat.png", cv2.IMREAD_UNCHANGED)

smoothed_angles = None

# --- Canvas ---


def create_work_canvas(img, scale=2.0):
    h, w = img.shape[:2]

    canvas_w = int(w * scale)
    canvas_h = int(h * scale)

    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    x = (canvas_w - w) // 2
    y = (canvas_h - h) // 2

    canvas[y : y + h, x : x + w] = img

    return canvas


# --- Settings ---


def load_settings(cam_index):
    defaults = {
        "head_offset_mult": 0.9,
        "head_x_nudge": 0,
        "head_y_nudge": 110,
        "head_scale_mult": 1.0,
    }
    if not os.path.exists(SETTINGS_FILE):
        return defaults
    with open(SETTINGS_FILE, "r") as f:
        data = json.load(f)
    return data.get(str(cam_index), defaults)


def save_settings(cam_index, settings):
    data = {}
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
    data[str(cam_index)] = settings
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Settings saved for camera {cam_index}")


# --- Camera selection ---


def choose_camera():
    print("Scanning cameras...")
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        ret, frame = cap.read()
        if ret:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append((i, f"Camera {i} ({w}x{h})"))
        cap.release()

    if not available:
        print("No cameras found!")
        return None

    sel = 0
    while True:
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(
            img,
            "SELECT CAMERA",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 255, 100),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "W/S to select, ENTER to confirm",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        for i, (idx, name) in enumerate(available):
            has_settings = os.path.exists(SETTINGS_FILE) and str(idx) in json.load(
                open(SETTINGS_FILE)
            )
            tag = " [saved]" if has_settings else ""
            color = (0, 255, 255) if i == sel else (200, 200, 200)
            prefix = "> " if i == sel else "  "
            cv2.putText(
                img,
                f"{prefix}{name}{tag}",
                (10, 90 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("PNGirl", img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("w"):
            sel = (sel - 1) % len(available)
        elif key == ord("s"):
            sel = (sel + 1) % len(available)
        elif key == 13:
            cv2.destroyAllWindows()
            return available[sel][0]
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return None


# --- Head pose ---


def get_head_pose(face_results, w, h):
    global smoothed_angles

    if not face_results.facial_transformation_matrixes:
        return smoothed_angles

    matrix = np.array(face_results.facial_transformation_matrixes[0])
    R = matrix[:3, :3]

    yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
    pitch = math.degrees(math.atan2(-R[1, 2], np.sqrt(R[0, 2] ** 2 + R[2, 2] ** 2)))
    roll = math.degrees(math.atan2(R[1, 0], R[1, 1]))

    pitch = -pitch
    roll = -roll

    if smoothed_angles is None:
        smoothed_angles = (roll, pitch, yaw)
    else:
        sr, sp, sy = smoothed_angles
        smoothed_angles = (
            sr + SMOOTH * (roll - sr),
            sp + SMOOTH * (pitch - sp),
            sy + SMOOTH * (yaw - sy),
        )

    return smoothed_angles


# --- Anchor + render ---


def get_head_anchor(landmarks, w, h, offset_mult, x_nudge, y_nudge, roll_deg):
    nose = landmarks[0]
    left_ear = landmarks[7]
    right_ear = landmarks[8]

    nose_y = int(nose.y * h)
    left_ear_x = int(left_ear.x * w)
    left_ear_y = int(left_ear.y * h)
    right_ear_x = int(right_ear.x * w)
    right_ear_y = int(right_ear.y * h)
    ear_mid_x = (left_ear_x + right_ear_x) // 2
    ear_mid_y = (left_ear_y + right_ear_y) // 2

    head_height = abs(nose_y - ear_mid_y)
    offset_dist = int(head_height * offset_mult) - y_nudge

    tilt_angle = math.radians(roll_deg)

    ax = int(ear_mid_x + offset_dist * math.sin(tilt_angle)) + x_nudge
    ay = int(ear_mid_y - offset_dist * math.cos(tilt_angle))

    ear_width = int(math.dist((left_ear_x, left_ear_y), (right_ear_x, right_ear_y)))

    return ax, ay, ear_width


def rotate_png(img, roll, pitch, yaw):
    img = create_work_canvas(img)
    h, w = img.shape[:2]

    cx, cy = w // 2, h // 2

    # Roll: 2D rotation
    M = cv2.getRotationMatrix2D((cx, cy), -roll, 1.0)
    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # Pitch: vertical perspective warp
    pitch_factor = np.clip(pitch / 90.0, -0.6, 0.6)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    shift = pitch_factor * (h * 0.3)
    dst = np.float32(
        [
            [w * 0.1, shift],
            [w * 0.9, shift],
            [w * 0.9, h - shift],
            [w * 0.1, h - shift],
        ]
    )
    M_persp = cv2.getPerspectiveTransform(src, dst)
    rotated = cv2.warpPerspective(
        rotated,
        M_persp,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # Yaw: horizontal perspective warp
    yaw_factor = np.clip(yaw / 90.0, -0.6, 0.6)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    shift_y = yaw_factor * (w * 0.3)
    dst = np.float32(
        [
            [shift_y, h * 0.1],
            [w - shift_y, h * 0.1],
            [w - shift_y, h * 0.9],
            [shift_y, h * 0.9],
        ]
    )
    M_yaw = cv2.getPerspectiveTransform(src, dst)
    rotated = cv2.warpPerspective(
        rotated, M_yaw, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )

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
    rgb = png_crop[:, :, :3]

    roi = frame[fy1:fy2, fx1:fx2]
    frame[fy1:fy2, fx1:fx2] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)


def draw_panel(frame, s, angles):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, 200), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        "HEAD ANCHOR TWEAKS",
        f"[W/S]  offset mult : {s['head_offset_mult']:.2f}",
        f"[A/D]  x nudge     : {s['head_x_nudge']}px",
        f"[I/K]  y nudge     : {s['head_y_nudge']}px",
        f"[Z/X]  scale mult  : {s['head_scale_mult']:.2f}",
        "[P] save    [Q] quit",
    ]
    for i, line in enumerate(lines):
        color = (100, 255, 100) if i == 0 else (220, 220, 220)
        if i == 5:
            color = (100, 200, 255)
        cv2.putText(
            frame,
            line,
            (10, 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )

    if angles:
        roll, pitch, yaw = angles
        cv2.putText(
            frame,
            f"roll:{roll:+.0f} pitch:{pitch:+.0f} yaw:{yaw:+.0f}",
            (10, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )


# --- Main ---

cam_index = choose_camera()
if cam_index is None:
    exit()

settings = load_settings(cam_index)
cap = cv2.VideoCapture(cam_index)

with PoseLandmarker.create_from_options(
    pose_options
) as pose_landmarker, FaceLandmarker.create_from_options(
    face_options
) as face_landmarker:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)

        pose_results = pose_landmarker.detect_for_video(mp_image, ts)
        face_results = face_landmarker.detect_for_video(mp_image, ts)

        angles = get_head_pose(face_results, *frame.shape[1::-1])

        if pose_results.pose_landmarks:
            h, w, _ = frame.shape
            landmarks = pose_results.pose_landmarks[0]

            for idx, lm in enumerate(landmarks):
                if idx not in RELEVANT_LANDMARKS:
                    continue
                if lm.visibility < 0.5:
                    continue
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 200, 0), -1)

            roll = angles[0] if angles else 0.0
            pitch = angles[1] if angles else 0.0
            yaw = angles[2] if angles else 0.0

            ax, ay, ear_width = get_head_anchor(
                landmarks,
                w,
                h,
                settings["head_offset_mult"],
                settings["head_x_nudge"],
                settings["head_y_nudge"],
                roll,
            )

            DEADZONE = 3

            if smoothed_anchor is None:
                smoothed_anchor = (ax, ay)
            else:
                sx, sy = smoothed_anchor

                dx = ax - sx
                dy = ay - sy

                if abs(dx) < DEADZONE:
                    dx = 0

                if abs(dy) < DEADZONE:
                    dy = 0

                sx += ANCHOR_SMOOTH * dx
                sy += ANCHOR_SMOOTH * dy

                smoothed_anchor = (sx, sy)

            ax = int(smoothed_anchor[0])
            ay = int(smoothed_anchor[1])

            scaled = scale_png(overlay_img, ear_width, settings["head_scale_mult"])
            rotated = rotate_png(scaled, roll, pitch, yaw)
            composite(frame, rotated, ax, ay)

            cv2.circle(frame, (ax, ay), 6, (0, 0, 255), -1)
            cv2.circle(frame, (ax, ay), 6, (255, 255, 255), 2)

        draw_panel(frame, settings, angles)
        cv2.imshow("PNGirl", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            save_settings(cam_index, settings)
        elif key == ord("w"):
            settings["head_offset_mult"] = round(settings["head_offset_mult"] + 0.1, 2)
        elif key == ord("s"):
            settings["head_offset_mult"] = round(settings["head_offset_mult"] - 0.1, 2)
        elif key == ord("a"):
            settings["head_x_nudge"] -= 5
        elif key == ord("d"):
            settings["head_x_nudge"] += 5
        elif key == ord("i"):
            settings["head_y_nudge"] -= 5
        elif key == ord("k"):
            settings["head_y_nudge"] += 5
        elif key == ord("z"):
            settings["head_scale_mult"] = round(settings["head_scale_mult"] + 0.1, 2)
        elif key == ord("x"):
            settings["head_scale_mult"] = round(settings["head_scale_mult"] - 0.1, 2)

cap.release()
cv2.destroyAllWindows()
