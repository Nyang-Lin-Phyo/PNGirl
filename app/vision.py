import cv2
import math
import numpy as np

SMOOTH = 0.4
ANCHOR_SMOOTH = 0.4


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
    nose_y    = int(nose.y * h)
    lx, ly    = int(left_ear.x  * w), int(left_ear.y * h)
    rx, ry    = int(right_ear.x * w), int(right_ear.y * h)
    ear_mid_x = (lx + rx) // 2
    ear_mid_y = (ly + ry) // 2
    head_height  = abs(nose_y - ear_mid_y)
    offset_dist  = int(head_height * offset_mult) - y_nudge
    tilt_angle   = math.radians(roll_deg)
    ax = int(ear_mid_x + offset_dist * math.sin(tilt_angle)) + x_nudge
    ay = int(ear_mid_y - offset_dist * math.cos(tilt_angle))
    ear_width = int(math.dist((lx, ly), (rx, ry)))
    return ax, ay, ear_width


def get_shoulder_anchor(landmarks, w, h, side, y_nudge, x_nudge):
    """
    side: 'left' (landmark 11) or 'right' (landmark 12).
    Returns (ax, ay) for the anchor point above the shoulder.
    """
    lm = landmarks[11] if side == "left" else landmarks[12]
    sx = int(lm.x * w) + x_nudge
    sy = int(lm.y * h) - y_nudge
    return sx, sy


def smooth_anchor(prev, target, deadzone=3):
    if prev is None:
        return target
    px, py = prev
    tx, ty = target
    dx = (tx - px) if abs(tx - px) >= deadzone else 0
    dy = (ty - py) if abs(ty - py) >= deadzone else 0
    return (px + ANCHOR_SMOOTH * dx, py + ANCHOR_SMOOTH * dy)


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
    sy2  = yaw_factor * (w * 0.3)
    dst2 = np.float32([[sy2,h*0.1],[w-sy2,h*0.1],[w-sy2,h*0.9],[sy2,h*0.9]])
    rotated = cv2.warpPerspective(rotated, cv2.getPerspectiveTransform(src, dst2), (w, h),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated


def scale_png(img, ref_width, scale_mult):
    ph, pw = img.shape[:2]
    target_w = int(ref_width * scale_mult)
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
    px1 = fx1 - x1
    py1 = fy1 - y1
    px2 = px1 + (fx2 - fx1)
    py2 = py1 + (fy2 - fy1)
    png_crop = png[py1:py2, px1:px2]
    alpha = png_crop[:, :, 3:4] / 255.0
    roi = frame[fy1:fy2, fx1:fx2]
    frame[fy1:fy2, fx1:fx2] = (png_crop[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)


def draw_crosshair(frame, ax, ay):
    r = 5
    cv2.line(frame,  (ax-r, ay), (ax+r, ay), (255,255,255), 1, cv2.LINE_AA)
    cv2.line(frame,  (ax, ay-r), (ax, ay+r), (255,255,255), 1, cv2.LINE_AA)
    cv2.circle(frame, (ax, ay), r+3, (180,180,180), 1, cv2.LINE_AA)