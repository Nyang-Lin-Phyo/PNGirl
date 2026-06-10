import json
import os

SETTINGS_FILE = "camera_settings.json"


def default_settings():
    return {
        "head_offset_mult":    0.9,
        "head_x_nudge":        0,
        "head_y_nudge":        110,
        "head_scale_mult":     1.0,
        "left_shoulder_y_nudge":   60,
        "left_shoulder_x_nudge":   0,
        "left_shoulder_scale_mult": 1.0,
        "right_shoulder_y_nudge":  60,
        "right_shoulder_x_nudge":  0,
        "right_shoulder_scale_mult": 1.0,
    }


def load_settings(cam_index):
    defaults = default_settings()
    if not os.path.exists(SETTINGS_FILE):
        return defaults
    with open(SETTINGS_FILE) as f:
        data = json.load(f)
    saved = data.get(str(cam_index), {})
    # Fill in any missing keys with defaults
    return {**defaults, **saved}


def save_settings(cam_index, settings):
    data = {}
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            data = json.load(f)
    data[str(cam_index)] = settings
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)