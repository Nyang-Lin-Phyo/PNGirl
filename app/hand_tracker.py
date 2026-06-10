"""
Wraps MediaPipe HandLandmarker.
Detects pinch (thumb tip <-> index tip distance) for up to 2 hands.
Emits per-hand state each frame.
"""

import math
import mediapipe as mp
import numpy as np

BaseOptions         = mp.tasks.BaseOptions
HandLandmarker      = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode   = mp.tasks.vision.RunningMode

PINCH_THRESHOLD  = 0.06   # normalised distance (fraction of image width)
PINCH_HYSTERESIS = 0.02   # must open this much further to un-pinch


def _build_options():
    return HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


class HandTracker:
    """
    Create once, call detect() every frame.
    Returns list of HandState dicts (one per detected hand).
    """

    def __init__(self):
        self._options  = _build_options()
        self._landmarker = HandLandmarker.create_from_options(self._options)
        # track pinch state per hand index to apply hysteresis
        self._pinching: dict[int, bool] = {}

    def close(self):
        self._landmarker.close()

    def detect(self, mp_image, timestamp_ms: int, frame_w: int, frame_h: int):
        """
        Returns list of dicts:
          {
            hand_id:   int,          # 0 or 1 (order of detection)
            pinching:  bool,
            pinch_x:   int,          # pixel x of pinch midpoint
            pinch_y:   int,          # pixel y of pinch midpoint
            landmarks: list          # raw normalised landmarks
          }
        """
        results = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        hands = []

        for i, hand_landmarks in enumerate(results.hand_landmarks):
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]

            # normalised distance
            dist = math.dist(
                (thumb_tip.x, thumb_tip.y),
                (index_tip.x, index_tip.y)
            )

            was_pinching = self._pinching.get(i, False)
            if was_pinching:
                is_pinching = dist < (PINCH_THRESHOLD + PINCH_HYSTERESIS)
            else:
                is_pinching = dist < PINCH_THRESHOLD
            self._pinching[i] = is_pinching

            # pinch midpoint in pixel space
            mx = int(((thumb_tip.x + index_tip.x) / 2) * frame_w)
            my = int(((thumb_tip.y + index_tip.y) / 2) * frame_h)

            hands.append({
                "hand_id":   i,
                "pinching":  is_pinching,
                "pinch_x":   mx,
                "pinch_y":   my,
                "landmarks": hand_landmarks,
            })

        # clear stale pinch state for hands that disappeared
        for i in list(self._pinching.keys()):
            if i >= len(results.hand_landmarks):
                self._pinching.pop(i, None)

        return hands