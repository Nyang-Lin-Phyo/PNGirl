from PyQt6.QtWidgets import QMainWindow, QStackedWidget

from app.settings import load_settings, save_settings
from app.worker import CameraWorker
from app.drag_state import DragState
from app.ui.picker import CameraPickerPage
from app.ui.viewer import ViewerPage


# How close a pinch must be to a PNG in the frame to "pick it up"
PINCH_PICKUP_RADIUS = 60


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PNGirl")
        self.setMinimumSize(1100, 620)

        self.cam_index  = None
        self.settings   = {}
        self.worker     = None
        self.drag_state = DragState()

        # Track which hands were pinching last frame for edge detection
        self._prev_pinching: dict[int, bool] = {}
        # Last known anchor positions from worker
        self._anchor_pos: dict = {}

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.picker = CameraPickerPage()
        self.picker.camera_selected.connect(self._start_camera)
        self.stack.addWidget(self.picker)

        self.viewer = ViewerPage(
            on_save=self._handle_settings_update,
            on_back=self._stop_camera,
        )
        self.viewer.set_drag_state(self.drag_state)
        self.stack.addWidget(self.viewer)

    # ── Camera lifecycle ──────────────────────────────────────────────────

    def _start_camera(self, cam_index):
        self.cam_index = cam_index
        self.settings  = load_settings(cam_index)

        self.viewer.load_settings(self.settings)
        self.stack.setCurrentWidget(self.viewer)

        self.worker = CameraWorker(
            cam_index,
            lambda: self.settings,
            self.drag_state,
        )
        self.worker.frame_ready.connect(self.viewer.update_frame)
        self.worker.hand_state.connect(self._on_hand_state)
        self.worker.anchor_updated.connect(self._on_anchor_updated)
        self.worker.start()

    def _stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.stack.setCurrentWidget(self.picker)

    # ── Anchor position sync ──────────────────────────────────────────────

    def _on_anchor_updated(self, positions: dict):
        self._anchor_pos = positions
        self.viewer.update_anchor_positions(positions)

    # ── Hand pinch state machine ──────────────────────────────────────────

    def _on_hand_state(self, hands: list):
        snap_threshold = int(self.settings.get("snap_threshold", 80))
        _, slots, _ = self.drag_state.snapshot()

        for hand in hands:
            hand_id  = hand["hand_id"]
            pinching = hand["pinching"]
            px, py   = hand["pinch_x"], hand["pinch_y"]

            was_pinching = self._prev_pinching.get(hand_id, False)

            # ── Pinch start (was open, now closed) ────────────────────────
            if pinching and not was_pinching:
                # Check if pinch is near an existing dragged item first,
                # then check if it's near a snapped anchor slot
                picked = False

                for key, (ax, ay) in self._anchor_pos.items():
                    dist = ((px - ax)**2 + (py - ay)**2) ** 0.5
                    if dist < PINCH_PICKUP_RADIUS and slots.get(key):
                        png_path = slots[key]
                        self.drag_state.clear_anchor(key)
                        self.drag_state.start_hand_drag(hand_id, png_path, px, py)
                        picked = True
                        break

            # ── Pinch held (dragging) ─────────────────────────────────────
            elif pinching and was_pinching:
                self.drag_state.move_hand_drag(hand_id, px, py)

            # ── Pinch release ─────────────────────────────────────────────
            elif not pinching and was_pinching:
                self.drag_state.release_hand_drag(
                    hand_id, self._anchor_pos, snap_threshold)

            self._prev_pinching[hand_id] = pinching

        # Clear state for hands that disappeared
        active_ids = {h["hand_id"] for h in hands}
        for hand_id in list(self._prev_pinching.keys()):
            if hand_id not in active_ids:
                # Hand disappeared while pinching — treat as release
                if self._prev_pinching.get(hand_id):
                    self.drag_state.release_hand_drag(
                        hand_id, self._anchor_pos, snap_threshold)
                self._prev_pinching.pop(hand_id, None)

    # ── Settings bridge ───────────────────────────────────────────────────

    def _handle_settings_update(self, new_settings, autosave=False):
        self.settings = new_settings
        if autosave and self.cam_index is not None:
            save_settings(self.cam_index, self.settings)

    # ── Cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()