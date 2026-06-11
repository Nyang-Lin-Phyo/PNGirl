from PyQt6.QtWidgets import QMainWindow, QStackedWidget

from app.settings import load_settings, save_settings
from app.worker import CameraWorker
from app.drag_state import DragState
from app.ui.picker import CameraPickerPage
from app.ui.viewer import ViewerPage

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

        self._prev_pinching: dict[int, bool] = {}
        self._anchor_pos: dict = {}
        # Cache frame dims so we can map pinch coords to screen space
        self._frame_size: tuple = (640, 480)

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

        self.worker = CameraWorker(cam_index, lambda: self.settings, self.drag_state)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.hand_state.connect(self._on_hand_state)
        self.worker.anchor_updated.connect(self._on_anchor_updated)
        self.worker.start()

    def _stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.stack.setCurrentWidget(self.picker)

    def _on_frame_ready(self, frame, angles):
        self._frame_size = (frame.shape[1], frame.shape[0])
        self.viewer.update_frame(frame, angles)

    # ── Anchor position sync ──────────────────────────────────────────────

    def _on_anchor_updated(self, positions: dict):
        self._anchor_pos = positions
        self.viewer.update_anchor_positions(positions)

    # ── Coordinate mapping ────────────────────────────────────────────────

    def _frame_to_screen(self, fx: int, fy: int) -> tuple[int, int]:
        """
        Convert a frame-space pixel to a screen-space pixel,
        accounting for the KeepAspectRatio scaling inside the video label.
        """
        vl = self.viewer.video_label
        fw, fh = self._frame_size
        lw, lh = vl.width(), vl.height()
        scale  = min(lw / fw, lh / fh)
        off_x  = (lw - fw * scale) / 2
        off_y  = (lh - fh * scale) / 2
        lx = fx * scale + off_x
        ly = fy * scale + off_y
        # Map from video_label-local coords to screen coords
        screen_pt = vl.mapToGlobal(vl.rect().topLeft())
        return int(screen_pt.x() + lx), int(screen_pt.y() + ly)

    def _pinch_over_asset_panel(self, fx: int, fy: int):
        """
        Returns the png_path of the thumbnail under the pinch,
        or None if the pinch is not over the asset panel.
        """
        sx, sy = self._frame_to_screen(fx, fy)
        panel  = self.viewer.asset_panel

        # Check if screen point is inside the asset panel's global rect
        panel_tl = panel.mapToGlobal(panel.rect().topLeft())
        panel_rect_global = panel.rect().translated(panel_tl)
        if not panel_rect_global.contains(sx, sy):
            return None

        # Convert to asset panel local coords and hit-test thumbnails
        local_x = sx - panel_tl.x()
        local_y = sy - panel_tl.y()

        for thumb in panel._thumbnails:
            # thumb geometry is relative to the grid widget inside the scroll area
            # mapTo gives coords relative to panel
            tl = thumb.mapTo(panel, thumb.rect().topLeft())
            tr = tl + thumb.rect().bottomRight()
            if tl.x() <= local_x <= tr.x() and tl.y() <= local_y <= tr.y():
                return thumb.png_path

        return None

    # ── Hand pinch state machine ──────────────────────────────────────────

    def _on_hand_state(self, hands: list):
        snap_threshold = int(self.settings.get("snap_threshold", 80))
        _, slots, _ = self.drag_state.snapshot()

        for hand in hands:
            hand_id  = hand["hand_id"]
            pinching = hand["pinching"]
            px, py   = hand["pinch_x"], hand["pinch_y"]

            was_pinching = self._prev_pinching.get(hand_id, False)

            # ── Pinch start ───────────────────────────────────────────────
            if pinching and not was_pinching:

                # 1. Check if pinching over the asset panel
                png_path = self._pinch_over_asset_panel(px, py)
                if png_path:
                    self.drag_state.start_hand_drag(hand_id, png_path, px, py)

                else:
                    # 2. Check if pinching near a snapped anchor
                    for key, (ax, ay) in self._anchor_pos.items():
                        dist = ((px - ax)**2 + (py - ay)**2) ** 0.5
                        if dist < PINCH_PICKUP_RADIUS and slots.get(key):
                            self.drag_state.clear_anchor(key)
                            self.drag_state.start_hand_drag(
                                hand_id, slots[key], px, py)
                            break

            # ── Pinch held ────────────────────────────────────────────────
            elif pinching and was_pinching:
                self.drag_state.move_hand_drag(hand_id, px, py)

            # ── Pinch release ─────────────────────────────────────────────
            elif not pinching and was_pinching:
                self.drag_state.release_hand_drag(
                    hand_id, self._anchor_pos, snap_threshold)

            self._prev_pinching[hand_id] = pinching

        # Hands that disappeared mid-pinch
        active_ids = {h["hand_id"] for h in hands}
        for hand_id in list(self._prev_pinching.keys()):
            if hand_id not in active_ids:
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