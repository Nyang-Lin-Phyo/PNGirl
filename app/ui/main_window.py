from PyQt6.QtWidgets import QMainWindow, QStackedWidget

from app.settings import load_settings, save_settings
from app.worker import CameraWorker
from app.ui.picker import CameraPickerPage
from app.ui.viewer import ViewerPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PNGirl")
        self.setMinimumSize(960, 580)

        self.cam_index = None
        self.settings  = {}
        self.worker    = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Page 0: camera picker
        self.picker = CameraPickerPage()
        self.picker.camera_selected.connect(self._start_camera)
        self.stack.addWidget(self.picker)

        # Page 1: viewer (built once, reused)
        self.viewer = ViewerPage(
            on_save=self._handle_settings_update,
            on_back=self._stop_camera,
        )
        self.stack.addWidget(self.viewer)

    # ── Camera lifecycle ──────────────────────────────────────────────────

    def _start_camera(self, cam_index):
        self.cam_index = cam_index
        self.settings  = load_settings(cam_index)

        self.viewer.load_settings(self.settings)
        self.stack.setCurrentWidget(self.viewer)

        self.worker = CameraWorker(cam_index, lambda: self.settings)
        self.worker.frame_ready.connect(self.viewer.update_frame)
        self.worker.start()

    def _stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.stack.setCurrentWidget(self.picker)

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