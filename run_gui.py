from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from bottle_detector.cameras import discover_cameras
from bottle_detector.config import AppConfig
from bottle_detector.openers import open_file, open_folder, reveal_in_file_manager
from bottle_detector.paths import resolve_app_path
from bottle_detector.runner import run_detector


class DetectorWorker(QObject):
    preview_ready = Signal(QImage)
    status_ready = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, source: str, stop_event: threading.Event) -> None:
        super().__init__()
        self.source = source
        self.stop_event = stop_event

    @Slot()
    def run(self) -> None:
        try:
            config = AppConfig(
                source=self.source,
                output_path=Path("outputs/detections.json"),
                crops_dir=Path("outputs/crops"),
                result_dir=Path("result"),
                display=False,
            )
            run_detector(
                config,
                stop_event=self.stop_event,
                preview_callback=self._on_preview,
                progress_callback=self._on_progress,
            )
        except Exception as exc:
            self.failed.emit(str(exc))

    def _on_preview(self, frame_bgr: Any) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self.preview_ready.emit(image)

    def _on_progress(self, payload: dict[str, Any]) -> None:
        event = payload.get("event")
        if event == "started":
            self.status_ready.emit("بدء تشغيل الكاميرا والتحليل...")
        elif event == "captured":
            self.status_ready.emit(f"تم التقاط العلبة رقم {payload.get('sequence')}")
        elif event == "progress":
            self.status_ready.emit(
                "العلب الملتقطة: "
                f"{payload.get('captured_count', 0)} | "
                f"المكتملة: {payload.get('completed_count', 0)} | "
                f"قيد التحليل: {payload.get('pending_count', 0)}"
            )
        elif event == "finished":
            self.finished_ok.emit(payload)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Bottle Defect Detector")
        self.resize(980, 720)

        self.stop_event: threading.Event | None = None
        self.thread: QThread | None = None
        self.worker: DetectorWorker | None = None
        self.last_excel_path: Path | None = None

        self.camera_combo = QComboBox()
        self.refresh_button = QPushButton("تحديث الكاميرات")
        self.start_button = QPushButton("تشغيل الكاميرا وبدء التحليل")
        self.stop_button = QPushButton("إيقاف الكاميرا وإنهاء التحليل")
        self.open_excel_button = QPushButton("فتح ملف Excel")
        self.open_folder_button = QPushButton("فتح مجلد النتائج")
        self.status_label = QLabel("اختر الكاميرا ثم اضغط تشغيل.")
        self.preview_label = QLabel("لا يوجد بث حالياً")

        self._setup_ui()
        self._connect_signals()
        self.refresh_cameras()

    def _setup_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("الكاميرا:"))
        top_bar.addWidget(self.camera_combo, 1)
        top_bar.addWidget(self.refresh_button)
        layout.addLayout(top_bar)

        button_bar = QHBoxLayout()
        button_bar.addWidget(self.start_button)
        button_bar.addWidget(self.stop_button)
        button_bar.addWidget(self.open_excel_button)
        button_bar.addWidget(self.open_folder_button)
        layout.addLayout(button_bar)

        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(480)
        self.preview_label.setStyleSheet(
            "QLabel { background: #101820; color: #ffffff; border-radius: 12px; font-size: 18px; }"
        )
        preview_frame = QFrame()
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_frame, 1)

        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.status_label.setStyleSheet("font-size: 16px; padding: 8px;")
        layout.addWidget(self.status_label)

        self.stop_button.setEnabled(False)
        self.open_excel_button.setEnabled(False)
        self.open_folder_button.setEnabled(True)
        self.setCentralWidget(root)

    def _connect_signals(self) -> None:
        self.refresh_button.clicked.connect(self.refresh_cameras)
        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.open_excel_button.clicked.connect(self.open_latest_excel)
        self.open_folder_button.clicked.connect(self.open_result_folder)

    @Slot()
    def refresh_cameras(self) -> None:
        self.camera_combo.clear()
        cameras = discover_cameras()
        if not cameras:
            self.camera_combo.addItem("0", "0")
            self.status_label.setText("لم يتم اكتشاف كاميرا، تم وضع camera 0 كخيار افتراضي.")
            return
        for index in cameras:
            self.camera_combo.addItem(f"Camera {index}", str(index))
        self.status_label.setText(f"تم اكتشاف {len(cameras)} كاميرا.")

    @Slot()
    def start_analysis(self) -> None:
        if self.thread is not None:
            return

        source = self.camera_combo.currentData() or "0"
        self.stop_event = threading.Event()
        self.thread = QThread()
        self.worker = DetectorWorker(str(source), self.stop_event)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.preview_ready.connect(self.update_preview)
        self.worker.status_ready.connect(self.status_label.setText)
        self.worker.finished_ok.connect(self.analysis_finished)
        self.worker.failed.connect(self.analysis_failed)
        self.worker.finished_ok.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_thread)

        self.start_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.open_excel_button.setEnabled(False)
        self.status_label.setText("جاري تشغيل الكاميرا...")
        self.thread.start()

    @Slot()
    def stop_analysis(self) -> None:
        if self.stop_event is not None:
            self.status_label.setText("جاري إيقاف الكاميرا وانتظار نتائج الذكاء الاصطناعي...")
            self.stop_event.set()
        self.stop_button.setEnabled(False)

    @Slot(QImage)
    def update_preview(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    @Slot(dict)
    def analysis_finished(self, payload: dict[str, Any]) -> None:
        report_path = payload.get("latest_path") or payload.get("report_path")
        self.last_excel_path = Path(report_path) if report_path else None
        detections_count = payload.get("detections_count", 0)
        self.status_label.setText(f"اكتمل التحليل. عدد العلب: {detections_count}. تم إنشاء ملف Excel.")
        self.open_excel_button.setEnabled(self.last_excel_path is not None)
        if self.last_excel_path:
            try:
                reveal_in_file_manager(self.last_excel_path)
            except Exception:
                pass

    @Slot(str)
    def analysis_failed(self, message: str) -> None:
        self.status_label.setText("فشل التشغيل.")
        QMessageBox.critical(self, "خطأ", message)

    @Slot()
    def cleanup_thread(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        if self.thread is not None:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None
        self.stop_event = None
        self.start_button.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    @Slot()
    def open_latest_excel(self) -> None:
        if self.last_excel_path and self.last_excel_path.exists():
            open_file(self.last_excel_path)
            return
        latest = resolve_app_path("result/detections_latest.xlsx")
        if latest.exists():
            self.last_excel_path = latest
            open_file(latest)
            return
        QMessageBox.information(self, "Excel", "لا يوجد ملف Excel جاهز حالياً.")

    @Slot()
    def open_result_folder(self) -> None:
        open_folder(resolve_app_path("result"))

    def closeEvent(self, event: Any) -> None:
        if self.thread is not None and self.stop_event is not None:
            self.stop_event.set()
            self.thread.quit()
            self.thread.wait(3000)
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Bottle Defect Detector")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

