# -*- coding: utf-8 -*-

import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFrame, QMessageBox
)
from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QFont

class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DepthLive3D Launcher")
        self.resize(600, 300)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QFrame {
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #d1d5db;
            }
            QLabel[class="title"] {
                color: #1f2937;
            }
            QLabel[class="desc"] {
                color: #6b7280;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_conv_title = QLabel("Conversion 3D")
        lbl_conv_title.setProperty("class", "title")
        lbl_conv_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_conv_title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))

        lbl_conv_desc = QLabel("Convert video files to 3D\nand encode with advanced options.")
        lbl_conv_desc.setProperty("class", "desc")
        lbl_conv_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_conv_desc.setFont(QFont("Segoe UI", 10))

        btn_conv = QPushButton("Launch Conversion 3D")
        btn_conv.setMinimumHeight(45)
        btn_conv.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_conv.clicked.connect(self.launch_conversion)

        left_layout.addWidget(lbl_conv_title)
        left_layout.addSpacing(10)
        left_layout.addWidget(lbl_conv_desc)
        left_layout.addSpacing(20)
        left_layout.addWidget(btn_conv)

        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_live_title = QLabel("Live 3D")
        lbl_live_title.setProperty("class", "title")
        lbl_live_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_live_title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))

        lbl_live_desc = QLabel("Convert real-time screen to 3D\nand render as an overlay.")
        lbl_live_desc.setProperty("class", "desc")
        lbl_live_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_live_desc.setFont(QFont("Segoe UI", 10))

        btn_live = QPushButton("Launch Live 3D")
        btn_live.setMinimumHeight(45)
        btn_live.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_live.clicked.connect(self.launch_live)

        right_layout.addWidget(lbl_live_title)
        right_layout.addSpacing(10)
        right_layout.addWidget(lbl_live_desc)
        right_layout.addSpacing(20)
        right_layout.addWidget(btn_live)

        main_layout.addWidget(left_frame)
        main_layout.addWidget(right_frame)

    def get_launch_command(self, arg):
        if getattr(sys, 'frozen', False):
            return sys.executable, [arg]
        else:
            return sys.executable, [os.path.abspath(sys.argv[0]), arg]

    def launch_conversion(self):
        program, args = self.get_launch_command("--run-conversion")
        ok = QProcess.startDetached(program, args)
        result = ok[0] if isinstance(ok, tuple) else ok
        if not result:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch Conversion 3D.\n{program} {args}")
        else:
            self.close()

    def launch_live(self):
        program, args = self.get_launch_command("--run-live")
        ok = QProcess.startDetached(program, args)
        result = ok[0] if isinstance(ok, tuple) else ok
        if not result:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch Live 3D.\n{program} {args}")
        else:
            self.close()


def run_conversion_app():
    try:
        import conversion3d
    except Exception as e:
        _show_fatal_error("Conversion 3D", e)
        return
    conversion3d.main()

def run_live_app():
    try:
        import live3d
    except Exception as e:
        _show_fatal_error("Live 3D", e)
        return
    if hasattr(live3d, 'main'):
        live3d.main()
    else:
        _show_fatal_error("Live 3D", RuntimeError("main() function is not defined in live3d.py."))

def _show_fatal_error(module_name: str, error: Exception):
    import sys
    import traceback
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        _app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, f"{module_name} Launch Failed", f"{error}\n\n{traceback.format_exc()}")
    except Exception:
        print(f"[Fatal] {module_name} failed to start: {error}")


if __name__ == "__main__":
    if "--run-conversion" in sys.argv:
        sys.argv.remove("--run-conversion")
        run_conversion_app()
        sys.exit(0)
    elif "--run-live" in sys.argv:
        sys.argv.remove("--run-live")
        run_live_app()
        sys.exit(0)
    elif "--run-engine" in sys.argv:
        run_live_app()
        sys.exit(0)

    else:
        app = QApplication(sys.argv)
        launcher = LauncherWindow()
        launcher.show()
        sys.exit(app.exec())
