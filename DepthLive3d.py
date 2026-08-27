# -*- coding: utf-8 -*-

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFrame, QMessageBox, QComboBox
)
from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QFont


# ------------------------------------------------------------------
# 언어(번역) 관련 설정
# ------------------------------------------------------------------
# 이 설정 파일은 런처(DepthLive3d.py)뿐 아니라 나중에 conversion3d.py,
# live3d.py 등에서도 같은 경로를 읽어 현재 선택된 언어를 따라갈 수 있도록
# 실행 파일과 같은 폴더에 저장합니다.
def _get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


CONFIG_PATH = os.path.join(_get_base_dir(), "depthlive3d_config.json")

SUPPORTED_LANGUAGES = ["en", "ko"]
DEFAULT_LANGUAGE = "en"

TRANSLATIONS = {
    "en": {
        "window_title": "DepthLive3D Launcher",
        "conv_title": "Conversion 3D",
        "conv_desc": "Convert video files to 3D\nand encode with advanced options.",
        "conv_btn": "Launch Conversion 3D",
        "live_title": "Live 3D",
        "live_desc": "Convert real-time screen to 3D\nand render as an overlay.",
        "live_btn": "Launch Live 3D",
        "language_label": "Language:",
        "launch_error_title": "Launch Error",
        "launch_error_conv": "Failed to launch Conversion 3D.\n{cmd}",
        "launch_error_live": "Failed to launch Live 3D.\n{cmd}",
        "fatal_error_title": "{module} Launch Failed",
    },
    "ko": {
        "window_title": "DepthLive3D 런처",
        "conv_title": "컨버전 3D",
        "conv_desc": "영상 파일을 3D로 변환하고\n고급 옵션으로 인코딩합니다.",
        "conv_btn": "컨버전 3D 실행",
        "live_title": "라이브 3D",
        "live_desc": "실시간 화면을 3D로 변환하여\n오버레이로 렌더링합니다.",
        "live_btn": "라이브 3D 실행",
        "language_label": "언어:",
        "launch_error_title": "실행 오류",
        "launch_error_conv": "컨버전 3D를 실행하지 못했습니다.\n{cmd}",
        "launch_error_live": "라이브 3D를 실행하지 못했습니다.\n{cmd}",
        "fatal_error_title": "{module} 실행 실패",
    },
}

LANGUAGE_DISPLAY_NAMES = {
    "en": "English",
    "ko": "한국어",
}


def load_language():
    """설정 파일에서 저장된 언어를 읽어옵니다. 없으면 기본값(영어)을 반환합니다."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            lang = data.get("language", DEFAULT_LANGUAGE)
            if lang in SUPPORTED_LANGUAGES:
                return lang
    except Exception:
        pass
    return DEFAULT_LANGUAGE


def save_language(lang: str):
    """선택된 언어를 설정 파일에 저장합니다.
    conversion3d.py / live3d.py 등 다른 모듈에서도 이 파일을 읽어
    동일한 언어 설정을 따라갈 수 있습니다."""
    try:
        data = {}
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data["language"] = lang
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to save language setting: {e}")


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_lang = load_language()

        self.resize(600, 340)
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
            QLabel[class="lang_label"] {
                color: #374151;
                font-weight: bold;
            }
            QComboBox {
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 4px 8px;
                background-color: #ffffff;
                min-width: 120px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(20, 20, 20, 20)
        outer_layout.setSpacing(15)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)

        # ---- Conversion 3D 카드 ----
        self.left_frame = QFrame()
        left_layout = QVBoxLayout(self.left_frame)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_conv_title = QLabel()
        self.lbl_conv_title.setProperty("class", "title")
        self.lbl_conv_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_conv_title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))

        self.lbl_conv_desc = QLabel()
        self.lbl_conv_desc.setProperty("class", "desc")
        self.lbl_conv_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_conv_desc.setFont(QFont("Segoe UI", 10))

        self.btn_conv = QPushButton()
        self.btn_conv.setMinimumHeight(45)
        self.btn_conv.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_conv.clicked.connect(self.launch_conversion)

        left_layout.addWidget(self.lbl_conv_title)
        left_layout.addSpacing(10)
        left_layout.addWidget(self.lbl_conv_desc)
        left_layout.addSpacing(20)
        left_layout.addWidget(self.btn_conv)

        # ---- Live 3D 카드 ----
        self.right_frame = QFrame()
        right_layout = QVBoxLayout(self.right_frame)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_live_title = QLabel()
        self.lbl_live_title.setProperty("class", "title")
        self.lbl_live_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_live_title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))

        self.lbl_live_desc = QLabel()
        self.lbl_live_desc.setProperty("class", "desc")
        self.lbl_live_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_live_desc.setFont(QFont("Segoe UI", 10))

        self.btn_live = QPushButton()
        self.btn_live.setMinimumHeight(45)
        self.btn_live.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_live.clicked.connect(self.launch_live)

        right_layout.addWidget(self.lbl_live_title)
        right_layout.addSpacing(10)
        right_layout.addWidget(self.lbl_live_desc)
        right_layout.addSpacing(20)
        right_layout.addWidget(self.btn_live)

        main_layout.addWidget(self.left_frame)
        main_layout.addWidget(self.right_frame)

        outer_layout.addLayout(main_layout)

        # ---- 언어 선택 영역 (하단) ----
        lang_bar = QHBoxLayout()
        lang_bar.addStretch()

        self.lbl_language = QLabel()
        self.lbl_language.setProperty("class", "lang_label")

        self.combo_language = QComboBox()
        for code in SUPPORTED_LANGUAGES:
            self.combo_language.addItem(LANGUAGE_DISPLAY_NAMES[code], userData=code)
        self.combo_language.currentIndexChanged.connect(self.on_language_changed)

        lang_bar.addWidget(self.lbl_language)
        lang_bar.addWidget(self.combo_language)
        lang_bar.addStretch()

        outer_layout.addLayout(lang_bar)

        # 초기 텍스트 및 콤보박스 상태 적용
        self.apply_language(self.current_lang, save=False)

    def tr_text(self, key):
        return TRANSLATIONS.get(self.current_lang, TRANSLATIONS[DEFAULT_LANGUAGE]).get(key, key)

    def apply_language(self, lang: str, save: bool = True):
        """선택된 언어를 즉시 UI에 반영합니다."""
        if lang not in SUPPORTED_LANGUAGES:
            lang = DEFAULT_LANGUAGE
        self.current_lang = lang

        self.setWindowTitle(self.tr_text("window_title"))
        self.lbl_conv_title.setText(self.tr_text("conv_title"))
        self.lbl_conv_desc.setText(self.tr_text("conv_desc"))
        self.btn_conv.setText(self.tr_text("conv_btn"))
        self.lbl_live_title.setText(self.tr_text("live_title"))
        self.lbl_live_desc.setText(self.tr_text("live_desc"))
        self.btn_live.setText(self.tr_text("live_btn"))
        self.lbl_language.setText(self.tr_text("language_label"))

        # 콤보박스가 현재 언어를 가리키도록 동기화 (신호 재귀 방지)
        idx = self.combo_language.findData(lang)
        if idx != -1 and self.combo_language.currentIndex() != idx:
            self.combo_language.blockSignals(True)
            self.combo_language.setCurrentIndex(idx)
            self.combo_language.blockSignals(False)

        if save:
            save_language(lang)

    def on_language_changed(self, index):
        lang = self.combo_language.itemData(index)
        self.apply_language(lang, save=True)

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
            QMessageBox.critical(
                self,
                self.tr_text("launch_error_title"),
                self.tr_text("launch_error_conv").format(cmd=f"{program} {args}"),
            )
        else:
            self.close()

    def launch_live(self):
        program, args = self.get_launch_command("--run-live")
        ok = QProcess.startDetached(program, args)
        result = ok[0] if isinstance(ok, tuple) else ok
        if not result:
            QMessageBox.critical(
                self,
                self.tr_text("launch_error_title"),
                self.tr_text("launch_error_live").format(cmd=f"{program} {args}"),
            )
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
        lang = load_language()
        title = TRANSLATIONS.get(lang, TRANSLATIONS[DEFAULT_LANGUAGE])["fatal_error_title"].format(module=module_name)
        QMessageBox.critical(None, title, f"{error}\n\n{traceback.format_exc()}")
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
