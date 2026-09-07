# -*- coding: utf-8 -*-

import sys
import os
import json
import shutil
import subprocess
import time
import tempfile
import zipfile
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFrame, QMessageBox, QComboBox,
    QLineEdit, QTextEdit, QFileDialog, QGroupBox, QGridLayout,
    QProgressDialog
)
from PySide6.QtCore import QProcess, Qt, QThread, Signal, QEventLoop
from PySide6.QtGui import QFont


# ------------------------------------------------------------------
# ffmpeg 실행 파일 경로 탐색 (별도 모듈 임포트 없이 자체적으로 처리)
# ------------------------------------------------------------------
def _get_base_dir_for_ffmpeg():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def get_executable(name: str) -> str:
    ext = ".exe" if os.name == "nt" else ""
    local = os.path.join(_get_base_dir_for_ffmpeg(), name + ext)
    if os.path.isfile(local):
        return local
    return shutil.which(name) or name


FFMPEG_EXE = get_executable("ffmpeg")
FFPROBE_EXE = get_executable("ffprobe")
FFMPEG_MISSING = (FFMPEG_EXE == "ffmpeg") or (FFPROBE_EXE == "ffprobe")

FFMPEG_BUILDS_API_URL = "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest"


def _ffmpeg_asset_name_for_platform():
    if os.name == "nt":
        return "ffmpeg-master-latest-win64-gpl.zip"
    if sys.platform.startswith("linux"):
        return "ffmpeg-master-latest-linux64-gpl.zip"
    return None


def fetch_latest_ffmpeg_asset_url():
    asset_name = _ffmpeg_asset_name_for_platform()
    if asset_name is None:
        return None
    req = urllib.request.Request(
        FFMPEG_BUILDS_API_URL,
        headers={"User-Agent": "depthlive3d-ffmpeg-installer", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    for asset in data.get("assets", []):
        if asset.get("name") == asset_name:
            return asset_name, asset.get("browser_download_url")
    return None


def download_and_install_ffmpeg(progress_cb=None, cancel_cb=None):
    """ffmpeg/ffprobe를 공식 FFmpeg-Builds 릴리스에서 내려받아
    프로그램 폴더에 설치합니다. (성공 여부, 에러 메시지) 튜플을 반환합니다."""
    base_dir = _get_base_dir_for_ffmpeg()
    try:
        result = fetch_latest_ffmpeg_asset_url()
    except Exception as e:
        return False, f"Failed to query the latest release info: {e}"
    if not result:
        return False, "Could not find a matching ffmpeg build for this platform."
    asset_name, download_url = result
    if not download_url:
        return False, "Could not find a matching ffmpeg build for this platform."

    tmp_dir = Path(tempfile.mkdtemp(prefix="ffmpeg_dl_"))
    zip_path = tmp_dir / asset_name

    def _reporthook(block_num, block_size, total_size):
        if progress_cb is None:
            return
        downloaded = block_num * block_size
        pct = int(downloaded * 100 / total_size) if total_size > 0 else 0
        progress_cb(min(100, pct))
        if cancel_cb is not None and cancel_cb():
            raise RuntimeError("Download cancelled by user.")

    try:
        urllib.request.urlretrieve(download_url, str(zip_path), reporthook=_reporthook)

        extract_dir = tmp_dir / "extracted"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        ext = ".exe" if os.name == "nt" else ""
        ffmpeg_src = next(extract_dir.rglob(f"ffmpeg{ext}"), None)
        ffprobe_src = next(extract_dir.rglob(f"ffprobe{ext}"), None)
        if ffmpeg_src is None or ffprobe_src is None:
            return False, "Downloaded archive did not contain ffmpeg/ffprobe executables."

        ffmpeg_dst = os.path.join(base_dir, f"ffmpeg{ext}")
        ffprobe_dst = os.path.join(base_dir, f"ffprobe{ext}")
        shutil.copy2(ffmpeg_src, ffmpeg_dst)
        shutil.copy2(ffprobe_src, ffprobe_dst)

        if os.name != "nt":
            os.chmod(ffmpeg_dst, 0o755)
            os.chmod(ffprobe_dst, 0o755)

        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


class FFmpegDownloadWorker(QThread):
    progress = Signal(int)
    finished_ok = Signal(bool, str)

    def __init__(self):
        super().__init__()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        ok, err_msg = download_and_install_ffmpeg(
            progress_cb=self.progress.emit,
            cancel_cb=lambda: self._cancelled,
        )
        self.finished_ok.emit(ok, err_msg)


def ensure_ffmpeg_available(parent, lang: str) -> bool:
    """ffmpeg가 없으면 (라이센스 안내 -> 다운로드 진행 -> 결과 안내) 순으로
    프로그램 창 안쪽에 알림창을 띄우며 자동으로 설치합니다."""
    global FFMPEG_EXE, FFPROBE_EXE, FFMPEG_MISSING

    FFMPEG_EXE = get_executable("ffmpeg")
    FFPROBE_EXE = get_executable("ffprobe")
    FFMPEG_MISSING = (FFMPEG_EXE == "ffmpeg") or (FFPROBE_EXE == "ffprobe")
    if not FFMPEG_MISSING:
        return True

    t = TRANSLATIONS.get(lang, TRANSLATIONS[DEFAULT_LANGUAGE])

    # 1) 라이센스 / 안내 알림창 (프로그램 창 안쪽)
    proceed = QMessageBox.question(
        parent,
        t["ffmpeg_license_title"],
        t["ffmpeg_license_msg"],
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes,
    )
    if proceed != QMessageBox.Yes:
        QMessageBox.warning(parent, t["ffmpeg_missing_title"], t["ffmpeg_missing_msg"].format(dir=_get_base_dir_for_ffmpeg()))
        return False

    # 2) 다운로드 진행 알림창 (프로그램 창 안쪽, 모달)
    dlg = QProgressDialog(t["ffmpeg_downloading_msg"], t["ffmpeg_cancel"], 0, 100, parent)
    dlg.setWindowTitle(t["ffmpeg_downloading_title"])
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumDuration(0)
    dlg.setValue(0)

    worker = FFmpegDownloadWorker()
    result = {"ok": False, "err": ""}

    def _on_progress(pct):
        dlg.setValue(pct)

    def _on_finished(ok, err_msg):
        result["ok"] = ok
        result["err"] = err_msg
        dlg.close()

    worker.progress.connect(_on_progress)
    worker.finished_ok.connect(_on_finished)
    dlg.canceled.connect(worker.cancel)

    loop = QEventLoop()
    worker.finished_ok.connect(loop.quit)
    worker.start()
    loop.exec()
    worker.wait()

    if result["ok"]:
        FFMPEG_EXE = get_executable("ffmpeg")
        FFPROBE_EXE = get_executable("ffprobe")
        FFMPEG_MISSING = (FFMPEG_EXE == "ffmpeg") or (FFPROBE_EXE == "ffprobe")
        if not FFMPEG_MISSING:
            QMessageBox.information(
                parent, t["ffmpeg_done_title"],
                t["ffmpeg_done_msg"].format(dir=_get_base_dir_for_ffmpeg())
            )
            return True
        result["err"] = result["err"] or t["ffmpeg_invalid_files"]

    QMessageBox.critical(
        parent,
        t["ffmpeg_fail_title"],
        t["ffmpeg_fail_msg"].format(err=result["err"], dir=_get_base_dir_for_ffmpeg()),
    )
    return False




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
        "sync_title": "Audio Sync",
        "sync_desc": "Shift or trim a video's audio track\nto fix sync with ffmpeg.",
        "sync_btn": "Launch Audio Sync",
        "language_label": "Language:",
        "launch_error_title": "Launch Error",
        "launch_error_conv": "Failed to launch Conversion 3D.\n{cmd}",
        "launch_error_live": "Failed to launch Live 3D.\n{cmd}",
        "fatal_error_title": "{module} Launch Failed",
        # ---- Audio Sync window ----
        "sync_window_title": "Audio Sync",
        "sync_path_group": " Path Settings ",
        "sync_input_label": "Input Video:",
        "sync_output_label": "Output Folder:",
        "sync_browse": "Browse",
        "sync_opt_group": " Audio Delay Settings ",
        "sync_delay_label": "Delay (ms):",
        "sync_delay_hint": "(just type a number for +, use - for negative, e.g. 350, -360, 0)",
        "sync_start": " Start ",
        "sync_stop": " Stop ",
        "sync_status_idle": "Status: Idle",
        "sync_status_running": "Status: Processing...",
        "sync_status_done": "Status: Done",
        "sync_status_stopped": "Status: Stopped",
        "sync_status_error": "Status: Error",
        "sync_warn_title": "Warning",
        "sync_warn_input": "Please select a valid input video file.",
        "sync_err_title": "Error",
        "sync_err_delay": "Invalid delay value. Please enter an integer in ms (e.g. 350, -360, 0).",
        "sync_err_ffmpeg": "ffmpeg executable was not found.\nPlace ffmpeg next to this program or add it to PATH.",
        "sync_log_start": "Starting audio sync...",
        "sync_log_input": "Input: {name}",
        "sync_log_delay": "Delay: {ms} ms -> Output: {name}",
        "sync_log_done": "Done. Saved to: {name}",
        "sync_done_title": "Audio Sync Complete",
        "sync_done_msg": "Audio sync has been completed successfully.\nSaved to: {name}",
        "sync_log_stopped": "Stopped by user.",
        "sync_log_error": "Error: {err}",
        # ---- ffmpeg auto-download ----
        "ffmpeg_license_title": "ffmpeg Not Found",
        "ffmpeg_license_msg": (
            "ffmpeg was not found in the program folder or system PATH.\n\n"
            "DepthLive3D can automatically download the latest official FFmpeg build\n"
            "(FFmpeg, licensed under GPL v3, https://ffmpeg.org) from the\n"
            "FFmpeg-Builds release page (https://github.com/BtbN/FFmpeg-Builds) and\n"
            "install it into the program folder.\n\n"
            "Do you want to download and install ffmpeg now?"
        ),
        "ffmpeg_missing_title": "ffmpeg Required",
        "ffmpeg_missing_msg": (
            "ffmpeg is required to run this feature.\n"
            "Please download ffmpeg/ffprobe manually and place them in:\n{dir}"
        ),
        "ffmpeg_downloading_title": "Downloading ffmpeg",
        "ffmpeg_downloading_msg": (
            "Downloading the latest ffmpeg build (GPL v3) from the\n"
            "official FFmpeg-Builds release page..."
        ),
        "ffmpeg_cancel": "Cancel",
        "ffmpeg_done_title": "ffmpeg Installed",
        "ffmpeg_done_msg": "FFmpeg (GPL v3) has been installed successfully to:\n{dir}",
        "ffmpeg_invalid_files": "Downloaded files were not recognized as valid executables.",
        "ffmpeg_fail_title": "ffmpeg Download Failed",
        "ffmpeg_fail_msg": (
            "Could not download or install ffmpeg automatically.\n"
            "Reason: {err}\n\n"
            "Please download it manually from:\n"
            "https://github.com/BtbN/FFmpeg-Builds/releases\n"
            "and place ffmpeg/ffprobe in:\n{dir}"
        ),
    },
    "ko": {
        "window_title": "DepthLive3D 런처",
        "conv_title": "컨버전 3D",
        "conv_desc": "영상 파일을 3D로 변환하고\n고급 옵션으로 인코딩합니다.",
        "conv_btn": "컨버전 3D 실행",
        "live_title": "라이브 3D",
        "live_desc": "실시간 화면을 3D로 변환하여\n오버레이로 렌더링합니다.",
        "live_btn": "라이브 3D 실행",
        "sync_title": "소리 싱크",
        "sync_desc": "영상의 오디오를 밀거나 당겨서\n싱크를 맞춥니다.",
        "sync_btn": "소리 싱크 실행",
        "language_label": "언어:",
        "launch_error_title": "실행 오류",
        "launch_error_conv": "컨버전 3D를 실행하지 못했습니다.\n{cmd}",
        "launch_error_live": "라이브 3D를 실행하지 못했습니다.\n{cmd}",
        "fatal_error_title": "{module} 실행 실패",
        # ---- 소리 싱크 창 ----
        "sync_window_title": "소리 싱크",
        "sync_path_group": " 경로 설정 ",
        "sync_input_label": "입력 영상:",
        "sync_output_label": "출력 폴더:",
        "sync_browse": "찾아보기",
        "sync_opt_group": " 오디오 딜레이 설정 ",
        "sync_delay_label": "딜레이 (ms):",
        "sync_delay_hint": "(숫자만 입력하면 +로 처리됩니다. -를 입력할 때만 -가 됩니다. 예: 350, -360, 0)",
        "sync_start": " 시작 ",
        "sync_stop": " 정지 ",
        "sync_status_idle": "상태: 대기 중",
        "sync_status_running": "상태: 처리 중...",
        "sync_status_done": "상태: 완료",
        "sync_status_stopped": "상태: 정지됨",
        "sync_status_error": "상태: 오류",
        "sync_warn_title": "경고",
        "sync_warn_input": "올바른 입력 영상 파일을 선택해주세요.",
        "sync_err_title": "오류",
        "sync_err_delay": "잘못된 딜레이 값입니다. ms 단위 정수로 입력해주세요 (예: 350, -360, 0).",
        "sync_err_ffmpeg": "ffmpeg 실행 파일을 찾을 수 없습니다.\n프로그램 폴더에 ffmpeg를 넣거나 PATH에 등록해주세요.",
        "sync_log_start": "소리 싱크를 시작합니다...",
        "sync_log_input": "입력: {name}",
        "sync_log_delay": "딜레이: {ms} ms -> 출력: {name}",
        "sync_log_done": "완료. 저장 위치: {name}",
        "sync_done_title": "소리 싱크 완료",
        "sync_done_msg": "소리 싱크가 성공적으로 완료되었습니다.\n저장 위치: {name}",
        "sync_log_stopped": "사용자에 의해 정지되었습니다.",
        "sync_log_error": "오류: {err}",
        # ---- ffmpeg 자동 다운로드 ----
        "ffmpeg_license_title": "ffmpeg를 찾을 수 없음",
        "ffmpeg_license_msg": (
            "프로그램 폴더와 시스템 PATH에서 ffmpeg를 찾을 수 없습니다.\n\n"
            "DepthLive3D는 공식 FFmpeg-Builds 릴리스 페이지\n"
            "(https://github.com/BtbN/FFmpeg-Builds)에서 최신 FFmpeg\n"
            "(GPL v3 라이센스, https://ffmpeg.org)를 자동으로 다운로드하여\n"
            "프로그램 폴더에 설치할 수 있습니다.\n\n"
            "지금 ffmpeg를 다운로드하여 설치하시겠습니까?"
        ),
        "ffmpeg_missing_title": "ffmpeg 필요",
        "ffmpeg_missing_msg": (
            "이 기능을 사용하려면 ffmpeg가 필요합니다.\n"
            "ffmpeg/ffprobe를 직접 다운로드하여 아래 폴더에 넣어주세요:\n{dir}"
        ),
        "ffmpeg_downloading_title": "ffmpeg 다운로드 중",
        "ffmpeg_downloading_msg": (
            "공식 FFmpeg-Builds 릴리스 페이지에서\n"
            "최신 ffmpeg 빌드(GPL v3)를 다운로드하는 중입니다..."
        ),
        "ffmpeg_cancel": "취소",
        "ffmpeg_done_title": "ffmpeg 설치 완료",
        "ffmpeg_done_msg": "FFmpeg(GPL v3)가 아래 위치에 성공적으로 설치되었습니다:\n{dir}",
        "ffmpeg_invalid_files": "다운로드된 파일이 올바른 실행 파일로 확인되지 않았습니다.",
        "ffmpeg_fail_title": "ffmpeg 다운로드 실패",
        "ffmpeg_fail_msg": (
            "ffmpeg를 자동으로 다운로드하거나 설치하지 못했습니다.\n"
            "사유: {err}\n\n"
            "아래 주소에서 직접 다운로드해주세요:\n"
            "https://github.com/BtbN/FFmpeg-Builds/releases\n"
            "그 후 다음 폴더에 넣어주세요:\n{dir}"
        ),
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


class AudioSyncWorker(QThread):
    """ffmpeg를 이용해 오디오 딜레이 작업을 백그라운드에서 실행하는 워커 스레드"""
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)  # (성공 여부, 결과 메시지 / 에러 메시지)

    def __init__(self, cmd, out_file):
        super().__init__()
        self.cmd = cmd
        self.out_file = out_file
        self._process = None
        self._stopped = False

    def run(self):
        si = None
        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        try:
            self._process = subprocess.Popen(
                self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                startupinfo=si, bufsize=1
            )

            for line in self._process.stdout:
                line = line.strip()
                if line:
                    self.log_signal.emit(line)

            self._process.wait()

            if self._stopped:
                self.finished_signal.emit(False, "stopped")
            elif self._process.returncode == 0:
                self.finished_signal.emit(True, self.out_file)
            else:
                self.finished_signal.emit(False, f"ffmpeg exit code {self._process.returncode}")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

    def stop(self):
        self._stopped = True
        if self._process and self._process.poll() is None:
            try:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(self._process.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                else:
                    self._process.terminate()
            except Exception:
                pass


class AudioSyncWindow(QMainWindow):
    """영상 오디오 딜레이(싱크) 조정 창.
    +값은 숫자만 입력해도 자동으로 +로 처리되고, -를 입력할 때만 음수로 처리됩니다."""

    def __init__(self, lang: str):
        super().__init__()
        self.current_lang = lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
        self.worker = None

        # Conversion 3D 창과 동일한 크기
        self.resize(880, 600)
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f2f5; }
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: bold;
                color: #1f2937;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLineEdit {
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 4px 6px;
                background-color: #ffffff;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 6px;
                font-weight: bold;
                padding: 6px 14px;
            }
            QPushButton:hover { background-color: #2563eb; }
            QPushButton:pressed { background-color: #1d4ed8; }
            QPushButton:disabled { background-color: #9ca3af; }
            QTextEdit {
                background-color: #111827;
                color: #22c55e;
                border-radius: 6px;
                font-family: Consolas, monospace;
            }
            QLabel[class="hint"] { color: #6b7280; }
            QLabel[class="status"] { color: #1d4ed8; font-weight: bold; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(15, 15, 15, 15)
        outer.setSpacing(10)

        # ---- 경로 설정 ----
        self.path_group = QGroupBox()
        path_grid = QGridLayout(self.path_group)

        self.lbl_input = QLabel()
        self.edt_input = QLineEdit()
        self.btn_browse_input = QPushButton()
        self.btn_browse_input.clicked.connect(self.browse_input)
        path_grid.addWidget(self.lbl_input, 0, 0)
        path_grid.addWidget(self.edt_input, 0, 1)
        path_grid.addWidget(self.btn_browse_input, 0, 2)

        self.lbl_output = QLabel()
        self.edt_output = QLineEdit()
        self.btn_browse_output = QPushButton()
        self.btn_browse_output.clicked.connect(self.browse_output)
        path_grid.addWidget(self.lbl_output, 1, 0)
        path_grid.addWidget(self.edt_output, 1, 1)
        path_grid.addWidget(self.btn_browse_output, 1, 2)

        path_grid.setColumnStretch(1, 1)
        outer.addWidget(self.path_group)

        # ---- 딜레이 설정 ----
        self.opt_group = QGroupBox()
        opt_layout = QHBoxLayout(self.opt_group)

        self.lbl_delay = QLabel()
        self.edt_delay = QLineEdit()
        self.edt_delay.setText("0")
        self.edt_delay.setFixedWidth(100)
        self.lbl_delay_hint = QLabel()
        self.lbl_delay_hint.setProperty("class", "hint")

        opt_layout.addWidget(self.lbl_delay)
        opt_layout.addWidget(self.edt_delay)
        opt_layout.addWidget(self.lbl_delay_hint)
        opt_layout.addStretch()
        outer.addWidget(self.opt_group)

        # ---- 상태 & 버튼 ----
        status_bar = QHBoxLayout()
        self.lbl_status = QLabel()
        self.lbl_status.setProperty("class", "status")
        status_bar.addWidget(self.lbl_status)
        status_bar.addStretch()
        outer.addLayout(status_bar)

        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        self.btn_start = QPushButton()
        self.btn_start.clicked.connect(self.start_conversion)
        self.btn_stop = QPushButton()
        self.btn_stop.clicked.connect(self.stop_conversion)
        self.btn_stop.setEnabled(False)
        btn_bar.addWidget(self.btn_start)
        btn_bar.addWidget(self.btn_stop)
        btn_bar.addStretch()
        outer.addLayout(btn_bar)

        # ---- 로그 ----
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        outer.addWidget(self.log_widget, 1)

        self.apply_language(self.current_lang)

    def tr_text(self, key):
        return TRANSLATIONS.get(self.current_lang, TRANSLATIONS[DEFAULT_LANGUAGE]).get(key, key)

    def apply_language(self, lang: str):
        if lang not in SUPPORTED_LANGUAGES:
            lang = DEFAULT_LANGUAGE
        self.current_lang = lang

        self.setWindowTitle(self.tr_text("sync_window_title"))
        self.path_group.setTitle(self.tr_text("sync_path_group"))
        self.lbl_input.setText(self.tr_text("sync_input_label"))
        self.lbl_output.setText(self.tr_text("sync_output_label"))
        self.btn_browse_input.setText(self.tr_text("sync_browse"))
        self.btn_browse_output.setText(self.tr_text("sync_browse"))
        self.opt_group.setTitle(self.tr_text("sync_opt_group"))
        self.lbl_delay.setText(self.tr_text("sync_delay_label"))
        self.lbl_delay_hint.setText(self.tr_text("sync_delay_hint"))
        self.btn_start.setText(self.tr_text("sync_start"))
        self.btn_stop.setText(self.tr_text("sync_stop"))
        self.lbl_status.setText(self.tr_text("sync_status_idle"))

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_widget.append(f"[{ts}] {msg}")

    def browse_input(self):
        f, _ = QFileDialog.getOpenFileName(
            self, self.tr_text("sync_input_label"), "",
            "Video files (*.mp4 *.mkv *.avi *.mov *.wmv *.flv);;All files (*.*)"
        )
        if f:
            self.edt_input.setText(f)
            if not self.edt_output.text().strip():
                self.edt_output.setText(os.path.dirname(f))

    def browse_output(self):
        d = QFileDialog.getExistingDirectory(self, self.tr_text("sync_output_label"))
        if d:
            self.edt_output.setText(d)

    @staticmethod
    def _parse_delay_ms(text: str) -> int:
        """숫자만 입력하면 +로, '-'를 입력할 때만 -로 처리합니다."""
        t = text.strip().replace(" ", "")
        if t.startswith("+"):
            t = t[1:]
        return int(t)

    @staticmethod
    def _get_unique_filepath(out_dir, in_path):
        base_name = os.path.basename(in_path)
        name, ext = os.path.splitext(base_name)
        target = os.path.join(out_dir, f"{name}{ext}")
        if not os.path.exists(target):
            return target
        counter = 1
        while True:
            target = os.path.join(out_dir, f"{name}_({counter}){ext}")
            if not os.path.exists(target):
                return target
            counter += 1

    def start_conversion(self):
        in_file = self.edt_input.text().strip()
        if not in_file or not os.path.exists(in_file):
            QMessageBox.warning(self, self.tr_text("sync_warn_title"), self.tr_text("sync_warn_input"))
            return

        out_dir = self.edt_output.text().strip()
        if not out_dir:
            out_dir = os.path.dirname(in_file)
            self.edt_output.setText(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        try:
            ms_val = self._parse_delay_ms(self.edt_delay.text())
        except ValueError:
            QMessageBox.critical(self, self.tr_text("sync_err_title"), self.tr_text("sync_err_delay"))
            return

        if not ensure_ffmpeg_available(self, self.current_lang):
            return
        ffmpeg = FFMPEG_EXE

        out_file = self._get_unique_filepath(out_dir, in_file)

        if ms_val > 0:
            cmd = [
                ffmpeg, "-y", "-i", in_file,
                "-c:v", "copy",
                "-af", f"adelay={ms_val}:all=1",
                "-c:a", "aac", "-b:a", "192k",
                out_file
            ]
        elif ms_val < 0:
            sec_val = abs(ms_val) / 1000.0
            cmd = [
                ffmpeg, "-y", "-i", in_file,
                "-c:v", "copy",
                "-af", f"atrim=start={sec_val:.3f},asetpts=PTS-STARTPTS",
                "-c:a", "aac", "-b:a", "192k",
                out_file
            ]
        else:
            cmd = [ffmpeg, "-y", "-i", in_file, "-c", "copy", out_file]

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText(self.tr_text("sync_status_running"))
        self.log(self.tr_text("sync_log_start"))
        self.log(self.tr_text("sync_log_input").format(name=os.path.basename(in_file)))
        self.log(self.tr_text("sync_log_delay").format(ms=ms_val, name=os.path.basename(out_file)))

        self.worker = AudioSyncWorker(cmd, out_file)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.start()

    def stop_conversion(self):
        if self.worker:
            self.worker.stop()

    def on_worker_finished(self, success: bool, message: str):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        if success:
            self.lbl_status.setText(self.tr_text("sync_status_done"))
            self.log(self.tr_text("sync_log_done").format(name=message))
            QMessageBox.information(
                self,
                self.tr_text("sync_done_title"),
                self.tr_text("sync_done_msg").format(name=message),
            )
        elif message == "stopped":
            self.lbl_status.setText(self.tr_text("sync_status_stopped"))
            self.log(self.tr_text("sync_log_stopped"))
        else:
            self.lbl_status.setText(self.tr_text("sync_status_error"))
            self.log(self.tr_text("sync_log_error").format(err=message))

        self.worker = None

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        event.accept()


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_lang = load_language()
        self.audio_sync_window = None

        self.resize(880, 340)
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

        # ---- Audio Sync 카드 (라이브 3D 옆) ----
        self.sync_frame = QFrame()
        sync_layout = QVBoxLayout(self.sync_frame)
        sync_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_sync_title = QLabel()
        self.lbl_sync_title.setProperty("class", "title")
        self.lbl_sync_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_sync_title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))

        self.lbl_sync_desc = QLabel()
        self.lbl_sync_desc.setProperty("class", "desc")
        self.lbl_sync_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_sync_desc.setFont(QFont("Segoe UI", 10))

        self.btn_sync = QPushButton()
        self.btn_sync.setMinimumHeight(45)
        self.btn_sync.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sync.clicked.connect(self.launch_audio_sync)

        sync_layout.addWidget(self.lbl_sync_title)
        sync_layout.addSpacing(10)
        sync_layout.addWidget(self.lbl_sync_desc)
        sync_layout.addSpacing(20)
        sync_layout.addWidget(self.btn_sync)

        main_layout.addWidget(self.left_frame)
        main_layout.addWidget(self.right_frame)
        main_layout.addWidget(self.sync_frame)

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
        self.lbl_sync_title.setText(self.tr_text("sync_title"))
        self.lbl_sync_desc.setText(self.tr_text("sync_desc"))
        self.btn_sync.setText(self.tr_text("sync_btn"))
        self.lbl_language.setText(self.tr_text("language_label"))

        if self.audio_sync_window is not None:
            self.audio_sync_window.apply_language(lang)

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

    def launch_audio_sync(self):
        """소리 싱크 창은 별도 프로세스로 실행하지 않고,
        현재 실행 중인 DepthLive3D 프로세스 안에서 바로 엽니다 (추가 임포트 없음)."""
        if self.audio_sync_window is None:
            self.audio_sync_window = AudioSyncWindow(self.current_lang)
        else:
            self.audio_sync_window.apply_language(self.current_lang)
        self.audio_sync_window.show()
        self.audio_sync_window.raise_()
        self.audio_sync_window.activateWindow()
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
