# -*- coding: utf-8 -*-


from __future__ import annotations

import os

_CPU_COUNT = os.cpu_count() or 4
_RESERVED_THREADS = 3
_CPU_HEADROOM_RATIO = 0.1
_CPU_THREADS = max(2, min(int((_CPU_COUNT - _RESERVED_THREADS) * _CPU_HEADROOM_RATIO), 12))

for _env_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_key, str(_CPU_THREADS))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import time
import uuid
import queue
import shutil
import tempfile
import subprocess
import threading
import sys
import gc
import zipfile
import urllib.request
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

cv2.setNumThreads(_CPU_THREADS)

torch.set_num_threads(_CPU_THREADS)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    from PySide6.QtCore import Qt, QThread, Signal as pyqtSignal, QTimer, QTime, QEventLoop
    from PySide6.QtGui import QFont, QFontMetrics, QPixmap, QPainter, QColor, QIntValidator
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QComboBox, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
        QCheckBox, QGroupBox, QScrollArea, QTextEdit, QFileDialog, QProgressBar, QMessageBox, QTimeEdit,
        QProgressDialog, QSplashScreen
    )
except ImportError:
    print("[Error] PySide6 is required: pip install PySide6")
    sys.exit(1)

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    APP_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent
    APP_DIR = BASE_DIR

ZIPDEPTH_ROOT = BASE_DIR / "ZipDepth"
sys.path.insert(0, str(ZIPDEPTH_ROOT))
sys.path.insert(0, str(BASE_DIR))

# Local install of Video-Depth-Anything (streaming), same path as live3d.py.
VDA_ROOT = BASE_DIR / "Video-Depth-Anything"
if VDA_ROOT.exists():
    sys.path.insert(0, str(VDA_ROOT))

def get_executable(name: str) -> str:
    ext = ".exe" if os.name == "nt" else ""
    local = BASE_DIR / (name + ext)
    if local.is_file():
        return str(local)
    return shutil.which(name) or name

FFMPEG_EXE = get_executable("ffmpeg")
FFPROBE_EXE = get_executable("ffprobe")

FFMPEG_MISSING = (FFMPEG_EXE == "ffmpeg") or (FFPROBE_EXE == "ffprobe")

FFMPEG_BUILDS_API_URL = "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest"


def _ffmpeg_asset_name_for_platform() -> str | None:
    if os.name == "nt":
        return "ffmpeg-master-latest-win64-gpl.zip"
    if sys.platform.startswith("linux"):
        return "ffmpeg-master-latest-linux64-gpl.zip"
    return None


def fetch_latest_ffmpeg_asset_url() -> tuple[str, str] | None:
    asset_name = _ffmpeg_asset_name_for_platform()
    if asset_name is None:
        return None
    req = urllib.request.Request(
        FFMPEG_BUILDS_API_URL,
        headers={"User-Agent": "conversion3d-ffmpeg-installer", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    for asset in data.get("assets", []):
        if asset.get("name") == asset_name:
            return asset_name, asset.get("browser_download_url")
    return None


def download_and_install_ffmpeg(progress_cb=None, cancel_cb=None) -> tuple[bool, str]:
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

        ffmpeg_dst = BASE_DIR / f"ffmpeg{ext}"
        ffprobe_dst = BASE_DIR / f"ffprobe{ext}"
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
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(bool, str)

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


def ensure_ffmpeg_available(parent=None) -> bool:
    global FFMPEG_EXE, FFPROBE_EXE, FFMPEG_MISSING
    if not FFMPEG_MISSING:
        return True

    if parent is not None and hasattr(parent, "_append_log"):
        parent._append_log(
            "ffmpeg not found. Downloading FFmpeg (GPL v3, https://ffmpeg.org) "
            "from the official FFmpeg-Builds release..."
        )

    dlg = QProgressDialog(
        "ffmpeg was not found. Downloading the latest build (GPL v3) from the official FFmpeg-Builds release page...",
        "Cancel", 0, 100, parent
    )
    dlg.setWindowTitle("Downloading ffmpeg")
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
            if parent is not None and hasattr(parent, "_append_log"):
                parent._append_log(f"FFmpeg (GPL v3) installed to: {BASE_DIR}")
            return True
        result["err"] = result["err"] or "Downloaded files were not recognized as valid executables."

    QMessageBox.critical(
        parent,
        "ffmpeg Not Found",
        "Could not download or install ffmpeg automatically.\n"
        f"Reason: {result['err']}\n\n"
        "Please download it manually from:\n"
        "https://github.com/BtbN/FFmpeg-Builds/releases\n"
        f"and place ffmpeg/ffprobe in:\n{BASE_DIR}"
    )
    return False

_HAS_DILATION = True

def clamp_ema_value(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.00
    if v < 0.0:
        return 0.00
    if v > 0.99:
        return 0.99
    return v


def compute_auto_edge_ema(divergence: float, depth_model: str | None = None) -> tuple[int, float]:
    """Divergence -> (edge-fix level, EMA decay) auto-mode curve.
    Mirrors live3d.py's compute_auto_edge_ema(): VDA-S uses its own curve,
    all other (ZipDepth-style) models use the shared curve below."""
    div = float(divergence)
    if depth_model in ("vda_s", "vda_s_metric"):
        if div <= 0.5 + 1e-9:
            edge, ema = 1, 0.00
        elif div <= 1.0 + 1e-9:
            edge, ema = 1, 0.00
        elif div <= 1.5 + 1e-9:
            edge, ema = 2, 0.00
        elif div <= 2.0 + 1e-9:
            edge, ema = 3, 0.00
        elif div <= 2.5 + 1e-9:
            edge, ema = 4, 0.00
        elif div <= 3.0 + 1e-9:
            edge, ema = 5, 0.00
        else:
            edge, ema = 5, 0.30
        edge = max(0, min(5, edge))
        return edge, clamp_ema_value(ema)

    if div <= 0.5 + 1e-9:
        edge, ema = 1, 0.00
    elif div <= 1.0 + 1e-9:
        edge, ema = 1, 0.30
    elif div <= 1.5 + 1e-9:
        edge, ema = 2, 0.40
    elif div <= 2.0 + 1e-9:
        edge, ema = 2, 0.50
    elif div <= 2.5 + 1e-9:
        edge, ema = 3, 0.50
    elif div <= 3.0 + 1e-9:
        edge, ema = 3, 0.60
    else:
        edge, ema = 3, 0.60
    edge = max(0, min(5, edge))
    return edge, clamp_ema_value(ema)


def edge_dilation_parse(edge_dilation):
    if isinstance(edge_dilation, (list, tuple)):
        if len(edge_dilation) == 0:
            x = y = 0
        elif len(edge_dilation) == 1:
            x = y = edge_dilation[0]
        else:
            x = edge_dilation[0]
            y = edge_dilation[1]
    elif isinstance(edge_dilation, int):
        x = y = edge_dilation
    elif edge_dilation is None:
        x = y = 0
    else:
        raise ValueError(f"Unsupported edge_dilation type {type(edge_dilation)}. "
                         "Supported types: int, list, tuple.")
    return x, y


def edge_dilation_is_enabled(edge_dilation):
    x, y = edge_dilation_parse(edge_dilation)
    return x != 0 or y != 0


# Edge-fix level -> (x, y) mapping for non-VDA models (e.g. ZipDepth).
# Y is fixed at 3 from level 3 onward. Matches live3d.py.
# level 1: x=3,y=1 / level 2: x=4,y=2 / level 3: x=5,y=3 / level 4: x=6,y=3 / level 5: x=7,y=3
_EDGE_FIX_LEVEL_MAP = {
    0: (0, 0),
    1: (3, 1),
    2: (4, 2),
    3: (5, 3),
    4: (6, 3),
    5: (7, 3),
}

# VDA-specific edge-fix level -> (x, y) mapping. Matches live3d.py.
# level 1: x=4,y=0 / level 2: x=5,y=1 / level 3: x=6,y=2 / level 4: x=8,y=2 / level 5: x=9,y=2
_EDGE_FIX_LEVEL_MAP_VDA = {
    0: (0, 0),
    1: (4, 0),
    2: (5, 1),
    3: (6, 2),
    4: (8, 2),
    5: (9, 2),
}


def edge_fix_level_to_xy(level, depth_model=None):
    try:
        lvl = int(level)
    except (TypeError, ValueError):
        return level
    level_map = _EDGE_FIX_LEVEL_MAP_VDA if depth_model in ("vda_s", "vda_s_metric") else _EDGE_FIX_LEVEL_MAP
    if lvl in level_map:
        return level_map[lvl]
    # Fallback for out-of-range levels: extrapolate using the level 4->5
    # step for VDA (x+1 per level beyond 5, y held at its level-5 value).
    # For non-VDA models, y stays locked at 3 (matching level 3+ above).
    if lvl > 5:
        if depth_model in ("vda_s", "vda_s_metric"):
            return (lvl + 3, 2)
        return (lvl + 2, 3)
    return (0, 0)


def _dilation_dilate(mask, kernel_size=3):
    if isinstance(kernel_size, (list, tuple)):
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        pad = kernel_size // 2
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)


def _dilation_erode(mask, kernel_size=3):
    if isinstance(kernel_size, (list, tuple)):
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        pad = kernel_size // 2
    return -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)


def _dilation_closing(mask, kernel_size=3, n_iter=2):
    for _ in range(n_iter):
        mask = _dilation_dilate(mask, kernel_size=kernel_size)
    for _ in range(n_iter):
        mask = _dilation_erode(mask, kernel_size=kernel_size)
    return mask


def _dilation_edge_weight(x):
    assert x.ndim == 4
    orig_dtype = x.dtype
    x32 = x.float()

    max_v = F.max_pool2d(x32, kernel_size=3, stride=1, padding=1)
    min_v = F.max_pool2d(x32.neg(), kernel_size=3, stride=1, padding=1).neg()
    range_v = max_v - min_v
    range_c = range_v - range_v.mean(dim=[1, 2, 3], keepdim=True)
    range_s = range_c.pow(2).mean(dim=[1, 2, 3], keepdim=True).sqrt()
    w = (range_c / (range_s + 1e-6)).clamp(-3, 3)
    w_min, w_max = w.amin(dim=[1, 2, 3], keepdim=True), w.amax(dim=[1, 2, 3], keepdim=True)
    w = (w - w_min) / ((w_max - w_min) + 1e-6)
    w = w * 1.2

    return w.to(orig_dtype)


@torch.inference_mode()
def _dilation_dilate_edge(
    x, n,
    small_kernel: int = 3,
    big_kernel: int = 9,
    kernel_step: int = 2,
    pre_closing_iter: int = 1,
    freeze_weight: bool = True,
    refresh_every: int = 2,
    weight_gain: float = 1.3,
):
    x_iter, y_iter = edge_dilation_parse(n)
    xy_iter = min(x_iter, y_iter)
    x_iter = x_iter - xy_iter
    y_iter = y_iter - xy_iter

    if xy_iter + x_iter + y_iter <= 0:
        return x

    orig_dtype = x.dtype
    x = x.float()

    if pre_closing_iter > 0:
        x = _dilation_closing(x, kernel_size=small_kernel, n_iter=pre_closing_iter)

    w = (_dilation_edge_weight(x) * weight_gain).clamp(0, 1) if freeze_weight else None

    for i in range(xy_iter):
        k = min(big_kernel, small_kernel + i * kernel_step)
        w_cur = w if freeze_weight else (_dilation_edge_weight(x) * weight_gain).clamp(0, 1)
        x2 = x * 0.9
        x2 = _dilation_dilate(x2, kernel_size=(k, k))
        x = (x * (1 - w_cur)) + (x2 * w_cur)
        if freeze_weight:
            w = _dilation_dilate(w, kernel_size=(k, k)).clamp(0, 1)
            if refresh_every > 0 and (i + 1) % refresh_every == 0:
                w_fresh = (_dilation_edge_weight(x) * weight_gain).clamp(0, 1)
                w = torch.maximum(w, w_fresh)

    for i in range(y_iter):
        w_cur = w if freeze_weight else (_dilation_edge_weight(x) * weight_gain).clamp(0, 1)
        x2 = x
        x2 = _dilation_dilate(x2, kernel_size=(3, 1))
        x = (x * (1 - w_cur)) + (x2 * w_cur)
        if freeze_weight:
            w = _dilation_dilate(w, kernel_size=(3, 1)).clamp(0, 1)

    for i in range(x_iter):
        w_cur = w if freeze_weight else (_dilation_edge_weight(x) * weight_gain).clamp(0, 1)
        x2 = x
        x2 = _dilation_dilate(x2, kernel_size=(1, 3))
        x = (x * (1 - w_cur)) + (x2 * w_cur)
        if freeze_weight:
            w = _dilation_dilate(w, kernel_size=(1, 3)).clamp(0, 1)

    return x.to(orig_dtype)


_HAS_IW3_EMA = True

def _depth_scaler_robust_minmax(frame: torch.Tensor, q_lo: float = 0.01, q_hi: float = 0.99):
    flat = frame.detach().float().reshape(-1)
    n = flat.numel()
    if n == 0:
        device = frame.device
        return (
            torch.tensor(0.0, device=device),
            torch.tensor(1.0, device=device),
        )

    sample = flat[::8] if n > 100_000 else flat

    try:
        lo = torch.quantile(sample, q_lo)
        hi = torch.quantile(sample, q_hi)
    except Exception:
        lo = sample.amin()
        hi = sample.amax()

    if (hi - lo) < 1e-6:
        lo = sample.amin()
        hi = sample.amax()
        if (hi - lo) < 1e-6:
            hi = lo + 1e-3

    return lo, hi


class EMAMinMaxScaler:
    


    def __init__(self, decay: float = 0.0, buffer_size: int = 1, mode: str = "minmax"):
        self.decay = float(max(0.0, min(0.99, decay)))
        self.buffer_size = 1
        self.mode = mode

        self.min_value = None
        self.max_value = None
        self.last_motion_score = 0.0
        self.last_effective_decay = self.decay

        self._prev_norm = None
        self._frame_count = 0

        self.motion_ema = 0.0
        self.motion_sensitivity = 2.5
        self.local_motion_sensitivity = 2.0
        self.motion_map_ema = None

        self._prev_depth_small = None
        self._motion_h = 90
        self._motion_w = 160

    def reset(self, decay=None, buffer_size=None, **kwargs):
        if decay is not None:
            self.decay = float(max(0.0, min(0.99, decay)))
        self.min_value = None
        self.max_value = None
        self._prev_norm = None
        self._frame_count = 0
        self.motion_ema = 0.0
        self.motion_map_ema = None
        self._prev_depth_small = None
        self.last_motion_score = 0.0
        self.last_effective_decay = self.decay

    def set_decay(self, decay):
        self.decay = float(max(0.0, min(0.99, decay)))
        self.last_effective_decay = self.decay

    def _is_adaptive_active(self) -> bool:
        return self.decay > 0.0

    def __call__(self, frame, return_minmax=False):
        return self.update(frame, return_minmax=return_minmax)

    def update(self, frame, return_minmax=False):
        if frame is None:
            if return_minmax:
                return None, None, None
            return None

        decay = float(self.decay)
        if decay < 0.0:
            decay = 0.0
        elif decay > 0.99:
            decay = 0.99

        motion_score, motion_map_t = self._compute_motion(frame)

        self.motion_ema = self.motion_ema * 0.7 + motion_score * 0.3
        self.last_motion_score = float(self.motion_ema)

        if decay > 0.0:
            motion_factor = max(0.0, 1.0 - self.motion_ema * self.motion_sensitivity)
        else:
            motion_factor = 1.0
        adaptive_decay = decay * motion_factor
        self.last_effective_decay = float(adaptive_decay)

        cur_min, cur_max = _depth_scaler_robust_minmax(frame)
        self.min_value = cur_min
        self.max_value = cur_max

        scale = (self.max_value - self.min_value).clamp(min=1e-8)
        norm = ((frame.float() - self.min_value) / scale).clamp(0.0, 1.0)

        if decay > 0.0:
            if (
                self._prev_norm is None
                or self._prev_norm.shape != norm.shape
                or self._prev_norm.device != norm.device
                or self._prev_norm.dtype != norm.dtype
            ):
                result = norm
                self._prev_norm = result.detach().clone()
            else:
                if motion_map_t is not None:
                    if (
                        self.motion_map_ema is None
                        or self.motion_map_ema.shape != motion_map_t.shape
                        or self.motion_map_ema.device != motion_map_t.device
                    ):
                        self.motion_map_ema = motion_map_t.clone()
                    else:
                        self.motion_map_ema.mul_(0.7).add_(motion_map_t, alpha=0.3)

                    motion_map_denoised = F.avg_pool2d(
                        self.motion_map_ema, kernel_size=3, stride=1, padding=1
                    )
                    motion_map_dilated = F.max_pool2d(
                        motion_map_denoised, kernel_size=5, stride=1, padding=2
                    )
                    local_factor = (
                        1.0 - motion_map_dilated * self.local_motion_sensitivity
                    ).clamp(0.0, 1.0)
                    local_decay_map = decay * local_factor

                    local_decay_full = F.interpolate(
                        local_decay_map,
                        size=norm.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    result = norm * (1.0 - local_decay_full) + self._prev_norm * local_decay_full
                    self._prev_norm = result.detach().clone()
                else:
                    result = norm * (1.0 - adaptive_decay) + self._prev_norm * adaptive_decay
                    self._prev_norm = result.detach().clone()
        else:
            result = norm
            self._prev_norm = None
            self.motion_map_ema = None

        self._frame_count += 1

        if return_minmax:
            return result, self.min_value, self.max_value
        return result

    def _compute_motion(self, frame: torch.Tensor):
        t = frame.detach().float()
        if t.dim() == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:
            if t.shape[0] in (1, 3):
                t = t[:1].unsqueeze(0)
            else:
                t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 4:
            t = t[:, :1]
        else:
            return 0.0, None

        small = F.interpolate(
            t,
            size=(self._motion_h, self._motion_w),
            mode="bilinear",
            align_corners=False,
        )

        if (
            self._prev_depth_small is None
            or self._prev_depth_small.shape != small.shape
            or self._prev_depth_small.device != small.device
        ):
            self._prev_depth_small = small.detach().clone()
            return 0.0, torch.zeros(
                (1, 1, self._motion_h, self._motion_w),
                device=small.device,
                dtype=small.dtype,
            )

        diff = (small - self._prev_depth_small).abs()
        self._prev_depth_small = small.detach().clone()

        d_max = diff.amax().clamp(min=1e-6)
        motion_map = (diff / d_max).clamp(0.0, 1.0)
        motion_score = float(motion_map.mean().clamp(0.0, 1.0))

        if motion_map.dim() == 4 and motion_map.shape[0] > 1:
            motion_map = motion_map[:1]
        if motion_map.shape[1] != 1:
            motion_map = motion_map[:, :1]

        return motion_score, motion_map

    def flush(self, return_minmax=False):
        self.reset()
        return []

CONFIG_FILE = APP_DIR / "DepthConversion3D_config.json"
PRESET_FILE = APP_DIR / "DepthConversion3D.json"
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".ts", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg", ".m2ts"}


def check_is_hdr(file_path: Path) -> bool:
    if not FFPROBE_EXE or FFPROBE_EXE == "ffprobe":
        return False
    try:
        si = None
        if os.name == 'nt':
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0

        cmd = [
            FFPROBE_EXE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=color_transfer",
            "-of", "json", str(file_path)
        ]
        result = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, startupinfo=si)
        data = json.loads(result)
        stream = data.get("streams", [{}])[0]
        
        color_transfer = stream.get("color_transfer", "").lower()
        if color_transfer in ["smpte2084", "arib-std-b67"]:
            return True
        return False
    except Exception:
        return False

def detect_letterbox(file_path: Path, sample_count=50, threshold=16, min_black_ratio=0.92) -> str | None:
    try:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 5:
            cap.release()
            return None

        indices = np.linspace(0, total - 1, min(sample_count, total), dtype=int)
        tops, bottoms = [], []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            h, w = gray.shape

            top = 0
            for y in range(h // 2):
                if np.mean(gray[y, :] < threshold) < min_black_ratio:
                    top = y
                    break

            bottom = h
            for y in range(h - 1, h // 2, -1):
                if np.mean(gray[y, :] < threshold) < min_black_ratio:
                    bottom = y + 1
                    break

            tops.append(top)
            bottoms.append(bottom)

        cap.release()

        if not tops:
            return None

        top = int(np.median(tops))
        bottom = int(np.median(bottoms))
        crop_h = bottom - top

        if crop_h < h * 0.55 or (top < 2 and (h - bottom) < 2):
            return None

        crop_h -= crop_h % 2
        top = max(0, top - (top % 2))

        return f"crop=iw:{crop_h}:0:{top}"
    except Exception:
        return None


def load_zipdepth(input_size=384, device="cuda", fp16=True):
    from zipdepth.inference.predictor import DepthInference
    import urllib.request

    ckpt = ZIPDEPTH_ROOT / "checkpoints" / "zipdepth_base.pth"

    if not ckpt.exists():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/fabiotosi92/ZipDepth/raw/main/checkpoints/zipdepth_base.pth"
        print(f"[Info] ZipDepth checkpoint file not found. Downloading from: {url}")
        urllib.request.urlretrieve(url, ckpt)
        print("[Info] ZipDepth checkpoint downloaded successfully.")

    use_cuda = device == "cuda" and torch.cuda.is_available()
    return DepthInference(
        checkpoint_path=str(ckpt),
        variant="base",
        device="cuda" if use_cuda else "cpu",
        use_half=fp16 and use_cuda,
        use_compile=False,
        input_size=input_size,
        ensure_multiple_of=32,
        warmup_iters=2,
        upsample_unfold=True,
    )

# --- Video Depth Anything (streaming) --------------------------------------
# Local install: https://github.com/DepthAnything/Video-Depth-Anything
# (VDA_ROOT, defined above, points at the folder containing run_streaming.py)

VDA_ENCODER_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

# Official checkpoint download links (see repo README "Pre-trained Models").
VDA_CHECKPOINT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true",
    "vitb": "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth?download=true",
    "vitl": "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth?download=true",
}

# Metric-depth checkpoint download links (Metric-Video-Depth-Anything repo family).
VDA_METRIC_CHECKPOINT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth?download=true",
    "vitb": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Base/resolve/main/metric_video_depth_anything_vitb.pth?download=true",
    "vitl": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth?download=true",
}


def _import_vda_streaming_class():
    """Locate the VideoDepthAnything model class inside the local
    Video-Depth-Anything checkout. Tries the real frame-by-frame streaming
    implementation first, falling back to older/offline-only module layouts."""
    candidates = [
        ("video_depth_anything.video_depth_stream", "VideoDepthAnything"),
        ("video_depth_anything.video_depth", "VideoDepthAnything"),
        ("video_depth_anything.video_depth_streaming", "VideoDepthAnythingStreaming"),
        ("video_depth_anything.video_depth_anything", "VideoDepthAnything"),
    ]
    last_err = None
    for module_name, cls_name in candidates:
        try:
            mod = __import__(module_name, fromlist=[cls_name])
            return getattr(mod, cls_name)
        except Exception as e:
            last_err = e
            continue
    raise ImportError(
        f"[VDA] Could not locate the VideoDepthAnything class under {VDA_ROOT}. "
        f"Check the Video-Depth-Anything installation / update _import_vda_streaming_class(). "
        f"Last import error: {last_err}"
    )


_VDA_STREAMING_METHOD_CANDIDATES = [
    "infer_video_depth_one",
    "infer_video_depth_online",
    "infer_video_depth_streaming_one",
    "infer_one_frame",
    "infer_frame",
    "streaming_infer",
    "infer_online",
]


def _find_vda_streaming_method(model):
    """Different checkouts of the Video-Depth-Anything repo have used
    different names for the frame-by-frame streaming call. Detect whichever
    one is actually present on the loaded model instead of hard-coding it."""
    for name in _VDA_STREAMING_METHOD_CANDIDATES:
        if callable(getattr(model, name, None)):
            return name
    return None


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _vda_forward_infer(model, rgb: np.ndarray, input_size: int, device: str, fp16: bool) -> torch.Tensor:
    """Direct single-frame forward pass through the VDA network (no sliding
    window / temporal cache). Used as the fallback when the local checkout
    doesn't expose a dedicated frame-by-frame streaming method."""
    orig_h, orig_w = rgb.shape[:2]
    scale = input_size / min(orig_h, orig_w)
    new_h = max(14, int(round(orig_h * scale / 14)) * 14)
    new_w = max(14, int(round(orig_w * scale / 14)) * 14)

    img = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # C,H,W

    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, C, H, W)

    if fp16 and device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            depth = model(x)
    else:
        depth = model(x.float())

    depth = depth.float()  # (1, 1, h, w)
    depth = F.interpolate(depth, size=(orig_h, orig_w), mode="bilinear", align_corners=True)
    return depth


class VDAStreamingAdapter:
    """Wraps the official Video-Depth-Anything model so it can be dropped
    in wherever the ZipDepth `predictor` object is used elsewhere in this
    file (see estimate_depth_raw())."""

    def __init__(self, model, device, input_size, fp16=True, streaming_method=None, temporal_smooth=0.35):
        self.model = model
        self.device = device
        self.input_size = input_size
        self.fp16 = fp16
        # Name of the frame-by-frame streaming method on `model`, or None
        # if this checkout doesn't expose one (see infer() fallback below).
        self.streaming_method = streaming_method
        # Only used in the no-streaming-method fallback: a simple per-pixel
        # EMA over consecutive depth maps to reduce frame-to-frame flicker,
        # since that path has no real temporal cache/state (0 = off).
        self.temporal_smooth = float(temporal_smooth) if streaming_method is None else 0.0
        self._prev_depth = None

    def reset_state(self):
        for name in ("reset_state", "reset_cache", "reset"):
            fn = getattr(self.model, name, None)
            if callable(fn):
                try:
                    fn()
                except TypeError:
                    pass
                break
        else:
            if hasattr(self.model, "frame_cache_list"):
                self.model.transform = None
                self.model.frame_id_list = []
                self.model.frame_cache_list = []
                self.model.id = -1
        self._prev_depth = None

    @torch.inference_mode()
    def infer(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.streaming_method is not None:
            fn = getattr(self.model, self.streaming_method)
            depth = fn(rgb, input_size=self.input_size, device=self.device, fp32=not self.fp16)
            if not torch.is_tensor(depth):
                depth = torch.from_numpy(np.asarray(depth))
            depth = depth.to(self.device).float()
            if depth.dim() == 2:
                depth = depth.unsqueeze(0).unsqueeze(0)
            elif depth.dim() == 3:
                depth = depth.unsqueeze(0)
            return depth
        else:
            depth = _vda_forward_infer(self.model, rgb, self.input_size, self.device, self.fp16)
            if self.temporal_smooth > 0:
                if self._prev_depth is not None and self._prev_depth.shape == depth.shape:
                    a = self.temporal_smooth
                    depth = a * self._prev_depth + (1.0 - a) * depth
                self._prev_depth = depth.detach()
            return depth


def load_vda_streaming(encoder="vits", input_size=392, fp16=True, metric=False):
    """Load the Video Depth Anything model (e.g. VDA-S) from the local
    checkout at VDA_ROOT, auto-downloading the checkpoint file into
    VDA_ROOT/checkpoints/ if it isn't there yet. When metric=True, loads the
    metric-depth checkpoint variant instead (e.g. VDA-S Metric streaming)."""
    import urllib.request

    if encoder not in VDA_ENCODER_CONFIGS:
        raise ValueError(f"Unsupported VDA encoder: {encoder}")

    if not VDA_ROOT.exists():
        raise FileNotFoundError(
            f"[VDA] Video-Depth-Anything installation not found at: {VDA_ROOT}\n"
            "Update VDA_ROOT near the top of this script to match your install path."
        )

    ckpt_dir = VDA_ROOT / "checkpoints"
    ckpt_prefix = "metric_video_depth_anything" if metric else "video_depth_anything"
    ckpt = ckpt_dir / f"{ckpt_prefix}_{encoder}.pth"

    if not ckpt.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        url = (VDA_METRIC_CHECKPOINT_URLS if metric else VDA_CHECKPOINT_URLS)[encoder]
        print(f"[Info] VDA-{encoder}{' (metric)' if metric else ''} checkpoint file not found. Downloading from: {url}")
        urllib.request.urlretrieve(url, ckpt)
        print("[Info] VDA checkpoint downloaded successfully.")

    StreamingClass = _import_vda_streaming_class()

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    model = StreamingClass(**VDA_ENCODER_CONFIGS[encoder])
    state_dict = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    isz = int(input_size)
    if isz % 14 != 0:
        isz += (14 - isz % 14)

    streaming_method = _find_vda_streaming_method(model)
    if streaming_method is None:
        public_methods = sorted(
            m for m in dir(model)
            if not m.startswith("_") and callable(getattr(model, m, None))
        )
        print(
            f"[Warning] VDA: no frame-by-frame streaming method found on "
            f"{type(model).__name__} (looked for: {', '.join(_VDA_STREAMING_METHOD_CANDIDATES)}). "
            f"Falling back to direct single-frame forward inference (no cross-frame temporal "
            f"smoothing, but fast enough for real-time). "
            f"Methods available on this checkout: {public_methods}"
        )
    else:
        print(f"[Info] VDA streaming method detected: {streaming_method}")

    return VDAStreamingAdapter(
        model, device=device, input_size=isz, fp16=(fp16 and use_cuda),
        streaming_method=streaming_method,
    )


def estimate_depth_raw(predictor, bgr: np.ndarray) -> torch.Tensor:
    if isinstance(predictor, VDAStreamingAdapter):
        return predictor.infer(bgr)

    image_tensor, _, _ = predictor.image2tensor(bgr)
    with torch.inference_mode():
        depth = predictor.model(image_tensor)
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)
    return depth


def upsample_depth(depth: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if depth.shape[-2:] == (h, w):
        return depth
    return F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)


DILATION_MAX_SHORT_SIDE = 630


def compute_dilation_target_dims(frame_h: int, frame_w: int, cap: int = DILATION_MAX_SHORT_SIDE) -> tuple:
    

    short_side = min(frame_h, frame_w)
    if short_side <= cap or cap <= 0:
        return frame_h, frame_w
    scale = cap / float(short_side)
    dh = max(2, int(round(frame_h * scale)))
    dw = max(2, int(round(frame_w * scale)))
    dh -= dh % 2
    dw -= dw % 2
    return max(2, dh), max(2, dw)


def estimate_depth(predictor, bgr: np.ndarray):
    

    h, w = bgr.shape[:2]
    depth = estimate_depth_raw(predictor, bgr)
    native_h, native_w = int(depth.shape[-2]), int(depth.shape[-1])
    depth_up = upsample_depth(depth, h, w)
    return depth_up, (native_w, native_h)

def depthmap_frame_to_tensor(bgr_frame: np.ndarray, target_w: int, target_h: int, device, dtype) -> torch.Tensor:
    

    if bgr_frame.ndim == 3:
        gray = bgr_frame[:, :, 0]
    else:
        gray = bgr_frame
    t = torch.from_numpy(np.ascontiguousarray(gray)).to(device=device, dtype=dtype)
    t = t.unsqueeze(0).unsqueeze(0).div_(255.0)
    if t.shape[-2:] != (target_h, target_w):
        t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=True)
    return t

def normalize_depth_gpu(depth: torch.Tensor, ema_lo=None, ema_hi=None, decay=0.0):
    flat = depth.float().view(-1)
    sample = flat[::8] if flat.numel() > 100_000 else flat
    lo = torch.quantile(sample, 0.01)
    hi = torch.quantile(sample, 0.99)

    if decay > 0.0 and ema_lo is not None and ema_hi is not None:
        lo = decay * ema_lo + (1.0 - decay) * lo
        hi = decay * ema_hi + (1.0 - decay) * hi

    d = depth.to(dtype=depth.dtype)
    lo = lo.to(dtype=d.dtype)
    hi = hi.to(dtype=d.dtype)
    eps = torch.tensor(1e-8, device=d.device, dtype=d.dtype)
    normalized = (d - lo) / (hi - lo + eps)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    return normalized, lo.detach(), hi.detach()

def dilate_edge(depth: torch.Tensor, edge_val) -> torch.Tensor:
    return _dilation_dilate_edge(depth, edge_val)

_GRID_CACHE = {}

def get_cached_grid_tensors(B: int, H: int, W: int, device: torch.device, dtype: torch.dtype):
    key = (B, H, W, device, dtype)
    if key not in _GRID_CACHE:
        src_index = torch.arange(0, W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)
        mesh_y = torch.linspace(-1, 1, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        _GRID_CACHE[key] = (src_index, mesh_y)
    return _GRID_CACHE[key]

_GAUSSIAN_KERNEL_CACHE = {}

def _get_gaussian_kernel_1d(size: int, device: torch.device, dtype: torch.dtype):
    key = (size, device, dtype)
    if key not in _GAUSSIAN_KERNEL_CACHE:
        sigma = size / 3.0
        coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        _GAUSSIAN_KERNEL_CACHE[key] = g.view(1, 1, 1, size)
    return _GAUSSIAN_KERNEL_CACHE[key]

def smooth_dest_index(dest_index_fix: torch.Tensor, dest_index_raw: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    

    if kernel_size <= 1:
        return dest_index_fix

    mask = dest_index_raw != dest_index_fix
    if not bool(mask.any()):
        return dest_index_fix

    x = dest_index_fix.unsqueeze(1)
    m = mask.unsqueeze(1).float()
    m = (F.max_pool2d(m, kernel_size=(1, 5), stride=1, padding=(0, 2)) > 0)

    kernel = _get_gaussian_kernel_1d(kernel_size, x.device, x.dtype)
    pad = (kernel_size - 1) // 2
    x_padded = F.pad(x, (pad, pad, 0, 0), mode="replicate")
    blurred = F.conv2d(x_padded, kernel)

    result = x.clone()
    result[m] = blurred[m]
    return result.squeeze(1)

def monobw_warp_one(img_t, depth, divergence, convergence, shift_sign=1.0, preserve_screen_border=False):
    B, _, H, W = img_t.shape
    device, dtype = img_t.device, img_t.dtype
    if depth.shape[-2:] != (H, W):
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)

    shift_size = divergence * 0.01 * W * 0.5
    index_shift = (depth[:, 0] * shift_size - shift_size * convergence) * shift_sign

    if preserve_screen_border:
        border_pix = max(1, round(divergence * 0.75 * 0.01 * W))
        if border_pix > 0 and border_pix * 2 < W:
            border_weight_l = torch.linspace(0.0, 1.0, border_pix, dtype=dtype, device=device)
            border_weight_r = torch.linspace(1.0, 0.0, border_pix, dtype=dtype, device=device)
            index_shift[..., :border_pix] *= border_weight_l
            index_shift[..., -border_pix:] *= border_weight_r

    src_index, mesh_y = get_cached_grid_tensors(B, H, W, device, dtype)
    dest_index_raw = src_index + index_shift
    dest_index = torch.cummax(dest_index_raw, dim=-1)[0]

    src_flat = src_index.reshape(B * H, W)
    dest_flat = dest_index.reshape(B * H, W)
    idx = torch.searchsorted(dest_flat.contiguous(), src_flat.contiguous(), right=False)
    idx0 = (idx - 1).clamp(0, W - 1)
    idx1 = idx.clamp(0, W - 1)
    d0 = torch.gather(dest_flat, 1, idx0)
    d1 = torch.gather(dest_flat, 1, idx1)
    s0 = torch.gather(src_flat, 1, idx0)
    s1 = torch.gather(src_flat, 1, idx1)
    denom = (d1 - d0).clamp(min=1e-6)
    w1 = (src_flat - d0) / denom
    index_back = (s0 * (1.0 - w1) + s1 * w1).reshape(B, 1, H, W)

    grid_x = (index_back / max(W - 1, 1)) * 2.0 - 1.0
    grid = torch.cat([grid_x, mesh_y], dim=1).permute(0, 2, 3, 1)
    return F.grid_sample(img_t, grid, mode="bilinear", padding_mode="border", align_corners=True)

def make_stereo(bgr_t, depth_t, divergence, convergence, device, already_normalized=False, preserve_screen_border=False):
    if already_normalized:
        depth_norm = depth_t.to(dtype=bgr_t.dtype)
    else:
        depth_norm = normalize_depth_gpu(depth_t)[0].to(dtype=bgr_t.dtype)

    with torch.inference_mode():
        left_t = monobw_warp_one(bgr_t, depth_norm, divergence, convergence, -1.0, preserve_screen_border)
        right_t = monobw_warp_one(bgr_t, depth_norm, divergence, convergence, +1.0, preserve_screen_border)

    return left_t, right_t

def pack_frame_gpu(left_t: torch.Tensor, right_t: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt == "hsbs":
        h, w = left_t.shape[2], left_t.shape[3]
        left_half = F.interpolate(left_t, size=(h, w // 2), mode="area")
        right_half = F.interpolate(right_t, size=(h, w // 2), mode="area")
        return torch.cat([left_half, right_half], dim=3)
    if fmt == "fsbs":
        return torch.cat([left_t, right_t], dim=3)
    if fmt == "tb":
        h, w = left_t.shape[2], left_t.shape[3]
        left_half = F.interpolate(left_t, size=(h // 2, w), mode="area")
        right_half = F.interpolate(right_t, size=(h // 2, w), mode="area")
        return torch.cat([left_half, right_half], dim=2)
    if fmt == "ftb":
        return torch.cat([left_t, right_t], dim=2)
    if fmt == "anaglyph":
        l, r = left_t, right_t
        out = torch.zeros_like(l)
        out[:, 0:1] = r[:, 0:1]
        out[:, 1:2] = r[:, 1:2]
        out[:, 2:3] = l[:, 2:3]
        return out
    if fmt == "half-anaglyph":
        l, r = left_t, right_t
        gray_l = 0.114 * l[:, 0:1] + 0.587 * l[:, 1:2] + 0.299 * l[:, 2:3]
        out = torch.zeros_like(l)
        out[:, 0:1] = r[:, 0:1]
        out[:, 1:2] = r[:, 1:2]
        out[:, 2:3] = gray_l
        return out
    raise ValueError(f"Unsupported format: {fmt}")

def tensor_to_bgr_bytes(packed_t: torch.Tensor) -> bytes:
    out_t = packed_t[0].clamp(0.0, 1.0).mul(255.0).byte()
    out_t = out_t.permute(1, 2, 0).contiguous()
    return out_t.cpu().numpy().tobytes()

_INVALID_FS_CHARS = set('\\/:*?"<>|') | {chr(c) for c in range(0, 32)}

def safe_stem(path: Path) -> str:
    s = "".join(c for c in path.stem if c not in _INVALID_FS_CHARS)
    s = s.strip().strip(".")
    return (s or "video")[:100]

_VR_PLAYER_FORMAT_NAMES = {
    "hsbs": "Half_SBS",
    "fsbs": "Full_SBS",
    "tb": "Half_TB",
    "ftb": "Full_TB",
}

def vr_player_format_label(fmt: str) -> str:
    return _VR_PLAYER_FORMAT_NAMES.get(fmt, fmt)

def make_unique_output_path(output_dir: Path, stem: str, fmt: str, ext: str) -> Path:
    candidate = output_dir / f"{stem}_{fmt}{ext}"
    if not candidate.exists(): return candidate
    i = 1
    while True:
        candidate = output_dir / f"{stem}_{fmt}_({i}){ext}"
        if not candidate.exists(): return candidate
        i += 1

def ensure_ascii_readable(path: Path) -> tuple[Path, Path | None]:
    try:
        str(path).encode("ascii")
        return path, None
    except UnicodeEncodeError:
        pass
    tmp_path = Path(tempfile.gettempdir()) / f"stereo_src_{uuid.uuid4().hex}{path.suffix.lower()}"
    try:
        os.link(path, tmp_path)
    except OSError:
        shutil.copy2(path, tmp_path)
    return tmp_path, tmp_path

def cleanup_temp_path(tmp_path: Path | None):
    if tmp_path is None: return
    try: tmp_path.unlink(missing_ok=True)
    except Exception: pass

def build_ffmpeg_encode_cmd(
    width: int, height: int, fps: float, video_only_out: Path,
    vcodec: str, preset: str, quality: int, pad_169: bool = False, stereo_format: str = "HSBS",
    threads: int | None = None
) -> list[str]:

    cmd = [
        FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "error",
        "-thread_queue_size", "32",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.6f}",
        "-i", "-",
        "-c:v", vcodec
    ]
    
    if "nvenc" in vcodec:
        cmd.extend([
            "-rc", "vbr", "-preset", preset, "-cq", str(quality),
            "-rc-lookahead", "20",
            "-multipass", "qres",
            "-bufsize", "50M",
            "-surfaces", "8",
            "-spatial-aq", "1", "-temporal-aq", "1", "-aq-strength", "8",
        ])
    else:
        if threads is not None:
            cmd.extend(["-threads", str(threads)])
        cmd.extend(["-preset", preset, "-crf", str(quality)])

    if pad_169:
        fmt_upper = stereo_format.upper()
        if fmt_upper == "FSBS":
            dar = "32/9"
        elif fmt_upper == "FTB":
            dar = "8/9"
        else:
            dar = "16/9"
            
        pad_filter = f"pad=max(iw\\,ih*({dar})):max(ih\\,iw/({dar})):(ow-iw)/2:(oh-ih)/2"
        cmd.extend(["-vf", pad_filter])

    cmd.extend(["-pix_fmt", "yuv420p", str(video_only_out)])
    return cmd

def build_audio_mux_cmd(
    video_only: Path, src_video_for_audio: Path, final_out: Path,
    audio_start_sec: float = None, audio_end_sec: float = None,
) -> list[str]:
    cmd = [FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_only)]
    if audio_start_sec is not None:
        cmd.extend(["-ss", str(audio_start_sec)])
    if audio_end_sec is not None:
        cmd.extend(["-to", str(audio_end_sec)])
    cmd.extend(["-i", str(src_video_for_audio)])
    cmd.extend([
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(final_out)
    ])
    return cmd


class _FrameReaderThread(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture, max_frames, stop_event: threading.Event, out_q: queue.Queue):
        super().__init__(daemon=True)
        self.cap = cap
        self.max_frames = max_frames
        self.stop_event = stop_event
        self.out_q = out_q

    def run(self):
        idx = 0
        try:
            while not self.stop_event.is_set():
                if self.max_frames and idx >= self.max_frames: break
                ok, frame = self.cap.read()
                if not ok: break
                while not self.stop_event.is_set():
                    try:
                        self.out_q.put(frame, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                idx += 1
        finally:
            try: self.out_q.put(None, timeout=1.0)
            except queue.Full: pass


class _PipeWriterThread(threading.Thread):
    


    def __init__(self, stdin_pipe, maxsize: int = 4):
        super().__init__(daemon=True)
        self.stdin_pipe = stdin_pipe
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.error: Exception | None = None
        self._stopped = threading.Event()

    def run(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                try:
                    self.stdin_pipe.write(item)
                except Exception as e:
                    self.error = e
                    break
        finally:
            self._stopped.set()

    def submit(self, data: bytes):
        if self.error is not None:
            raise self.error
        self.queue.put(data)

    def close(self):
        self.queue.put(None)
        self.join(timeout=10.0)
        if self.error is not None:
            raise self.error


class _NvencStallWatchdog(threading.Thread):
    """
    Watches one or more NVENC output files for growth after the encoder starts.
    If none of the watched files have grown past 0 bytes within `stall_seconds`,
    this is treated as a hung NVENC session -- most commonly caused by an
    outdated/incompatible NVIDIA display driver for the ffmpeg build in use --
    and the associated process(es) are killed so the conversion can fail fast
    with a clear message instead of hanging indefinitely.

    Safe by design: it only ever kills processes it was explicitly given, only
    fires once after the full stall window (no premature action), and exits
    quietly the moment `mark_done()` is called (normal completion / user abort)
    or as soon as any watched file shows real progress.
    """

    def __init__(self, worker: "ConvertWorker", targets, procs, stall_seconds: float = 120.0, poll_interval: float = 2.0):
        super().__init__(daemon=True)
        self._worker = worker
        self._targets = [t for t in targets if t is not None]
        self._procs = [p for p in procs if p is not None]
        self._stall_seconds = stall_seconds
        self._poll_interval = poll_interval
        self._done_event = threading.Event()

    def mark_done(self):
        self._done_event.set()

    def run(self):
        if not self._targets:
            return
        elapsed = 0.0
        try:
            while elapsed < self._stall_seconds:
                if self._done_event.wait(timeout=self._poll_interval):
                    return  # conversion already finished / aborted normally
                elapsed += self._poll_interval
                try:
                    if any(t.exists() and t.stat().st_size > 0 for t in self._targets):
                        return  # encoder produced data -> it's alive, stop watching
                except Exception:
                    return

            if self._done_event.is_set():
                return

            self._worker._stall_message = (
                f"NVENC encoding stalled: the output file stayed at 0 bytes for "
                f"{int(self._stall_seconds)} seconds with no progress. This is a common "
                f"symptom of an outdated or incompatible NVIDIA display driver for the "
                f"current ffmpeg NVENC build. Please update your NVIDIA driver, or switch "
                f"the video codec to libx264/libx265, and try again."
            )
            self._worker.log.emit(f"[Error] {self._worker._stall_message}")
            for p in self._procs:
                try:
                    p.kill()
                except Exception:
                    pass
            self._worker._stop.set()
        except Exception:
            # Watchdog must never crash the conversion; just give up silently.
            pass


class ConvertWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(str)
    finished_error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._stop = threading.Event()
        self._temp_files = []
        self._stall_message: str | None = None
        self._last_logged_progress_percent = -1

    def stop(self):
        self._stop.set()

    def run(self):
        p = self.params
        try:
            self._run_convert(p)
        except Exception as e:
            self.finished_error.emit(self._stall_message if self._stall_message else str(e))
        finally:
            for tmp in self._temp_files:
                cleanup_temp_path(tmp)

    def _run_convert(self, p: dict):
        input_path_orig: Path = p["input_path"]
        output_dir: Path = p["output_dir"]
        divergence = p["divergence"]
        convergence = 1.0 - p["convergence"]
        input_size = p["input_size"]
        fmt = p["format"]
        invert_depth = p["invert_depth"]
        fp16 = p["fp16"]
        low_ram = False
        depth_model = p.get("depth_model", "zipdepth") or "zipdepth"
        edge_dilation = p["edge_dilation"]
        edge_dilation = (
            edge_fix_level_to_xy(edge_dilation, depth_model=depth_model)
            if not isinstance(edge_dilation, (list, tuple)) else edge_dilation
        )
        ema_decay = p["ema_decay"]
        preserve_border = p["preserve_border"]
        extract_depthmap = bool(p.get("extract_depthmap", False))
        raw_depthmap = bool(p.get("raw_depthmap", False))
        corrected_depthmap = bool(p.get("corrected_depthmap", False))
        depthmap_input_str = str(p.get("depthmap_input_path", "") or "").strip()
        depthmap_input_path = Path(depthmap_input_str) if depthmap_input_str else None
        use_external_depthmap = bool(depthmap_input_path and depthmap_input_path.is_file())
        if depthmap_input_path is not None and not use_external_depthmap:
            self.log.emit(f"[Warning] Depthmap input file not found, ignoring: {depthmap_input_path}")
        if use_external_depthmap and extract_depthmap:
            self.log.emit("[Info] A depthmap input file was provided, so depthmap extraction will be skipped (already have one).")
        
        use_start = p.get("use_start_time", False)
        start_sec = p.get("start_time_sec", 0.0)
        use_end = p.get("use_end_time", False)
        end_sec = p.get("end_time_sec", 0.0)
        
        vcodec = p.get("vcodec", "libx265")
        preset = p.get("preset", "ultrafast")
        quality = p.get("quality", 16)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        name = safe_stem(input_path_orig)
        orig_ext = input_path_orig.suffix.lower() or ".mp4"
        if orig_ext not in VIDEO_EXTS: orig_ext = ".mp4"

        final_video = make_unique_output_path(output_dir, name, vr_player_format_label(fmt), orig_ext)

        self.log.emit(f"Input: {input_path_orig}")
        self.log.emit(f"Output: {final_video}")

        vf_filters = []
        
        if p.get("hdr_norm") and check_is_hdr(input_path_orig):
            self.log.emit("HDR video detected: Adding tonemapping filter.")
            vf_filters.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
        
        if p.get("auto_crop"):
            crop_filter = detect_letterbox(input_path_orig)
            if crop_filter:
                self.log.emit(f"Letterbox detected: Adding {crop_filter} filter.")
                vf_filters.append(crop_filter)

        if p.get("resize"):
            resize_value = str(p.get("resize_value", "1920x1080")).strip().lower().replace(" ", "")
            rw_str, rh_str = resize_value.split("x")
            
            try:
                target_w = int(rw_str)
                target_h = int(rh_str)
                
                cap_temp = cv2.VideoCapture(str(input_path_orig))
                orig_w = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_temp.release()
                
                if orig_w > 0 and orig_h > 0 and orig_w <= target_w and orig_h <= target_h:
                    self.log.emit(f"Original resolution ({orig_w}x{orig_h}) is lower or equal to target ({target_w}x{target_h}). Skipping resize.")
                else:
                    self.log.emit(f"Resize enabled: Adding scale={rw_str}:{rh_str} filter.")
                    vf_filters.append(f"scale={rw_str}:{rh_str}")
                    
            except Exception as e:
                self.log.emit(f"Resize enabled (fallback): Adding scale={rw_str}:{rh_str} filter.")
                vf_filters.append(f"scale={rw_str}:{rh_str}")

        working_input_path = input_path_orig
        is_pre_trimmed = False

        if vf_filters or use_start or use_end:
            self.log.emit("Generating temporary preprocessed / trimmed video. Please wait...")
            tmp_pre = Path(tempfile.gettempdir()) / f"stereo_preprocess_{uuid.uuid4().hex}.mp4"
            self._temp_files.append(tmp_pre)

            cmd = [FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "error", "-stats"]
            
            if use_start: cmd.extend(["-ss", str(start_sec)])
            if use_end: cmd.extend(["-to", str(end_sec)])
            
            cmd.extend(["-i", str(input_path_orig)])
            
            if vf_filters:
                cmd.extend(["-vf", ",".join(vf_filters)])
                
            pre_threads = 1

            self.log.emit(f"Creating temporary preprocessing file (codec={vcodec}, preset={preset}, quality={quality})")
            cmd.extend(["-c:v", vcodec])

            if "nvenc" in vcodec:
                cmd.extend([
                    "-rc", "vbr", "-preset", preset, "-cq", str(quality),
                    "-rc-lookahead", "20",
                    "-bufsize", "50M",
                    "-spatial-aq", "1", "-temporal-aq", "1", "-aq-strength", "8",
                ])
            else:
                cmd.extend([
                    "-preset", preset, "-crf", str(quality),
                    "-threads", str(pre_threads),
                ])

            cmd.extend(["-c:a", "aac", str(tmp_pre)])
            
            si = None
            if os.name == 'nt':
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                si.wShowWindow = 0

            total_duration = 0.0
            try:
                probe_cmd = [
                    FFPROBE_EXE, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(input_path_orig)
                ]
                duration_str = subprocess.check_output(probe_cmd, text=True, startupinfo=si).strip()
                total_duration = float(duration_str)

                if use_start or use_end:
                    s = start_sec if use_start else 0.0
                    e = end_sec if use_end else total_duration
                    total_duration = max(0.1, e - s)
            except Exception:
                total_duration = 0.0

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                startupinfo=si
            )

            pre_watchdog = None
            if "nvenc" in vcodec:
                pre_watchdog = _NvencStallWatchdog(self, [tmp_pre], [proc], stall_seconds=120.0)
                pre_watchdog.start()

            last_percent = -1
            try:
                while True:
                    if self._stop.is_set():
                        proc.kill()
                        break

                    line = proc.stderr.readline()
                    if not line and proc.poll() is not None:
                        break

                    if "time=" in line:
                        try:
                            time_str = line.split("time=")[1].split()[0]
                            h, m, s = time_str.split(":")
                            current_sec = int(h) * 3600 + int(m) * 60 + float(s)

                            if total_duration > 0:
                                percent = min(99, int(current_sec / total_duration * 100))
                                if percent != last_percent:
                                    self.log.emit(f"Preprocessing encoding... {percent}%")
                                    last_percent = percent
                        except Exception:
                            pass

                ret = proc.wait()
            except Exception:
                proc.kill()
                ret = -1
            finally:
                if pre_watchdog is not None:
                    pre_watchdog.mark_done()

            if ret == 0:
                working_input_path = tmp_pre
                is_pre_trimmed = True
                self.log.emit("Preprocessing and segment extraction complete. (100%)")
            else:
                err_txt = ""
                try:
                    err_txt = proc.stderr.read()[-300:] if proc.stderr else ""
                except Exception:
                    pass
                self.log.emit(f"[Warning] Preprocessing failed. Proceeding with original video.\nReason: {err_txt}")

        read_path, tmp_link = ensure_ascii_readable(working_input_path)
        if tmp_link is not None:
            self._temp_files.append(tmp_link)
            if working_input_path == input_path_orig:
                self.log.emit("Non-ASCII or special characters in path -> Created temporary link for reading")

        cap = cv2.VideoCapture(str(read_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {working_input_path}")

        if low_ram:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            original_duration = total / fps if fps > 0 else 0

            audio_start_sec = None
            audio_end_sec = None
            max_frames = None

            if not is_pre_trimmed:
                if use_start:
                    target_msec = start_sec * 1000.0
                    cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
                    actual_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if actual_msec < target_msec - 500:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
                    audio_start_sec = start_sec
                    
                if use_end:
                    audio_end_sec = end_sec
                
                if use_start or use_end:
                    s = start_sec if use_start else 0.0
                    e = end_sec if use_end else original_duration
                    if e > s:
                        max_frames = max(1, int((e - s) * fps))
                        total = max_frames
            else:
                audio_start_sec = None
                audio_end_sec = None
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                max_frames = total if total > 0 else None

            self.log.emit(f"Processing resolution: {width}x{height} @ {fps:.3f} fps, frames≈{total}")
            self.log.emit(f"Encoding config: codec={vcodec}, preset={preset}, quality={quality}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if self._stop.is_set():
                if self._stall_message:
                    self.log.emit("Aborted (NVENC stall detected before encoding started)")
                else:
                    self.log.emit("Aborted by user (before encoding started)")
                self.finished_error.emit(self._stall_message if self._stall_message else "Conversion was aborted.")
                return

            predictor = None
            depth_cap = None
            if use_external_depthmap:
                self.log.emit(f"Using external depthmap video (skipping ZipDepth): {depthmap_input_path}")
                depth_read_path, depth_tmp_link = ensure_ascii_readable(depthmap_input_path)
                if depth_tmp_link is not None:
                    self._temp_files.append(depth_tmp_link)
                depth_cap = cv2.VideoCapture(str(depth_read_path))
                if not depth_cap.isOpened():
                    raise RuntimeError(f"Failed to open depthmap video: {depthmap_input_path}")
                if use_start:
                    depth_cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

                # The main loop reads exactly one depthmap frame per source frame, in lockstep.
                # If the depthmap video is shorter than the source (or vice versa), that mismatch
                # would previously surface mid-encode as silently dropped frames, desyncing video
                # and audio. Instead, pre-check both frame counts here and clamp processing to
                # whichever is shorter, trimming the tail so video/audio/depthmap all end together.
                depth_total_raw = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                depth_start_pos = int(depth_cap.get(cv2.CAP_PROP_POS_FRAMES))
                depth_frames_available = (depth_total_raw - depth_start_pos) if depth_total_raw > 0 else None

                if depth_frames_available is not None and depth_frames_available > 0:
                    main_frames_available = max_frames if max_frames is not None else max(0, total)
                    if main_frames_available > 0 and depth_frames_available < main_frames_available:
                        clamped = max(1, depth_frames_available)
                        dropped_sec = (main_frames_available - clamped) / fps if fps > 0 else 0.0
                        self.log.emit(
                            f"[Info] Depthmap video is shorter than the source video "
                            f"({depth_frames_available} vs {main_frames_available} frames after the start offset). "
                            f"Trimming the tail by ~{dropped_sec:.2f}s so video/audio/depthmap stay in sync."
                        )
                        max_frames = clamped
                        total = clamped
                        base_start = start_sec if use_start else 0.0
                        audio_start_sec = base_start
                        audio_end_sec = base_start + (clamped / fps if fps > 0 else 0.0)
                    elif main_frames_available > 0 and depth_frames_available > main_frames_available:
                        extra_sec = (depth_frames_available - main_frames_available) / fps if fps > 0 else 0.0
                        self.log.emit(
                            f"[Info] Depthmap video is longer than the source video by ~{extra_sec:.2f}s; "
                            f"the extra depthmap tail will simply be unused."
                        )
                    # else: lengths already match (or main is unknown) -- nothing to clamp.
                elif depth_total_raw == 0:
                    self.log.emit(
                        "[Warning] Could not determine the depthmap video's frame count in advance; "
                        "falling back to a per-frame sync check during encoding."
                    )
            else:
                if depth_model in ("vda_s", "vda_s_metric"):
                    _is_metric = depth_model == "vda_s_metric"
                    self.log.emit(f"Loading Video Depth Anything (VDA-S{' Metric' if _is_metric else ''}, streaming)...")
                    predictor = load_vda_streaming(encoder="vits", input_size=input_size, fp16=fp16, metric=_is_metric)
                else:
                    self.log.emit("Loading ZipDepth...")
                    predictor = load_zipdepth(input_size=input_size, device=str(device), fp16=fp16)

            start_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ok, frame0 = cap.read()
            if not ok: raise RuntimeError("Failed to read first frame")

            depth_frame0 = None
            if use_external_depthmap:
                ok_d, depth_frame0 = depth_cap.read()
                if not ok_d: raise RuntimeError("Failed to read first frame of depthmap video")

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_pos)

            dtype_tensor = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
            bgr_t0 = torch.from_numpy(frame0).to(device=device, dtype=dtype_tensor).permute(2, 0, 1).unsqueeze(0).div_(255.0)

            depth_native_wh = None
            _ema_decay_clamped = float(max(0.0, min(0.99, float(ema_decay))))

            if use_external_depthmap:
                depth_ext0 = depthmap_frame_to_tensor(depth_frame0, frame0.shape[1], frame0.shape[0], device, dtype_tensor)
                depth_ext0 = 1.0 - depth_ext0
                with torch.inference_mode():
                    depth_norm0, _, _ = normalize_depth_gpu(depth_ext0.float(), decay=0.0)
                try:
                    depth_cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame_pos))
                except Exception:
                    try:
                        depth_cap.release()
                        depth_cap = cv2.VideoCapture(str(depth_read_path))
                        if use_start:
                            depth_cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
                    except Exception:
                        pass
            else:
                depth_raw0 = estimate_depth_raw(predictor, frame0)
                depth_native_wh = (int(depth_raw0.shape[-1]), int(depth_raw0.shape[-2]))
                if depth_model != "vda_s_metric":
                    depth_raw0 = depth_raw0.max() - depth_raw0
                depth0 = upsample_depth(depth_raw0.float(), frame0.shape[0], frame0.shape[1])
                with torch.inference_mode():
                    depth_norm0, _, _ = normalize_depth_gpu(depth0, decay=0.0)
                del depth_raw0, depth0

                # The warm-up pass above already fed frame0 through the predictor once
                # (to determine output shape before the encoder/pipe are set up). The
                # capture position is then rewound to start_frame_pos, so the main
                # encoding loop below will feed frame0 through the predictor a second
                # time. For stateless predictors (ZipDepth) this is harmless, but VDA
                # streaming models keep a temporal cache, and re-feeding frame0 would
                # corrupt that cache right at the start of the stream, causing a
                # depth/brightness jump in the first frames. Reset the predictor's
                # temporal state here so the main loop starts with a clean cache.
                _reset_fn = getattr(predictor, "reset_state", None)
                if callable(_reset_fn):
                    try:
                        _reset_fn()
                    except Exception as _reset_err:
                        self.log.emit(f"[Warning] predictor.reset_state() after warm-up frame failed: {_reset_err}")

            depth_for_warp = depth_norm0
            if edge_dilation_is_enabled(edge_dilation):
                _f0_h, _f0_w = frame0.shape[0], frame0.shape[1]
                _dh0, _dw0 = compute_dilation_target_dims(_f0_h, _f0_w)
                with torch.inference_mode():
                    depth_for_dilate0 = upsample_depth(depth_norm0, _dh0, _dw0)
                    inv = 1.0 - depth_for_dilate0
                    inv = dilate_edge(inv, edge_dilation)
                    depth_dilated0 = 1.0 - inv
                    depth_for_warp = (
                        upsample_depth(depth_dilated0, _f0_h, _f0_w)
                        if (_dh0, _dw0) != (_f0_h, _f0_w)
                        else depth_dilated0
                    )

            l0, r0 = make_stereo(
                bgr_t0, depth_for_warp, divergence, convergence, device,
                already_normalized=True, preserve_screen_border=preserve_border
            )
            packed0 = pack_frame_gpu(l0, r0, fmt)
            oh, ow = packed0.shape[2], packed0.shape[3]
            ow = ow - (ow % 2)
            oh = oh - (oh % 2)

            if low_ram:
                del l0, r0, packed0, depth_norm0, depth_for_warp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if _HAS_IW3_EMA:
                depth_scaler = EMAMinMaxScaler(decay=_ema_decay_clamped, buffer_size=1, mode="minmax")
                self.log.emit(f"[EMA] depth_scaler active (IW3 EMAMinMaxScaler), decay={_ema_decay_clamped}")
            else:
                depth_scaler = None
                self.log.emit("[EMA] depth_scaler unavailable — fallback to plain min/max (decay partially ignored)")

            current_format = p.get("format", "HSBS")

            threads_val = 1 if "nvenc" not in vcodec else None

            video_only_path = Path(tempfile.gettempdir()) / f"stereo_video_only_{uuid.uuid4().hex}.mp4"
            self._temp_files.append(video_only_path)

            ffmpeg_cmd = build_ffmpeg_encode_cmd(
                ow, oh, fps, video_only_path,
                vcodec=vcodec, preset=preset, quality=quality,
                pad_169=p.get("pad_169", False),
                stereo_format=current_format,
                threads=threads_val
            )

            do_extract_depthmap = extract_depthmap and not use_external_depthmap
            depthmap_video_only_path = None
            depthmap_proc = None
            depthmap_stderr_thread = None
            depthmap_stderr_buf = None
            depthmap_final_video = None
            if do_extract_depthmap:
                if depth_native_wh is not None:
                    dw, dh = depth_native_wh
                else:
                    dw, dh = width, height
                dw = dw - (dw % 2)
                dh = dh - (dh % 2)
                self.log.emit(f"Depthmap output resolution: {dw}x{dh} (ZipDepth native resolution)")
                depthmap_final_video = output_dir / f"{name}_depthmap.mp4"
                if depthmap_final_video.exists():
                    di = 1
                    while True:
                        cand = output_dir / f"{name}_depthmap_({di}).mp4"
                        if not cand.exists():
                            depthmap_final_video = cand
                            break
                        di += 1
                depthmap_video_only_path = Path(tempfile.gettempdir()) / f"stereo_depthmap_only_{uuid.uuid4().hex}.mp4"
                self._temp_files.append(depthmap_video_only_path)
                depthmap_ffmpeg_cmd = build_ffmpeg_encode_cmd(
                    dw, dh, fps, depthmap_video_only_path,
                    vcodec=vcodec, preset=preset, quality=quality,
                    pad_169=False,
                    stereo_format="hsbs",
                    threads=threads_val
                )
                self.log.emit(f"Depthmap output: {depthmap_final_video}")

            do_extract_corrected = extract_depthmap and not use_external_depthmap and corrected_depthmap
            corrected_video_only_path = None
            corrected_proc = None
            corrected_stderr_thread = None
            corrected_stderr_buf = None
            corrected_final_video = None
            corrected_dw = corrected_dh = None
            if do_extract_corrected:
                if depth_native_wh is not None:
                    corrected_dw, corrected_dh = depth_native_wh
                else:
                    corrected_dw, corrected_dh = width, height
                corrected_dw = corrected_dw - (corrected_dw % 2)
                corrected_dh = corrected_dh - (corrected_dh % 2)
                corrected_final_video = output_dir / f"{name}_depthmap_corrected.mp4"
                if corrected_final_video.exists():
                    ci = 1
                    while True:
                        cand = output_dir / f"{name}_depthmap_corrected_({ci}).mp4"
                        if not cand.exists():
                            corrected_final_video = cand
                            break
                        ci += 1
                corrected_video_only_path = Path(tempfile.gettempdir()) / f"stereo_depthmap_corrected_only_{uuid.uuid4().hex}.mp4"
                self._temp_files.append(corrected_video_only_path)
                corrected_ffmpeg_cmd = build_ffmpeg_encode_cmd(
                    corrected_dw, corrected_dh, fps, corrected_video_only_path,
                    vcodec=vcodec, preset=preset, quality=quality,
                    pad_169=False,
                    stereo_format="hsbs",
                    threads=threads_val
                )
                self.log.emit(f"Corrected depthmap output: {corrected_final_video}")

            self.log.emit("Starting encoder...")
            
            si = None
            if os.name == 'nt':
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                si.wShowWindow = 0
                
            try:
                proc = subprocess.Popen(
                    ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, startupinfo=si
                )
            except FileNotFoundError:
                raise RuntimeError("ffmpeg executable not found.")

            stderr_buf = []
            def _drain_stderr(pipe):
                try:
                    for line in iter(pipe.readline, b""):
                        stderr_buf.append(line)
                        if len(stderr_buf) > 500:
                            stderr_buf.pop(0)
                except Exception:
                    pass

            stderr_thread = threading.Thread(target=_drain_stderr, args=(proc.stderr,), daemon=True)
            stderr_thread.start()

            if do_extract_depthmap:
                try:
                    depthmap_proc = subprocess.Popen(
                        depthmap_ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE, startupinfo=si
                    )
                except FileNotFoundError:
                    raise RuntimeError("ffmpeg executable not found. (depthmap encoder)")

                depthmap_stderr_buf = []
                def _drain_depthmap_stderr(pipe):
                    try:
                        for line in iter(pipe.readline, b""):
                            depthmap_stderr_buf.append(line)
                            if len(depthmap_stderr_buf) > 500:
                                depthmap_stderr_buf.pop(0)
                    except Exception:
                        pass
                depthmap_stderr_thread = threading.Thread(target=_drain_depthmap_stderr, args=(depthmap_proc.stderr,), daemon=True)
                depthmap_stderr_thread.start()

            if do_extract_corrected:
                try:
                    corrected_proc = subprocess.Popen(
                        corrected_ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE, startupinfo=si
                    )
                except FileNotFoundError:
                    raise RuntimeError("ffmpeg executable not found. (corrected depthmap encoder)")

                corrected_stderr_buf = []
                def _drain_corrected_stderr(pipe):
                    try:
                        for line in iter(pipe.readline, b""):
                            corrected_stderr_buf.append(line)
                            if len(corrected_stderr_buf) > 500:
                                corrected_stderr_buf.pop(0)
                    except Exception:
                        pass
                corrected_stderr_thread = threading.Thread(target=_drain_corrected_stderr, args=(corrected_proc.stderr,), daemon=True)
                corrected_stderr_thread.start()

            nvenc_watchdog = None
            if "nvenc" in vcodec:
                watch_targets = [video_only_path]
                watch_procs = [proc]
                if do_extract_depthmap and depthmap_video_only_path is not None and depthmap_proc is not None:
                    watch_targets.append(depthmap_video_only_path)
                    watch_procs.append(depthmap_proc)
                if do_extract_corrected and corrected_video_only_path is not None and corrected_proc is not None:
                    watch_targets.append(corrected_video_only_path)
                    watch_procs.append(corrected_proc)
                nvenc_watchdog = _NvencStallWatchdog(self, watch_targets, watch_procs, stall_seconds=120.0)
                nvenc_watchdog.start()

            frame_q: queue.Queue = queue.Queue(maxsize=1 if low_ram else 8)
            reader = _FrameReaderThread(cap, max_frames, self._stop, frame_q)
            reader.start()

            pipe_writer = _PipeWriterThread(proc.stdin, maxsize=2 if low_ram else 6)
            pipe_writer.start()
            depthmap_pipe_writer = None
            if do_extract_depthmap:
                depthmap_pipe_writer = _PipeWriterThread(depthmap_proc.stdin, maxsize=2 if low_ram else 6)
                depthmap_pipe_writer.start()
            corrected_pipe_writer = None
            if do_extract_corrected:
                corrected_pipe_writer = _PipeWriterThread(corrected_proc.stdin, maxsize=2 if low_ram else 6)
                corrected_pipe_writer.start()

            idx = 0
            t0 = time.perf_counter()
            aborted = False

            try:
                while True:
                    if self._stop.is_set():
                        self.log.emit("Aborted by user")
                        aborted = True
                        break

                    frame = frame_q.get()
                    if frame is None: break

                    dtype_tensor = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
                    frame_cpu_t = torch.from_numpy(frame)
                    if device.type == "cuda":
                        frame_cpu_t = frame_cpu_t.pin_memory()
                    bgr_t = frame_cpu_t.to(device=device, dtype=dtype_tensor, non_blocking=(device.type == "cuda")).permute(2, 0, 1).unsqueeze(0).div_(255.0)

                    decay = float(max(0.0, min(0.99, float(ema_decay))))
                    if depth_scaler is not None and abs(depth_scaler.decay - decay) > 1e-9:
                        depth_scaler.set_decay(decay)
                        _ema_decay_clamped = decay

                    raw_depth_snapshot = None
                    depth_norm = None
                    try:
                        if use_external_depthmap:
                            ok_d, depth_frame = depth_cap.read()
                            if not ok_d:
                                depth_frame = None
                            if depth_frame is not None:
                                depth_ext = depthmap_frame_to_tensor(
                                    depth_frame, frame.shape[1], frame.shape[0], device, dtype_tensor
                                )
                                depth_ext = (1.0 - depth_ext).float()
                                if do_extract_depthmap and raw_depthmap:
                                    with torch.inference_mode():
                                        raw_depth_snapshot, _, _ = normalize_depth_gpu(depth_ext, decay=0.0)
                                if depth_scaler is not None:
                                    with torch.inference_mode():
                                        depth_norm = depth_scaler.update(depth_ext)
                                else:
                                    with torch.inference_mode():
                                        depth_norm, _, _ = normalize_depth_gpu(depth_ext, decay=0.0)
                            else:
                                depth_norm = None
                        else:
                            depth_raw = estimate_depth_raw(predictor, frame)
                            if depth_model != "vda_s_metric":
                                depth_raw = depth_raw.max() - depth_raw
                            depth_raw = depth_raw.float()

                            if do_extract_depthmap and raw_depthmap:
                                with torch.inference_mode():
                                    raw_snap_raw, _, _ = normalize_depth_gpu(depth_raw, decay=0.0)
                                raw_depth_snapshot = upsample_depth(
                                    raw_snap_raw, frame.shape[0], frame.shape[1]
                                )

                            if depth_scaler is not None:
                                with torch.inference_mode():
                                    depth_norm_raw = depth_scaler.update(depth_raw)
                                depth_norm = upsample_depth(
                                    depth_norm_raw, frame.shape[0], frame.shape[1]
                                )
                            else:
                                depth_up = upsample_depth(depth_raw, frame.shape[0], frame.shape[1])
                                with torch.inference_mode():
                                    depth_norm, _, _ = normalize_depth_gpu(depth_up, decay=0.0)
                            del depth_raw
                    except Exception as _depth_err:
                        self.log.emit(f"[Error] Depth/EMA failed at frame {idx}: {_depth_err}")
                        raise

                    if depth_norm is None:
                        # Should be rare after the pre-loop length clamp above, but guards against
                        # edge cases (e.g. frame-count metadata was inaccurate). Stop cleanly here
                        # instead of silently dropping frames for the remainder of the video, which
                        # would desync video/audio without any indication of what happened.
                        self.log.emit(
                            f"[Warning] Depthmap video ran out of frames at frame {idx}. "
                            f"Stopping here and finalizing the output so video/audio stay in sync."
                        )
                        break

                    depth_for_warp = depth_norm
                    if edge_dilation_is_enabled(edge_dilation):
                        _fr_h, _fr_w = frame.shape[0], frame.shape[1]
                        _dh, _dw = compute_dilation_target_dims(_fr_h, _fr_w)
                        with torch.inference_mode():
                            depth_for_dilate = upsample_depth(depth_norm, _dh, _dw)
                            inv = 1.0 - depth_for_dilate
                            inv = dilate_edge(inv, edge_dilation)
                            depth_dilated = 1.0 - inv
                            depth_for_warp = (
                                upsample_depth(depth_dilated, _fr_h, _fr_w)
                                if (_dh, _dw) != (_fr_h, _fr_w)
                                else depth_dilated
                            )

                    if do_extract_depthmap:
                        with torch.inference_mode():
                            dm_source = raw_depth_snapshot if (raw_depthmap and raw_depth_snapshot is not None) else depth_for_warp
                            if dm_source.shape[-1] != dw or dm_source.shape[-2] != dh:
                                dm_source = F.interpolate(dm_source, size=(dh, dw), mode="bilinear", align_corners=False)
                            dm_u8 = (1.0 - dm_source).clamp(0.0, 1.0).mul(255.0).byte()
                            dm_u8 = dm_u8.expand(-1, 3, -1, -1).permute(0, 2, 3, 1).contiguous()
                            dm_bytes = dm_u8[0].cpu().numpy().tobytes()
                        try:
                            depthmap_pipe_writer.submit(dm_bytes)
                        except Exception:
                            raise RuntimeError("ffmpeg depthmap encoding pipe broken.")

                    if do_extract_corrected:
                        with torch.inference_mode():
                            corr_source = depth_for_warp
                            if corr_source.shape[-1] != corrected_dw or corr_source.shape[-2] != corrected_dh:
                                corr_source = F.interpolate(corr_source, size=(corrected_dh, corrected_dw), mode="bilinear", align_corners=False)
                            corr_u8 = (1.0 - corr_source).clamp(0.0, 1.0).mul(255.0).byte()
                            corr_u8 = corr_u8.expand(-1, 3, -1, -1).permute(0, 2, 3, 1).contiguous()
                            corr_bytes = corr_u8[0].cpu().numpy().tobytes()
                        try:
                            corrected_pipe_writer.submit(corr_bytes)
                        except Exception:
                            raise RuntimeError("ffmpeg corrected depthmap encoding pipe broken.")

                    left_t, right_t = make_stereo(
                        bgr_t, depth_for_warp, divergence, convergence, device,
                        already_normalized=True, preserve_screen_border=preserve_border,
                    )
                    packed_t = pack_frame_gpu(left_t, right_t, fmt)

                    frame_bytes = tensor_to_bgr_bytes(packed_t)

                    del left_t, right_t, packed_t, depth_norm, depth_for_warp, bgr_t, frame
                    if raw_depth_snapshot is not None:
                        del raw_depth_snapshot

                    try:
                        pipe_writer.submit(frame_bytes)
                    except Exception:
                        raise RuntimeError("ffmpeg encoding pipe broken.")

                    if low_ram:
                        if idx % 50 == 0 and torch.cuda.is_available():
                            try:
                                total_mem = torch.cuda.get_device_properties(0).total_memory
                                reserved_mem = torch.cuda.memory_reserved(0)
                                if (reserved_mem / total_mem) > 0.80:
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass

                        if idx % 200 == 0:
                            gc.collect()
                    else:
                        if idx % 300 == 0 and torch.cuda.is_available():
                            try:
                                total_mem = torch.cuda.get_device_properties(0).total_memory
                                reserved_mem = torch.cuda.memory_reserved(0)
                                if (reserved_mem / total_mem) > 0.92:
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass

                    idx += 1
                    if idx % 10 == 0 or (total > 0 and idx == total):
                        elapsed = time.perf_counter() - t0
                        fps_now = idx / max(elapsed, 1e-6)
                        msg = f"{idx}/{total if total > 0 else '?'} ({fps_now:.1f} fps)"
                        self.progress.emit(idx, total if total > 0 else 0, msg)

                        # Throttle the text log (separate from the progress bar) so that
                        # frequent progress lines don't push older messages (e.g. scene
                        # cut markers) out of the log widget's line buffer on long videos.
                        if total > 0:
                            percent = min(100, int(idx / total * 100))
                            if percent != self._last_logged_progress_percent or idx == total:
                                self.log.emit(f"Progress: {msg}")
                                self._last_logged_progress_percent = percent
                        else:
                            if idx % 200 == 0:
                                self.log.emit(f"Progress: {msg}")
            finally:
                self._stop.set()
                if nvenc_watchdog is not None:
                    nvenc_watchdog.mark_done()
                reader.join(timeout=2.0)
                try:
                    if not aborted:
                        pipe_writer.close()
                    else:
                        pipe_writer.queue.put(None)
                except Exception:
                    pass
                try: proc.stdin.close()
                except Exception: pass
                if depthmap_proc is not None:
                    try:
                        if depthmap_pipe_writer is not None:
                            if not aborted:
                                depthmap_pipe_writer.close()
                            else:
                                depthmap_pipe_writer.queue.put(None)
                    except Exception:
                        pass
                    try: depthmap_proc.stdin.close()
                    except Exception: pass
                if corrected_proc is not None:
                    try:
                        if corrected_pipe_writer is not None:
                            if not aborted:
                                corrected_pipe_writer.close()
                            else:
                                corrected_pipe_writer.queue.put(None)
                    except Exception:
                        pass
                    try: corrected_proc.stdin.close()
                    except Exception: pass
                if depth_cap is not None:
                    try: depth_cap.release()
                    except Exception: pass

            if aborted:
                proc.kill()
                proc.wait(timeout=3)
                stderr_thread.join(timeout=2.0)
                if final_video.exists(): final_video.unlink(missing_ok=True)
                if depthmap_proc is not None:
                    depthmap_proc.kill()
                    depthmap_proc.wait(timeout=3)
                    depthmap_stderr_thread.join(timeout=2.0)
                    if depthmap_final_video and depthmap_final_video.exists():
                        depthmap_final_video.unlink(missing_ok=True)
                if corrected_proc is not None:
                    corrected_proc.kill()
                    corrected_proc.wait(timeout=3)
                    corrected_stderr_thread.join(timeout=2.0)
                    if corrected_final_video and corrected_final_video.exists():
                        corrected_final_video.unlink(missing_ok=True)
                self.finished_error.emit(self._stall_message if self._stall_message else "Conversion was aborted.")
                return
            
            ret = proc.wait()
            stderr_thread.join(timeout=2.0)
            if ret != 0:
                err_txt = b"".join(stderr_buf).decode("utf-8", errors="ignore")[-2000:]
                raise RuntimeError(f"ffmpeg encoding failed (code={ret}): {err_txt}")

            depthmap_ret = None
            if depthmap_proc is not None:
                depthmap_ret = depthmap_proc.wait()
                depthmap_stderr_thread.join(timeout=2.0)
                if depthmap_ret != 0:
                    err_txt = b"".join(depthmap_stderr_buf).decode("utf-8", errors="ignore")[-2000:]
                    self.log.emit(f"[Warning] Depthmap encoding failed (code={depthmap_ret}): {err_txt}")

            corrected_ret = None
            if corrected_proc is not None:
                corrected_ret = corrected_proc.wait()
                corrected_stderr_thread.join(timeout=2.0)
                if corrected_ret != 0:
                    err_txt = b"".join(corrected_stderr_buf).decode("utf-8", errors="ignore")[-2000:]
                    self.log.emit(f"[Warning] Corrected depthmap encoding failed (code={corrected_ret}): {err_txt}")

            self.log.emit("Video encoding complete. Muxing audio...")
            mux_cmd = build_audio_mux_cmd(
                video_only_path, working_input_path, final_video,
                audio_start_sec=audio_start_sec, audio_end_sec=audio_end_sec,
            )
            mux_ret = subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, startupinfo=si)
            if mux_ret.returncode != 0:
                raise RuntimeError(f"Audio muxing failed: {mux_ret.stderr.decode('utf-8', errors='ignore')[-1000:]}")

            self.log.emit(f"Completed: {final_video}")

            if depthmap_proc is not None and depthmap_ret == 0 and depthmap_video_only_path.exists():
                self.log.emit("Depthmap video encoding complete. Muxing audio...")
                dm_mux_cmd = build_audio_mux_cmd(
                    depthmap_video_only_path, working_input_path, depthmap_final_video,
                    audio_start_sec=audio_start_sec, audio_end_sec=audio_end_sec,
                )
                dm_mux_ret = subprocess.run(dm_mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, startupinfo=si)
                if dm_mux_ret.returncode != 0:
                    self.log.emit(f"[Warning] Depthmap audio muxing failed: {dm_mux_ret.stderr.decode('utf-8', errors='ignore')[-1000:]}")
                else:
                    self.log.emit(f"Depthmap video completed: {depthmap_final_video}")

            if corrected_proc is not None and corrected_ret == 0 and corrected_video_only_path.exists():
                self.log.emit("Corrected depthmap video encoding complete. Muxing audio...")
                corr_mux_cmd = build_audio_mux_cmd(
                    corrected_video_only_path, working_input_path, corrected_final_video,
                    audio_start_sec=audio_start_sec, audio_end_sec=audio_end_sec,
                )
                corr_mux_ret = subprocess.run(corr_mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, startupinfo=si)
                if corr_mux_ret.returncode != 0:
                    self.log.emit(f"[Warning] Corrected depthmap audio muxing failed: {corr_mux_ret.stderr.decode('utf-8', errors='ignore')[-1000:]}")
                else:
                    self.log.emit(f"Corrected depthmap video completed: {corrected_final_video}")

            self.finished_ok.emit(str(final_video))

        finally:
            cap.release()


# ------------------------------------------------------------------
# 언어(번역) 관련 설정
# ------------------------------------------------------------------
# 런처(DepthLive3d.py)에서 선택/저장한 언어 설정을 그대로 따라갑니다.
# 런처가 영어로 표시되어 있으면 이 창도 영어로, 런처가 한국어로
# 표시되어 있으면 이 창도 한국어로 표시됩니다.
_LAUNCHER_SUPPORTED_LANGUAGES = ("en", "ko")
_LAUNCHER_DEFAULT_LANGUAGE = "en"


def _launcher_config_path() -> Path:
    return APP_DIR / "depthlive3d_config.json"


def _load_ui_language() -> str:
    """런처가 저장한 depthlive3d_config.json 에서 현재 언어를 읽어옵니다.
    설정 파일이 없거나 읽기에 실패하면 기본값(영어)을 사용합니다."""
    try:
        with open(_launcher_config_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        lang = data.get("language", _LAUNCHER_DEFAULT_LANGUAGE)
        if lang in _LAUNCHER_SUPPORTED_LANGUAGES:
            return lang
    except Exception:
        pass
    return _LAUNCHER_DEFAULT_LANGUAGE


CURRENT_LANG = _load_ui_language()


TRANSLATIONS = {
    "en": {
        "title": "Depth Conversion 3D  |  © 2026 Wake-82",
        "file_grp": "Input / Output",
        "input_label": "Input File",
        "input_placeholder": "Select input video file...",
        "btn_input": "Select File...",
        "depthmap_input_label": "Depthmap Input File",
        "depthmap_input_placeholder": "(Optional) Select a pre-made depthmap video...",
        "btn_depthmap_input": "Select File...",
        "btn_depthmap_clear": "Clear",
        "output_label": "Output Folder",
        "output_placeholder": "Select output folder...",
        "btn_output": "Select Folder...",
        "preset_grp": "Preset",
        "preset_name": "Preset Name",
        "preset_save": "Save",
        "preset_load": "Load",
        "preset_delete": "Delete",
        "preset_saved": "Preset saved.",
        "preset_loaded": "Preset loaded.",
        "preset_deleted": "Preset deleted.",
        "param_grp": "",
        "fmt_label": "Output Format",
        "div_label": "3D Strength",
        "conv_label": "Convergence",
        "edge_label": "Edge Fix",
        "ema_label": "Flicker Reduction",
        "depth_model_label": "Depth Model",
        "size_label": "Depth Resolution",
        "chk_extract_depthmap": "Extract Depthmap Video",
        "chk_raw_depthmap": "Extract Raw Depthmap",
        "chk_corrected_depthmap": "Extract Corrected Depthmap",
        "chk_preserve": "Screen Border Protection",
        "chk_fp16": "Use FP16",
        "chk_auto_mode": "Auto Mode",
        "chk_low_ram": "Use Memory Saver",
        "vcodec_label": "Video Codec:",
        "preset_label": "Preset:",
        "quality_crf": "Quality (CRF):",
        "quality_cq": "Quality (CQ):",
        "chk_resize": "Resize Resolution",
        "chk_hdr_norm": "MKV HDR Normalize (Auto HDR Tonemapping)",
        "chk_auto_crop": "Auto Letterbox Crop (Remove Black Bars)",
        "chk_pad_169": "Pad to 16:9 (Auto Aspect Ratio Fill)",
        "chk_start_time": "Start Time",
        "chk_end_time": "End Time",
        "btn_start": "Start",
        "btn_stop": "Stop",
        "status_idle": "Idle",
        "status_starting": "Starting conversion...",
        "status_done": "Completed",
        "status_error": "Error / Aborted",
        "dialog_input_title": "Select Input Video",
        "dialog_output_title": "Select Output Folder",
        "btn_input_folder": "Select Folder...",
        "err_no_videos_in_folder": "No supported video files found in the selected folder.",
        "batch_progress": "Batch Progress: {} / {} - {}",
        "err_title": "Input Error",
        "err_no_input": "Please select an input file.",
        "err_no_output": "Please select an output folder.",
        "err_file_not_found": "Input file does not exist: {}",
        "err_unsupported_ext": "Unsupported file extension: {}",
        "err_invalid_size_range": "Input Size must be between 64 and 630.",
        "err_invalid_size_num": "Input Size is not a valid number.",
        "err_invalid_resize": "Resize Resolution must be in WIDTHxHEIGHT format (e.g. 1920x1080).",
        "notice_title": "Notice",
        "notice_already_running": "Conversion is already in progress.",
        "notice_preset_load_auto_mode": "Please turn off Auto Mode before loading a preset.",
        "time_err_title": "Time Input Error",
        "time_err_start_exceed": "Start time ({}s) exceeds total video length ({:.1f}s).",
        "time_err_end_exceed": "End time ({}s) exceeds total video length ({:.1f}s).",
        "time_err_order": "Start time cannot be greater than or equal to end time.",
        "done_title": "Completed",
        "done_msg": "Conversion completed.\n\n{}",
        "err_dialog_title": "Error",
        "err_depthmap_folder_conflict": "A depthmap input file is selected, but the input is set to a folder.\nPlease select a single input video file instead of a folder.",
    },
    "ko": {
        "title": "Depth Conversion 3D  |  © 2026 Wake-82",
        "file_grp": "입력 / 출력",
        "input_label": "입력 파일",
        "input_placeholder": "변환할 입력 영상 파일을 선택하세요...",
        "btn_input": "파일 선택...",
        "depthmap_input_label": "뎁스맵 입력 파일",
        "depthmap_input_placeholder": "(선택) 미리 만들어진 뎁스맵 영상을 선택하세요...",
        "btn_depthmap_input": "파일 선택...",
        "btn_depthmap_clear": "지우기",
        "output_label": "출력 폴더",
        "output_placeholder": "출력 폴더를 선택하세요...",
        "btn_output": "폴더 선택...",
        "preset_grp": "프리셋",
        "preset_name": "프리셋 이름",
        "preset_save": "저장",
        "preset_load": "불러오기",
        "preset_delete": "삭제",
        "preset_saved": "프리셋이 저장되었습니다.",
        "preset_loaded": "프리셋을 불러왔습니다.",
        "preset_deleted": "프리셋이 삭제되었습니다.",
        "param_grp": "",
        "fmt_label": "출력 형식",
        "div_label": "3D 강도",
        "conv_label": "컨버전스",
        "edge_label": "엣지 보정",
        "ema_label": "깜빡임 감소",
        "depth_model_label": "뎁스맵 모델",
        "size_label": "뎁스 해상도",
        "chk_extract_depthmap": "뎁스맵 영상 추출",
        "chk_raw_depthmap": "원본 뎁스맵 추출",
        "chk_corrected_depthmap": "보정 뎁스맵 추출",
        "chk_preserve": "화면 테두리 보호",
        "chk_fp16": "FP16 사용",
        "chk_auto_mode": "자동 모드",
        "chk_low_ram": "메모리 절약 모드 사용",
        "vcodec_label": "비디오 코덱:",
        "preset_label": "프리셋:",
        "quality_crf": "품질 (CRF):",
        "quality_cq": "품질 (CQ):",
        "chk_resize": "해상도 조정",
        "chk_hdr_norm": "MKV HDR 정규화 (자동 HDR 톤매핑)",
        "chk_auto_crop": "자동 레터박스 크롭 (검은 여백 제거)",
        "chk_pad_169": "16:9 비율로 패딩 (자동 화면 비율 채우기)",
        "chk_start_time": "시작 시간",
        "chk_end_time": "종료 시간",
        "btn_start": "시작",
        "btn_stop": "중지",
        "status_idle": "대기 중",
        "status_starting": "변환을 시작합니다...",
        "status_done": "완료",
        "status_error": "오류 / 중단됨",
        "dialog_input_title": "입력 영상 선택",
        "dialog_output_title": "출력 폴더 선택",
        "btn_input_folder": "폴더 선택...",
        "err_no_videos_in_folder": "선택한 폴더에 지원되는 영상 파일이 없습니다.",
        "batch_progress": "일괄 진행: {} / {} - {}",
        "err_title": "입력 오류",
        "err_no_input": "입력 파일을 선택해 주세요.",
        "err_no_output": "출력 폴더를 선택해 주세요.",
        "err_file_not_found": "입력 파일이 존재하지 않습니다: {}",
        "err_unsupported_ext": "지원하지 않는 파일 확장자입니다: {}",
        "err_invalid_size_range": "입력 크기는 64~630 사이여야 합니다.",
        "err_invalid_size_num": "입력 크기가 올바른 숫자가 아닙니다.",
        "err_invalid_resize": "해상도 조정 값은 '가로x세로' 형식이어야 합니다 (예: 1920x1080).",
        "notice_title": "알림",
        "notice_already_running": "이미 변환이 진행 중입니다.",
        "notice_preset_load_auto_mode": "프리셋을 불러오기 전에 자동 모드를 꺼주세요.",
        "time_err_title": "시간 입력 오류",
        "time_err_start_exceed": "시작 시간({}초)이 전체 영상 길이({:.1f}초)를 초과합니다.",
        "time_err_end_exceed": "종료 시간({}초)이 전체 영상 길이({:.1f}초)를 초과합니다.",
        "time_err_order": "시작 시간은 종료 시간보다 크거나 같을 수 없습니다.",
        "done_title": "완료",
        "done_msg": "변환이 완료되었습니다.\n\n{}",
        "err_dialog_title": "오류",
        "err_depthmap_folder_conflict": "뎁스맵 입력 파일이 선택되어 있지만, 입력이 폴더로 설정되어 있습니다.\n폴더 대신 단일 입력 영상 파일을 선택해 주세요.",
    },
}


class StereoVideoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(880, 600)
        self.worker: ConvertWorker | None = None

        self.setStyleSheet(self.styleSheet() + """
            QComboBox {
                background-color: #ffffff;
                color: #000000;
            }
            QComboBox:disabled {
                background-color: #ffffff;
                color: #808080;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #000000;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QVBoxLayout(central)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_content = QWidget()
        content_lay = QVBoxLayout(scroll_content)
        content_lay.setSpacing(10)
        scroll.setWidget(scroll_content)
        main_lay.addWidget(scroll, stretch=1)

        self.batch_queue = []
        self.batch_total = 0
        self.current_params = {}

        self.file_grp = QGroupBox()
        file_lay = QGridLayout(self.file_grp)

        self.lbl_input = QLabel()
        self.edt_input = QLineEdit()
        self.edt_input.setReadOnly(True)
        
        input_btn_lay = QHBoxLayout()
        input_btn_lay.setContentsMargins(0, 0, 0, 0)
        self.btn_input = QPushButton()
        self.btn_input.clicked.connect(self._select_input)
        self.btn_input_folder = QPushButton()
        self.btn_input_folder.clicked.connect(self._select_input_folder)
        input_btn_lay.addWidget(self.btn_input)
        input_btn_lay.addWidget(self.btn_input_folder)

        self.lbl_depthmap_input = QLabel()
        self.edt_depthmap_input = QLineEdit()
        self.edt_depthmap_input.setReadOnly(True)

        depthmap_input_btn_lay = QHBoxLayout()
        depthmap_input_btn_lay.setContentsMargins(0, 0, 0, 0)
        self.btn_depthmap_input = QPushButton()
        self.btn_depthmap_input.clicked.connect(self._select_depthmap_input)
        self.btn_depthmap_clear = QPushButton()
        self.btn_depthmap_clear.clicked.connect(self._clear_depthmap_input)
        depthmap_input_btn_lay.addWidget(self.btn_depthmap_input)
        depthmap_input_btn_lay.addWidget(self.btn_depthmap_clear)

        self.lbl_output = QLabel()
        self.edt_output = QLineEdit()
        self.edt_output.setReadOnly(True)
        self.btn_output = QPushButton()
        self.btn_output.clicked.connect(self._select_output)

        file_lay.addWidget(self.lbl_input, 0, 0)
        file_lay.addWidget(self.edt_input, 0, 1)
        file_lay.addLayout(input_btn_lay, 0, 2)
        file_lay.addWidget(self.lbl_depthmap_input, 1, 0)
        file_lay.addWidget(self.edt_depthmap_input, 1, 1)
        file_lay.addLayout(depthmap_input_btn_lay, 1, 2)
        file_lay.addWidget(self.lbl_output, 2, 0)
        file_lay.addWidget(self.edt_output, 2, 1)
        file_lay.addWidget(self.btn_output, 2, 2)
        content_lay.addWidget(self.file_grp)

        self.preset_grp = QGroupBox()
        preset_lay = QHBoxLayout(self.preset_grp)
        self.cmb_preset_name = QComboBox()
        self.cmb_preset_name.setMinimumWidth(160)
        self.cmb_preset_name.setEditable(False)
        self.cmb_preset_name.currentIndexChanged.connect(self._on_preset_selected)
        self.edt_preset_name = QLineEdit()
        self.btn_preset_save = QPushButton()
        self.btn_preset_save.clicked.connect(self._preset_save)
        self.btn_preset_load = QPushButton()
        self.btn_preset_load.clicked.connect(self._preset_load)
        self.btn_preset_delete = QPushButton()
        self.btn_preset_delete.clicked.connect(self._preset_delete)
        preset_lay.addWidget(self.cmb_preset_name)
        preset_lay.addWidget(self.edt_preset_name, 1)
        preset_lay.addWidget(self.btn_preset_save)
        preset_lay.addWidget(self.btn_preset_load)
        preset_lay.addWidget(self.btn_preset_delete)
        content_lay.addWidget(self.preset_grp)

        self.param_grp = QGroupBox()
        param_lay = QGridLayout(self.param_grp)

        self.lbl_format = QLabel()
        self.cmb_format = QComboBox()
        self.cmb_format.addItems(["hsbs", "fsbs", "tb", "ftb", "anaglyph", "half-anaglyph"])
        self.cmb_format.setCurrentText("hsbs")

        self.lbl_div = QLabel()
        self.spin_div = QDoubleSpinBox(); self.spin_div.setRange(0.5, 3.0); self.spin_div.setSingleStep(0.5); self.spin_div.setValue(1.0); self.spin_div.setDecimals(1)
        
        self.lbl_conv = QLabel()
        self.spin_conv = QDoubleSpinBox(); self.spin_conv.setRange(0.0, 1.0); self.spin_conv.setSingleStep(0.1); self.spin_conv.setValue(0.5); self.spin_conv.setDecimals(1)
        
        self.lbl_edge = QLabel()
        self.spin_edge = QSpinBox(); self.spin_edge.setRange(0, 5); self.spin_edge.setValue(2)
        
        self.lbl_ema = QLabel()
        self._EMA_LEVEL_VALUES = [0.0, 0.30, 0.40, 0.50, 0.60]
        self.spin_ema = QComboBox()
        self.spin_ema.addItems(["OFF", "Low", "Medium", "High", "Ultra"])
        self.spin_ema.setCurrentIndex(0)
        
        self.lbl_depth_model = QLabel()
        # Resolution choices differ per depth backend: ZipDepth is happy
        # with multiples of 32, while VDA needs multiples of 14 (its ViT
        # patch size). Keep both lists + a default here so the resolution
        # combo can be repopulated when the model choice changes.
        self._depth_model_input_sizes = {
            "zipdepth": ["256", "384", "512"],
            "vda_s": ["336", "392", "518"],
            "vda_s_metric": ["336", "392", "518"],
        }
        self._depth_model_default_size = {
            "zipdepth": "256",
            "vda_s": "392",
            "vda_s_metric": "392",
        }
        self.cmb_depth_model = QComboBox()
        self.cmb_depth_model.setEditable(False)
        self.cmb_depth_model.addItem("ZipDepth", "zipdepth")
        self.cmb_depth_model.addItem("VDA-S", "vda_s")
        self.cmb_depth_model.addItem("VDA-S (Metric)", "vda_s_metric")
        self.cmb_depth_model.setCurrentIndex(0)

        self.lbl_input_size = QLabel()
        self.cmb_input_size = QComboBox()
        self.cmb_input_size.addItems(self._depth_model_input_sizes["zipdepth"])
        self.cmb_input_size.setCurrentText(self._depth_model_default_size["zipdepth"])
        self.cmb_input_size.setEditable(True)
        self.cmb_input_size.lineEdit().setValidator(QIntValidator(64, 630, self.cmb_input_size))

        self.cmb_depth_model.currentIndexChanged.connect(self._on_depth_model_changed)

        self.chk_raw_depthmap = QCheckBox()
        self.chk_corrected_depthmap = QCheckBox()

        self.chk_preserve = QCheckBox()
        self.chk_preserve.setChecked(True)
        self.chk_fp16 = QCheckBox()
        self.chk_fp16.setChecked(True)
        self.chk_auto_mode = QCheckBox()
        self.chk_auto_mode.setChecked(True)
        self.chk_low_ram = QCheckBox()
        
        self.chk_auto_mode.toggled.connect(self._update_auto_mode_state)

        self.lbl_vcodec = QLabel()
        self.cmb_vcodec = QComboBox()
        self.cmb_vcodec.addItems(["libx264", "libx265", "h264_nvenc", "hevc_nvenc"])
        self.cmb_vcodec.setCurrentText("libx265")
        self.cmb_vcodec.currentTextChanged.connect(self._update_codec_options)

        self.lbl_preset = QLabel()
        self.cmb_preset = QComboBox()
        
        self.spin_quality = QSpinBox()
        self.spin_quality.setRange(0, 51)
        self.spin_quality.setValue(16)
        self.lbl_quality = QLabel()


        self.chk_resize = QCheckBox()
        self.edt_resize = QLineEdit()
        self.edt_resize.setText("1920x1080")
        self.edt_resize.setPlaceholderText("1920x1080")
        self.edt_resize.setEnabled(False)
        self.edt_resize.setMaximumWidth(95)
        self.chk_resize.toggled.connect(self.edt_resize.setEnabled)
        
        self.spin_div.valueChanged.connect(self._on_div_changed)

        self.chk_hdr_norm = QCheckBox()
        self.chk_auto_crop = QCheckBox()
        self.chk_pad_169 = QCheckBox()

        r = 0
        param_lay.addWidget(self.lbl_format, r, 0); param_lay.addWidget(self.cmb_format, r, 1); r += 1
        param_lay.addWidget(self.lbl_div, r, 0); param_lay.addWidget(self.spin_div, r, 1); r += 1
        param_lay.addWidget(self.lbl_conv, r, 0); param_lay.addWidget(self.spin_conv, r, 1); r += 1
        param_lay.addWidget(self.lbl_edge, r, 0); param_lay.addWidget(self.spin_edge, r, 1); r += 1
        param_lay.addWidget(self.lbl_ema, r, 0); param_lay.addWidget(self.spin_ema, r, 1); r += 1
        param_lay.addWidget(self.lbl_depth_model, r, 0); param_lay.addWidget(self.cmb_depth_model, r, 1); r += 1
        param_lay.addWidget(self.lbl_input_size, r, 0); param_lay.addWidget(self.cmb_input_size, r, 1); r += 1
        depthmap_opt_lay = QHBoxLayout()
        depthmap_opt_lay.addWidget(self.chk_raw_depthmap)
        depthmap_opt_lay.addWidget(self.chk_corrected_depthmap)
        depthmap_opt_lay.addStretch(1)
        param_lay.addLayout(depthmap_opt_lay, r, 0, 1, 2); r += 1

        preserve_lay = QHBoxLayout()
        preserve_lay.addWidget(self.chk_preserve)
        preserve_lay.addWidget(self.chk_fp16)
        preserve_lay.addWidget(self.chk_auto_mode)
        preserve_lay.addStretch(1)
        param_lay.addLayout(preserve_lay, r, 0, 1, 2); r += 1

        encode_lay = QHBoxLayout()
        encode_lay.addWidget(self.lbl_vcodec)
        encode_lay.addWidget(self.cmb_vcodec)
        encode_lay.addWidget(self.lbl_preset)
        encode_lay.addWidget(self.cmb_preset, 1)
        encode_lay.addWidget(self.lbl_quality)
        encode_lay.addWidget(self.spin_quality, 1)
        param_lay.addLayout(encode_lay, r, 0, 1, 2); r += 1

        resize_lay = QHBoxLayout()
        resize_lay.addWidget(self.chk_resize)
        resize_lay.addWidget(self.edt_resize)
        resize_lay.addWidget(self.chk_hdr_norm)
        resize_lay.addStretch(1)
        param_lay.addLayout(resize_lay, r, 0, 1, 2); r += 1

        crop_lay = QHBoxLayout()
        crop_lay.addWidget(self.chk_auto_crop)
        crop_lay.addWidget(self.chk_pad_169)
        crop_lay.addStretch(1)
        param_lay.addLayout(crop_lay, r, 0, 1, 2); r += 1
        
        self.chk_start_time = QCheckBox()
        self.time_start = QTimeEdit()
        self.time_start.setDisplayFormat("HH:mm:ss")
        self.time_start.setEnabled(False)
        self.chk_start_time.toggled.connect(self.time_start.setEnabled)

        self.chk_end_time = QCheckBox()
        self.time_end = QTimeEdit()
        self.time_end.setDisplayFormat("HH:mm:ss")
        self.time_end.setEnabled(False)
        self.chk_end_time.toggled.connect(self.time_end.setEnabled)

        time_lay = QHBoxLayout()
        time_lay.addWidget(self.chk_start_time)
        time_lay.addWidget(self.time_start, 1)
        time_lay.addWidget(self.chk_end_time)
        time_lay.addWidget(self.time_end, 1)
        main_lay.addLayout(time_lay)

        content_lay.addWidget(self.param_grp)

        btn_lay = QHBoxLayout()
        self.btn_start = QPushButton()
        self.btn_start.setMinimumHeight(36)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton()
        self.btn_stop.setMinimumHeight(36)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        btn_lay.addWidget(self.btn_start)
        btn_lay.addWidget(self.btn_stop)
        main_lay.addLayout(btn_lay)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        main_lay.addWidget(self.progress)

        self.lbl_status = QLabel()
        main_lay.addWidget(self.lbl_status)

        content_lay.addStretch()

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(2000)
        log_font = QFont("Consolas", 9)
        self.log.setFont(log_font)
        fm = QFontMetrics(log_font)
        log_h = fm.lineSpacing() * 2 + 14
        self.log.setFixedHeight(56)
        main_lay.addWidget(self.log)

        self._update_ui_language()
        self._update_codec_options(self.cmb_vcodec.currentText())

        self._append_log("Stereo Video Converter ready.")
        
        if FFMPEG_EXE and FFMPEG_EXE != "ffmpeg":
            self._append_log(f"FFmpeg recognized: {FFMPEG_EXE}")
        else:
            self._append_log("[Warning] FFmpeg is not recognized. It will be downloaded automatically when you press Start.")

        if 'FFPROBE_EXE' in globals() and FFPROBE_EXE and FFPROBE_EXE != "ffprobe":
            self._append_log(f"FFprobe recognized: {FFPROBE_EXE}")
        else:
            self._append_log("[Warning] FFprobe is not recognized. It will be downloaded automatically when you press Start.")

        if not torch.cuda.is_available():
            self._append_log("[Warning] CUDA is not available. Running on CPU.")

        self.presets: dict = {}
        self._load_presets()
        self._refresh_preset_combo()

        self._load_config()
        
        self._update_auto_mode_state()
        if self.chk_auto_mode.isChecked():
            self._apply_auto_mode_values()

    def _tr(self, key: str) -> str:
        return TRANSLATIONS.get(CURRENT_LANG, TRANSLATIONS[_LAUNCHER_DEFAULT_LANGUAGE]).get(key, key)

    def _show_message_box(self, icon, title: str, text: str, buttons=QMessageBox.Ok):
        """Show a QMessageBox centered on this window's current geometry.

        QMessageBox's default centering can end up anchored to the top-left
        when the parent window has been maximized or moved, since it relies
        on stale geometry in some window managers. Centering it manually
        against the parent's live frameGeometry() avoids that.
        """
        box = QMessageBox(icon, title, text, buttons, self)
        box.setWindowModality(Qt.WindowModal)
        geo = box.frameGeometry()
        geo.moveCenter(self.frameGeometry().center())
        box.move(geo.topLeft())
        return box.exec()

    def _ema_value(self) -> float:
        idx = self.spin_ema.currentIndex()
        if 0 <= idx < len(self._EMA_LEVEL_VALUES):
            return self._EMA_LEVEL_VALUES[idx]
        return 0.0

    def _set_ema_value(self, value: float):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        best_idx = 0
        best_diff = abs(self._EMA_LEVEL_VALUES[0] - value)
        for i, v in enumerate(self._EMA_LEVEL_VALUES):
            diff = abs(v - value)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        self.spin_ema.setCurrentIndex(best_idx)

    def _update_ui_language(self):
        t = TRANSLATIONS.get(CURRENT_LANG, TRANSLATIONS[_LAUNCHER_DEFAULT_LANGUAGE])

        self.setWindowTitle(t["title"])
        self.file_grp.setTitle(t["file_grp"])
        self.lbl_input.setText(t["input_label"])
        self.edt_input.setPlaceholderText(t["input_placeholder"])
        self.btn_input.setText(t["btn_input"])
        self.btn_input_folder.setText(t.get("btn_input_folder", "Select Folder..."))
        self.lbl_depthmap_input.setText(t["depthmap_input_label"])
        self.edt_depthmap_input.setPlaceholderText(t["depthmap_input_placeholder"])
        self.btn_depthmap_input.setText(t["btn_depthmap_input"])
        self.btn_depthmap_clear.setText(t["btn_depthmap_clear"])
        self.lbl_output.setText(t["output_label"])
        self.edt_output.setPlaceholderText(t["output_placeholder"])
        self.btn_output.setText(t["btn_output"])

        self.preset_grp.setTitle(t["preset_grp"])
        self.edt_preset_name.setPlaceholderText(t["preset_name"])
        self.btn_preset_save.setText(t["preset_save"])
        self.btn_preset_load.setText(t["preset_load"])
        self.btn_preset_delete.setText(t["preset_delete"])

        self.param_grp.setTitle(t["param_grp"])
        self.lbl_format.setText(t["fmt_label"])
        self.lbl_div.setText(t["div_label"])
        self.lbl_conv.setText(t["conv_label"])
        self.lbl_edge.setText(t["edge_label"])
        self.lbl_ema.setText(t["ema_label"])
        self.lbl_depth_model.setText(t["depth_model_label"])
        self.lbl_input_size.setText(t["size_label"])
        self.chk_raw_depthmap.setText(t["chk_raw_depthmap"])
        self.chk_corrected_depthmap.setText(t["chk_corrected_depthmap"])

        self.chk_preserve.setText(t["chk_preserve"])
        self.chk_fp16.setText(t["chk_fp16"])
        self.chk_auto_mode.setText(t["chk_auto_mode"])

        self.lbl_vcodec.setText(t["vcodec_label"])
        self.lbl_preset.setText(t["preset_label"])
        
        codec = self.cmb_vcodec.currentText()
        self.lbl_quality.setText(t["quality_cq"] if "nvenc" in codec else t["quality_crf"])

        self.chk_resize.setText(t["chk_resize"])
        self.chk_hdr_norm.setText(t["chk_hdr_norm"])
        self.chk_auto_crop.setText(t["chk_auto_crop"])
        self.chk_pad_169.setText(t["chk_pad_169"])

        self.chk_start_time.setText(t["chk_start_time"])
        self.chk_end_time.setText(t["chk_end_time"])

        self.btn_start.setText(t["btn_start"])
        self.btn_stop.setText(t["btn_stop"])

        if self.worker is None or not self.worker.isRunning():
            cur = self.lbl_status.text()
            if cur in ["Idle", ""]:
                self.lbl_status.setText(t["status_idle"])

    def _update_auto_mode_state(self):
        auto_enabled = self.chk_auto_mode.isChecked()
        
        self.spin_edge.setEnabled(not auto_enabled)
        self.spin_ema.setEnabled(not auto_enabled)
        
        if auto_enabled:
            self._apply_auto_mode_values()
    
    def _apply_auto_mode_values(self):
        div = self.spin_div.value()
        model_key = self.cmb_depth_model.currentData() if hasattr(self, "cmb_depth_model") else None
        auto_edge, auto_ema = compute_auto_edge_ema(div, depth_model=model_key)
        self.spin_edge.setValue(auto_edge)
        self._set_ema_value(auto_ema)

    def _on_div_changed(self):
        if self.chk_auto_mode.isChecked():
            self._apply_auto_mode_values()

    def _round_input_size(self, val: int) -> int:
        model_key = self.cmb_depth_model.currentData() if hasattr(self, "cmb_depth_model") else "zipdepth"
        if model_key in ("vda_s", "vda_s_metric"):
            # VDA needs input size to be a multiple of 14 (ViT patch size).
            return max(70, int(round(val / 14.0) * 14))
        return max(64, int(round(val / 32.0) * 32))

    def _on_depth_model_changed(self, _index=None):
        """Swap the depth-resolution choices to match the selected backend
        (ZipDepth: multiples of 32, VDA: multiples of 14)."""
        model_key = self.cmb_depth_model.currentData() or "zipdepth"
        sizes = self._depth_model_input_sizes.get(model_key, self._depth_model_input_sizes["zipdepth"])
        default_size = self._depth_model_default_size.get(model_key, sizes[0])

        prev_text = self.cmb_input_size.currentText().strip()

        self.cmb_input_size.blockSignals(True)
        self.cmb_input_size.clear()
        self.cmb_input_size.addItems(sizes)

        if prev_text in sizes:
            self.cmb_input_size.setCurrentText(prev_text)
        else:
            try:
                prev_val = int(prev_text)
                closest = min(sizes, key=lambda s: abs(int(s) - prev_val))
                self.cmb_input_size.setCurrentText(closest)
            except ValueError:
                self.cmb_input_size.setCurrentText(default_size)
        self.cmb_input_size.blockSignals(False)

        if hasattr(self, "chk_auto_mode") and self.chk_auto_mode.isChecked():
            self._apply_auto_mode_values()

    def _update_codec_options(self, codec: str):
        self.cmb_preset.clear()
        if "nvenc" in codec:
            self.cmb_preset.addItems(["p1", "p2", "p3", "p4", "p5", "p6", "p7"])
            self.cmb_preset.setCurrentText("p4")
            self.lbl_quality.setText(self._tr("quality_cq"))
        else:
            self.cmb_preset.addItems(["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"])
            self.cmb_preset.setCurrentText("ultrafast")
            self.lbl_quality.setText(self._tr("quality_crf"))

    def _append_log(self, text: str):
        self.log.append(text)
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)

    def _collect_config(self) -> dict:
        return {
            "input_path": self.edt_input.text(),
            "depthmap_input_path": self.edt_depthmap_input.text(),
            "extract_depthmap": self.chk_raw_depthmap.isChecked() or self.chk_corrected_depthmap.isChecked(),
            "raw_depthmap": self.chk_raw_depthmap.isChecked(),
            "corrected_depthmap": self.chk_corrected_depthmap.isChecked(),
            "output_dir": self.edt_output.text(),
            "format": self.cmb_format.currentText(),
            "divergence": self.spin_div.value(),
            "convergence": self.spin_conv.value(),
            "edge_dilation": self.spin_edge.value(),
            "ema_decay": self._ema_value(),
            "depth_model": self.cmb_depth_model.currentData(),
            "input_size": self.cmb_input_size.currentText(),
            "preserve_border": self.chk_preserve.isChecked(),
            "invert_depth": True,
            "fp16": self.chk_fp16.isChecked(),
            "auto_mode": self.chk_auto_mode.isChecked(),
            "use_start_time": self.chk_start_time.isChecked(),
            "start_time_sec": self.time_start.time().hour() * 3600 + self.time_start.time().minute() * 60 + self.time_start.time().second(),
            "start_time_str": self.time_start.time().toString("HH:mm:ss"),
            "use_end_time": self.chk_end_time.isChecked(),
            "end_time_sec": self.time_end.time().hour() * 3600 + self.time_end.time().minute() * 60 + self.time_end.time().second(),
            "end_time_str": self.time_end.time().toString("HH:mm:ss"),
            "vcodec": self.cmb_vcodec.currentText(),
            "preset": self.cmb_preset.currentText(),
            "quality": self.spin_quality.value(),
            "resize": self.chk_resize.isChecked(),
            "resize_value": self.edt_resize.text().strip(),
            "hdr_norm": self.chk_hdr_norm.isChecked(),
            "auto_crop": self.chk_auto_crop.isChecked(),
            "pad_169": self.chk_pad_169.isChecked(),
        }

    def _apply_config(self, cfg: dict):
        self.edt_input.setText(cfg.get("input_path", ""))
        self.edt_depthmap_input.setText(cfg.get("depthmap_input_path", ""))
        self.chk_raw_depthmap.setChecked(
            cfg.get("raw_depthmap", cfg.get("extract_depthmap", False))
        )
        self.chk_corrected_depthmap.setChecked(cfg.get("corrected_depthmap", False))
        self.edt_output.setText(cfg.get("output_dir", ""))
        self.cmb_format.setCurrentText(cfg.get("format", "hsbs"))
        self.spin_div.setValue(cfg.get("divergence", 1.0))
        self.spin_conv.setValue(cfg.get("convergence", 0.5))
        self.spin_edge.setValue(cfg.get("edge_dilation", 2))
        self._set_ema_value(cfg.get("ema_decay", 0.0))

        _saved_depth_model = cfg.get("depth_model", "zipdepth")
        _dm_idx = self.cmb_depth_model.findData(_saved_depth_model)
        self.cmb_depth_model.blockSignals(True)
        self.cmb_depth_model.setCurrentIndex(_dm_idx if _dm_idx >= 0 else 0)
        self.cmb_depth_model.blockSignals(False)
        self._on_depth_model_changed()

        try:
            _loaded_isz = max(64, min(630, int(cfg.get("input_size", 384))))
        except (TypeError, ValueError):
            _loaded_isz = 384
        self.cmb_input_size.setCurrentText(str(_loaded_isz))
        self.chk_preserve.setChecked(cfg.get("preserve_border", True))
        self.chk_fp16.setChecked(cfg.get("fp16", True))
        self.chk_auto_mode.setChecked(cfg.get("auto_mode", True))
        
        self.chk_start_time.setChecked(cfg.get("use_start_time", False))
        t_s = QTime.fromString(cfg.get("start_time_str", "00:00:00"), "HH:mm:ss")
        if t_s.isValid(): self.time_start.setTime(t_s)

        self.chk_end_time.setChecked(cfg.get("use_end_time", False))
        t_e = QTime.fromString(cfg.get("end_time_str", "00:00:00"), "HH:mm:ss")
        if t_e.isValid(): self.time_end.setTime(t_e)
        
        codec = cfg.get("vcodec", "libx265")
        self.cmb_vcodec.setCurrentText(codec)
        self._update_codec_options(codec)
        self.cmb_preset.setCurrentText(cfg.get("preset", "ultrafast"))
        self.spin_quality.setValue(cfg.get("quality", 16))
        self.edt_resize.setText(cfg.get("resize_value", "1920x1080") or "1920x1080")
        self.chk_resize.setChecked(cfg.get("resize", False))
        self.edt_resize.setEnabled(self.chk_resize.isChecked())
        self.chk_hdr_norm.setChecked(cfg.get("hdr_norm", False))
        self.chk_auto_crop.setChecked(cfg.get("auto_crop", False))
        self.chk_pad_169.setChecked(cfg.get("pad_169", False))

        self._update_ui_language()

    def _save_config(self):
        try:
            CONFIG_FILE.write_text(json.dumps(self._collect_config(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self._append_log(f"[Warning] Failed to save config: {e}")

    def _load_config(self):
        if not CONFIG_FILE.exists(): return
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            self._apply_config(cfg)
            self._append_log("Loaded previous settings.")
        except Exception as e:
            self._append_log(f"[Warning] Failed to load config: {e}")

    def _load_presets(self):
        self.presets = {}
        if PRESET_FILE.exists():
            try:
                data = json.loads(PRESET_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.presets = data
            except Exception:
                self.presets = {}

    def _save_presets(self):
        try:
            PRESET_FILE.write_text(json.dumps(self.presets, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self._append_log(f"[Warning] Failed to save preset: {e}")

    def _refresh_preset_combo(self):
        current = self.cmb_preset_name.currentText()
        self.cmb_preset_name.blockSignals(True)
        self.cmb_preset_name.clear()
        self.cmb_preset_name.addItems(sorted(self.presets.keys()))
        if current in self.presets:
            self.cmb_preset_name.setCurrentText(current)
        self.cmb_preset_name.blockSignals(False)

    def _on_preset_selected(self, index: int):
        name = self.cmb_preset_name.currentText()
        if name:
            self.edt_preset_name.setText(name)

    def _preset_save(self):
        name = self.edt_preset_name.text().strip() or self.cmb_preset_name.currentText().strip()
        if not name:
            return
        self.presets[name] = {
            "divergence": self.spin_div.value(),
            "convergence": self.spin_conv.value(),
            "edge_dilation": self.spin_edge.value(),
            "ema_decay": self._ema_value(),
        }
        self._save_presets()
        self._refresh_preset_combo()
        self.cmb_preset_name.setCurrentText(name)
        self._append_log(self._tr("preset_saved") + f" : {name}")

    def _preset_load(self):
        if self.chk_auto_mode.isChecked():
            self._show_message_box(
                QMessageBox.Warning,
                self._tr("notice_title"),
                self._tr("notice_preset_load_auto_mode")
            )
            return
        name = self.cmb_preset_name.currentText()
        if name not in self.presets:
            return
        p = self.presets[name]
        self.spin_div.setValue(p.get("divergence", self.spin_div.value()))
        self.spin_conv.setValue(p.get("convergence", self.spin_conv.value()))
        self.spin_edge.setValue(p.get("edge_dilation", self.spin_edge.value()))
        self._set_ema_value(p.get("ema_decay", self._ema_value()))
        self.edt_preset_name.setText(name)
        self._append_log(self._tr("preset_loaded") + f" : {name}")

    def _preset_delete(self):
        name = self.cmb_preset_name.currentText()
        if name in self.presets:
            del self.presets[name]
            self._save_presets()
            self._refresh_preset_combo()
            self._append_log(self._tr("preset_deleted") + f" : {name}")

    def _select_input(self):
        filters = ("Video Files (*.mp4 *.mkv *.avi *.ts *.mov *.wmv *.flv *.webm *.m4v *.mpg *.mpeg *.m2ts);;All Files (*)")
        path, _ = QFileDialog.getOpenFileName(self, self._tr("dialog_input_title"), "", filters)
        if path: self.edt_input.setText(path)

    def _select_depthmap_input(self):
        filters = ("Video Files (*.mp4 *.mkv *.avi *.ts *.mov *.wmv *.flv *.webm *.m4v *.mpg *.mpeg *.m2ts);;All Files (*)")
        path, _ = QFileDialog.getOpenFileName(self, self._tr("dialog_input_title"), "", filters)
        if path: self.edt_depthmap_input.setText(path)

    def _clear_depthmap_input(self):
        self.edt_depthmap_input.setText("")

    def _select_output(self):
        path = QFileDialog.getExistingDirectory(self, self._tr("dialog_output_title"))
        if path: self.edt_output.setText(path)

    def _select_input_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if path: self.edt_input.setText(path)

    def _validate(self) -> str | None:
        if not self.edt_input.text().strip(): return self._tr("err_no_input")
        if not self.edt_output.text().strip(): return self._tr("err_no_output")
        p = Path(self.edt_input.text().strip())
        if not p.exists(): return self._tr("err_file_not_found").format(p)
        
        if p.is_file():
            if p.suffix.lower() not in VIDEO_EXTS: return self._tr("err_unsupported_ext").format(p.suffix)
        elif p.is_dir():
            videos = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
            if not videos: return self._tr("err_no_videos_in_folder")
            if self.edt_depthmap_input.text().strip():
                return self._tr("err_depthmap_folder_conflict")
            
        try:
            isz = int(self.cmb_input_size.currentText().strip())
            if isz < 64 or isz > 630: return self._tr("err_invalid_size_range")
        except ValueError: return self._tr("err_invalid_size_num")
        if self.chk_resize.isChecked():
            rv = self.edt_resize.text().strip().lower()
            parts = rv.split("x")
            if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
                return self._tr("err_invalid_resize")
        return None

    def _start(self):
        err = self._validate()
        if err:
            self._show_message_box(QMessageBox.Warning, self._tr("err_title"), err)
            return

        if FFMPEG_MISSING:
            if not ensure_ffmpeg_available(self):
                return

        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    self._show_message_box(QMessageBox.Information, self._tr("notice_title"), self._tr("notice_already_running"))
                    return
            except RuntimeError:
                self.worker = None

        try:
            isz = int(self.cmb_input_size.currentText().strip())
            isz = max(64, min(630, self._round_input_size(isz)))
            self.cmb_input_size.setCurrentText(str(isz))
        except Exception:
            isz = 384

        self._save_config()

        self.current_params = self._collect_config()
        self.current_params["output_dir"] = Path(self.current_params["output_dir"])
        self.current_params["input_size"] = isz

        input_path = Path(self.current_params["input_path"])
        self.batch_queue = []
        
        if input_path.is_file():
            self.batch_queue.append(input_path)
        elif input_path.is_dir():
            videos = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
            videos.sort(key=lambda x: x.name)
            self.batch_queue.extend(videos)

        self.batch_total = len(self.batch_queue)
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._append_log("=" * 50)
        self._append_log(f"Starting batch conversion. Total files: {self.batch_total}")
        
        self._process_next_in_batch()

    def _process_next_in_batch(self):
        if not self.batch_queue:
            self.progress.setValue(100)
            self.lbl_status.setText(self._tr("status_done"))
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._append_log("All batch conversions completed.")
            self._show_message_box(QMessageBox.Information, self._tr("done_title"), self._tr("done_msg").format("All files processed."))
            return

        next_file = self.batch_queue.pop(0)
        current_index = self.batch_total - len(self.batch_queue)
        
        batch_msg = self._tr("batch_progress").format(current_index, self.batch_total, next_file.name)
        self.lbl_status.setText(batch_msg)
        self._append_log(f"\n[{current_index}/{self.batch_total}] Processing: {next_file.name}")
        self.progress.setValue(0)

        params = self.current_params.copy()
        params["input_path"] = next_file

        use_start = params.get("use_start_time", False)
        start_sec = params.get("start_time_sec", 0.0)
        use_end = params.get("use_end_time", False)
        end_sec = params.get("end_time_sec", 0.0)

        if use_start or use_end:
            cap = cv2.VideoCapture(str(params["input_path"]))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            duration = total_frames / fps if fps > 0 else 0.0

            if duration > 0:
                if use_start and start_sec > duration:
                    self._append_log(f"[Skip] {self._tr('time_err_start_exceed').format(start_sec, duration)}")
                    self._process_next_in_batch()
                    return
                if use_end and end_sec > duration:
                    self._append_log(f"[Skip] {self._tr('time_err_end_exceed').format(end_sec, duration)}")
                    self._process_next_in_batch()
                    return
            if use_start and use_end and start_sec >= end_sec:
                self._append_log(f"[Skip] {self._tr('time_err_order')}")
                self._process_next_in_batch()
                return

        if self.worker is not None:
            self.worker.wait()

        self.worker = ConvertWorker(params)
        _w = self.worker
        _w.finished.connect(_w.deleteLater)
        _w.finished.connect(lambda: setattr(self, 'worker', None) if self.worker is _w else None)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._append_log)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished_error.connect(self._on_finished_error)
        self.worker.start()

    def _stop(self):
        self.batch_queue.clear()
        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    self._append_log("Requesting stop...")
                    self.worker.stop()
            except RuntimeError:
                self.worker = None
            self.btn_stop.setEnabled(False)

    def _on_progress(self, current: int, total: int, msg: str):
        if total > 0:
            pct = int(current / total * 100)
            self.progress.setValue(min(100, pct))

    def _on_finished_ok(self, path: str):
        self._append_log(f"Conversion completed: {path}")
        if self.worker is not None:
            self.worker.wait()
        self._process_next_in_batch()

    def _on_finished_error(self, err_msg):
        if self.worker is not None:
            self.worker.wait()
        if hasattr(self, 'batch_queue') and len(self.batch_queue) > 0:
            self._append_log(f"[Skip] Skipping file due to an error. Reason: {err_msg}")
            
            self._process_next_in_batch()
            
        else:
            self._show_message_box(QMessageBox.Critical, self._tr("err_dialog_title"), f"An error occurred during conversion:\n{err_msg}")
            
            self.lbl_status.setText(self._tr("status_error"))
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def closeEvent(self, event):
        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    self.worker.stop()
                    self.worker.wait(3000)
            except RuntimeError:
                self.worker = None
        self._save_config()
        event.accept()

def _build_splash_pixmap() -> "QPixmap":
    """Simple placeholder splash image (no external image file required)."""
    pix = QPixmap(420, 220)
    pix.fill(QColor("#1e1e1e"))
    painter = QPainter(pix)
    try:
        painter.setPen(QColor("#ffffff"))
        font = painter.font()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pix.rect(), Qt.AlignCenter, "Conversion3D\nLoading...")
    finally:
        painter.end()
    return pix


def main():
    app = QApplication(sys.argv)

    splash = QSplashScreen(_build_splash_pixmap())
    splash.show()
    app.processEvents()

    state = {"gui": None}

    def _init_and_show():
        gui = StereoVideoGUI()
        state["gui"] = gui
        gui.show()
        splash.finish(gui)

    # Defer the heavy GUI construction until after the event loop starts,
    # so the OS sees the app pumping messages immediately instead of
    # showing a "busy" (hourglass) cursor while widgets are being built.
    QTimer.singleShot(0, _init_and_show)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
