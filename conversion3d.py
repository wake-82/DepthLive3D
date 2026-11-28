
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
import math
import time
import uuid
import queue
import shutil
import tempfile
import subprocess
import threading
import sys
import gc
import re
import zipfile
import urllib.request
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Dynamo diagnostics (conversion3d only): print recompiles and graph breaks to the
# console so it is visible whether torch.compile is really being used.
# Set to False to silence.
TORCH_COMPILE_DEBUG_LOGS = True
if TORCH_COMPILE_DEBUG_LOGS:
    try:
        torch._logging.set_logs(recompiles=True, graph_breaks=True)
    except Exception as _log_err:
        print(f"[torch.compile] Could not enable Dynamo logs: {_log_err}", flush=True)

cv2.setNumThreads(_CPU_THREADS)

torch.set_num_threads(_CPU_THREADS)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    from PySide6.QtCore import Qt, QThread, Signal as pyqtSignal, QTimer, QTime, QEventLoop
    from PySide6.QtGui import QFont, QFontMetrics, QPixmap, QPainter, QColor, QIntValidator, QValidator
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

VDA_ROOT = BASE_DIR / "Video-Depth-Anything"
if VDA_ROOT.exists():
    sys.path.insert(0, str(VDA_ROOT))

DISTILL_ROOT = BASE_DIR / "Depth-Anything-V2"
if DISTILL_ROOT.exists():
    sys.path.insert(0, str(DISTILL_ROOT))

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

    t = TRANSLATIONS.get(CURRENT_LANG, TRANSLATIONS[_LAUNCHER_DEFAULT_LANGUAGE])

    proceed = QMessageBox.question(
        parent,
        t["ffmpeg_license_title"],
        t["ffmpeg_license_msg"],
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes,
    )
    if proceed != QMessageBox.Yes:
        QMessageBox.warning(
            parent,
            t["ffmpeg_license_title"],
            t["ffmpeg_missing_msg"].format(dir=BASE_DIR),
        )
        return False

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

EMA_MAX = 0.60

def clamp_ema_value(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.00
    if v < 0.0:
        return 0.00
    if v > EMA_MAX:
        return EMA_MAX
    return v

# ---------------------------------------------------------------------------
# Auto Mode EMA table. Each row is (min strength, EMA decay): the LAST row whose min strength <= the slider
# value is used (below the first non-zero row -> OFF 0.00).
# Levels: OFF 0.00 / Low 0.30 / Medium 0.40 / High 0.50 / Ultra 0.60.
# The EMA value does NOT depend on Depth Linear Compression (ON uses exactly the same rows as OFF).
# ---------------------------------------------------------------------------
_EMA_ROWS_STD = [(0.0, 0.00), (1.5, 0.30), (2.0, 0.40), (2.5, 0.50), (3.0, 0.60)]
# VDA models (VDA-S, VDA-S Metric): flicker reduction (EMA) is OFF at every strength in Auto Mode.
_EMA_ROWS_VDA = [(0.0, 0.00)]
_EMA_ROWS_VDA_METRIC = [(0.0, 0.00)]
# Other models: previous curve, unchanged (old rule was "strength <= max", hence the +1e-6).
_EMA_ROWS_DEFAULT = [(0.0, 0.00), (0.5 + 1e-6, 0.30), (1.0 + 1e-6, 0.40), (1.5 + 1e-6, 0.50), (2.5 + 1e-6, 0.60)]

_AUTO_EMA_TABLE = {
    "zipdepth":      _EMA_ROWS_STD,
    "dav2_s":        _EMA_ROWS_STD,
    "distill_any_s": _EMA_ROWS_STD,
    "da3_mono_l":    _EMA_ROWS_STD,
    "vda_s":         _EMA_ROWS_VDA,
    "vda_s_metric":  _EMA_ROWS_VDA_METRIC,
    "default":       _EMA_ROWS_DEFAULT,
}

def auto_ema_from_table(depth_model, divergence: float, comp_on: bool = False) -> float:
    # comp_on is accepted for compatibility but ignored: same EMA with Depth Linear Compression ON or OFF.
    rows = _AUTO_EMA_TABLE.get(depth_model) or _AUTO_EMA_TABLE["default"]
    d = float(divergence)
    ema = rows[0][1]
    for min_div, val in rows:
        if d >= min_div - 1e-9:
            ema = val
        else:
            break
    return ema

# ---------------------------------------------------------------------------
# Auto Mode edge fix table (measured 'recommended edge fix' of the Edge Fix Meter).
# Key: warp/fill method -> depth model -> (compression OFF, compression ON); each list holds the edge fix for
# strength 0.5, 1.0, 1.5 ... 5.0 (10 steps). A strength in between uses the next step up (ceil to 0.5).
# Depth Linear Compression ON only applies to models in DEPTH_LINEAR_COMP_PARAMS. EMA is NOT affected.
# Combinations missing from the table fall back to the previous rule (_compute_auto_edge_ema_legacy).
# ---------------------------------------------------------------------------
_AUTO_EDGE_TABLE = {
    "stretch": {
        "dav2_s":        ([ 3,  5,  7,  9, 12, 14, 16, 18, 21, 22],
                          [ 3,  5,  6,  7,  9, 10, 11, 13, 14, 15]),
        "distill_any_s": ([ 3,  5,  7, 10, 12, 14, 16, 19, 21, 22],
                          [ 3,  5,  6,  7,  9, 10, 11, 13, 14, 15]),
        "vda_s":         ([ 3,  5,  7,  9, 12, 14, 16, 18, 20, 22],
                          [ 3,  5,  6,  7,  7,  8,  9, 10, 11, 11]),
        "zipdepth":      ([ 5,  7,  9, 11, 13, 15, 17, 19, 21, 22],
                          [ 5,  7,  9, 11, 13, 15, 16, 17, 18, 19]),
        "vda_s_metric":  ([ 3,  5,  8, 10, 12, 15, 17, 19, 21, 22],
                          [ 3,  5,  7,  8,  9, 10, 11, 13, 14, 15]),
    },
    "fast": {
        "distill_any_s": ([ 2,  3,  5,  7,  8, 10, 12, 13, 15, 16],
                          [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11]),
        "dav2_s":        ([ 2,  4,  5,  7,  8, 10, 11, 13, 15, 16],
                          [ 2,  4,  4,  5,  6,  7,  8,  9, 10, 11]),
        "vda_s":         ([ 2,  4,  5,  7,  8, 10, 11, 13, 14, 16],
                          [ 2,  4,  4,  5,  5,  6,  6,  7,  8,  8]),
        "zipdepth":      ([ 4,  6,  7,  9, 10, 12, 13, 14, 16, 17],
                          [ 4,  6,  7,  9, 10, 12, 12, 13, 14, 14]),
        "vda_s_metric":  ([ 3,  4,  6,  7,  9, 10, 12, 14, 15, 17],
                          [ 3,  4,  5,  6,  7,  7,  8,  9, 10, 11]),
    },
    "medium": {
        "dav2_s":        ([ 1,  2,  3,  4,  5,  6,  7,  8, 10, 11],
                          [ 1,  2,  3,  4,  4,  5,  5,  6,  6,  7]),
        "distill_any_s": ([ 1,  2,  3,  4,  5,  7,  8,  9, 10, 11],
                          [ 1,  2,  3,  3,  4,  5,  5,  6,  6,  7]),
        "vda_s":         ([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
                          [ 2,  3,  3,  3,  4,  4,  4,  5,  5,  5]),
        "zipdepth":      ([ 4,  5,  6,  7,  8,  8,  9, 10, 11, 12],
                          [ 4,  5,  6,  7,  8,  8,  9,  9, 10, 10]),
        "vda_s_metric":  ([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
                          [ 2,  3,  4,  4,  5,  5,  6,  6,  7,  7]),
    },
    "light": {
        "vda_s":         ([ 3,  5,  7,  9, 11, 13, 15, 17, 19, 21],
                          [ 3,  5,  6,  6,  7,  8,  9,  9, 10, 11]),
        "dav2_s":        ([ 3,  5,  7,  9, 11, 13, 15, 17, 20, 22],
                          [ 3,  5,  6,  7,  8, 10, 11, 12, 13, 14]),
        "distill_any_s": ([ 2,  5,  7,  9, 11, 13, 16, 18, 20, 22],
                          [ 2,  5,  6,  7,  8, 10, 11, 12, 13, 15]),
        "zipdepth":      ([ 5,  7,  9, 11, 13, 15, 17, 19, 20, 22],
                          [ 5,  7,  9, 11, 13, 15, 16, 17, 18, 19]),
        "vda_s_metric":  ([ 3,  5,  7, 10, 12, 14, 16, 18, 21, 22],
                          [ 3,  5,  6,  7,  9, 10, 11, 12, 13, 14]),
    },
    "light_stream": {
        "distill_any_s": ([ 1,  1,  2,  2,  3,  3,  3,  4,  4,  5],
                          [ 1,  1,  2,  2,  2,  2,  2,  3,  3,  3]),
        "dav2_s":        ([ 1,  1,  2,  2,  3,  3,  4,  4,  4,  5],
                          [ 1,  1,  2,  2,  2,  2,  3,  3,  3,  3]),
        "vda_s":         ([ 1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                          [ 1,  1,  2,  2,  2,  2,  2,  2,  3,  3]),
        "zipdepth":      ([ 3,  4,  4,  5,  5,  5,  6,  6,  7,  7],
                          [ 3,  4,  4,  5,  5,  5,  6,  6,  6,  6]),
        "vda_s_metric":  ([ 0,  2,  2,  3,  3,  4,  4,  5,  5,  5],
                          [ 0,  2,  2,  2,  2,  3,  3,  4,  4,  4]),
    },
}
_AUTO_EDGE_MODE_ALIAS = {'stretch': 'stretch', 'fast': 'fast', 'medium': 'medium', 'light': 'light', 'light_stream': 'light_stream'}

# ---------------------------------------------------------------------------
# Auto Mode edge fix MEASUREMENT (runs inside this program right before the conversion / live stream starts).
# The embedded reference image is run through the SAME depth model / input size / frame size that will be used,
# the edge-fix dilation is measured for every level 0..EDGE_FIX_MAX, and for each 3D strength the level whose
# ratio (expansion width / disocclusion gap) is closest to the method's target ratio is recommended.
# The measurement does not depend on strength or on Depth Linear Compression (they only change the gap), so ONE
# measurement gives the recommendations for every strength and for compression OFF and ON.
# To tune a method later, change only its number in EDGEFIX_METHOD_TARGET_RATIO.
# If the measurement fails, the built-in table (_AUTO_EDGE_TABLE) is used.
# ---------------------------------------------------------------------------
EDGEFIX_AUTO_MEASURE = True
EDGEFIX_METHOD_TARGET_RATIO = {
    "fast":         0.50,   # Fast Inpaint
    "light_stream": 0.50,   # Light Inpaint (stream)
    "light":        0.50,   # Light Inpaint
    "stretch":      1.00,   # Method Normal (stretch)
    "medium":       0.50,   # Medium Inpaint
}
_EDGEFIX_STRENGTH_STEPS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
_EDGEFIX_THRESHOLDS = (0.3, 0.5, 0.7)
_EDGEFIX_MIN_BOUNDARY_PX = 50
_EDGEFIX_BAND_IN = (4, 10)
_EDGEFIX_PROFILE = None          # current measured profile: {"model", "base", "tables", "key"}
_EDGEFIX_PROFILE_CACHE = {}      # key (model, input size, depth hw, frame hw) -> base (same condition = no re-measure)
_EDGEFIX_CROSS = None

def edgefix_reset_profile():
    global _EDGEFIX_PROFILE
    _EDGEFIX_PROFILE = None

def edgefix_profile_ready(depth_model=None) -> bool:
    p = _EDGEFIX_PROFILE
    return bool(p) and (depth_model is None or p.get("model") == depth_model)

def _edgefix_reference_bgr(width: int, height: int) -> np.ndarray:
    """Embedded reference image, centre-cropped to the target aspect ratio and resized to width x height."""
    import base64
    arr = np.frombuffer(base64.b64decode(_EDGEFIX_REF_B64), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("embedded edge fix reference image could not be decoded")
    h, w = img.shape[:2]
    want = float(width) / float(height)
    if w / h > want:
        nw = max(1, int(round(h * want)))
        x0 = (w - nw) // 2
        img = img[:, x0:x0 + nw]
    else:
        nh = max(1, int(round(w / want)))
        y0 = (h - nh) // 2
        img = img[y0:y0 + nh, :]
    interp = cv2.INTER_AREA if (img.shape[1] > width or img.shape[0] > height) else cv2.INTER_CUBIC
    return np.ascontiguousarray(cv2.resize(img, (int(width), int(height)), interpolation=interp))

def _edgefix_measure_pair(inv0: np.ndarray, invl: np.ndarray):
    """(expansion width px, depth step) of one dilation level, averaged over the thresholds."""
    global _EDGEFIX_CROSS
    if _EDGEFIX_CROSS is None:
        _EDGEFIX_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    expands, steps = [], []
    for t in _EDGEFIX_THRESHOLDS:
        m0 = inv0 > t
        ml = invl > t
        m0u = m0.astype(np.uint8)
        boundary = m0 & ~cv2.erode(m0u, _EDGEFIX_CROSS).astype(bool)
        pcount = int(boundary.sum())
        if pcount < _EDGEFIX_MIN_BOUNDARY_PX or m0.all() or (~m0).all():
            continue
        expands.append((int(ml.sum()) - int(m0.sum())) / pcount)
        dist_out = cv2.distanceTransform((~m0).astype(np.uint8), cv2.DIST_L2, 3)
        dist_in = cv2.distanceTransform(m0u, cv2.DIST_L2, 3)
        band_in = m0 & (dist_in >= _EDGEFIX_BAND_IN[0]) & (dist_in <= _EDGEFIX_BAND_IN[1])
        band_out = (~m0) & (dist_out >= _EDGEFIX_BAND_IN[0]) & (dist_out <= _EDGEFIX_BAND_IN[1])
        if band_in.any() and band_out.any():
            steps.append(float(inv0[band_in].mean() - inv0[band_out].mean()))
    expand = float(np.mean(expands)) if expands else float("nan")
    step = abs(float(np.mean(steps))) if steps else float("nan")
    return expand, step

def edgefix_measure_profile(predictor, depth_model, depth_hw, frame_hw, input_size, invert_depth=True, log=None):
    """Measure with the loaded depth predictor and make the result the current profile.
    depth_hw = (h, w) fed to the depth model, frame_hw = (h, w) of the frame the dilation is applied to."""
    global _EDGEFIX_PROFILE
    depth_hw = (int(depth_hw[0]), int(depth_hw[1]))
    frame_hw = (int(frame_hw[0]), int(frame_hw[1]))
    key = (str(depth_model), int(input_size), depth_hw, frame_hw)
    base = _EDGEFIX_PROFILE_CACHE.get(key)
    cached = base is not None
    if base is None:
        ref = _edgefix_reference_bgr(depth_hw[1], depth_hw[0])
        fh, fw = frame_hw
        with torch.inference_mode():
            raw = estimate_depth_raw(predictor, ref)
            if invert_depth:
                raw = raw.max() - raw
            d = upsample_depth(raw.float(), fh, fw)
            norm, _, _ = normalize_depth_gpu(d, decay=0.0)
            dh, dw = compute_dilation_target_dims(fh, fw)
            inv0 = 1.0 - upsample_depth(norm, dh, dw)
        base = {}
        for level in range(0, EDGE_FIX_MAX + 1):
            with torch.inference_mode():
                if level > 0:
                    xy = edge_fix_level_to_xy(level, depth_model=depth_model)
                    invl = dilate_edge(inv0.clone(), xy)
                else:
                    invl = inv0
                a0 = inv0[0, 0].float().cpu().numpy()
                al = invl[0, 0].float().cpu().numpy()
            expand, step = _edgefix_measure_pair(a0, al)
            base[level] = (expand, step, float(dw))
        _reset = getattr(predictor, "reset_state", None)      # the streaming (VDA) state must not keep the reference
        if callable(_reset):
            try:
                _reset()
            except Exception:
                pass
        _EDGEFIX_PROFILE_CACHE[key] = base
    _EDGEFIX_PROFILE = {"model": str(depth_model), "base": base, "tables": {}, "key": key}
    if log is not None:
        log(f"[Auto Edge Fix] {'cached' if cached else 'measured'}: {depth_model}, depth {depth_hw[1]}x{depth_hw[0]}, "
            f"frame {frame_hw[1]}x{frame_hw[0]}, input {int(input_size)}")
    return _EDGEFIX_PROFILE

def _edgefix_recommend_level(base: dict, div: float, k: float, target: float):
    best = None
    for level in sorted(base):
        expand, step, w = base[level]
        if not (expand == expand and step == step):
            continue
        gap = div * 0.005 * w * step * k
        if gap <= 1e-9:
            continue
        err = abs(expand / gap - target)
        if best is None or err < best[0]:
            best = (err, level)          # on a tie the smaller level is kept
    return None if best is None else best[1]

def _edgefix_profile_edge(depth_model, warp_fill_mode, comp_on: bool, divergence: float):
    """Measured recommendation for (model, method, compression, strength), or None when nothing is measured."""
    prof = _EDGEFIX_PROFILE
    if not prof or prof.get("model") != depth_model:
        return None
    mk = _AUTO_EDGE_MODE_ALIAS.get(warp_fill_mode)
    target = EDGEFIX_METHOD_TARGET_RATIO.get(mk)
    if target is None:
        return None
    tkey = (mk, bool(comp_on))
    table = prof["tables"].get(tkey)
    if table is None:
        table, run_max = [], 0
        for s in _EDGEFIX_STRENGTH_STEPS:
            kk = (depth_linear_comp_width(s, depth_model) if comp_on else 1.0) * depth_base_scale(depth_model)
            lv = _edgefix_recommend_level(prof["base"], s, kk, float(target))
            if lv is None:
                table = None
                break
            run_max = max(run_max, int(lv))      # a stronger 3D strength never gets a smaller edge fix
            table.append(run_max)
        prof["tables"][tkey] = table if table is not None else False
    if not table:
        return None
    idx = int(math.ceil(float(divergence) / 0.5 - 1e-9)) - 1
    idx = max(0, min(len(table) - 1, idx))
    return max(0, min(EDGE_FIX_MAX, int(table[idx])))

def auto_edge_from_table(depth_model, warp_fill_mode, comp_on: bool, divergence: float):
    """Edge fix from _AUTO_EDGE_TABLE, or None when this model / method is not in the table."""
    mk = _AUTO_EDGE_MODE_ALIAS.get(warp_fill_mode)
    _tm = "vda_s" if depth_model == "da3_mono_l" else depth_model   # DA3-Mono-L is strength-matched to VDA-S
    row = (_AUTO_EDGE_TABLE.get(mk) or {}).get(_tm)
    if row is None:
        return None
    vals = row[1] if comp_on else row[0]
    idx = int(math.ceil(float(divergence) / 0.5 - 1e-9)) - 1
    idx = max(0, min(len(vals) - 1, idx))
    return max(0, min(EDGE_FIX_MAX, int(vals[idx])))

def compute_auto_edge_ema(divergence: float, depth_model: str | None = None, warp_fill_mode: str | None = None,
                         depth_linear_comp: bool = False) -> tuple[int, float]:
    edge, ema = _compute_auto_edge_ema_legacy(divergence, depth_model=depth_model, warp_fill_mode=warp_fill_mode,
                                              depth_linear_comp=depth_linear_comp)
    _comp_on = bool(depth_linear_comp and depth_model in DEPTH_LINEAR_COMP_PARAMS
                    and depth_model not in DEPTH_LINEAR_COMP_NO_EDGE_OVERRIDE)
    _t = None
    if EDGEFIX_AUTO_MEASURE:
        _t = _edgefix_profile_edge(depth_model, warp_fill_mode, _comp_on, divergence)   # measured right before start
    if _t is None:
        _t = auto_edge_from_table(depth_model, warp_fill_mode, _comp_on, divergence)    # built-in table (fallback)
    if _t is not None:
        edge = _t          # EMA stays the value of the previous rule (_AUTO_EMA_TABLE)
    return edge, ema


def _compute_auto_edge_ema_legacy(divergence: float, depth_model: str | None = None, warp_fill_mode: str | None = None,
                         depth_linear_comp: bool = False) -> tuple[int, float]:
    div = float(divergence)
    _orig_model = depth_model
    _comp_edge_on = False
    _orig_warp_mode = warp_fill_mode   # ORIGINAL warp mode (the compression override below replaces it)
    # Light Inpaint (NOT stream) is decided from the ORIGINAL warp mode (the compression override below replaces it).
    _light_nonstream = (warp_fill_mode == WARP_FILL_MODE_LIGHT)
    # Depth Linear Compression ON (models that support it): use the Light Inpaint (stream) edge values,
    # whatever warp/fill method is selected, because the compressed depth map needs far less edge fix.
    if (depth_linear_comp and depth_model in DEPTH_LINEAR_COMP_PARAMS
            and depth_model not in DEPTH_LINEAR_COMP_NO_EDGE_OVERRIDE):
        warp_fill_mode = WARP_FILL_MODE_LIGHT_STREAM
        # Edge fix then follows the Auto Mode (inpaint) rule applied to the EFFECTIVE strength, see
        # depth_linear_comp_auto_edge. EMA is the same as with compression OFF (_AUTO_EMA_TABLE).
        _comp_edge_on = True
    is_fast_fill = warp_fill_mode in (WARP_FILL_MODE_FAST, WARP_FILL_MODE_MEDIUM, WARP_FILL_MODE_LIGHT, WARP_FILL_MODE_LIGHT_STREAM)

    def _apply_fast_fill_offset(edge: int, curve: str = "generic") -> int:
        # Auto Mode + Light Inpaint (not stream), every model EXCEPT VDA-S Metric (unchanged):
        #   Depth Linear Compression OFF: +1 per 0.5 strength (0.5->1, 1.0->2 ... 5.0->10)
        #   Depth Linear Compression ON : +1 per 1.0 strength (0.5->1, 1.0->1, 1.5->2 ... 5.0->5), slider value (not effective)
        # Auto Mode + Normal (stretch) + Depth Linear Compression ON: edge fix = (compression-OFF value) / 2 + 1
        # (compression-OFF Normal value = the base curve value passed in), for every model that supports the compression.
        if _comp_edge_on and _orig_warp_mode == WARP_FILL_MODE_STRETCH:
            return max(1, min(EDGE_FIX_MAX, int(edge) // 2 + 1))
        if _light_nonstream and _orig_model != "vda_s_metric":
            _unit = 1.0 if _comp_edge_on else 0.5
            return max(1, min(EDGE_FIX_MAX, int(math.ceil(div / _unit - 1e-9))))
        if _comp_edge_on:
            return depth_linear_comp_auto_edge(div, _orig_model)
        if is_fast_fill:
            # Inpaint modes: edge fix +1 per 1.0 strength (VDA-S Metric: +1 per 0.5 strength)
            if curve == "metric":
                edge = max(1, int(math.ceil(div / 0.5 - 1e-9)))
            else:
                edge = max(1, int(math.ceil(div - 1e-9)))
        return max(0, min(EDGE_FIX_MAX, edge))

    if depth_model == "vda_s":
        if div <= 0.5 + 1e-9:
            edge = 2
        elif div <= 1.0 + 1e-9:
            edge = 4
        elif div <= 1.5 + 1e-9:
            edge = 6
        elif div <= 2.0 + 1e-9:
            edge = 8
        elif div <= 2.5 + 1e-9:
            edge = 10
        elif div <= 3.0 + 1e-9:
            edge = 12
        elif div <= 3.5 + 1e-9:
            edge = 14
        elif div <= 4.0 + 1e-9:
            edge = 16
        elif div <= 4.5 + 1e-9:
            edge = 18
        else:
            edge = 20
        edge = _apply_fast_fill_offset(edge, curve="vda_s")
        ema = auto_ema_from_table("vda_s", div, _comp_edge_on)
        return edge, clamp_ema_value(ema)

    if depth_model == "vda_s_metric":
        if div <= 0.5 + 1e-9:
            edge = 4
        elif div <= 1.0 + 1e-9:
            edge = 6
        elif div <= 1.5 + 1e-9:
            edge = 8
        elif div <= 2.0 + 1e-9:
            edge = 10
        elif div <= 2.5 + 1e-9:
            edge = 12
        elif div <= 3.0 + 1e-9:
            edge = 14
        elif div <= 3.5 + 1e-9:
            edge = 16
        elif div <= 4.0 + 1e-9:
            edge = 18
        elif div <= 4.5 + 1e-9:
            edge = 20
        else:
            edge = 22
        edge = _apply_fast_fill_offset(edge, curve="metric")
        ema = auto_ema_from_table("vda_s_metric", div, _comp_edge_on)
        return edge, clamp_ema_value(ema)

    if depth_model == "distill_any_s":
        if div <= 0.5 + 1e-9:
            edge = 2
        elif div <= 1.0 + 1e-9:
            edge = 4
        elif div <= 1.5 + 1e-9:
            edge = 6
        elif div <= 2.0 + 1e-9:
            edge = 8
        elif div <= 2.5 + 1e-9:
            edge = 10
        elif div <= 3.0 + 1e-9:
            edge = 12
        elif div <= 3.5 + 1e-9:
            edge = 14
        elif div <= 4.0 + 1e-9:
            edge = 16
        elif div <= 4.5 + 1e-9:
            edge = 18
        else:
            edge = 20
        edge = _apply_fast_fill_offset(edge)
        ema = auto_ema_from_table("distill_any_s", div, _comp_edge_on)
        return edge, clamp_ema_value(ema)

    if depth_model == "dav2_s":
        # Edge curve: same as VDA-S (streaming, non-metric). EMA: Depth Anything V2 table (_AUTO_EMA_TABLE).
        if div <= 0.5 + 1e-9:
            edge = 2
        elif div <= 1.0 + 1e-9:
            edge = 4
        elif div <= 1.5 + 1e-9:
            edge = 6
        elif div <= 2.0 + 1e-9:
            edge = 8
        elif div <= 2.5 + 1e-9:
            edge = 10
        elif div <= 3.0 + 1e-9:
            edge = 12
        elif div <= 3.5 + 1e-9:
            edge = 14
        elif div <= 4.0 + 1e-9:
            edge = 16
        elif div <= 4.5 + 1e-9:
            edge = 18
        else:
            edge = 20
        edge = _apply_fast_fill_offset(edge, curve="vda_s")
        ema = auto_ema_from_table("dav2_s", div, _comp_edge_on)
        return edge, clamp_ema_value(ema)

    if depth_model == "da3_mono_l":
        # Relative-depth model (not metric): same edge curve as Distill-Any-S / DA-V2-S
        # (0.5 -> 2, 1.0 -> 4 ... 4.5 -> 18, above -> 20). NOT empirically tuned yet; with the Edge Fix
        # Meter (EDGEFIX_AUTO_MEASURE) the measured value is used instead of this fallback.
        edge = min(20, 2 * max(1, int(math.ceil(div / 0.5 - 1e-9))))
        edge = _apply_fast_fill_offset(edge)
        ema = auto_ema_from_table("da3_mono_l", div, _comp_edge_on)
        return edge, clamp_ema_value(ema)

    if div <= 0.5 + 1e-9:
        edge = 2
    elif div <= 1.0 + 1e-9:
        edge = 4
    elif div <= 1.5 + 1e-9:
        edge = 6
    elif div <= 2.0 + 1e-9:
        edge = 8
    elif div <= 2.5 + 1e-9:
        edge = 10
    elif div <= 3.0 + 1e-9:
        edge = 12
    elif div <= 3.5 + 1e-9:
        edge = 14
    elif div <= 4.0 + 1e-9:
        edge = 16
    elif div <= 4.5 + 1e-9:
        edge = 18
    else:
        edge = 20
    edge = _apply_fast_fill_offset(edge)
    ema = auto_ema_from_table("zipdepth" if depth_model == "zipdepth" else "default", div, _comp_edge_on)
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

EDGE_FIX_MAX = 26
DIVERGENCE_MAX = 5.0

_EDGE_FIX_LEVEL_MAP = {
    0: (0, 0),
    1: (1, 1),
    2: (2, 2),
    3: (3, 2),
    4: (4, 2),
    5: (5, 2),
    6: (6, 2),
    7: (7, 2),
    8: (8, 2),
    9: (9, 2),
    10: (10, 2),
    11: (11, 2),
    12: (12, 2),
    13: (13, 2),
    14: (14, 2),
    15: (15, 2),
    16: (16, 2),
    17: (17, 2),
    18: (18, 2),
    19: (19, 2),
    20: (20, 2),
    21: (21, 2),
    22: (22, 2),
    23: (23, 2),
    24: (24, 2),
    25: (25, 2),
    26: (26, 2),
}

def edge_fix_level_to_xy(level, depth_model=None):
    try:
        lvl = int(level)
    except (TypeError, ValueError):
        return level
    level_map = _EDGE_FIX_LEVEL_MAP
    if lvl in level_map:
        return level_map[lvl]

    if lvl > EDGE_FIX_MAX:
        return (lvl, 2)
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

_HAS_EMA_SCALER = True

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

# Depth EMA, absolute-difference gate (anti-flicker for high stereo strength).
# ON : per pixel, the blend weight depends on the ABSOLUTE change of the normalized depth between frames, measured in
#      output disparity pixels (change * strength-px). Changes below GATE_PX[0] px are treated as model noise and get the
#      full (raised) decay; the decay fades to 0 at GATE_PX[1] px, so real motion / scene cuts leave no ghost.
#      The slider/auto EMA value (<= EMA_MAX) is scaled to the raised maximum: EMA_MAX -> DEPTH_EMA_ABS_MAX.
# OFF: the previous motion-map based blend (decay <= EMA_MAX, gate relative to the frame's largest change).
DEPTH_EMA_ABS_GATE = True
DEPTH_EMA_ABS_MAX = 0.85
DEPTH_EMA_GATE_PX = (0.25, 6.0)

class EMAMinMaxScaler:

    def __init__(self, decay: float = 0.0, buffer_size: int = 1, mode: str = "minmax"):
        self.decay = float(max(0.0, min(EMA_MAX, decay)))
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

        self.px_per_unit = 48.0      # disparity pixels per 1.0 of normalized depth (set per run via set_px_per_unit)
        self._prev_depth_small = None
        self._motion_h = 90
        self._motion_w = 160

    def reset(self, decay=None, buffer_size=None, **kwargs):
        if decay is not None:
            self.decay = float(max(0.0, min(EMA_MAX, decay)))
        self.min_value = None
        self.max_value = None
        self._prev_norm = None
        self._frame_count = 0
        self.motion_ema = 0.0
        self.motion_map_ema = None
        self._prev_depth_small = None
        self.last_motion_score = 0.0
        self.last_effective_decay = self.decay

    def set_px_per_unit(self, px_per_unit):
        try:
            v = float(px_per_unit)
        except (TypeError, ValueError):
            return
        if v > 1e-6:
            self.px_per_unit = v

    def set_decay(self, decay):
        self.decay = float(max(0.0, min(EMA_MAX, decay)))
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
        elif decay > EMA_MAX:
            decay = EMA_MAX

        _abs_gate = bool(DEPTH_EMA_ABS_GATE) and decay > 0.0
        if _abs_gate:
            motion_score, motion_map_t = 0.0, None     # motion map not needed (absolute gate below)
        else:
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
            elif _abs_gate:
                d = norm - self._prev_norm
                t0 = float(DEPTH_EMA_GATE_PX[0]) / self.px_per_unit
                t1 = max(t0 + 1e-6, float(DEPTH_EMA_GATE_PX[1]) / self.px_per_unit)
                t = ((d.abs() - t0) / (t1 - t0)).clamp(0.0, 1.0)
                gate = 1.0 - t * t * (3.0 - 2.0 * t)                       # smoothstep fade-out
                d_hi = float(DEPTH_EMA_ABS_MAX) * min(1.0, decay / EMA_MAX)
                dec = gate * d_hi
                result = norm * (1.0 - dec) + self._prev_norm * dec
                self._prev_norm = result.detach().clone()
                self.last_effective_decay = float(d_hi)
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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

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

def get_video_duration_sec(file_path: Path) -> float | None:
    """Container duration in seconds via ffprobe (None if unknown)."""
    if not FFPROBE_EXE or FFPROBE_EXE == "ffprobe":
        return None
    try:
        si = None
        if os.name == 'nt':
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0
        cmd = [FFPROBE_EXE, "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, startupinfo=si).strip()
        val = float(out.splitlines()[0])
        return val if val > 0 else None
    except Exception:
        return None

def get_video_fps(file_path: Path, default_fps: float = 30.0) -> float:

    if not FFPROBE_EXE or FFPROBE_EXE == "ffprobe":
        return default_fps
    try:
        si = None
        if os.name == 'nt':
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0

        cmd = [
            FFPROBE_EXE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate",
            "-of", "json", str(file_path)
        ]
        result = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, startupinfo=si)
        data = json.loads(result)
        stream = data.get("streams", [{}])[0]

        def _parse_rate(rate_str: str) -> float:
            if not rate_str or rate_str == "0/0":
                return 0.0
            if "/" in rate_str:
                num, den = rate_str.split("/")
                num, den = float(num), float(den)
                return num / den if den else 0.0
            return float(rate_str)

        fps = _parse_rate(stream.get("avg_frame_rate", ""))
        if fps <= 0:
            fps = _parse_rate(stream.get("r_frame_rate", ""))
        if fps <= 0:
            fps = default_fps
        return fps
    except Exception:
        return default_fps

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

# ---------------------------------------------------------------------------
# torch.compile support (iw3-style, same as live3d.py): run a tiny real
# compile once per device and cache the result, so a broken/missing
# compiler toolchain (triton, MSVC, etc.) never crashes the program - it
# just silently falls back to eager mode. Used both by the GUI (to
# auto-uncheck the "torch.compile" checkbox when unsupported) and by the
# conversion worker (to actually compile the depth model).
# ---------------------------------------------------------------------------
_COMPILER_SUPPORTED_DEVICES = {}


def _compile_test_func(x):
    return x + 1.0


def check_compile_support(device) -> bool:
    device_name = device if isinstance(device, str) else str(device)
    if device_name not in _COMPILER_SUPPORTED_DEVICES:
        try:
            model = torch.nn.Linear(32, 32, bias=False)
            model.weight.data.zero_()
            model = torch.compile(model.eval().to(device))
            func = torch.compile(_compile_test_func)
            with torch.inference_mode():
                model(torch.zeros((1, 32), device=device))
                func(torch.zeros((32,), dtype=torch.float32, device=device))
            _COMPILER_SUPPORTED_DEVICES[device_name] = True
        except Exception as e:
            print(f"[torch.compile] Not supported on this environment, falling back to eager mode: {e}", flush=True)
            _COMPILER_SUPPORTED_DEVICES[device_name] = False
    return _COMPILER_SUPPORTED_DEVICES[device_name]


def is_compiled_model(model) -> bool:
    return hasattr(model, "_orig_mod")


def compile_model(model, device=None, **kwargs):
    device = device or next(model.parameters()).device
    if not is_compiled_model(model) and check_compile_support(device):
        print(f"[torch.compile] Compiling {model.__class__.__name__}...", flush=True)
        model = torch.compile(model, **kwargs)
    return model


def load_zipdepth(input_size=384, device="cuda", fp16=True, compile_model_flag=False, tensorrt_flag=False):
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
    # TensorRT (if requested and available) takes priority over torch.compile for this model.
    want_trt = bool(tensorrt_flag and use_cuda and check_tensorrt_support())
    want_compile = bool(compile_model_flag and use_cuda and not want_trt and check_compile_support("cuda"))
    # NOTE: DepthInference's own use_compile path warms up with grad enabled and lets
    # Dynamo fall into symbolic (dynamic-shape) recompiles, which crashes Inductor's
    # adaptive_avg_pool2d lowering ("cannot determine truth value of Relational").
    # So the model is always built eager here and compiled below with dynamic=False.
    predictor = DepthInference(
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
    if want_trt:
        print("[TensorRT] ZipDepth: TensorRT backend ON (engines are built lazily per input shape).", flush=True)
        _eager = predictor.model
        predictor.model = TensorRTDepthModel(_eager, fallback=_eager, tag="ZipDepth",
                                             fp16=bool(fp16 and use_cuda), ckpt_path=ckpt,
                                             export_factory=_zipdepth_export_factory(_eager))
    elif tensorrt_flag:
        print("[TensorRT] Requested but unavailable -> using PyTorch.", flush=True)
    if want_compile:
        predictor.model = _compile_depth_model_static(predictor.model, tag="ZipDepth")
        predictor._cd3d_compiled = True
        predictor._cd3d_warm_shapes = set()
        predictor._cd3d_tag = "ZipDepth"
    return predictor


def _compile_depth_model_static(model, tag="Depth model"):
    """torch.compile for a per-frame depth model, with static shapes (dynamic=False).

    This is shared by every depth backend that goes through estimate_depth_raw()'s generic
    _cd3d_compiled / _cd3d_warm_shapes path (ZipDepth, Distill-Any-S, MiDaS-v2.1-Small).

    dynamic=False prevents Dynamo from switching to symbolic shapes on a recompile, which is
    what originally triggered an Inductor LoweringException ("cannot determine truth value of
    Relational") in ZipDepth's adaptive_avg_pool2d the moment the input aspect ratio changed
    (e.g. toggling letterbox-crop removal mid-stream). The same failure mode applies to any
    depth model whose input tensor shape can change at runtime -- i.e. all of them, since
    enabling/disabling letterbox-crop removal changes the frame's height. Each distinct shape
    is instead compiled once (see _warmup_compiled_depth_model) and cached, so a shape change
    costs one warmup stall rather than a crash or a silent fallback to dynamic shapes.
    """
    try:
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = max(int(getattr(_dynamo.config, "cache_size_limit", 8)), 64)
        _dynamo.config.automatic_dynamic_shapes = False
    except Exception as _e:
        print(f"[torch.compile] Could not tune Dynamo config: {_e}", flush=True)
    print(f"[torch.compile] Compiling {tag} (mode=reduce-overhead, dynamic=False)...", flush=True)
    return torch.compile(model, mode="reduce-overhead", dynamic=False)


def _warmup_compiled_depth_model(predictor, image_tensor, iters=3):
    """Warm up the compiled model under inference_mode for THIS input shape.

    Called once per distinct input shape (e.g. full frame, then letterbox-cropped frame).
    Same grad mode / shape as the real calls -> no dispatch-key-mismatch recompile later,
    and CUDA graphs are recorded here. Marks _COMPILE_BUSY so watchdogs don't treat the
    compile as a stall.
    """
    tag = getattr(predictor, "_cd3d_tag", "Depth model")
    shp = tuple(image_tensor.shape)
    print(f"[torch.compile] {tag}: new input shape {shp} -> compiling/warming up "
          f"({iters} iters, can take a while)...", flush=True)
    t0 = time.time()
    _COMPILE_BUSY.set()
    try:
        with torch.inference_mode():
            for _ in range(iters):
                try:
                    torch.compiler.cudagraph_mark_step_begin()
                except Exception:
                    pass
                predictor.model(image_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        _COMPILE_BUSY.clear()
    predictor._cd3d_warm_shapes.add(shp)
    print(f"[torch.compile] {tag}: shape {shp} ready ({time.time() - t0:.1f}s)", flush=True)

VDA_ENCODER_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

VDA_CHECKPOINT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true",
    "vitb": "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth?download=true",
    "vitl": "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth?download=true",
}

VDA_METRIC_CHECKPOINT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth?download=true",
    "vitb": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Base/resolve/main/metric_video_depth_anything_vitb.pth?download=true",
    "vitl": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth?download=true",
}

def _import_vda_streaming_class():

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

    for name in _VDA_STREAMING_METHOD_CANDIDATES:
        if callable(getattr(model, name, None)):
            return name
    return None

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _vda_forward_infer(model, rgb: np.ndarray, input_size: int, device: str, fp16: bool) -> torch.Tensor:

    orig_h, orig_w = rgb.shape[:2]
    new_h, new_w = vda_native_hw(orig_h, orig_w, input_size)

    img = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))         

    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)                   

    if fp16 and device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            depth = model(x)
    else:
        depth = model(x.float())

    depth = depth.float()

    return depth

def vda_native_hw(orig_h: int, orig_w: int, input_size: int) -> tuple[int, int]:

    scale = input_size / min(orig_h, orig_w)
    new_h = max(14, int(round(orig_h * scale / 14)) * 14)
    new_w = max(14, int(round(orig_w * scale / 14)) * 14)
    return new_h, new_w


# --- VDA torch.compile support -------------------------------------------
# Why compiling the VDA model/encoder "did nothing" before:
#   * torch.compile(model) / torch.compile(model.pretrained) only compile
#     forward()/__call__. The streaming path never calls those: it runs
#     model.infer_video_depth_one() -> self.forward_features() ->
#     self.pretrained.get_intermediate_layers(...), i.e. plain methods that
#     execute on the original, eager modules.
# So we compile the actual callables used in that path:
#   * the encoder call (get_intermediate_layers) wrapped in a function,
#     installed as model.forward_features
#   * the temporal head (called through __call__, so torch.compile(head) works)
VDA_COMPILE_HEAD = True     # set False if the temporal head causes problems
VDA_FAST_STREAM = True      # GPU-resident per-frame path (no CPU<->GPU depth round trip)

def compile_vda_parts(model, device, compile_encoder=True):
    """Compile the VDA encoder call and temporal head in place.
    Returns a restore() callable, or None if nothing was compiled."""
    if not check_compile_support(device):
        return None
    pre = getattr(model, "pretrained", None)
    head = getattr(model, "head", None)
    layer_idx = getattr(model, "intermediate_layer_idx", None)
    enc = getattr(model, "encoder", None)
    saved = {}

    if compile_encoder and pre is not None and hasattr(pre, "get_intermediate_layers") and layer_idx is not None and enc in layer_idx:
        idx = layer_idx[enc]

        def _vda_features(x):
            return pre.get_intermediate_layers(x.flatten(0, 1), idx, return_class_token=True)

        print("[torch.compile] Compiling VDA encoder (get_intermediate_layers)...", flush=True)
        _compiled_feat = torch.compile(_vda_features, dynamic=False)
        model.forward_features = lambda x: _compiled_feat(x)
        saved["forward_features"] = True

    if VDA_COMPILE_HEAD and isinstance(head, torch.nn.Module) and not is_compiled_model(head):
        print(f"[torch.compile] Compiling VDA temporal head ({head.__class__.__name__})...", flush=True)
        saved["head"] = head
        model.head = torch.compile(head, dynamic=False)

    if not saved:
        print("[torch.compile] VDA: nothing compilable found (expected model.pretrained / model.head); "
              "running in eager mode.", flush=True)
        return None

    def restore():
        if saved.get("forward_features"):
            model.__dict__.pop("forward_features", None)
        if "head" in saved:
            model.head = saved["head"]
    return restore

# iw3 (nunif) VDA Metric handling, applied to the streaming adapter:
#   input : resize so that (short side + 2*PAD) == input_size, then reflection-pad PAD px on every side
#   output: depth -> disparity (1 / (depth + EPS)), then crop PAD px on every side
# The adapter output is disparity-like (near = large), exactly like the relative VDA output,
# so the metric model goes through the same polarity handling as vda_s.
VDA_METRIC_PADDING = 14
VDA_METRIC_DISP_EPS = 0.1

class VDAStreamingAdapter:

    def __init__(self, model, device, input_size, fp16=True, streaming_method=None, temporal_smooth=0.35, metric=False):
        self.model = model
        self.metric = bool(metric)
        self.device = device
        self.input_size = input_size
        self.fp16 = fp16

        self.streaming_method = streaming_method

        self.temporal_smooth = float(temporal_smooth) if streaming_method is None else 0.0
        self._prev_depth = None
        self._restore_fn = None      # undoes compile_vda_parts()
        self._fast = False           # GPU-resident streaming path enabled
        self._fast_hw = None
        self._fast_mean = None
        self._fast_std = None

    def _fallback_to_eager(self, err):
        if self._restore_fn is not None:
            try:
                self._restore_fn()
            except Exception:
                pass
        self._restore_fn = None
        self._fast = False
        print(f"[torch.compile] VDA compiled/fast path failed, falling back to the original eager path: {err}", flush=True)

    def _infer_fast(self, rgb):
        """Same math as VideoDepthAnything.infer_video_depth_one() for frames after the
        first one, but preprocessing is done with a uint8 resize + GPU normalization and
        the depth stays on the GPU (the original returns a full-res numpy array, which
        forced a GPU->CPU copy and a CPU->GPU copy of a full-res frame every frame)."""
        m = self.model
        dev = self.device
        fh, fw = rgb.shape[:2]
        if fh != m.frame_height or fw != m.frame_width:
            raise RuntimeError("frame size changed during streaming")
        if self._fast_hw is None:
            # learn the model's native input size once, using its own transform
            sample = m.transform({'image': rgb.astype(np.float32) / 255.0})['image']
            self._fast_hw = (int(sample.shape[-2]), int(sample.shape[-1]))
            self._fast_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=dev).view(3, 1, 1)
            self._fast_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=dev).view(3, 1, 1)
        nh, nw = self._fast_hw
        infer_len = getattr(sys.modules.get(type(m).__module__), "INFER_LEN", 32)

        m.id += 1
        small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        x = torch.from_numpy(small).to(dev).permute(2, 0, 1).float().div_(255.0)
        x = ((x - self._fast_mean) / self._fast_std).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            with torch.autocast(device_type=dev, enabled=self.fp16):
                feat = m.forward_features(x)
            cur_list = m.frame_cache_list[0:2] + m.frame_cache_list[-infer_len + 3:]
            cur_cache = [torch.cat([h[i] for h in cur_list], dim=1) for i in range(len(cur_list[0]))]
            with torch.autocast(device_type=dev, enabled=self.fp16):
                depth, new_cache = m.forward_depth(feat, x.shape, cached_hidden_state_list=cur_cache)
            depth = depth.to(x.dtype)
            depth = torch.nn.functional.interpolate(
                depth.flatten(0, 1).unsqueeze(1), size=(fh, fw), mode='bilinear', align_corners=True)

        m.frame_cache_list.append(new_cache)
        m.frame_id_list.append(m.id)
        if m.id + infer_len > m.gap + 1:
            del m.frame_id_list[1]
            del m.frame_cache_list[1]
        return depth.float()

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
        self._fast_hw = None

    def infer(self, bgr: np.ndarray) -> torch.Tensor:
        if self._restore_fn is None and not self._fast:
            return self._infer_impl(bgr)
        try:
            return self._infer_impl(bgr)
        except Exception as e:
            self._fallback_to_eager(e)
            self.reset_state()
            return self._infer_impl(bgr)

    @torch.inference_mode()
    def _infer_impl(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if not self.metric:
            return self._infer_raw(rgb)

        pad = VDA_METRIC_PADDING
        fh, fw = rgb.shape[:2]
        eff = int(self.input_size)
        _ratio = max(fh, fw) / float(max(1, min(fh, fw)))
        if _ratio > 1.78:
            # same rule as VideoDepthAnythingStreaming.infer_video_depth_one()
            eff = int(round(int(eff * 1.777 / _ratio) / 14) * 14)
        inner = max(14, eff - pad * 2)
        _scale = inner / float(min(fh, fw))
        nh = max(14, int(round(fh * _scale / 14)) * 14)
        nw = max(14, int(round(fw * _scale / 14)) * 14)
        if (nh, nw) != (fh, fw):
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)

        depth = self._infer_raw(rgb)
        depth = torch.nan_to_num(depth.float())
        depth = 1.0 / (depth + VDA_METRIC_DISP_EPS)
        depth = depth[..., pad:-pad, pad:-pad]
        if tuple(depth.shape[-2:]) != (fh, fw):
            depth = torch.nn.functional.interpolate(depth, size=(fh, fw), mode="bilinear", align_corners=True)
        return depth.contiguous()

    @torch.inference_mode()
    def _infer_raw(self, rgb: np.ndarray) -> torch.Tensor:
        if self.streaming_method is not None:
            if self._fast and getattr(self.model, "transform", None) is not None:
                return self._infer_fast(rgb)
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

def load_vda_streaming(encoder="vits", input_size=392, fp16=True, metric=False, compile_model_flag=False,
                       tensorrt_flag=False):

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

    restore_fn = None
    use_fast = False

    # TensorRT: only the (stateless) DINOv2 encoder; the temporal head keeps its cache in PyTorch.
    trt_restore = None
    want_trt = bool(tensorrt_flag and use_cuda and check_tensorrt_support())
    if want_trt:
        if (streaming_method == "infer_video_depth_one" and hasattr(model, "pretrained")
                and hasattr(model, "intermediate_layer_idx") and model.encoder in model.intermediate_layer_idx):
            _tag = "VDA-S-Metric-enc" if metric else "VDA-S-enc"
            trt_restore = install_vda_tensorrt(model, fp16=bool(fp16 and use_cuda), ckpt_path=ckpt, tag=_tag)
            print("[TensorRT] VDA: encoder runs on TensorRT (engines are built lazily per input shape); "
                  "temporal head stays in PyTorch.", flush=True)
        else:
            print("[TensorRT] VDA: unsupported streaming layout on this checkout -> using PyTorch.", flush=True)
    elif tensorrt_flag:
        print("[TensorRT] Requested but unavailable -> using PyTorch.", flush=True)

    if (compile_model_flag and use_cuda) or trt_restore is not None:
        if streaming_method is None:
            # forward() path: compiling the whole model works here
            if compile_model_flag and use_cuda:
                model = compile_model(model, device=device)
        else:
            _compile_restore = None
            if compile_model_flag and use_cuda:
                # with TensorRT the encoder is already handled; only the head may be compiled
                _compile_restore = compile_vda_parts(model, device, compile_encoder=(trt_restore is None))
            if trt_restore is not None and _compile_restore is not None:
                def restore_fn(_a=trt_restore, _b=_compile_restore):
                    _a()
                    _b()
            else:
                restore_fn = trt_restore or _compile_restore
            use_fast = (
                VDA_FAST_STREAM and restore_fn is not None
                and streaming_method == "infer_video_depth_one"
                and all(hasattr(model, a) for a in (
                    "forward_features", "forward_depth", "frame_cache_list",
                    "frame_id_list", "transform", "gap", "id"))
            )
            if use_fast:
                print("[Info] VDA GPU-resident streaming path enabled.", flush=True)

    adapter = VDAStreamingAdapter(
        model, device=device, input_size=isz, fp16=(fp16 and use_cuda),
        streaming_method=streaming_method, metric=bool(metric),
    )
    adapter._restore_fn = restore_fn
    adapter._fast = use_fast
    return adapter

DISTILL_ANY_S_DAV2_CONFIG = VDA_ENCODER_CONFIGS["vits"]  # encoder="vits", features=64, out_channels=[48, 96, 192, 384]
DISTILL_ANY_S_HF_REPO_ID = "xingyang1/Distill-Any-Depth"
DISTILL_ANY_S_HF_FILENAME = "small/model.safetensors"

# Distill-Any-Depth-Small is trained on the plain Depth-Anything-V2 (vits)
# architecture and its safetensors checkpoint loads straight into
# depth_anything_v2.dpt.DepthAnythingV2 (confirmed by the DA-V2 authors'
# own repo, not the much heavier Westlake-AGI-Lab/Distill-Any-Depth repo
# which wraps it in extra packaging + a detectron2 dependency we don't need
# just for inference). The distillanydepth.* / DAM candidates are kept as a
# fallback in case DISTILL_ROOT ends up pointing at that repo instead.
# The Depth-Anything-V2 source itself is fetched by install.bat (cloned into
# DISTILL_ROOT) - this module only imports it, it does not clone anything.
_DISTILL_ANY_S_IMPORT_CANDIDATES = [
    ("depth_anything_v2.dpt", "DepthAnythingV2", "dav2"),
    ("distillanydepth.depth_anything_v2.dpt", "DepthAnythingV2", "dav2"),
    ("distillanydepth.modeling.archs.dam.dam", "DepthAnything", "dam"),
    ("modeling.archs.dam.dam", "DepthAnything", "dam"),
]

def _import_distill_any_depth_class():
    if not DISTILL_ROOT.exists():
        raise RuntimeError(
            f"[Distill-Any-S] {DISTILL_ROOT} not found. Run install.bat first to fetch "
            f"the Depth-Anything-V2 source."
        )
    if str(DISTILL_ROOT) not in sys.path:
        sys.path.insert(0, str(DISTILL_ROOT))

    last_err = None
    for module_name, cls_name, kind in _DISTILL_ANY_S_IMPORT_CANDIDATES:
        try:
            mod = __import__(module_name, fromlist=[cls_name])
            return getattr(mod, cls_name), kind
        except Exception as e:
            last_err = e
            continue
    raise ImportError(
        f"[Distill-Any-S] Could not locate the DepthAnything(V2) class under {DISTILL_ROOT}. "
        f"Check the Depth-Anything-V2 installation / update _DISTILL_ANY_S_IMPORT_CANDIDATES(). "
        f"Last import error: {last_err}"
    )

def _distill_any_s_checkpoint_path() -> Path:
    ckpt_dir = DISTILL_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "distill_any_depth_small.safetensors"
    if ckpt.exists():
        return ckpt

    print(f"[Info] Distill-Any-Depth Small checkpoint not found. Downloading "
          f"({DISTILL_ANY_S_HF_REPO_ID}/{DISTILL_ANY_S_HF_FILENAME})...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=DISTILL_ANY_S_HF_REPO_ID,
            filename=DISTILL_ANY_S_HF_FILENAME,
        )
        import shutil
        shutil.copyfile(downloaded, ckpt)
    except Exception:
        import urllib.request
        url = (
            f"https://huggingface.co/{DISTILL_ANY_S_HF_REPO_ID}/resolve/main/"
            f"{DISTILL_ANY_S_HF_FILENAME}?download=true"
        )
        urllib.request.urlretrieve(url, ckpt)
    print("[Info] Distill-Any-Depth Small checkpoint downloaded successfully.")
    return ckpt

class _DamForwardWrapper:
    """DAM's DepthAnything.forward() returns (pred_disp, feature). Wrapping
    it so predictor.model(x) yields just the depth tensor lets it plug into
    the shared estimate_depth_raw() dispatch used by ZipDepth, unchanged."""

    def __init__(self, dam_model):
        self._m = dam_model

    def __call__(self, x):
        out = self._m(x)
        return out[0] if isinstance(out, (tuple, list)) else out

    def to(self, *a, **kw):
        self._m = self._m.to(*a, **kw)
        return self

    def eval(self):
        self._m.eval()
        return self

    def half(self):
        self._m = self._m.half()
        return self

class DistillAnyDepthAdapter:
    """Mirrors the ZipDepth predictor interface (image2tensor + model) so
    estimate_depth_raw() can dispatch to it through its generic (non-VDA)
    branch without any extra isinstance() handling."""

    def __init__(self, model, device: str, input_size: int, fp16: bool = True, ensure_multiple_of: int = 14):
        self.model = model
        self.device = device
        self.input_size = input_size
        self.fp16 = fp16 and device == "cuda"
        self.ensure_multiple_of = ensure_multiple_of

    def image2tensor(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]
        m = self.ensure_multiple_of
        scale = self.input_size / min(orig_h, orig_w)
        new_h = max(m, int(round(orig_h * scale / m)) * m)
        new_w = max(m, int(round(orig_w * scale / m)) * m)

        img = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))

        x = torch.from_numpy(img).unsqueeze(0).to(self.device)
        if self.fp16:
            x = x.half()
        return x, orig_h, orig_w

def load_distill_any_s(input_size=518, device="cuda", fp16=True, compile_model_flag=False, tensorrt_flag=False):
    from safetensors.torch import load_file

    ModelClass, kind = _import_distill_any_depth_class()
    ckpt = _distill_any_s_checkpoint_path()
    state_dict = load_file(str(ckpt))

    use_cuda = torch.cuda.is_available() and str(device) == "cuda"
    dev = "cuda" if use_cuda else "cpu"

    if kind == "dam":
        try:
            raw_model = ModelClass(checkpoint_path=None)
        except TypeError:
            raw_model = ModelClass()
    else:
        raw_model = ModelClass(**DISTILL_ANY_S_DAV2_CONFIG)

    missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[Distill-Any-S] load_state_dict: {len(missing)} missing / {len(unexpected)} unexpected keys "
              f"(usually harmless buffers, but check if inference looks wrong).")

    raw_model = raw_model.to(dev).eval()
    if fp16 and use_cuda:
        raw_model = raw_model.half()

    # Same approach as ZipDepth: the generic compile_model() traces with dynamic shapes, which
    # can crash Inductor (adaptive_avg_pool2d lowering) when the depth input shape changes at
    # runtime (letterbox-crop removal toggle). So compile with dynamic=False and warm up once
    # per distinct input shape (see _warmup_compiled_depth_model / estimate_depth_raw).
    # TensorRT (if requested and available) takes priority over torch.compile for this model.
    want_trt = bool(tensorrt_flag and use_cuda and check_tensorrt_support())
    want_compile = bool(compile_model_flag and use_cuda and not want_trt and check_compile_support(dev))
    if want_compile:
        raw_model = _compile_depth_model_static(raw_model, tag="Distill-Any-S")

    model = _DamForwardWrapper(raw_model) if kind == "dam" else raw_model
    if want_trt:
        print("[TensorRT] Distill-Any-S: TensorRT backend ON (engines are built lazily per input shape).", flush=True)
        model = TensorRTDepthModel(raw_model, fallback=model, tag="Distill-Any-S",
                                   fp16=bool(fp16 and use_cuda), ckpt_path=ckpt)
    elif tensorrt_flag:
        print("[TensorRT] Requested but unavailable -> using PyTorch.", flush=True)

    isz = int(input_size)
    if isz % 14 != 0:
        isz += (14 - isz % 14)

    adapter = DistillAnyDepthAdapter(
        model, device=dev, input_size=isz, fp16=fp16, ensure_multiple_of=14,
    )
    if want_compile:
        adapter._cd3d_compiled = True
        adapter._cd3d_warm_shapes = set()
        adapter._cd3d_tag = "Distill-Any-S"
    return adapter

# ---------------------------------------------------------------------------
# DA-V2-S (official Depth-Anything-V2 Small). Same architecture / adapter as
# Distill-Any-S (both load straight into depth_anything_v2.dpt.DepthAnythingV2
# "vits"), just with the official DA-V2 checkpoint instead of the distilled
# one. The Depth-Anything-V2 source under DISTILL_ROOT is shared with
# Distill-Any-S (already installed; this module only imports it).
# torch.compile and TensorRT are supported (same paths as Distill-Any-S; TensorRT takes priority).
# ---------------------------------------------------------------------------
DAV2_S_HF_REPO_ID = "depth-anything/Depth-Anything-V2-Small"
DAV2_S_HF_FILENAME = "depth_anything_v2_vits.pth"

def _dav2_s_checkpoint_path() -> Path:
    ckpt_dir = DISTILL_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "depth_anything_v2_vits.pth"
    if ckpt.exists():
        return ckpt

    print(f"[Info] Depth-Anything-V2 Small checkpoint not found. Downloading "
          f"({DAV2_S_HF_REPO_ID}/{DAV2_S_HF_FILENAME})...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=DAV2_S_HF_REPO_ID,
            filename=DAV2_S_HF_FILENAME,
        )
        import shutil
        shutil.copyfile(downloaded, ckpt)
    except Exception:
        import urllib.request
        url = (
            f"https://huggingface.co/{DAV2_S_HF_REPO_ID}/resolve/main/"
            f"{DAV2_S_HF_FILENAME}?download=true"
        )
        urllib.request.urlretrieve(url, ckpt)
    print("[Info] Depth-Anything-V2 Small checkpoint downloaded successfully.")
    return ckpt

def load_dav2_s(input_size=518, device="cuda", fp16=True, compile_model_flag=False, tensorrt_flag=False):
    ModelClass, kind = _import_distill_any_depth_class()
    ckpt = _dav2_s_checkpoint_path()
    state_dict = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict and "state_dict" not in state_dict \
            and not any(str(k).startswith(("pretrained.", "depth_head.")) for k in state_dict.keys()):
        state_dict = state_dict["model"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    use_cuda = torch.cuda.is_available() and str(device) == "cuda"
    dev = "cuda" if use_cuda else "cpu"

    if kind == "dam":
        try:
            raw_model = ModelClass(checkpoint_path=None)
        except TypeError:
            raw_model = ModelClass()
    else:
        raw_model = ModelClass(**DISTILL_ANY_S_DAV2_CONFIG)

    missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[DA-V2-S] load_state_dict: {len(missing)} missing / {len(unexpected)} unexpected keys "
              f"(usually harmless buffers, but check if inference looks wrong).")

    raw_model = raw_model.to(dev).eval()
    if fp16 and use_cuda:
        raw_model = raw_model.half()

    # torch.compile: same approach as Distill-Any-S / ZipDepth (identical DepthAnythingV2 ViT-S
    # architecture). dynamic=False + one warmup per distinct input shape (see
    # _warmup_compiled_depth_model / estimate_depth_raw) so a shape change (e.g. letterbox-crop
    # toggle) costs one warmup instead of crashing Inductor or silently going dynamic.
    # TensorRT (if requested and available) takes priority over torch.compile for this model.
    want_trt = bool(tensorrt_flag and use_cuda and check_tensorrt_support())
    want_compile = bool(compile_model_flag and use_cuda and not want_trt and check_compile_support(dev))
    if compile_model_flag and not want_compile and not want_trt:
        print("[DA-V2-S] torch.compile requested but not available here -> running eager PyTorch.", flush=True)
    if want_compile:
        raw_model = _compile_depth_model_static(raw_model, tag="DA-V2-S")

    model = _DamForwardWrapper(raw_model) if kind == "dam" else raw_model
    if want_trt:
        print("[TensorRT] DA-V2-S: TensorRT backend ON (engines are built lazily per input shape).", flush=True)
        model = TensorRTDepthModel(raw_model, fallback=model, tag="DA-V2-S",
                                   fp16=bool(fp16 and use_cuda), ckpt_path=ckpt)
    elif tensorrt_flag:
        print("[TensorRT] Requested but unavailable -> using PyTorch.", flush=True)

    isz = int(input_size)
    if isz % 14 != 0:
        isz += (14 - isz % 14)

    adapter = DistillAnyDepthAdapter(
        model, device=dev, input_size=isz, fp16=fp16, ensure_multiple_of=14,
    )
    if want_compile:
        adapter._cd3d_compiled = True
        adapter._cd3d_warm_shapes = set()
        adapter._cd3d_tag = "DA-V2-S"
    return adapter

# ---------------------------------------------------------------------------
# TensorRT support (Distill-Any-S, DA-V2-S, ZipDepth, VDA-S encoder, DA3-Mono-L)
#
# Pipeline: PyTorch model -> ONNX (fixed input shape) -> TensorRT engine (.trt,
# cached on disk per input shape / GPU / TensorRT version) -> execute_async_v3()
# directly on PyTorch CUDA tensors (no host copies).
#
# The depth input shape can change at runtime (e.g. letterbox-crop removal), so
# engines are built lazily, once per distinct shape. A first-time build takes
# from tens of seconds to a few minutes; later runs just load the cached file.
# Any failure (missing tensorrt, ONNX export error, build error, runtime error)
# silently falls back to the original eager PyTorch model for that shape.
# ---------------------------------------------------------------------------
TRT_CACHE_DIR = BASE_DIR / "trt_cache"
_TRT_SUPPORTED = None


def check_tensorrt_support() -> bool:
    global _TRT_SUPPORTED
    if _TRT_SUPPORTED is None:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            import tensorrt as trt
            _major = int(str(trt.__version__).split(".")[0])
            if _major < 10:
                raise RuntimeError(f"TensorRT 10.x is required (found {trt.__version__})")
            trt.Builder(trt.Logger(trt.Logger.ERROR))
            _TRT_SUPPORTED = True
        except Exception as e:
            print(f"[TensorRT] Not supported on this environment, falling back to PyTorch: {e}", flush=True)
            _TRT_SUPPORTED = False
    return _TRT_SUPPORTED


def _make_trt_logger(trt):
    """TensorRT logger that keeps warnings/errors (the console is hidden, so they would otherwise be lost)."""
    class _CaptureLogger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)
            self.msgs = []

        def log(self, severity, msg):
            if severity <= trt.ILogger.Severity.WARNING:
                self.msgs.append(f"[{str(severity).split('.')[-1]}] {msg}")
                if len(self.msgs) > 400:
                    del self.msgs[:200]
                if severity <= trt.ILogger.Severity.ERROR:
                    print(f"[TensorRT] {msg}", flush=True)
    return _CaptureLogger()


def _trt_dump_error(tag, text):
    """Append a failure report to trt_cache/trt_build_error.log (readable without the console)."""
    try:
        TRT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(TRT_CACHE_DIR / "trt_build_error.log", "a", encoding="utf-8") as f:
            f.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} [{tag}] ====\n{text}\n\n")
    except Exception:
        pass


class _TRTExportModule(torch.nn.Module):
    """Makes the exported graph return a single depth tensor."""

    def __init__(self, module):
        super().__init__()
        self.m = module

    def forward(self, x):
        out = self.m(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


class TensorRTDepthModel:
    """Drop-in replacement for a model callable: callable(input_tensor) -> tensor (or tuple of tensors).

    Engines are built lazily per input shape/dtype. Models with several outputs (e.g. the VDA
    encoder: 4 patch-token maps + 4 class tokens) return a tuple in ONNX output order.
    """

    def __init__(self, torch_module, fallback, tag="Depth model", fp16=True, ckpt_path=None, workspace_gb=2,
                 export_factory=None, export_context=None):
        import tensorrt as trt
        self._trt = trt
        self._module = torch_module
        # export_factory(x) -> (nn.Module, dummy_input[, output_names]). Lets a model prepare a
        # static-shape, ONNX-friendly copy of itself (e.g. ZipDepth). Default: export as-is.
        self._export_factory = export_factory
        # export_context() -> context manager active during torch.onnx.export (e.g. disable xformers).
        self._export_context = export_context
        self._fallback = fallback
        self._tag = tag
        self._fp16 = bool(fp16)
        self._workspace = int(workspace_gb) << 30
        self._logger = _make_trt_logger(trt)
        self._runtime = trt.Runtime(self._logger)
        self._engines = {}
        self._lock = threading.Lock()
        try:
            _p = torch.cuda.get_device_properties(torch.cuda.current_device())
            _gpu = f"{_p.name}_sm{_p.major}{_p.minor}"
        except Exception:
            _gpu = "gpu"
        self._gpu_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", _gpu)
        self._ckpt_sig = "0"
        try:
            if ckpt_path is not None:
                _st = Path(ckpt_path).stat()
                self._ckpt_sig = f"{_st.st_size}_{int(_st.st_mtime)}"
        except Exception:
            pass
        self._trt_dtype_map = {
            trt.DataType.HALF: torch.float16,
            trt.DataType.FLOAT: torch.float32,
        }

    # -- cache -------------------------------------------------------------
    def _engine_path(self, shape, dtype) -> Path:
        TRT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dt = "fp16" if self._fp16 else "fp32"
        name = (f"{self._tag}_{dt}_{shape[-2]}x{shape[-1]}_{self._gpu_key}"
                f"_trt{self._trt.__version__}_{self._ckpt_sig}.trt")
        return TRT_CACHE_DIR / re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    # -- build -------------------------------------------------------------
    def _export_onnx(self, x, onnx_path: Path):
        out_names = ["predicted_depth"]
        if self._export_factory is not None:
            res = self._export_factory(x)
            wrapper, dummy = res[0], res[1]
            if len(res) > 2 and res[2]:
                out_names = list(res[2])
        else:
            wrapper = _TRTExportModule(self._module).eval()
            dummy = torch.randn(tuple(x.shape), device=x.device, dtype=x.dtype)
        attempts = [
            dict(opset_version=17, dynamo=False),   # legacy TorchScript exporter
            dict(),                                 # whatever the installed torch defaults to
        ]
        last_err = None
        for kw in attempts:
            try:
                import contextlib
                ctx = self._export_context() if self._export_context is not None else contextlib.nullcontext()
                with torch.no_grad(), ctx:
                    torch.onnx.export(
                        wrapper, dummy, str(onnx_path),
                        input_names=["pixel_values"], output_names=out_names,
                        do_constant_folding=True, **kw,
                    )
                if onnx_path.exists():
                    return
            except Exception as e:
                last_err = e
        raise RuntimeError(f"ONNX export failed: {last_err}")

    def _configure_builder(self, builder, network, config):
        """Hook for subclasses (extra builder-config settings). The generic engines need none."""
        return None

    def _build_engine(self, x, engine_path: Path):
        trt = self._trt
        onnx_path = engine_path.with_suffix(".onnx")
        stage = "ONNX export"
        try:
            print(f"[TensorRT] {self._tag}: exporting ONNX for input {tuple(x.shape)} {x.dtype}...", flush=True)
            self._export_onnx(x, onnx_path)

            print(f"[TensorRT] {self._tag}: building engine (first time only, may take a few minutes)...", flush=True)
            t0 = time.time()
            self._logger.msgs.clear()
            stage = "ONNX parse"
            builder = trt.Builder(self._logger)
            network = builder.create_network(0)  # TensorRT 10: explicit batch is the only mode
            parser = trt.OnnxParser(network, self._logger)
            if not parser.parse_from_file(str(onnx_path)):
                errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError("ONNX parse failed: " + " | ".join(errs[:8]))
            stage = "engine build"
            config = builder.create_builder_config()
            if self._fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self._workspace)
            self._configure_builder(builder, network, config)
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("build_serialized_network returned None")
            engine_path.write_bytes(bytes(serialized))
        except Exception as e:
            tail = " | ".join(self._logger.msgs[-12:])
            _trt_dump_error(self._tag, f"stage: {stage}\ninput: {tuple(x.shape)} {x.dtype} fp16={self._fp16}\n"
                                       f"trt: {self._trt.__version__}\nerror: {e}\n"
                                       f"trt log (last):\n" + "\n".join(self._logger.msgs[-40:])
                                       + f"\nkept ONNX: {onnx_path}")
            raise RuntimeError(f"{stage}: {e}" + (f" || TRT: {tail}" if tail else "")
                               + " (details: trt_cache/trt_build_error.log)")
        print(f"[TensorRT] {self._tag}: engine saved ({time.time() - t0:.1f}s) -> {engine_path.name}", flush=True)
        for p in (onnx_path, Path(str(onnx_path) + ".data")):
            try:
                p.unlink()
            except Exception:
                pass

    def _load_engine(self, engine_path: Path):
        trt = self._trt
        engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError("deserialize_cuda_engine returned None")
        context = engine.create_execution_context()
        in_name = None
        out_names = []
        for i in range(engine.num_io_tensors):
            n = engine.get_tensor_name(i)
            if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                in_name = n
            else:
                out_names.append(n)
        if in_name is None or not out_names:
            raise RuntimeError("engine has no input/output tensor")
        if len(out_names) > 1:
            out_names.sort()  # outputs were named out00, out01, ... in export order
        in_dtype = self._trt_dtype_map.get(engine.get_tensor_dtype(in_name), torch.float32)
        outs = []
        for n in out_names:
            shp = tuple(context.get_tensor_shape(n))
            if any(d < 0 for d in shp):
                raise RuntimeError(f"engine output shape is dynamic: {shp}")
            outs.append((n, shp, self._trt_dtype_map.get(engine.get_tensor_dtype(n), torch.float32)))
        return dict(engine=engine, context=context, in_name=in_name, in_dtype=in_dtype, outs=outs)

    def _prepare(self, x):
        key = (tuple(x.shape), x.dtype)
        _COMPILE_BUSY.set()  # a first-time build can take minutes; don't let watchdogs treat it as a stall
        try:
            path = self._engine_path(x.shape, x.dtype)
            entry = None
            if path.exists():
                try:
                    entry = self._load_engine(path)
                    print(f"[TensorRT] {self._tag}: loaded cached engine {path.name}", flush=True)
                except Exception as e:
                    print(f"[TensorRT] {self._tag}: cached engine unusable ({e}); rebuilding...", flush=True)
                    try:
                        path.unlink()
                    except Exception:
                        pass
            if entry is None:
                self._build_engine(x, path)
                entry = self._load_engine(path)
            self._engines[key] = entry
            return entry
        except Exception as e:
            print(f"[TensorRT] {self._tag}: failed for input {tuple(x.shape)} -> using PyTorch for this shape: {e}",
                  flush=True)
            self._engines[key] = None
            return None
        finally:
            _COMPILE_BUSY.clear()

    # -- inference ---------------------------------------------------------
    def _enqueue(self, entry, x):
        """Run one engine on x and return its output tensor(s). Generic path: PyTorch's current stream."""
        xin = x
        if xin.dtype != entry["in_dtype"]:
            xin = xin.to(entry["in_dtype"])
        xin = xin.contiguous()  # NCHW (ZipDepth hands over a channels_last buffer)
        ctx = entry["context"]
        ctx.set_tensor_address(entry["in_name"], xin.data_ptr())
        outs = []
        for name, shp, dt in entry["outs"]:
            o = torch.empty(shp, device=xin.device, dtype=dt)
            ctx.set_tensor_address(name, o.data_ptr())
            outs.append(o)
        ok = ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")
        return outs[0] if len(outs) == 1 else tuple(outs)

    def __call__(self, x):
        key = (tuple(x.shape), x.dtype)
        if key in self._engines:
            entry = self._engines[key]
        else:
            with self._lock:
                entry = self._engines[key] if key in self._engines else self._prepare(x)
        if entry is None:
            return self._fallback(x)
        try:
            return self._enqueue(entry, x)
        except Exception as e:
            print(f"[TensorRT] {self._tag}: runtime error -> falling back to PyTorch: {e}", flush=True)
            self._engines[key] = None
            return self._fallback(x)


def _zipdepth_export_factory(model):
    """Returns export_factory(x) for TensorRTDepthModel: a static-shape fp32 CPU copy of ZipDepth.

    Mirrors ZipDepth's own scripts/export.py (opset 17, fixed shape) but works on a deep copy so
    the eager model used as fallback stays untouched. The only graph change is MinimalCrossScale:
    adaptive_avg_pool2d to exactly half size is replaced by the equivalent avg_pool2d(2, 2), because
    adaptive pooling with traced sizes is fragile in ONNX. Inputs are multiples of 32, so the
    H/16 and H/32 feature maps differ by exactly 2x and the result is identical.
    """
    def _factory(x):
        import copy
        import types
        import torch.nn.functional as F
        m = copy.deepcopy(model).float().cpu().eval()
        m = m.to(memory_format=torch.contiguous_format)
        cs = getattr(getattr(m, "encoder", None), "cross_scale", None)
        if cs is not None and hasattr(cs, "low_to_high") and hasattr(cs, "high_to_low"):
            def _cs_fwd(self_cs, x_high, x_low):
                lo = F.interpolate(self_cs.low_to_high(x_low), size=x_high.shape[2:], mode="nearest")
                hi = F.avg_pool2d(self_cs.high_to_low(x_high), 2, 2)
                return x_high + lo * 0.3, x_low + hi * 0.3
            cs.forward = types.MethodType(_cs_fwd, cs)
        dummy = torch.randn(tuple(x.shape), dtype=torch.float32)
        return m, dummy
    return _factory


def _vda_export_context():
    """Disable xformers attention (if present) while exporting the DINOv2 encoder to ONNX."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        toggled = []
        for mod_name in ("video_depth_anything.dinov2_layers.attention",
                         "video_depth_anything.dinov2_layers.block"):
            mod = sys.modules.get(mod_name)
            if mod is not None and hasattr(mod, "XFORMERS_AVAILABLE"):
                toggled.append((mod, mod.XFORMERS_AVAILABLE))
                mod.XFORMERS_AVAILABLE = False
        try:
            yield
        finally:
            for mod, orig in toggled:
                mod.XFORMERS_AVAILABLE = orig
    return _ctx()


def install_vda_tensorrt(model, fp16, ckpt_path, tag):
    """Run the VDA DINOv2 encoder (stateless) through TensorRT.

    Replaces model.forward_features (same trick as compile_vda_parts). The temporal head keeps its
    frame-to-frame hidden-state cache and stays in PyTorch (optionally torch.compile'd).
    Returns a restore() callable.
    """
    pre = model.pretrained
    idx = model.intermediate_layer_idx[model.encoder]
    n = len(idx)

    def _flat_eager(x4):
        feats = pre.get_intermediate_layers(x4, idx, return_class_token=True)
        return tuple(t for pair in feats for t in pair)

    def _factory(x):
        class _Enc(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pre = pre

            def forward(self, t):
                feats = self.pre.get_intermediate_layers(t, idx, return_class_token=True)
                flat = []
                for p, c in feats:
                    flat += [p, c]
                return tuple(flat)

        _dt = next(pre.parameters()).dtype
        dummy = torch.randn(tuple(x.shape), device=x.device, dtype=_dt)
        return _Enc().eval(), dummy, [f"out{i:02d}" for i in range(2 * n)]

    trt_model = TensorRTDepthModel(pre, fallback=_flat_eager, tag=tag, fp16=fp16, ckpt_path=ckpt_path,
                                   export_factory=_factory, export_context=_vda_export_context)

    def _forward_features(x):
        outs = trt_model(x.flatten(0, 1))
        return tuple((outs[2 * i], outs[2 * i + 1]) for i in range(n))

    model.forward_features = _forward_features

    def restore():
        model.__dict__.pop("forward_features", None)
    return restore


# ---------------------------------------------------------------------------
# DA3-Mono-L (Depth Anything 3, DA3MONO-LARGE = relative depth, NOT the metric model)
#   https://huggingface.co/depth-anything/DA3MONO-LARGE
# The DA3 source is NOT pip-installed (its requirements would break numpy / torch / opencv): it lives in
# <BASE_DIR>/Depth-Anything-3/src and is added to sys.path lazily when the model is loaded
# (install-da3-mono-large.bat). Only the network (DINOv2 ViT-L backbone + DPT head) is built from
# configs/da3mono-large.yaml; depth_anything_3.api (export / gradio / open3d ...) is never imported.
# The model outputs TRUE depth (near = small), while every other depth model of this program returns a
# disparity-like value (near = large; the converter inverts it with `max - x`). The wrapper below therefore
# returns -depth, so after the converter's inversion the result is exactly the depth itself (near = small),
# i.e. the same direction as the DA3-SMALL implementation in the old live3d.py (verified there).
# torch.compile and TensorRT are both supported (TensorRT takes priority when both are requested): see
# "TensorRT for DA3-Mono-L" below. Without them the model runs as eager PyTorch.
# ---------------------------------------------------------------------------
DA3_ROOT = BASE_DIR / "Depth-Anything-3"
DA3_SRC = DA3_ROOT / "src"
DA3_MONO_L_CKPT_DIR = DA3_ROOT / "checkpoints" / "DA3MONO-LARGE"
DA3_MONO_L_CKPT_NAME = "model.safetensors"
DA3_MONO_L_HF_REPO_ID = "depth-anything/DA3MONO-LARGE"
DA3_MONO_L_CONFIG_NAME = "da3mono-large"        # <DA3_SRC>/depth_anything_3/configs/da3mono-large.yaml
# True : use the inverse depth (1 / depth, disparity-like; near = large before the converter's `max - x`).
# False: use the depth as it is (-depth, same as the verified DA3-SMALL of live3d.py).
DA3_MONO_L_USE_INVERSE_DEPTH = True
# iw3-style disparity: 1 / (depth + SHIFT) * (1 - sky_weight). SHIFT keeps the near range from exploding
# (pure 1/depth gives huge disparity at near edges -> excessive edge artifacts). iw3 uses 0.2.
DA3_MONO_L_DISPARITY_SHIFT = 0.4   # iw3 uses 0.2; larger = weaker near-range disparity (less foreground pop-out)
# sky channel of the DA3 mono head: sky > thresh is treated as sky and its disparity is faded to 0 (far).
DA3_MONO_L_SKY_THRESH = 0.3
# Distribution matching to VDA-S (stream) (measured from the raw, uncompressed depth maps of the same clip, both normalized per
# frame with the 1% / 99% quantiles): pooled CDF matching DA3 -> VDA gave y = x ** 0.9 (mean abs error 0.009 in
# normalized depth). < 1.0 raises the mid/background values = less foreground/background gap. 1.0 = off.
# Calibrated for DA3_MONO_L_DISPARITY_SHIFT = 0.4; re-measure if the shift is changed.
DA3_MONO_L_MATCH_GAMMA = 0.9
# load_state_dict(strict=False) may report harmless leftovers; more than this ratio of missing keys is an error.
DA3_MONO_L_MAX_MISSING_RATIO = 0.02

def _da3_ensure_einops_einsum():
    """einops.einsum() does not exist in the pinned einops==0.4.1, but depth_anything_3.utils.geometry imports it
    at module load time. Patch in an equivalent built on torch.einsum (nothing in the environment is upgraded)."""
    import einops
    if hasattr(einops, "einsum"):
        return

    def _einops_einsum_polyfill(*args):
        *tensors, pattern = args
        lhs, rhs = pattern.split("->")
        rhs = rhs.strip()
        lhs_terms = [s.strip() for s in lhs.split(",")]
        mapping = {}
        used = set()
        letters = iter("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

        def convert(term):
            out = []
            for tok in term.split():
                if tok == "...":
                    out.append("...")
                    continue
                if tok not in mapping:
                    letter = next(letters)
                    while letter in used:
                        letter = next(letters)
                    used.add(letter)
                    mapping[tok] = letter
                out.append(mapping[tok])
            return "".join(out)

        new_lhs = ",".join(convert(s) for s in lhs_terms)
        new_rhs = convert(rhs)
        return torch.einsum(f"{new_lhs}->{new_rhs}", *tensors)

    einops.einsum = _einops_einsum_polyfill

def _da3_prepare_import():
    if not (DA3_SRC / "depth_anything_3").is_dir():
        raise FileNotFoundError(
            f"[DA3-Mono-L] Depth-Anything-3 source not found at: {DA3_SRC}\n"
            "Run install-da3-mono-large.bat first."
        )
    _s = str(DA3_SRC)
    if _s not in sys.path:
        sys.path.insert(0, _s)
    _da3_ensure_einops_einsum()
    try:
        import warnings
        # omegaconf 2.4: DA3's cfg.py still calls the deprecated register_new_resolver(); harmless noise.
        warnings.filterwarnings("ignore", message=".*register_new_resolver.*", category=UserWarning)
    except Exception:
        pass

def _da3_mono_l_checkpoint_path() -> Path:
    ckpt = DA3_MONO_L_CKPT_DIR / DA3_MONO_L_CKPT_NAME
    if ckpt.exists():
        return ckpt
    DA3_MONO_L_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Info] DA3-Mono-L checkpoint not found. Downloading "
          f"({DA3_MONO_L_HF_REPO_ID}/{DA3_MONO_L_CKPT_NAME}, about 1.4 GB)...", flush=True)
    part = ckpt.with_name(ckpt.name + ".part")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=DA3_MONO_L_HF_REPO_ID, filename=DA3_MONO_L_CKPT_NAME)
        import shutil
        shutil.copyfile(downloaded, part)
    except Exception:
        import urllib.request
        url = (f"https://huggingface.co/{DA3_MONO_L_HF_REPO_ID}/resolve/main/"
               f"{DA3_MONO_L_CKPT_NAME}?download=true")
        urllib.request.urlretrieve(url, part)
    os.replace(part, ckpt)
    print("[Info] DA3-Mono-L checkpoint downloaded successfully.", flush=True)
    return ckpt

# ---- TensorRT for DA3-Mono-L -------------------------------------------------------------------
# What runs inside TensorRT: the whole network (DINOv2 ViT-L backbone + DPT head), static shape
# (1, 1, 3, H, W) -> (depth, sky), both (1, 1, H, W). Engines are built lazily per input shape by the
# generic TensorRTDepthModel above (PyTorch -> ONNX -> engine, cached in trt_cache/).
# What stays in PyTorch (fp32, exactly like the eager path): DA3's own sky post-processing
# (_process_mono_sky_estimation), the 1/(depth+shift) disparity conversion, the sky fade and the VDA match curve.
#
# Export-time changes (only while torch.onnx.export runs, restored right after, the eager fallback is untouched):
#   * Attention.fused_attn is turned off -> plain  softmax(q k^T) v  (TensorRT fuses this pattern itself and it
#     avoids depending on the ONNX symbolic of F.scaled_dot_product_attention of the installed torch).
#   * DinoVisionTransformer.interpolate_pos_encoding returns a tensor that was pre-computed once in eager PyTorch
#     for the fixed (H, W) -> no bicubic Resize node in the graph, numerically identical.
# Safeguards against a broken depth map (see _DA3TensorRTModel): dedicated CUDA stream + fixed I/O buffers + a host
# wait after every run (the generic path enqueues on PyTorch's default stream and never waits), an engine built
# without auxiliary streams, and a continuous check against eager PyTorch. A bad engine is rejected on the spot, the
# frame is replaced by the PyTorch result and that input shape runs in PyTorch from then on.
DA3_MONO_L_TRT_VALIDATE = True
DA3_MONO_L_TRT_MIN_CORR = 0.98        # Pearson correlation of the depth map (TensorRT vs PyTorch) must reach this
DA3_MONO_L_TRT_WORKSPACE_GB = 4
DA3_MONO_L_TRT_ENGINE_REV = "r2"      # bump when the exported graph / build config changes -> old cached engines are not reused
                                      # (r2: engine built with max_aux_streams = 0)
DA3_MONO_L_TRT_MAX_REL_ERR = 0.12     # mean |TensorRT - PyTorch| / mean |PyTorch| of the depth map must stay below this
DA3_MONO_L_TRT_SKY_MAX_ERR = 0.25     # mean |TensorRT - PyTorch| of the sky map must stay below this
DA3_MONO_L_TRT_CHECK_FIRST = 48       # compare every frame with PyTorch for the first N frames of an input shape ...
DA3_MONO_L_TRT_CHECK_EVERY = 60       # ... then every N-th frame (0-cost finite check runs on every frame anyway)
DA3_MONO_L_TRT_LOG_CHECKS = 3         # how many passed checks are printed (failures are always printed)
# Troubleshooting switch (no code edit): set the environment variable CD3D_DA3_TRT_FP32=1 to build/use an fp32 engine
# (slower, but rules out fp16 overflow as the cause of a broken depth map). CD3D_DA3_TRT_CHECK_EVERY overrides the interval.
DA3_MONO_L_TRT_FORCE_FP32 = os.environ.get("CD3D_DA3_TRT_FP32", "0").strip() == "1"
try:
    DA3_MONO_L_TRT_CHECK_EVERY = int(os.environ.get("CD3D_DA3_TRT_CHECK_EVERY", DA3_MONO_L_TRT_CHECK_EVERY))
except Exception:
    pass


class _AttrDict(dict):
    """Minimal stand-in for addict.Dict: DA3's _process_mono_sky_estimation uses output.depth / output.sky."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


class _DA3MonoLTRTExport(torch.nn.Module):
    """backbone + DPT head with fixed (H, W) as python ints: (1, 1, 3, H, W) -> (depth, sky).
    Shares the weights with the eager net (no copy). The sky post-processing is NOT part of the graph."""

    def __init__(self, net, H: int, W: int):
        super().__init__()
        self.backbone = net.backbone
        self.head = net.head
        self.H = int(H)
        self.W = int(W)

    def forward(self, x):
        feats, _ = self.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced")
        out = self.head(feats, self.H, self.W, patch_start_idx=0)
        depth = out["depth"]
        sky = out["sky"] if "sky" in out else torch.zeros_like(depth)
        return depth, sky


def _da3_trt_export_factory(net, holder):
    def _factory(x):
        H, W = int(x.shape[-2]), int(x.shape[-1])
        mod = _DA3MonoLTRTExport(net, H, W).eval()
        dummy = torch.randn(tuple(x.shape), device=x.device, dtype=torch.float32)
        # position embedding for this fixed size, computed by the model's own (eager) code
        bb = net.backbone.pretrained
        with torch.no_grad():
            tokens = torch.zeros(1, 1 + (H // bb.patch_size) * (W // bb.patch_size), bb.embed_dim,
                                 device=x.device, dtype=torch.float32)
            holder["pe"] = bb.interpolate_pos_encoding(tokens, H, W).detach().clone()
        return mod, dummy, ["out00", "out01"]
    return _factory


def _da3_trt_export_context(net, holder):
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        bb = net.backbone.pretrained
        toggled = []
        for m in net.modules():
            if getattr(m, "fused_attn", False):
                toggled.append(m)
                m.fused_attn = False
        pe = holder.get("pe")
        had_override = "interpolate_pos_encoding" in bb.__dict__
        orig_override = bb.__dict__.get("interpolate_pos_encoding")
        if pe is not None:
            bb.interpolate_pos_encoding = lambda x, w, h, _pe=pe: _pe.to(x.dtype)
        try:
            yield
        finally:
            for m in toggled:
                m.fused_attn = True
            if pe is not None:
                if had_override:
                    bb.interpolate_pos_encoding = orig_override
                else:
                    bb.__dict__.pop("interpolate_pos_encoding", None)
    return _ctx


class _DA3TensorRTModel(TensorRTDepthModel):
    """TensorRTDepthModel for DA3-Mono-L plus the safeguards that keep a broken depth map from reaching the output.

    (a) The engine is built outside inference_mode (ONNX tracing needs normal tensors; the caller runs under
        torch.inference_mode()) and without auxiliary CUDA streams (max_aux_streams = 0).
    (b) Execution runs on a dedicated CUDA stream with fixed input/output buffers. PyTorch's stream is made to
        wait for the input, the host waits until the engine has finished, and copies of the outputs are returned.
        The generic path enqueues on PyTorch's current (usually the default) stream with fresh buffers and never
        waits; a consumer that reads before the engine is done sees a half-written map (the top rows right, the rows
        below a tile boundary stale/garbage) - exactly the symptom seen in the broken videos.
    (c) Every frame is checked for non-finite values; the first CHECK_FIRST frames and then every CHECK_EVERY-th
        frame are also compared with eager PyTorch. A failing frame is replaced by the PyTorch result and that input
        shape runs in PyTorch from then on."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._exec_lock = threading.Lock()
        self._streams = {}
        self._frames = {}
        self._passed_logged = 0

    # -- build ---------------------------------------------------------------
    def _configure_builder(self, builder, network, config):
        try:
            config.max_aux_streams = 0      # no inter-stream scheduling inside the engine
        except Exception as e:
            print(f"[TensorRT] {self._tag}: could not disable auxiliary streams ({e}).", flush=True)

    def _report_build_warnings(self):
        """Print TensorRT's warnings (the console is hidden, so they were lost before) and keep them in the log file."""
        msgs = list(self._logger.msgs)
        self._logger.msgs.clear()
        warns = [m for m in msgs if m.startswith("[WARNING]")]
        if not warns:
            return
        seen, uniq = set(), []
        for m in warns:
            k = m[:120]
            if k not in seen:
                seen.add(k)
                uniq.append(m)
        shown = [m[:240] for m in uniq[:8]]
        print(f"[TensorRT] {self._tag}: {len(warns)} TensorRT warning(s), first {len(shown)} distinct "
              f"(full list: trt_cache/trt_build_error.log):", flush=True)
        for m in shown:
            print("    " + m, flush=True)
        _trt_dump_error(self._tag, "TensorRT warnings:\n" + "\n".join(uniq[:60]))

    # -- run -----------------------------------------------------------------
    def _stream_for(self, device):
        key = str(device)
        st = self._streams.get(key)
        if st is None:
            st = torch.cuda.Stream(device=device)
            self._streams[key] = st
        return st

    def _enqueue(self, entry, x):
        with self._exec_lock:
            dev = x.device
            st = self._stream_for(dev)
            bufs = entry.get("bufs")
            if bufs is None:
                # persistent buffers (normal tensors, so they also work outside inference_mode)
                with torch.inference_mode(False):
                    in_buf = torch.empty(tuple(x.shape), device=dev, dtype=entry["in_dtype"])
                    out_bufs = [torch.empty(shp, device=dev, dtype=dt) for _n, shp, dt in entry["outs"]]
                bufs = entry["bufs"] = (in_buf, out_bufs)
            in_buf, out_bufs = bufs
            # 1) the TensorRT stream waits for everything PyTorch has queued so far (the input frame, the previous clones)
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(dev))
            st.wait_event(ev)
            with torch.cuda.stream(st):
                in_buf.copy_(x)                    # dtype / contiguity conversion
            ctx = entry["context"]
            ctx.set_tensor_address(entry["in_name"], in_buf.data_ptr())
            for (name, _shp, _dt), o in zip(entry["outs"], out_bufs):
                ctx.set_tensor_address(name, o.data_ptr())
            if not ctx.execute_async_v3(st.cuda_stream):
                raise RuntimeError("execute_async_v3 returned False")
            # 2) the host waits until the engine is completely done: whatever stream reads the outputs sees all rows
            st.synchronize()
            outs = [o.clone() for o in out_bufs]   # the buffers are reused by the next call
        return outs[0] if len(outs) == 1 else tuple(outs)

    # -- checks ----------------------------------------------------------------
    @staticmethod
    def _bad_row_hint(ref_depth, depth):
        try:
            e = (ref_depth.float() - depth.float()).abs()
            e = e.reshape(-1, e.shape[-2], e.shape[-1]).mean(dim=(0, 2))
            level = float(ref_depth.float().abs().mean()) + 1e-8
            bad = (e > 0.2 * level).nonzero()
            if bad.numel() == 0:
                return ""
            return f"; first bad row {int(bad[0])} of {int(e.numel())} ({int(bad.numel())} bad rows)"
        except Exception:
            return ""

    @classmethod
    def _compare(cls, ref, got):
        rd, rs = ref
        d, s = got
        a = rd.float().flatten()
        b = d.float().flatten()
        if not (bool(torch.isfinite(b).all()) and bool(torch.isfinite(s.float()).all())):
            return False, "non-finite values in the TensorRT output (fp16 overflow?)"
        corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
        rel = float((a - b).abs().mean() / (a.abs().mean() + 1e-8))
        sky_err = float((rs.float() - s.float()).abs().mean())
        info = f"depth corr={corr:.5f}, mean rel. err={rel * 100:.2f}%, sky mean abs err={sky_err:.4f}"
        if not (corr >= DA3_MONO_L_TRT_MIN_CORR) or not (rel <= DA3_MONO_L_TRT_MAX_REL_ERR):
            return False, info + cls._bad_row_hint(rd, d)
        if not (sky_err <= DA3_MONO_L_TRT_SKY_MAX_ERR):
            return False, info + " (sky map differs)"
        return True, info

    def _validate(self, x, key):
        with torch.inference_mode():
            ref = self._fallback(x)
            got = TensorRTDepthModel.__call__(self, x)
        if self._engines.get(key) is None:
            return False, "TensorRT execution failed"
        return self._compare(ref, got)

    def _prepare(self, x):
        key = (tuple(x.shape), x.dtype)
        with torch.inference_mode(False):
            entry = super()._prepare(x)
        self._report_build_warnings()
        if entry is None or not DA3_MONO_L_TRT_VALIDATE:
            return entry
        _COMPILE_BUSY.set()
        try:
            ok, info = self._validate(x, key)
        except Exception as e:
            ok, info = False, f"validation error: {type(e).__name__}: {e}"
        finally:
            _COMPILE_BUSY.clear()
        if ok:
            print(f"[TensorRT] {self._tag}: engine check vs PyTorch OK ({info}).", flush=True)
            return entry
        print(f"[TensorRT] {self._tag}: engine REJECTED ({info}) -> using PyTorch for input {tuple(x.shape)}.",
              flush=True)
        _trt_dump_error(self._tag, f"engine rejected for input {tuple(x.shape)}: {info}")
        self._engines[key] = None
        return None

    def __call__(self, x):
        key = (tuple(x.shape), x.dtype)
        out = super().__call__(x)          # engine result, or the PyTorch fallback when the engine is unusable
        if self._engines.get(key) is None:
            return out
        n = self._frames.get(key, 0) + 1
        self._frames[key] = n
        every = int(DA3_MONO_L_TRT_CHECK_EVERY)
        due = n <= int(DA3_MONO_L_TRT_CHECK_FIRST) or (every > 0 and n % every == 0)
        ref = None
        try:
            d, s = out
            if due:
                ref = self._fallback(x)
                ok, info = self._compare(ref, out)
            else:
                ok = bool(torch.isfinite(d).all() & torch.isfinite(s).all())
                info = "non-finite values in the TensorRT output (fp16 overflow?)"
        except Exception as e:
            print(f"[TensorRT] {self._tag}: frame check skipped ({type(e).__name__}: {e})", flush=True)
            return out
        if ok:
            if due and self._passed_logged < int(DA3_MONO_L_TRT_LOG_CHECKS):
                self._passed_logged += 1
                print(f"[TensorRT] {self._tag}: frame {n} check vs PyTorch OK ({info}).", flush=True)
            return out
        msg = (f"engine REJECTED at frame {n} ({info}) -> this and all following frames of input "
               f"{tuple(x.shape)} use PyTorch. (If this keeps happening try CD3D_DA3_TRT_FP32=1.)")
        print(f"[TensorRT] {self._tag}: {msg}", flush=True)
        _trt_dump_error(self._tag, msg)
        self._engines[key] = None
        return ref if ref is not None else self._fallback(x)


class _DA3MonoLForward:
    """predictor.model for estimate_depth_raw(): (1, 1, 3, H, W) float tensor in -> (1, H, W) depth-like out.
    Output direction: -depth (near = large) so that the converter's `max - x` gives the depth (near = small)."""

    def __init__(self, net, device: str, use_amp: bool, amp_dtype, inverse: bool = False):
        self.net = net
        self.device = device
        self.use_amp = bool(use_amp)
        self.amp_dtype = amp_dtype
        self.inverse = bool(inverse)
        self._compiled = False
        self._orig_backbone = None
        self._orig_sky_fn = None
        self._trt = None

    def enable_trt(self, fp16: bool, ckpt_path=None) -> None:
        """Run backbone + DPT head through TensorRT (engines are built lazily per input shape, see
        _DA3TensorRTModel). The eager PyTorch path stays available as the per-shape fallback."""
        holder = {}
        self._trt = _DA3TensorRTModel(
            self.net, fallback=self._eager_raw, tag=f"DA3-Mono-L-{DA3_MONO_L_TRT_ENGINE_REV}",
            fp16=bool(fp16) and not DA3_MONO_L_TRT_FORCE_FP32,
            ckpt_path=ckpt_path, workspace_gb=DA3_MONO_L_TRT_WORKSPACE_GB,
            export_factory=_da3_trt_export_factory(self.net, holder),
            export_context=_da3_trt_export_context(self.net, holder))

    def _eager_raw(self, x):
        """Eager PyTorch twin of the TensorRT graph: (depth, sky) before DA3's sky post-processing, with the same
        precision split as DepthAnything3Net.forward (backbone under autocast, DPT head in fp32)."""
        net = self.net
        H, W = int(x.shape[-2]), int(x.shape[-1])
        with torch.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.use_amp):
            feats, _ = net.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced")
            with torch.autocast(device_type=self.device, enabled=False):
                out = net.head(feats, H, W, patch_start_idx=0)
        depth = out["depth"]
        sky = out["sky"] if "sky" in out else torch.zeros_like(depth)
        return depth, sky

    def _run_trt(self, x):
        depth, sky = self._trt(x)      # TensorRT engine, or _eager_raw for a shape whose engine is unusable
        out = _AttrDict(depth=depth.float(), sky=sky.float())
        # same sky handling as the eager net.forward() (sky pixels -> 99% non-sky depth)
        return self.net._process_mono_sky_estimation(out)

    def enable_compile(self) -> bool:
        """torch.compile the heavy parts only (DINOv2 backbone + DPT head body), static shapes.

        The net's own forward stays eager on purpose: its tail (_process_mono_sky_estimation) uses boolean-mask
        indexing, a data-dependent `.sum()` branch, torch.quantile and torch.randint, which break / recompile
        torch.compile. The sky handling is done by this wrapper anyway (iw3 style), so that step is bypassed while
        compiled (iw3 also keeps all sky processing outside of the compiled model)."""
        if self._compiled:
            return True
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.cache_size_limit = max(int(getattr(_dynamo.config, "cache_size_limit", 8)), 64)
            _dynamo.config.automatic_dynamic_shapes = False
        except Exception as _e:
            print(f"[torch.compile] Could not tune Dynamo config: {_e}", flush=True)
        net = self.net
        print("[torch.compile] Compiling DA3-Mono-L (backbone + DPT head, mode=default, dynamic=False)...", flush=True)
        self._orig_backbone = net.backbone
        self._orig_sky_fn = getattr(net, "_process_mono_sky_estimation", None)
        net.backbone = torch.compile(net.backbone, dynamic=False)
        net.head._forward_impl = torch.compile(net.head._forward_impl, dynamic=False)
        net._process_mono_sky_estimation = lambda output: output
        self._compiled = True
        return True

    def _disable_compile(self):
        net = self.net
        if self._orig_backbone is not None:
            net.backbone = self._orig_backbone
        try:
            if "_forward_impl" in net.head.__dict__:
                del net.head.__dict__["_forward_impl"]
        except Exception:
            pass
        # keep the sky bypass (same result as while compiled); eager and compiled then match
        self._compiled = False

    def _run_net(self, x):
        if self._trt is not None:
            return self._run_trt(x)
        if not self._compiled:
            return self.net(x)
        # The input may be a normal tensor (warmup / edge-fix measurement) or an inference tensor (real frames);
        # Dynamo guards on that (DispatchKeySet mismatch -> an extra recompile). Always feed an inference tensor
        # (same idea as iw3's "preserve x.is_inference()"). The copy is ~8 MB, negligible.
        with torch.inference_mode():
            x = x.clone()
        try:
            return self.net(x)
        except Exception as e:
            print(f"[Warning] DA3-Mono-L torch.compile failed ({type(e).__name__}: {str(e)[:300]}) "
                  f"-> falling back to eager PyTorch.", flush=True)
            self._disable_compile()
            return self.net(x)

    def __call__(self, x):
        with torch.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.use_amp):
            out = self._run_net(x)
        depth = out["depth"].float()
        depth = depth.reshape(-1, depth.shape[-2], depth.shape[-1])[:1]
        depth = torch.nan_to_num(depth)
        if self.inverse:
            # iw3 (depth_anything_v3_model.py _forward): disparity = 1 / (depth + shift) * (1 - sky_weight)
            disp = 1.0 / (depth + DA3_MONO_L_DISPARITY_SHIFT).clamp_min(1e-6)
            sky = None
            try:
                sky = out["sky"] if "sky" in out else None
            except Exception:
                sky = None
            if sky is None:
                if not getattr(self, "_warned_no_sky", False):
                    print("[Warning] DA3-Mono-L: 'sky' output not found; sky handling is skipped.", flush=True)
                    self._warned_no_sky = True
                return self._match_vda(disp)
            sky = torch.nan_to_num(sky.float().reshape(-1, sky.shape[-2], sky.shape[-1])[:1])
            th = DA3_MONO_L_SKY_THRESH
            sky_mask = sky > th
            sky_weight = (sky.clamp(th, 1.0) - th) / (1.0 - th)
            disp = disp * (1.0 - sky_weight)
            non_sky_pixels = sky_mask.shape[-2] * sky_mask.shape[-1] - sky_mask.sum(dim=(-2, -1), keepdim=True)
            disp = torch.where(non_sky_pixels < 10, torch.zeros_like(disp), disp)
            return self._match_vda(disp)
        return -depth

    @staticmethod
    def _match_vda(disp):
        # Same per-frame normalization as normalize_depth_gpu() (1% / 99% quantiles, clamp 0..1) so that the curve acts
        # on the values the converter really sees. A monotone curve with f(0)=0, f(1)=1 commutes with the quantiles
        # and with the converter's `max - x` inversion, so the later normalization does not undo it.
        g = float(DA3_MONO_L_MATCH_GAMMA)
        if abs(g - 1.0) < 1e-6:
            return disp
        flat = disp.reshape(-1)
        sample = flat[::8] if flat.numel() > 100_000 else flat
        lo = torch.quantile(sample, 0.01)
        hi = torch.quantile(sample, 0.99)
        x = ((disp - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
        return x.pow(g)

    def to(self, *a, **kw):
        self.net = self.net.to(*a, **kw)
        return self

    def eval(self):
        self.net.eval()
        return self

class DA3MonoLAdapter:
    """Same interface as DistillAnyDepthAdapter (image2tensor + model) so estimate_depth_raw() dispatches to it
    through its generic branch. The input is a 5-D tensor (B=1, N=1, 3, H, W) as DA3 expects."""

    def __init__(self, model, device: str, input_size: int, fp16: bool = True, ensure_multiple_of: int = 14):
        self.model = model
        self.device = device
        self.input_size = input_size
        self.fp16 = fp16 and device == "cuda"
        self.ensure_multiple_of = ensure_multiple_of

    def image2tensor(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]
        m = self.ensure_multiple_of
        scale = self.input_size / min(orig_h, orig_w)
        new_h = max(m, int(round(orig_h * scale / m)) * m)
        new_w = max(m, int(round(orig_w * scale / m)) * m)

        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        img = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
        img = img.astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))

        # fp32 input: the autocast context inside _DA3MonoLForward casts it (the weights stay fp32, like DA3's own api)
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
        return x, orig_h, orig_w

def load_da3_mono_l(input_size=504, device="cuda", fp16=True, compile_model_flag=False, tensorrt_flag=False):
    from safetensors.torch import load_file

    _da3_prepare_import()
    from depth_anything_3.cfg import load_config, create_object

    cfg_path = DA3_SRC / "depth_anything_3" / "configs" / f"{DA3_MONO_L_CONFIG_NAME}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"[DA3-Mono-L] config not found: {cfg_path}")
    ckpt = _da3_mono_l_checkpoint_path()

    use_cuda = torch.cuda.is_available() and str(device) == "cuda"
    dev = "cuda" if use_cuda else "cpu"

    net = create_object(load_config(str(cfg_path)))

    def _strip_prefix(sd):
        # The Hub checkpoint is the state dict of DepthAnything3 (self.model = network): keys start with "model."
        return {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}

    state_dict = load_file(str(ckpt))
    model_keys = set(net.state_dict().keys())
    candidates = [("direct", _strip_prefix(state_dict))]
    try:
        from depth_anything_3.utils.model_loading import convert_general_state_dict
        candidates.append(("converted", _strip_prefix(convert_general_state_dict(dict(state_dict)))))
    except Exception:
        pass
    _kind, best = max(candidates, key=lambda c: len(model_keys & set(c[1].keys())))
    missing, unexpected = net.load_state_dict(best, strict=False)
    _ratio = len(missing) / max(1, len(model_keys))
    if _ratio > DA3_MONO_L_MAX_MISSING_RATIO:
        raise RuntimeError(
            f"[DA3-Mono-L] checkpoint does not match the network: {len(missing)}/{len(model_keys)} keys missing "
            f"(e.g. {list(missing)[:3]}), {len(unexpected)} unexpected (e.g. {list(unexpected)[:3]})."
        )
    print(f"[DA3-Mono-L] weights loaded ({_kind} keys): {len(missing)} missing / {len(unexpected)} unexpected.",
          flush=True)

    net = net.to(dev).eval()

    # TensorRT (if requested and available) takes priority over torch.compile for this model.
    want_trt = bool(tensorrt_flag and use_cuda and check_tensorrt_support())
    want_compile = bool(compile_model_flag and use_cuda and not want_trt and check_compile_support(dev))
    if tensorrt_flag and not want_trt:
        print("[TensorRT] Requested but unavailable -> using PyTorch.", flush=True)
    if compile_model_flag and not want_compile and not want_trt:
        print("[DA3-Mono-L] torch.compile is unavailable here -> running eager PyTorch.", flush=True)

    use_amp = bool(fp16 and use_cuda)
    amp_dtype = torch.float16
    if use_amp:
        try:
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16     # same choice as DA3's own api (bf16 when the GPU supports it)
        except Exception:
            pass

    model = _DA3MonoLForward(net, device=dev, use_amp=use_amp, amp_dtype=amp_dtype,
                             inverse=DA3_MONO_L_USE_INVERSE_DEPTH)

    isz = int(input_size)
    if isz % 14 != 0:
        isz += (14 - isz % 14)

    print(f"[DA3-Mono-L] ready (device={dev}, autocast={'off' if not use_amp else str(amp_dtype).split('.')[-1]}, "
          f"depth resolution={isz}).", flush=True)
    adapter = DA3MonoLAdapter(model, device=dev, input_size=isz, fp16=use_amp, ensure_multiple_of=14)
    if want_trt:
        try:
            model.enable_trt(fp16=use_amp, ckpt_path=ckpt)
            print("[TensorRT] DA3-Mono-L: TensorRT backend ON (engines are built lazily per input shape; "
                  f"{'fp16' if getattr(model._trt, '_fp16', use_amp) else 'fp32'}).", flush=True)
        except Exception as e:
            model._trt = None
            print(f"[Warning] DA3-Mono-L TensorRT setup failed ({type(e).__name__}: {e}) -> eager PyTorch.",
                  flush=True)
    if want_compile:
        try:
            model.enable_compile()
            # estimate_depth_raw() warms the compiled model up once per distinct input shape
            adapter._cd3d_compiled = True
            adapter._cd3d_warm_shapes = set()
            adapter._cd3d_tag = "DA3-Mono-L"
        except Exception as e:
            print(f"[Warning] DA3-Mono-L torch.compile setup failed ({type(e).__name__}: {e}) -> eager PyTorch.",
                  flush=True)
            model._disable_compile()
    return adapter

def estimate_depth_raw(predictor, bgr: np.ndarray) -> torch.Tensor:
    if isinstance(predictor, VDAStreamingAdapter):
        return predictor.infer(bgr)

    image_tensor, _, _ = predictor.image2tensor(bgr)
    _compiled = getattr(predictor, "_cd3d_compiled", False)
    if _compiled and tuple(image_tensor.shape) not in predictor._cd3d_warm_shapes:
        _warmup_compiled_depth_model(predictor, image_tensor)
    with torch.inference_mode():
        if _compiled:
            try:
                torch.compiler.cudagraph_mark_step_begin()
            except Exception:
                pass
        depth = predictor.model(image_tensor)
        if _compiled:
            # reduce-overhead (CUDA graphs) reuses its output buffer on the next call
            depth = depth.clone()
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

_DEPTH_AA_SIGMA = 1.00
_DEPTH_AA_KERNEL_CACHE = {}

def _get_depth_aa_kernel_2d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (sigma, device, dtype)
    if key not in _DEPTH_AA_KERNEL_CACHE:
        radius = int(math.ceil(sigma * 3.0))
        size = 2 * radius + 1
        coords = torch.arange(size, device=device, dtype=torch.float32) - radius
        g1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g1d = g1d / g1d.sum()
        g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)
        g2d = g2d / g2d.sum()
        _DEPTH_AA_KERNEL_CACHE[key] = g2d.view(1, 1, size, size).to(dtype=dtype)
    return _DEPTH_AA_KERNEL_CACHE[key]

def apply_depth_aa_gpu(depth: torch.Tensor, sigma: float = _DEPTH_AA_SIGMA) -> torch.Tensor:
    if sigma <= 0.0:
        return depth
    kernel = _get_depth_aa_kernel_2d(sigma, depth.device, depth.dtype)
    pad = kernel.shape[-1] // 2
    depth_padded = F.pad(depth, (pad, pad, pad, pad), mode="replicate")
    return F.conv2d(depth_padded, kernel)

# ---------------------------------------------------------------------------
# Depth linear compression (per model, follows the stereo strength)
#   Same behaviour / calibration as live3d.py (keep the two tables in sync).
#   Up to the limit strength T nothing changes. Above it the effective strength only grows with slope s:
#       effective(D) = T + s * (D - T)        width k(D) = effective(D) / D        (D <= T -> k = 1)
#   The normalized depth [0,1] is squeezed around 0.5 into [0.5 - k/2, 0.5 + k/2], so the foreground/background
#   gap (= the width of the disocclusion / inpaint band) stops growing as fast as the strength.
#   Models that are not listed here are never touched. ZipDepth: T = 3.0, s = 0.5 (nothing up to 3.0; 5.0 -> perceived 4.0).
# ---------------------------------------------------------------------------
DEPTH_LINEAR_COMP_PARAMS = {
    "vda_s":         (1.0, 0.25),
    "vda_s_metric":  (1.0, 0.5),
    "distill_any_s": (1.0, 0.5),
    "dav2_s":        (1.0, 0.5),
    "zipdepth":      (1.0, 0.5),
    # DA3-Mono-L: same compression as VDA-S (stream) (its depth distribution is matched to VDA-S by DA3_MONO_L_MATCH_GAMMA).
    "da3_mono_l":    (1.0, 0.25),
}
# Models whose Auto Mode edge fix / EMA values are NOT switched to the Light Inpaint (stream) values when the
# compression is ON. (ZipDepth now also uses the Light Inpaint (stream) edge/EMA values when compression is ON.)
DEPTH_LINEAR_COMP_NO_EDGE_OVERRIDE = set()

# Auto Mode edge fix while the compression is ON (any warp method): the same rule as Auto Mode + compression OFF +
# inpaint (+1 per 1.0 strength; VDA-S Metric: +1 per 0.5 strength), but applied to the EFFECTIVE
# strength (limit + slope * (D - limit)) instead of the slider value. At or below the limit nothing is compressed,
# so the values are identical to the compression-OFF values there. EMA is unchanged.
DEPTH_LINEAR_COMP_EDGE_HALF_STEP = ("vda_s_metric",)

def depth_linear_comp_auto_edge(divergence, depth_model) -> int:
    d = float(divergence)
    p = DEPTH_LINEAR_COMP_PARAMS.get(depth_model)
    if p is None:
        eff = d
    else:
        limit, slope = p
        eff = d if d <= limit else limit + slope * (d - limit)
    unit = 0.5 if depth_model in DEPTH_LINEAR_COMP_EDGE_HALF_STEP else 1.0
    return max(1, min(EDGE_FIX_MAX, int(math.ceil(eff / unit - 1e-9))))

DEPTH_LINEAR_COMP_MIN_WIDTH = 0.05

def depth_linear_comp_width(divergence, depth_model) -> float:
    p = DEPTH_LINEAR_COMP_PARAMS.get(depth_model)
    if p is None:
        return 1.0
    try:
        d = float(divergence)
    except Exception:
        return 1.0
    limit, slope = p
    if not (d == d) or d <= limit:
        return 1.0
    eff = limit + slope * (d - limit)
    return min(1.0, max(DEPTH_LINEAR_COMP_MIN_WIDTH, eff / d))

# Fixed depth range scale that is ALWAYS applied (also with Depth Linear Compression OFF), to equalize the raw 3D
# strength of a model with VDA-S (stream). Measured by eye on the raw (uncompressed) result: DA3-Mono-L at strength 3.0
# looks like VDA-S at strength 5.0, so DA3-Mono-L's depth range is squeezed to 3/5 = 0.6 (same mechanism as the
# compression: normalized depth squeezed around 0.5, convergence follows). Then DA3 at strength S == VDA-S at S.
DEPTH_BASE_SCALE = {
    "da3_mono_l": 0.93,
}

def depth_base_scale(depth_model) -> float:
    return float(DEPTH_BASE_SCALE.get(depth_model, 1.0))

def apply_depth_linear_comp_gpu(depth: torch.Tensor, width: float) -> torch.Tensor:
    if width >= 1.0 - 1e-4:
        return depth
    # out-of-place on purpose: `depth` may share memory with other buffers
    return depth.mul(width).add(0.5 - 0.5 * width)

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

WARP_FILL_MODE_STRETCH   = "stretch"
WARP_FILL_MODE_FAST      = "fast"
WARP_FILL_MODE_MEDIUM    = "medium"
WARP_FILL_MODE_LIGHT     = "light"

# Light Inpaint (iw3 light_video_inpaint_v1, video model): same overlap / dilation as Medium Inpaint,
# plus the stretched-band mask (reduces background stretching).
LIGHT_INPAINT_MODEL_FILENAME   = "light_video_inpaint.pth"

# Light Inpaint (stream): image model ported from live3d.py (separate model file / structure from "Light Inpaint")
WARP_FILL_MODE_LIGHT_STREAM    = "light_stream"
LIGHT_INPAINT_STREAM_MODEL_FILENAME     = "light_inpaint.pth"
LIGHT_INPAINT_STREAM_INNER_DILATION     = 2
LIGHT_INPAINT_STREAM_OUTER_DILATION     = 2
# temporal stabilisation (the model is an IMAGE model: every call hallucinates a different texture)
LIGHT_INPAINT_STREAM_CACHE_MAX_INTERVAL = 8     # force a model re-run at least every N frames
LIGHT_INPAINT_STREAM_CACHE_CHANGE_THRESH = 0.06  # re-run when this fraction of the hole area is newly uncovered
LIGHT_INPAINT_STREAM_HOLD_FRAMES        = 1      # mask hysteresis: a pixel stays "inpainted" for N frames after it leaves the hole (0 = off)
LIGHT_INPAINT_STREAM_FADE_FRAMES        = 1      # on a model refresh, cross-fade old -> new hole texture over N frames (1 = hard switch)

# Medium Inpaint (iw3 light video inpaint) - fixed settings, not exposed in the UI
MEDIUM_INPAINT_MODEL_FILENAME  = "medium_inpaint.pth"
MEDIUM_INPAINT_PRE_PADDING     = 3
MEDIUM_INPAINT_POST_PADDING    = 3
MEDIUM_INPAINT_INNER_DILATION  = 2
MEDIUM_INPAINT_OUTER_DILATION  = 2

# Inpaint Resolution (ported from live3d.py): shared by Light Inpaint, Medium Inpaint (video models) and
# Light Inpaint (stream) (image model). The model only ever runs on a frame downscaled to this height, the
# full-resolution picture is kept outside the hole. conversion3d.py reads this value fresh every time a
# conversion job starts, so no restart notice is needed.
# Options: 512 / 720 / 1080 / MAX. MAX (= 0, stored as blank in config/presets) means NO limit: the model runs at the source/output resolution.
# Default: MAX for Light Inpaint (stream), Light Inpaint and Medium Inpaint.
INPAINT_RES_PRESETS        = (512, 720, 1080)
INPAINT_RES_UNLIMITED      = 0
INPAINT_RES_MIN            = 64
INPAINT_RES_MAX            = 4320
INPAINT_RES_MAX_LABEL      = "MAX"  # shown in the UI (English only, in every language) for 'no limit'
INPAINT_RES_DEFAULT_STREAM = 0      # MAX = no limit
INPAINT_RES_DEFAULT_VIDEO  = 0      # MAX = no limit
INPAINT_RES_DEFAULT        = INPAINT_RES_DEFAULT_VIDEO
_INPAINT_RES = [INPAINT_RES_DEFAULT]


def clamp_inpaint_res(value) -> int:
    """'' / None / 0 -> 0 (no limit). Any number is accepted and clamped to 64..4320 (custom input allowed)."""
    s_ = "" if value is None else str(value).strip()
    if s_ == "" or s_.lower() == INPAINT_RES_MAX_LABEL.lower():
        return INPAINT_RES_UNLIMITED
    try:
        v = int(float(s_))
    except Exception:
        return INPAINT_RES_DEFAULT
    if v <= 0:
        return INPAINT_RES_UNLIMITED
    return max(INPAINT_RES_MIN, min(INPAINT_RES_MAX, v))


def inpaint_res_text(value) -> str:
    """Storage form (config / presets): blank = no limit."""
    v = clamp_inpaint_res(value)
    return "" if v == INPAINT_RES_UNLIMITED else str(v)


def inpaint_res_display(value) -> str:
    """UI form: 'MAX' (English only) = no limit."""
    v = clamp_inpaint_res(value)
    return INPAINT_RES_MAX_LABEL if v == INPAINT_RES_UNLIMITED else str(v)


def inpaint_res_default_for_mode(mode):
    """Default Inpaint Resolution for a warp/fill mode (None = mode does not use it, leave as is)."""
    if mode == WARP_FILL_MODE_LIGHT_STREAM:
        return INPAINT_RES_DEFAULT_STREAM
    if mode in (WARP_FILL_MODE_LIGHT, WARP_FILL_MODE_MEDIUM):
        return INPAINT_RES_DEFAULT_VIDEO
    return None


def set_inpaint_res(value) -> int:
    _INPAINT_RES[0] = clamp_inpaint_res(value)
    return _INPAINT_RES[0]


def get_inpaint_res() -> int:
    """0 = no limit (blank option)."""
    return int(_INPAINT_RES[0])

FAST_INPAINT_RADIUS   = 2
FAST_INPAINT_MAX_ITERS = 24
FAST_INPAINT_COLOR_SIGMA = 0.10
FAST_BLUR_KERNEL_PX   = 5
FAST_BLUR_FEATHER_MIN = 2
FAST_BLUR_FEATHER_MAX = 6
FAST_GRAIN_STRENGTH   = 0.020

TRUE_FAST_INPAINT_RADIUS    = 3
TRUE_FAST_SAFETY_MAX_ITERS  = 4096

INPAINT_BLUR_KERNEL_PX              = 5

INPAINT_GRAIN_STRENGTH = 0.015
_INPAINT_GRAIN_CACHE: dict = {}

def _get_inpaint_grain(device, dtype, h: int, w: int) -> torch.Tensor:
    key = (str(device), str(dtype), int(h), int(w))
    cached = _INPAINT_GRAIN_CACHE.get(key)
    if cached is None or cached.device != device or cached.dtype != dtype or cached.shape[-2:] != (h, w):
        cached = torch.randn((1, 1, h, w), device=device, dtype=dtype)
        _INPAINT_GRAIN_CACHE[key] = cached
    return cached

@torch.inference_mode()
def blur_hole_region_gpu(rgb: torch.Tensor, mask: torch.Tensor,
                          kernel_px: int = INPAINT_BLUR_KERNEL_PX,
                          feather_px: "int | None" = None) -> torch.Tensor:
    kernel_px = int(kernel_px)
    if kernel_px <= 1:
        return rgb
    mask = mask.to(dtype=rgb.dtype)
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    hole = mask > 0.5
    if not hole.any():
        return rgb
    B, C, H, W = rgb.shape
    device = rgb.device
    if feather_px is None or feather_px <= 0:
        feather_px = max(1, kernel_px)
    idx          = torch.arange(W, device=device, dtype=torch.long).view(1, 1, 1, W).expand(B, 1, H, W)
    valid        = ~hole
    left_idx     = torch.cummax(torch.where(valid, idx, torch.full_like(idx, -1)), dim=-1).values
    right_idx    = torch.flip(torch.cummin(torch.flip(torch.where(valid, idx, torch.full_like(idx, W)), dims=[-1]),
                                            dim=-1).values, dims=[-1])
    dist_to_edge = torch.minimum(idx - left_idx, right_idx - idx).clamp(min=0).to(rgb.dtype)
    blend        = (dist_to_edge / float(feather_px)).clamp(0.0, 1.0)
    pad          = kernel_px // 2
    padded       = F.pad(rgb, (pad, pad, pad, pad), mode="replicate")
    blurred      = F.avg_pool2d(padded, kernel_size=kernel_px, stride=1)
    softened     = rgb * (1.0 - blend) + blurred * blend

    noise = _get_inpaint_grain(device, rgb.dtype, H, W)
    softened = softened + noise * (INPAINT_GRAIN_STRENGTH * blend)

    return torch.where(hole.expand_as(rgb), softened, rgb)

@torch.inference_mode()
def nearest_fill_inpaint_gpu(rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=rgb.dtype)
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    B, C, H, W = rgb.shape
    device      = rgb.device
    hole        = mask > 0.5
    if not hole.any():
        return rgb
    idx        = torch.arange(W, device=device, dtype=torch.long).view(1, 1, 1, W).expand(B, 1, H, W)
    valid      = ~hole
    left_src   = torch.where(valid, idx, torch.full_like(idx, -1))
    left_idx   = torch.cummax(left_src, dim=-1).values
    right_src  = torch.where(valid, idx, torch.full_like(idx, W))
    right_idx  = torch.flip(torch.cummin(torch.flip(right_src, dims=[-1]), dim=-1).values, dims=[-1])
    left_dist  = idx - left_idx
    right_dist = right_idx - idx
    has_left   = left_idx >= 0
    has_right  = right_idx < W
    use_left   = has_left & (~has_right | (left_dist <= right_dist))
    fill_idx   = torch.where(use_left, left_idx, right_idx).clamp(0, W - 1)
    gathered   = torch.gather(rgb, dim=-1, index=fill_idx.expand(-1, C, -1, -1))
    return torch.where(hole.expand_as(rgb), gathered, rgb)

@torch.inference_mode()
def true_fast_isotropic_inpaint_gpu(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    radius: int = TRUE_FAST_INPAINT_RADIUS,
    max_iters: int = TRUE_FAST_SAFETY_MAX_ITERS,
) -> torch.Tensor:
    mask = mask.to(dtype=rgb.dtype)
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    B, C, H, W = rgb.shape
    device, dtype = rgb.device, rgb.dtype
    hole = mask > 0.5
    if not hole.any():
        return rgb

    out = rgb.clone()
    k = 2 * int(radius) + 1
    ones_c = torch.ones((C, 1, k, k), device=device, dtype=dtype)
    ones_1 = torch.ones((1, 1, k, k), device=device, dtype=dtype)

    for _ in range(int(max_iters)):
        valid = (~hole).to(dtype)
        color_sum = F.conv2d(out * valid, ones_c, padding=radius, groups=C)
        count = F.conv2d(valid, ones_1, padding=radius)
        newly = hole & (count > 0.5)
        if not newly.any():
            break
        filled = color_sum / count.clamp(min=1e-6)
        out = torch.where(newly.expand_as(out), filled, out)
        hole = hole & ~newly
        if not hole.any():
            break
    return out

def monobw_warp_one_fast(
    rgb, depth, divergence, convergence, shift_sign=1.0,
    preserve_screen_border: bool = False,
    hole_threshold_px: float = 1.5,
    true_fast_radius: int = TRUE_FAST_INPAINT_RADIUS,
):
    B, C, H, W = rgb.shape
    device, dtype = rgb.device, rgb.dtype
    if depth.shape[-2:] != (H, W):
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)

    depth0 = depth[:, 0]
    shift_size = divergence * 0.01 * W * 0.5
    index_shift = (depth0 * shift_size - shift_size * convergence) * shift_sign

    border_pix = 0
    if preserve_screen_border:
        border_pix = max(1, round(divergence * 0.75 * 0.01 * W))
        if border_pix > 0 and border_pix * 2 < W:
            border_weight_l = torch.linspace(0.0, 1.0, border_pix, dtype=dtype, device=device)
            border_weight_r = torch.linspace(1.0, 0.0, border_pix, dtype=dtype, device=device)
            index_shift[..., :border_pix] *= border_weight_l
            index_shift[..., -border_pix:] *= border_weight_r
        else:
            border_pix = 0

    src_col = torch.arange(0, W, device=device, dtype=torch.long).view(1, 1, W).expand(B, H, W)
    dest_col_raw = src_col.to(dtype) + torch.trunc(index_shift)
    dest_col = dest_col_raw.trunc().long()
    valid = dest_col >= 0
    dest_col_c = dest_col.clamp(0, W - 1)
    src_col_masked = torch.where(valid, src_col, torch.full_like(src_col, -1))

    winner = torch.full((B, H, W), -1, dtype=torch.long, device=device)
    winner.scatter_reduce_(-1, dest_col_c, src_col_masked, reduce="amax", include_self=True)

    hole_mask = (winner < 0).unsqueeze(1)
    winner_c = winner.clamp(0, W - 1)
    warped = torch.gather(rgb, dim=-1, index=winner_c.unsqueeze(1).expand(-1, C, -1, -1))
    warped = torch.where(hole_mask.expand_as(warped), torch.zeros_like(warped), warped)

    if border_pix > 0:
        border_strip = torch.zeros_like(hole_mask)
        border_strip[..., :border_pix] = hole_mask[..., :border_pix]
        border_strip[..., -border_pix:] = hole_mask[..., -border_pix:]
        if border_strip.any():
            warped = nearest_fill_inpaint_gpu(warped, border_strip)
        hole_mask[..., :border_pix] = False
        hole_mask[..., -border_pix:] = False

    warped = true_fast_isotropic_inpaint_gpu(warped, hole_mask, radius=int(true_fast_radius))

    shift_size_px = abs(divergence) * 0.01 * float(W) * 0.5
    feather = int(max(FAST_BLUR_FEATHER_MIN, min(FAST_BLUR_FEATHER_MAX, round(shift_size_px * 0.15))))
    global INPAINT_GRAIN_STRENGTH
    _prev_grain = INPAINT_GRAIN_STRENGTH
    try:
        INPAINT_GRAIN_STRENGTH = float(FAST_GRAIN_STRENGTH)
        warped = blur_hole_region_gpu(
            warped, hole_mask,
            kernel_px=int(FAST_BLUR_KERNEL_PX),
            feather_px=feather,
        )
    finally:
        INPAINT_GRAIN_STRENGTH = _prev_grain
    return warped

# =============================================================================
# Medium Inpaint (iw3 light_video_inpaint_v1, nunif-free standalone port)
#   - Model file : medium_inpaint.pth  (Rowan_High-Depth_Inpaint_e789-Medium.pth = nt_video_inpaint_v2_b;
#                  the old iw3_light_video_inpaint_v1 file is still detected and loads too)
#   - Overlap    : PRE 3 / POST 3 frames (12-frame sequence window)
#   - Mask dilate: inner 1 / outer 2 (fixed, not exposed in the UI)
# =============================================================================
import torch.nn as nn

_MI_SEQ_LEN = 12


def _mi_bchw_to_bnc(x, window_size):
    B, C, H, W = x.shape
    SH, SW = window_size
    oh, ow = H // SH, W // SW
    x = x.reshape(B, C, oh, SH, ow, SW)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B * oh * ow, SH * SW, C)


def _mi_bnc_to_bchw(x, out_shape, window_size):
    _, _, C = x.shape
    OB, OC, OH, OW = out_shape
    SH, SW = window_size
    H, W = OH // SH, OW // SW
    x = x.reshape(OB, H, W, SH, SW, C)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(OB, C, OH, OW)


def _mi_bcdhw_to_bnc(x, window_size):
    B, C, D, H, W = x.shape
    SD, SH, SW = window_size
    OD, OH, OW = D // SD, H // SH, W // SW
    x = x.reshape(B, C, OD, SD, OH, SH, OW, SW)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
    return x.reshape(B * OD * OH * OW, SD * SH * SW, C)


def _mi_bnc_to_bcdhw(x, out_shape, window_size):
    OB, OC, OD_tot, OH_tot, OW_tot = out_shape
    SD, SH, SW = window_size
    OD, OH, OW = OD_tot // SD, OH_tot // SH, OW_tot // SW
    _, _, C = x.shape
    x = x.reshape(OB, OD, OH, OW, SD, SH, SW, C)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return x.reshape(OB, OC, OD_tot, OH_tot, OW_tot)


def _mi_pixel_unshuffle(x, s):
    B, C, H, W = x.shape
    oh, ow = H // s, W // s
    x = x.reshape(B, C, oh, s, ow, s)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(B, C * s * s, oh, ow)


def _mi_pixel_shuffle(x, s):
    B, C, H, W = x.shape
    oc = C // (s * s)
    x = x.reshape(B, oc, s, s, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3)
    return x.reshape(B, oc, H * s, W * s)


def _mi_replication_pad2d(x, padding):
    # padding = (left, right, top, bottom); negative values crop. Same result as nn.ReplicationPad2d.
    left, right, top, bottom = padding
    if left > 0:
        x = torch.cat((*((x[:, :, :, :1],) * left), x), dim=3)
    elif left < 0:
        x = x[:, :, :, -left:]
    if right > 0:
        x = torch.cat((x, *((x[:, :, :, -1:],) * right)), dim=3)
    elif right < 0:
        x = x[:, :, :, :right]
    if top > 0:
        x = torch.cat((*((x[:, :, :1, :],) * top), x), dim=2)
    elif top < 0:
        x = x[:, :, -top:, :]
    if bottom > 0:
        x = torch.cat((x, *((x[:, :, -1:, :],) * bottom)), dim=2)
    elif bottom < 0:
        x = x[:, :, :bottom, :]
    return x.contiguous()


class _MIReplicationPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = tuple(padding)

    def forward(self, x):
        return _mi_replication_pad2d(x, self.padding)


class _MIFastLayerNorm(nn.LayerNorm):
    """LayerNorm that does not upcast to fp32 under autocast (same as nunif FastLayerNorm)."""

    def forward(self, input):
        if torch.is_autocast_enabled(input.device.type):
            dtype = torch.get_autocast_dtype(input.device.type)
            input = input.to(dtype)
            weight = self.weight.to(dtype) if self.weight is not None else None
            bias = self.bias.to(dtype) if self.bias is not None else None
            with torch.amp.autocast(device_type=input.device.type, enabled=False):
                return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
        return super().forward(input)


class _MISeparableGaussianFilter2d(nn.Module):
    def __init__(self, kernel_size=15):
        super().__init__()
        sigma = kernel_size * 0.15 + 0.35
        half = (kernel_size - 1) * 0.5
        xs = torch.linspace(-half, half, steps=kernel_size)
        k = torch.exp(-0.5 * (xs / sigma).pow(2))
        k = k / k.sum()
        self.register_buffer("kernel_h", k.reshape(1, 1, 1, kernel_size), persistent=False)
        self.register_buffer("kernel_v", k.reshape(1, 1, kernel_size, 1), persistent=False)
        p = kernel_size // 2
        self.pad = _MIReplicationPad2d((p, p, p, p))

    def forward(self, x):
        x = self.pad(x)
        x = F.conv2d(x, weight=self.kernel_h, bias=None)
        x = F.conv2d(x, weight=self.kernel_v, bias=None)
        return x


class _MIGMLP(nn.Module):
    def __init__(self, embed_dim, seq_len, mlp_ratio=1):
        super().__init__()
        self.proj_in = nn.Linear(embed_dim, int(embed_dim * mlp_ratio * 2))
        self.proj_spatial = nn.Conv1d(seq_len, seq_len, kernel_size=1, stride=1, bias=True)
        self.proj_out = nn.Linear(int(embed_dim * mlp_ratio * 2) // 2, embed_dim)

    def forward(self, x, norm1=None, norm2=None):
        shortcut = x
        if norm1 is not None:
            x = norm1(x)
        x = self.proj_in(x)
        x = F.gelu(x)
        u, v = x.chunk(2, dim=-1)
        if norm2 is not None:
            v = norm2(v)
        v = self.proj_spatial(v)
        x = u * v
        x = self.proj_out(x)
        return x + shortcut


class _MIWindowGMLP2d(nn.Module):
    def __init__(self, in_channels, window_size=(4, 4), mlp_ratio=2, shift=False):
        super().__init__()
        self.window_size = window_size if isinstance(window_size, (tuple, list)) else (window_size, window_size)
        self.shift = shift
        if self.shift:
            self.pad_h = self.window_size[0] // 2
            self.pad_w = self.window_size[1] // 2
        self.seq_len = self.window_size[0] * self.window_size[1]
        self.gmlp = _MIGMLP(in_channels, seq_len=self.seq_len, mlp_ratio=mlp_ratio)

    def forward(self, x, norm1=None, norm2=None):
        if self.shift:
            # zero pad via cat / crop via slice (same result as F.pad): TensorRT turns a constant Pad into a Slice
            # with a fill value whose type does not match the fp16 input and the engine build fails.
            ph, pw = self.pad_h, self.pad_w
            B, C, H, W = x.shape
            if pw:
                z = x.new_zeros((B, C, H, pw))
                x = torch.cat((z, x, z), dim=3)
            if ph:
                z = x.new_zeros((B, C, ph, x.shape[3]))
                x = torch.cat((z, x, z), dim=2)
        out_shape = x.shape
        x = _mi_bchw_to_bnc(x, self.window_size)
        x = self.gmlp(x, norm1, norm2)
        x = _mi_bnc_to_bchw(x, out_shape, self.window_size)
        if self.shift:
            x = x[:, :, self.pad_h:x.shape[2] - self.pad_h, self.pad_w:x.shape[3] - self.pad_w]
        return x


class _MIWindowGMLP3d(nn.Module):
    def __init__(self, in_channels, window_size=(4, 4, 4), mlp_ratio=2, shift=False):
        super().__init__()
        self.window_size = tuple(window_size)
        self.shift = shift if isinstance(shift, (tuple, list)) else (shift, shift, shift)
        self.pad_h = self.pad_w = self.pad_d = 0
        if any(self.shift):
            if self.shift[0]:
                self.pad_d = self.window_size[0] // 2
            if self.shift[1]:
                self.pad_h = self.window_size[1] // 2
            if self.shift[2]:
                self.pad_w = self.window_size[2] // 2
        self.seq_len = self.window_size[0] * self.window_size[1] * self.window_size[2]
        self.gmlp = _MIGMLP(in_channels, seq_len=self.seq_len, mlp_ratio=mlp_ratio)

    def forward(self, x, norm1=None, norm2=None):
        if any(self.shift):
            x = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h, 0, 0), mode="constant", value=0)
            x = F.pad(x, (0, 0, 0, 0, self.pad_d, self.pad_d), mode="reflect")
        out_shape = x.shape
        x = _mi_bcdhw_to_bnc(x, self.window_size)
        x = self.gmlp(x, norm1, norm2)
        x = _mi_bnc_to_bcdhw(x, out_shape, self.window_size)
        if any(self.shift):
            x = F.pad(x, (-self.pad_w, -self.pad_w, -self.pad_h, -self.pad_h, -self.pad_d, -self.pad_d))
        return x


class _MIGLUConvMLP(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_ratio=1):
        super().__init__()
        self.pad = _MIReplicationPad2d((1, 1, 1, 1))
        self.w1 = nn.Conv2d(in_channels, in_channels * mlp_ratio, kernel_size=1, stride=1, padding=0)
        self.w2 = nn.Conv2d(in_channels * mlp_ratio // 2, in_channels, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.w1(x)
        x = F.glu(x, dim=1)
        x = self.pad(x)
        return self.w2(x)


class _MIGMLPBlock(nn.Module):
    def __init__(self, in_channels, window_size, mlp_ratio=2, shift=False):
        super().__init__()
        self.gmlp = _MIWindowGMLP2d(in_channels, window_size=window_size, mlp_ratio=mlp_ratio, shift=shift)
        self.norm1 = _MIFastLayerNorm(in_channels, bias=False)
        self.norm2 = _MIFastLayerNorm(in_channels * mlp_ratio, bias=False)
        self.glu_conv = _MIGLUConvMLP(in_channels, in_channels, mlp_ratio=1)

    def forward(self, x):
        x = x + self.gmlp(x, self.norm1, self.norm2)
        x = x + self.glu_conv(x)
        return x


class _MIGMLP3DBlock(nn.Module):
    _flat = False      # True only on the ONNX export copy (rank<=4 temporal path, identical result)

    def __init__(self, in_channels, window_size, mlp_ratio=2, shift=False):
        super().__init__()
        self.gmlp = _MIWindowGMLP3d(in_channels, window_size=window_size, mlp_ratio=mlp_ratio, shift=shift)
        self.norm1 = _MIFastLayerNorm(in_channels, bias=False)
        self.norm2 = _MIFastLayerNorm(in_channels * mlp_ratio, bias=False)
        self.glu_conv = _MIGLUConvMLP(in_channels, in_channels, mlp_ratio=1)

    def forward(self, x):
        B, C, H, W = x.shape
        if self._flat:
            x = x + _mi_temporal_flat(self.gmlp, x, self.norm1, self.norm2)
            return x + self.glu_conv(x)
        x = x.permute(1, 0, 2, 3).reshape(1, C, B, H, W)
        x = x + self.gmlp(x, self.norm1, self.norm2)
        x = x.permute(0, 2, 1, 3, 4).reshape(B, C, H, W)
        x = x + self.glu_conv(x)
        return x


def _mi_temporal_flat(g3d, x, norm1, norm2):
    """Same maths as _MIWindowGMLP3d with window (S,1,1) and no shift, but with rank<=4 tensors only.
    x: (S, C, H, W) -> one window per (h, w) position: (H*W, S, C). The 5D / 8D reshape+permute chain of the
    generic 3D window code is what the TensorRT ONNX parser / builder choked on, so the export graph uses this."""
    S, C, H, W = x.shape
    t = x.permute(2, 3, 0, 1).reshape(H * W, S, C)
    t = g3d.gmlp(t, norm1, norm2)
    return t.reshape(H, W, S, C).permute(2, 3, 0, 1)


def _mi_chunked_forward(module, x, chunk):
    return torch.cat([module(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)], dim=0)


class LightVideoInpaintV1(nn.Module):
    """Port of iw3 `inpaint.light_video_inpaint_v1` (inference only)."""

    def __init__(self, base_dim=96, lv2_mlp_ratio=1):
        super().__init__()
        self.downscaling_factor = 4
        self.mod = 16
        pack = self.downscaling_factor ** 2
        C = base_dim
        C2 = C * 2
        S = _MI_SEQ_LEN
        self.mask_bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.patch = nn.Conv2d(3, C, kernel_size=self.downscaling_factor, stride=self.downscaling_factor, padding=0)
        self.enc1 = _MIGMLPBlock(C, mlp_ratio=2, window_size=16, shift=False)
        self.down = nn.Conv2d(C, C2, kernel_size=2, stride=2, padding=0)
        self.enc2 = nn.ModuleList([
            _MIGMLPBlock(C2, window_size=(8, 8), mlp_ratio=lv2_mlp_ratio, shift=True),
            _MIGMLP3DBlock(C2, window_size=(S, 1, 1), mlp_ratio=2, shift=False),
            _MIGMLPBlock(C2, window_size=(8, 8), mlp_ratio=lv2_mlp_ratio, shift=False),
            _MIGMLP3DBlock(C2, window_size=(S, 1, 1), mlp_ratio=2, shift=False),
            _MIGMLPBlock(C2, window_size=(8, 8), mlp_ratio=lv2_mlp_ratio, shift=True),
        ])
        self.up = nn.Conv2d(C2, C * 4, kernel_size=1, stride=1, padding=0)
        self.dec1 = _MIGMLPBlock(C, mlp_ratio=2, window_size=16, shift=False)
        self.to_image = nn.Conv2d(C, 3 * pack, kernel_size=1, stride=1, padding=0)
        self.mask_blur = _MISeparableGaussianFilter2d(kernel_size=15)

    def preprocess(self, x, mask):
        mask = mask.float()
        x = x * (1 - mask)
        mask = torch.clamp(self.mask_blur(mask) + mask, 0, 1)
        return x, mask

    @torch.inference_mode()
    def infer(self, x, mask):
        S = _MI_SEQ_LEN
        if x.shape[0] % S != 0:
            pad_b = S - x.shape[0] % S
            pad_b1 = pad_b // 2
            pad_b2 = pad_b - pad_b1
            x = torch.cat([x[0:1]] * pad_b1 + [x] + [x[-1:]] * pad_b2, dim=0)
            mask = torch.cat([mask[0:1]] * pad_b1 + [mask] + [mask[-1:]] * pad_b2, dim=0)
        else:
            pad_b1 = pad_b2 = 0
        x, mask = self.preprocess(x, mask)
        out = self.forward(x, mask, micro_batch_size=2)
        if pad_b1 > 0:
            out = out[pad_b1:]
        if pad_b2 > 0:
            out = out[:-pad_b2]
        return out

    def _forward(self, x, mask, micro_batch_size=_MI_SEQ_LEN):
        assert x.shape[0] == _MI_SEQ_LEN
        mask = _mi_pixel_unshuffle(mask, self.downscaling_factor).amax(dim=1, keepdim=True) > 0.99

        x1s, x2s = [], []
        for i in range(0, x.shape[0], micro_batch_size):
            x0 = F.leaky_relu(self.patch(x[i:i + micro_batch_size]), 0.1, inplace=True)
            x0 = torch.where(mask[i:i + micro_batch_size], self.mask_bias.to(x0.dtype), x0)
            x1 = self.enc1(x0)
            x2 = self.down(x1)
            x1s.append(x1)
            x2s.append(x2)

        x2 = torch.cat(x2s, dim=0)
        del x2s
        for mod in self.enc2:
            if isinstance(mod, _MIGMLP3DBlock):
                x2 = mod(x2)
            else:
                x2 = _mi_chunked_forward(mod, x2, micro_batch_size)

        outs = []
        for i in range(0, x.shape[0], micro_batch_size):
            x3 = self.up(x2[i:i + micro_batch_size])
            x3 = F.pixel_shuffle(x3, 2)
            out = self.dec1(x1s[i // micro_batch_size] + x3)
            outs.append(self.to_image(out))
        x = torch.cat(outs, dim=0)
        return _mi_pixel_shuffle(x, self.downscaling_factor)

    def forward(self, x, mask, micro_batch_size=_MI_SEQ_LEN):
        src = x
        x = (x - 0.5) / 0.5
        input_height, input_width = x.shape[2:]
        unit = self.mod * self.downscaling_factor
        pad1 = unit - input_width % unit
        pad2 = unit - input_height % unit
        padding = (0, pad1, 0, pad2)
        x = _mi_replication_pad2d(x, padding)
        mask = _mi_replication_pad2d(mask, padding)

        x = self._forward(x, mask, micro_batch_size=micro_batch_size)
        x = F.pad(x, (0, -pad1, 0, -pad2))
        mask = F.pad(mask, (0, -pad1, 0, -pad2))

        mask = mask.expand_as(src)
        src = src * (1 - mask) + x * mask
        return src.clamp(0, 1)


# -----------------------------------------------------------------------------
# Rowan High-Depth Inpaint (nt_video_inpaint_v2, nunif-free standalone port)
#   - Same file name (medium_inpaint.pth); the architecture is detected from the
#     checkpoint, so the old iw3 light_video_inpaint_v1 file still loads too.
#   - This model was trained on the "stretched band" mask, not on iw3's tiny
#     gen_mask2 mask, so when it is loaded the warp also marks stretched runs
#     (index-gradient detector, same as the iw3-inpaint extras' iw3_mask.py).
# -----------------------------------------------------------------------------
MEDIUM_INPAINT_MICRO_BATCH = 2      # frames per pass through the per-frame parts (VRAM knob)


def _mi_env(name, default, cast=float):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return cast(value)
    except ValueError:
        return default


MEDIUM_STRETCH_MAX_GRAD = _mi_env("NT_MASK_GRAD", 0.4)       # index advance below this = stretched
MEDIUM_STRETCH_MIN_RUN  = _mi_env("NT_MASK_RUN", 3, int)     # ignore runs narrower than this
MEDIUM_STRETCH_GROW     = _mi_env("NT_MASK_GROW", 2, int)    # widen the marked band (px)

# Light Inpaint: stretched-band mask is OFF (these values are only used if it is switched back on)
LIGHT_STRETCH_MAX_GRAD = _mi_env("NT_MASK_GRAD", 0.6)
LIGHT_STRETCH_MIN_RUN  = _mi_env("NT_MASK_RUN", 4, int)
LIGHT_STRETCH_GROW     = _mi_env("NT_MASK_GROW", 2, int)

_MI_STRETCH_MASK_ON = False         # set by load_medium_inpaint_model() (ON for the Rowan model = Medium Inpaint, forced OFF for Light Inpaint)
# Values actually used by _mi_stretch_mask(); switched per mode when a conversion starts.
_MI_STRETCH_ACTIVE = {"grad": MEDIUM_STRETCH_MAX_GRAD, "run": MEDIUM_STRETCH_MIN_RUN, "grow": MEDIUM_STRETCH_GROW}


def _mi_set_stretch_params(grad, run, grow):
    _MI_STRETCH_ACTIVE["grad"] = grad
    _MI_STRETCH_ACTIVE["run"] = run
    _MI_STRETCH_ACTIVE["grow"] = grow


def _mi_padb(t, left, right):
    """F.pad for a bool tensor along x (goes through float, always supported)."""
    return F.pad(t.float(), (left, right, 0, 0)) > 0.5


def _mi_stretch_mask(index):
    """Target pixels that the warp stretched or invented. index: (B,1,H,W) source x per pixel."""
    index = index.float()
    grad = index[..., 1:] - index[..., :-1]
    stretched = grad < _MI_STRETCH_ACTIVE["grad"]
    mask = _mi_padb(stretched, 1, 0) | _mi_padb(stretched, 0, 1)
    mask = mask & ~_mi_padb(grad < -1.0, 1, 0)           # backward jump = occlusion edge, not a stretch
    k = int(_MI_STRETCH_ACTIVE["run"])
    if k > 1:                                            # drop runs shorter than k (erode, then dilate along x)
        pad = k // 2
        f = mask.float()
        eroded = -F.max_pool2d(-f, (1, k), stride=1, padding=(0, pad))
        kept = F.max_pool2d(eroded, (1, k), stride=1, padding=(0, pad))
        kept = kept[..., :mask.shape[-1]]
        mask = (kept > 0) & mask
    n = int(_MI_STRETCH_ACTIVE["grow"])
    if n > 0:
        mask = F.max_pool2d(mask.float(), (1, 2 * n + 1), stride=1, padding=(0, n)) > 0
    return mask


class _NTGMLPBlock(nn.Module):
    """nt_inpaint_v2 block: the shortcut is added ONCE (inside the gMLP), unlike light_inpaint_v1."""

    def __init__(self, in_channels, window_size, mlp_ratio=2, shift=False):
        super().__init__()
        self.gmlp = _MIWindowGMLP2d(in_channels, window_size=window_size, mlp_ratio=mlp_ratio, shift=shift)
        self.norm1 = _MIFastLayerNorm(in_channels, bias=False)
        self.norm2 = _MIFastLayerNorm(in_channels * mlp_ratio, bias=False)
        self.glu_conv = _MIGLUConvMLP(in_channels, in_channels, mlp_ratio=1)

    def forward(self, x):
        x = self.gmlp(x, self.norm1, self.norm2)      # residual is inside
        return x + self.glu_conv(x)


class _NTGMLP3DBlock(nn.Module):
    """Temporal-only block (window = seq_len x 1 x 1): mixes along time, per feature position."""
    _flat = False      # True only on the ONNX export copy (rank<=4 temporal path, identical result)

    def __init__(self, in_channels, seq_len=_MI_SEQ_LEN, mlp_ratio=2):
        super().__init__()
        self.gmlp = _MIWindowGMLP3d(in_channels, window_size=(seq_len, 1, 1), mlp_ratio=mlp_ratio, shift=False)
        self.norm1 = _MIFastLayerNorm(in_channels, bias=False)
        self.norm2 = _MIFastLayerNorm(in_channels * mlp_ratio, bias=False)
        self.glu_conv = _MIGLUConvMLP(in_channels, in_channels, mlp_ratio=1)

    def forward(self, x):
        B, C, H, W = x.shape
        if self._flat:
            x = _mi_temporal_flat(self.gmlp, x, self.norm1, self.norm2)      # residual is inside
            return x + self.glu_conv(x)
        x = x.permute(1, 0, 2, 3).reshape(1, C, B, H, W)
        x = self.gmlp(x, self.norm1, self.norm2)      # residual is inside
        x = x.permute(0, 2, 1, 3, 4).reshape(B, C, H, W)
        return x + self.glu_conv(x)


def _mi_nt_stack(n, channels, window, mlp_ratio, shift_first=False):
    return nn.Sequential(*[
        _NTGMLPBlock(channels, window_size=window, mlp_ratio=mlp_ratio,
                     shift=((i % 2 == 0) == shift_first))
        for i in range(n)
    ])


def _mi_level_slots(n_blocks, n_temporal):
    """2D block index each temporal block fires after (same rule as nt_video_inpaint_v2)."""
    if not n_temporal:
        return []
    return sorted({int(round((i + 1) * n_blocks / n_temporal)) - 1 for i in range(n_temporal)})


class NTVideoInpaintV2Port(nn.Module):
    """Port of nunif-side `inpaint.nt_video_inpaint_v2[_s/_b/_l]` (inference only, no window cache)."""

    def __init__(self, dims=(128, 256, 384), blocks=(1, 4, 6), dec_blocks=(1, 2),
                 windows=(16, 8, 8), mlp_ratio=2, mask_channel=True,
                 seq_len=_MI_SEQ_LEN, temporal=(2, 2)):
        super().__init__()
        if int(seq_len) != _MI_SEQ_LEN:
            raise ValueError(f"Medium Inpaint expects seq_len={_MI_SEQ_LEN}, model has {seq_len}")
        self.downscaling_factor = 4
        pack = self.downscaling_factor ** 2
        C1, C2, C3 = [int(c) for c in dims]
        w1, w2, w3 = [tuple(w) if isinstance(w, (tuple, list)) else (w, w) for w in windows]
        self.mask_channel = bool(mask_channel)
        self.seq_len = int(seq_len)
        mod = max(w1[0], w1[1], 2 * w2[0], 2 * w2[1], 4 * w3[0], 4 * w3[1])
        self.align = mod * self.downscaling_factor

        in_ch = 3 * pack + (pack if self.mask_channel else 0)
        self.patch = nn.Sequential(nn.Conv2d(in_ch, C1, kernel_size=1), nn.LeakyReLU(0.2, inplace=True))
        self.mask_bias = nn.Parameter(torch.zeros(1, C1, 1, 1))
        self.enc1 = _mi_nt_stack(blocks[0], C1, w1, mlp_ratio, shift_first=True)
        self.down1 = nn.Conv2d(C1, C2, kernel_size=2, stride=2, padding=0)
        self.enc2 = _mi_nt_stack(blocks[1], C2, w2, mlp_ratio)
        self.down2 = nn.Conv2d(C2, C3, kernel_size=2, stride=2, padding=0)
        self.enc3 = _mi_nt_stack(blocks[2], C3, w3, mlp_ratio)
        self.up2 = nn.Conv2d(C3, C2 * 4, kernel_size=1)
        self.dec2 = _mi_nt_stack(dec_blocks[1], C2, w2, mlp_ratio)
        self.up1 = nn.Conv2d(C2, C1 * 4, kernel_size=1)
        self.dec1 = _mi_nt_stack(dec_blocks[0], C1, w1, mlp_ratio)
        self.to_image = nn.Sequential(_MIReplicationPad2d((1, 1, 1, 1)),
                                      nn.Conv2d(C1, 3 * pack, kernel_size=3, stride=1, padding=0))
        self.temporal2 = nn.ModuleList([_NTGMLP3DBlock(C2, self.seq_len, mlp_ratio) for _ in range(temporal[0])])
        self.temporal3 = nn.ModuleList([_NTGMLP3DBlock(C3, self.seq_len, mlp_ratio) for _ in range(temporal[1])])
        self.mask_blur = _MISeparableGaussianFilter2d(kernel_size=15)

    @classmethod
    def kwargs_from_state(cls, state, saved=None):
        """Architecture kwargs: from the checkpoint's own `kwargs` when present, else from tensor shapes."""
        def count(prefix):
            return len({int(k[len(prefix):].split(".")[0]) for k in state if k.startswith(prefix)})
        if isinstance(saved, dict) and "dims" in saved:
            keys = ("dims", "blocks", "dec_blocks", "windows", "mlp_ratio", "mask_channel", "seq_len", "temporal")
            return {k: saved[k] for k in keys if k in saved}
        C1, C2, C3 = (int(state["mask_bias"].shape[1]), int(state["down1.weight"].shape[0]),
                      int(state["down2.weight"].shape[0]))
        win = lambda key: int(round(math.sqrt(int(state[key].shape[0]))))
        return dict(
            dims=(C1, C2, C3),
            blocks=(count("enc1."), count("enc2."), count("enc3.")),
            dec_blocks=(count("dec1."), count("dec2.")),
            windows=(win("enc1.0.gmlp.gmlp.proj_spatial.weight"),
                     win("enc2.0.gmlp.gmlp.proj_spatial.weight"),
                     win("enc3.0.gmlp.gmlp.proj_spatial.weight")),
            mlp_ratio=int(state["enc1.0.gmlp.gmlp.proj_in.weight"].shape[0]) // (4 * C1),
            mask_channel=int(state["patch.0.weight"].shape[1]) > 3 * 16,
            seq_len=int(state["temporal2.0.gmlp.gmlp.proj_spatial.weight"].shape[0]),
            temporal=(count("temporal2."), count("temporal3.")),
        )

    def preprocess(self, x, mask):
        mask = mask.float()
        x = x * (1 - mask)
        mask = torch.clamp(self.mask_blur(mask) + mask, 0, 1)
        return x, mask

    @torch.inference_mode()
    def infer(self, x, mask):
        S = self.seq_len
        pad_b1 = pad_b2 = 0
        if x.shape[0] % S != 0:
            pad_b = (-x.shape[0]) % S
            pad_b1 = pad_b // 2
            pad_b2 = pad_b - pad_b1
            x = torch.cat([x[0:1]] * pad_b1 + [x] + [x[-1:]] * pad_b2, dim=0)
            mask = torch.cat([mask[0:1]] * pad_b1 + [mask] + [mask[-1:]] * pad_b2, dim=0)
        x, mask = self.preprocess(x, mask)
        out = self.forward(x, mask, micro_batch_size=MEDIUM_INPAINT_MICRO_BATCH)
        if pad_b1 > 0:
            out = out[pad_b1:]
        if pad_b2 > 0:
            out = out[:-pad_b2]
        return out

    def _run_level(self, blocks2d, temporal, x, chunk):
        n = len(blocks2d)
        slots = _mi_level_slots(n, len(temporal))
        if not slots:
            for i in range(n):
                x = _mi_chunked_forward(blocks2d[i], x, chunk)
            return x
        for ti, s in enumerate(slots):
            start = 0 if ti == 0 else slots[ti - 1] + 1
            for i in range(start, s + 1):
                x = _mi_chunked_forward(blocks2d[i], x, chunk)
            x = temporal[ti](x)                       # needs the whole window
        for i in range(slots[-1] + 1, n):
            x = _mi_chunked_forward(blocks2d[i], x, chunk)
        return x

    def _forward(self, x, mask, chunk):
        n = x.shape[0]
        s = self.downscaling_factor
        x1s, ds = [], []
        for i in range(0, n, chunk):
            xi = _mi_pixel_unshuffle(x[i:i + chunk], s)
            mi = _mi_pixel_unshuffle(mask[i:i + chunk], s)
            if self.mask_channel:
                xi = torch.cat([xi, mi.to(xi.dtype)], dim=1)
            xi = self.patch(xi)
            hole = mi.amax(dim=1, keepdim=True) > 0.99
            xi = torch.where(hole, self.mask_bias.to(xi.dtype), xi)
            x1 = self.enc1(xi)
            x1s.append(x1)
            ds.append(self.down1(x1))
        d = torch.cat(ds, dim=0)
        del ds
        x2 = self._run_level(self.enc2, self.temporal2, d, chunk)
        del d
        x3 = self._run_level(self.enc3, self.temporal3, self.down2(x2), chunk)

        outs = []
        for k, i in enumerate(range(0, n, chunk)):
            y2 = self.dec2(x2[i:i + chunk] + F.pixel_shuffle(self.up2(x3[i:i + chunk]), 2))
            y1 = self.dec1(x1s[k] + F.pixel_shuffle(self.up1(y2), 2))
            outs.append(self.to_image(y1))
        return _mi_pixel_shuffle(torch.cat(outs, dim=0), s)

    def forward(self, x, mask, micro_batch_size=MEDIUM_INPAINT_MICRO_BATCH):
        src = x
        x = (x - 0.5) / 0.5
        H, W = x.shape[2:]
        pad1 = (-W) % self.align
        pad2 = (-H) % self.align
        if pad1 or pad2:
            padding = (0, pad1, 0, pad2)
            x = _mi_replication_pad2d(x, padding)
            mask = _mi_replication_pad2d(mask, padding)
        x = self._forward(x, mask, micro_batch_size)
        if pad1 or pad2:
            x = F.pad(x, (0, -pad1, 0, -pad2))
            mask = F.pad(mask, (0, -pad1, 0, -pad2))
        mask = mask.expand_as(src)
        src = src * (1 - mask) + x * mask
        return src.clamp(0, 1)


def _mi_torch_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        # nunif checkpoints can carry extra metadata that the strict loader refuses.
        return torch.load(str(path), map_location="cpu", weights_only=False)



def _medium_inpaint_model_path() -> Path:
    # Looked up next to conversion3d.py (or next to the exe when frozen).
    for base in (APP_DIR, BASE_DIR):
        cand = Path(base) / MEDIUM_INPAINT_MODEL_FILENAME
        if cand.is_file():
            return cand
    return Path(APP_DIR) / MEDIUM_INPAINT_MODEL_FILENAME


MEDIUM_INPAINT_URL = "https://github.com/wake-82/DepthLive3D/releases/download/VR/medium_inpaint.pth"
LIGHT_INPAINT_URL  = "https://github.com/wake-82/DepthLive3D/releases/download/VR/light_video_inpaint.pth"


def ensure_inpaint_model(path, url, log=print):
    """Downloads an inpaint model file next to the program when it is missing.
    Downloads to '<name>.part' first and renames on success, so a broken
    download never leaves a corrupt model file behind."""
    import os as _os
    import urllib.request as _ur
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    log(f"[Inpaint] '{path.name}' not found. Downloading from: {url}")
    try:
        req = _ur.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with _ur.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            done, last_pct = 0, -10
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total > 0:
                    pct = int(done * 100 / total)
                    if pct >= last_pct + 10:
                        last_pct = pct
                        log(f"[Inpaint] Downloading {path.name}: {pct}% ({done // (1 << 20)} / {total // (1 << 20)} MB)")
        if total > 0 and done != total:
            raise IOError(f"incomplete download ({done} / {total} bytes)")
        _os.replace(str(tmp), str(path))
        log(f"[Inpaint] Download complete: {path}")
        return path
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise RuntimeError(
            f"Failed to download '{path.name}': {e}\n"
            f"Please download it manually from:\n{url}\nand place it in:\n{path.parent}"
        ) from e


def medium_inpaint_model_available() -> bool:
    return _medium_inpaint_model_path().is_file()


def _light_inpaint_model_path() -> Path:
    # Looked up next to conversion3d.py (or next to the exe when frozen).
    for base in (APP_DIR, BASE_DIR):
        cand = Path(base) / LIGHT_INPAINT_MODEL_FILENAME
        if cand.is_file():
            return cand
    return Path(APP_DIR) / LIGHT_INPAINT_MODEL_FILENAME


def light_inpaint_model_available() -> bool:
    return _light_inpaint_model_path().is_file()


def load_medium_inpaint_model(device, path=None, force_stretch_mask=False, disable_stretch_mask=False):
    """Loads medium_inpaint.pth. Two architectures are recognised from the tensors themselves:
       - Rowan High-Depth Inpaint (nt_video_inpaint_v2): has temporal2./temporal3. blocks
       - iw3 light_video_inpaint_v1 (legacy)"""
    global _MI_STRETCH_MASK_ON
    path = Path(path) if path is not None else _medium_inpaint_model_path()
    if not path.is_file():
        raise FileNotFoundError(f"Inpaint model file not found: {path}")
    ckpt = _mi_torch_load(path)
    is_ckpt = isinstance(ckpt, dict) and "state_dict" in ckpt
    state = ckpt["state_dict"] if is_ckpt else ckpt

    if "temporal2.0.gmlp.gmlp.proj_in.weight" in state:
        kw = NTVideoInpaintV2Port.kwargs_from_state(state, ckpt.get("kwargs") if is_ckpt else None)
        model = NTVideoInpaintV2Port(**kw)
        model.load_state_dict(state, strict=True)
        _MI_STRETCH_MASK_ON = True       # trained on the stretched-band mask
    else:
        # Infer the architecture from tensor shapes (small/medium/large all load correctly).
        base_dim = int(state["mask_bias"].shape[1])
        lv2_mlp_ratio = int(state["enc2.0.gmlp.gmlp.proj_in.weight"].shape[0]) // (2 * base_dim * 2)
        model = LightVideoInpaintV1(base_dim=base_dim, lv2_mlp_ratio=max(1, lv2_mlp_ratio))
        model.load_state_dict(state, strict=True)
        _MI_STRETCH_MASK_ON = False
    if force_stretch_mask:
        _MI_STRETCH_MASK_ON = True
    if disable_stretch_mask:
        _MI_STRETCH_MASK_ON = False     # Light Inpaint: stretched-band mask OFF (Medium Inpaint keeps it ON)
    return model.to(device).eval()


def _mi_mask_closing(mask, kernel_size=3, n_iter=2):
    org = mask.float()
    m = org
    pad = kernel_size // 2
    for _ in range(n_iter):
        m = F.max_pool2d(m, kernel_size=kernel_size, stride=1, padding=pad)
    for _ in range(n_iter):
        m = -F.max_pool2d(-m, kernel_size=kernel_size, stride=1, padding=pad)
    return (m + org).clamp(0, 1)


def _mi_dilate_outer(mask, n_iter):
    # right-eye based (left eye is flipped before this call)
    if n_iter <= 0:
        return mask
    m = mask.bool()
    for _ in range(n_iter):
        m = m | F.pad(m, (1, 0, 0, 0))[:, :, :, :-1]
    return m.to(mask.dtype)


def _mi_dilate_inner(mask, n_iter):
    if n_iter <= 0:
        return mask
    m = mask.bool()
    for _ in range(n_iter):
        m = m | F.pad(m, (0, 1, 0, 0))[:, :, :, 1:]
    return m.to(mask.dtype)


def _mi_preprocess_mask(mask):
    mask = mask > 0
    mask = _mi_mask_closing(mask)
    mask = _mi_dilate_outer(mask, MEDIUM_INPAINT_OUTER_DILATION)
    mask = _mi_dilate_inner(mask, MEDIUM_INPAINT_INNER_DILATION)
    return mask


def monobw_warp_one_with_hole_mask(
    rgb, depth, divergence, convergence, shift_sign=1.0, preserve_screen_border: bool = False,
):
    """[LEGACY - no longer used by Medium Inpaint; see medium_forward_warp_one] Same integer forward warp as monobw_warp_one_fast, but the holes are NOT filled.
    Returns (warped_rgb, hole_mask[bool, B x 1 x H x W])."""
    B, C, H, W = rgb.shape
    device, dtype = rgb.device, rgb.dtype
    if depth.shape[-2:] != (H, W):
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)

    depth0 = depth[:, 0]
    shift_size = divergence * 0.01 * W * 0.5
    index_shift = (depth0 * shift_size - shift_size * convergence) * shift_sign

    border_pix = 0
    if preserve_screen_border:
        border_pix = max(1, round(divergence * 0.75 * 0.01 * W))
        if border_pix > 0 and border_pix * 2 < W:
            border_weight_l = torch.linspace(0.0, 1.0, border_pix, dtype=dtype, device=device)
            border_weight_r = torch.linspace(1.0, 0.0, border_pix, dtype=dtype, device=device)
            index_shift[..., :border_pix] *= border_weight_l
            index_shift[..., -border_pix:] *= border_weight_r
        else:
            border_pix = 0

    src_col = torch.arange(0, W, device=device, dtype=torch.long).view(1, 1, W).expand(B, H, W)
    dest_col_raw = src_col.to(dtype) + torch.trunc(index_shift)
    dest_col = dest_col_raw.trunc().long()
    valid = dest_col >= 0
    dest_col_c = dest_col.clamp(0, W - 1)
    src_col_masked = torch.where(valid, src_col, torch.full_like(src_col, -1))

    winner = torch.full((B, H, W), -1, dtype=torch.long, device=device)
    winner.scatter_reduce_(-1, dest_col_c, src_col_masked, reduce="amax", include_self=True)

    hole_mask = (winner < 0).unsqueeze(1)
    winner_c = winner.clamp(0, W - 1)
    warped = torch.gather(rgb, dim=-1, index=winner_c.unsqueeze(1).expand(-1, C, -1, -1))
    warped = torch.where(hole_mask.expand_as(warped), torch.zeros_like(warped), warped)

    if border_pix > 0:
        border_strip = torch.zeros_like(hole_mask)
        border_strip[..., :border_pix] = hole_mask[..., :border_pix]
        border_strip[..., -border_pix:] = hole_mask[..., -border_pix:]
        if border_strip.any():
            warped = nearest_fill_inpaint_gpu(warped, border_strip)
        hole_mask[..., :border_pix] = False
        hole_mask[..., -border_pix:] = False

    return warped, hole_mask


# =============================================================================
# Medium Forward Warp  (iw3 forward_warp.py : depth_order_bilinear_forward_warp, nunif-free port)
#   - Used ONLY by Medium Inpaint (it produces the hole masks for the AI video inpaint model).
#   - Fast Inpaint keeps using monobw_warp_one_fast (integer forward warp). Do NOT mix them up.
#   - Single-eye path of iw3 (synthetic_view="right"/"left", fill=False, return_mask=True, width_base=False).
#   - Sign convention of this app: shift_sign=+1 -> iw3 "right" eye, shift_sign=-1 -> iw3 "left" eye.
#     (this app's depth is "near = small", iw3's is "near = large"; the geometry is identical, only the
#      sign/ordering of the formulas below is adapted, see medium_forward_warp_one)
# =============================================================================

MEDIUM_FORWARD_SYNC_EVERY = 4   # check "any holes left?" (a GPU->CPU sync) only every N iterations


def medium_forward_shift_fill(x, sign, max_tries=100):
    # Same result as the original loop. Iterations after convergence are no-ops, so the
    # expensive `.any().item()` sync only needs to run every few iterations. torch.where
    # replaces boolean-mask indexing (which syncs by itself).
    mask = x < 0
    while max_tries > 0:
        if not mask.any().item():
            break
        for _ in range(min(MEDIUM_FORWARD_SYNC_EVERY, max_tries)):
            if sign > 0:
                shifted = F.pad(x[:, :, :, 1:], (0, 1, 0, 0))
            else:
                shifted = F.pad(x[:, :, :, :-1], (1, 0, 0, 0))
            x.copy_(torch.where(mask, shifted, x))
            mask = x < 0
            max_tries -= 1
    return x


def medium_forward_fix_layered_holes(side_image, index_image, sign, max_tries=100):
    def _mask():
        m = (index_image[:, :, :, :-1] - index_image[:, :, :, 1:]) > 0
        return F.pad(m, (0, 1, 0, 0)) if sign > 0 else F.pad(m, (1, 0, 0, 0))

    mask = _mask()
    while max_tries > 0:
        if not mask.any().item():
            break
        for _ in range(min(MEDIUM_FORWARD_SYNC_EVERY, max_tries)):
            side_image.masked_fill_(mask.expand_as(side_image), -2)  # set undefined value
            if sign > 0:
                shifted = F.pad(index_image[:, :, :, 1:], (0, 1, 0, 0))
            else:
                shifted = F.pad(index_image[:, :, :, :-1], (1, 0, 0, 0))
            index_image.copy_(torch.where(mask, shifted, index_image))
            mask = _mask()
            max_tries -= 1


def medium_forward_to_flat_index(batch, width, height, index):
    index = index + torch.arange(0, height, device=index.device).view(1, height, 1) * width
    index = index + torch.arange(0, batch, device=index.device).view(batch, 1, 1) * height * width
    index = index.view(-1)
    return index


def medium_forward_make_bilinear_data(batch, width, height, index, index_shift):
    float_index = torch.clamp(index + index_shift, 0, width - 1)
    floor_index = torch.clamp(float_index.floor(), 0, width - 1)
    ceil_index = torch.clamp(float_index.ceil(), 0, width - 1)
    ceil_weight = (float_index - floor_index).reshape(batch, 1, height, width)
    ceil_weight = torch.clamp(ceil_weight, min=1e-5, max=1.0 - 1e-5)
    floor_weight = 1.0 - ceil_weight
    floor_index = medium_forward_to_flat_index(batch, width, height, floor_index.long())
    ceil_index = medium_forward_to_flat_index(batch, width, height, ceil_index.long())
    return floor_index, ceil_index, floor_weight, ceil_weight


def medium_forward_ordered_index_copy(c, src_index, dest_index, index_order, undefined_value=-1):
    B, _, H, W = c.shape
    c = c.permute(0, 2, 3, 1).reshape(-1, c.shape[1])
    if torch.is_tensor(undefined_value):
        out = undefined_value.view(1, -1).repeat(c.shape[0], 1)
    else:
        out = torch.empty_like(c).fill_(undefined_value)

    # Deterministically resolve "last write wins" without global torch.use_deterministic_algorithms
    ordered_dest = dest_index[index_order]
    pos = torch.arange(ordered_dest.numel(), device=c.device)
    best_pos = torch.full((out.shape[0],), -1, dtype=torch.long, device=c.device)
    best_pos.scatter_reduce_(0, ordered_dest, pos, reduce="amax", include_self=False)

    valid = best_pos >= 0
    src_values = c[src_index[index_order]]
    # gather + where instead of out[valid] = src_values[best_pos[valid]] (no GPU sync, same values)
    picked = src_values[best_pos.clamp(min=0)]
    out = torch.where(valid.unsqueeze(1), picked, out)
    return out.view(B, H, W, -1).permute(0, 3, 1, 2)


def medium_forward_warp(batch, width, height, c, x_index, index_shift, src_index, index_order):
    floor_index, ceil_index, floor_weight, ceil_weight = medium_forward_make_bilinear_data(
        batch, width, height, x_index, index_shift)

    # pack for optimization
    floor_data = torch.cat([floor_weight, c], dim=1)
    ceil_data = torch.cat([ceil_weight, c], dim=1)

    # 0 for weight, -1 for pixel
    undefined_value = torch.tensor([0] + [-1] * c.shape[1], dtype=c.dtype, device=c.device)
    floor_warp = medium_forward_ordered_index_copy(
        floor_data, src_index, floor_index, index_order, undefined_value=undefined_value)
    ceil_warp = medium_forward_ordered_index_copy(
        ceil_data, src_index, ceil_index, index_order, undefined_value=undefined_value)

    # unpack
    floor_weight_warp, floor_warp = floor_warp[:, 0:1, :, :], floor_warp[:, 1:, :, :]
    ceil_weight_warp, ceil_warp = ceil_warp[:, 0:1, :, :], ceil_warp[:, 1:, :, :]

    out = (floor_warp * floor_weight_warp + ceil_warp * ceil_weight_warp) / (floor_weight_warp + ceil_weight_warp)
    out = torch.nan_to_num(out, -1)
    return out


def medium_forward_gen_mask(eye):
    eye = eye[:, 0:1]
    return torch.clamp((eye == -1).float() + (eye == -2).float() * 0.5, 0, 1)


def medium_forward_warp_one(
    rgb, depth, divergence, convergence, shift_sign=1.0, preserve_screen_border: bool = False,
):
    """iw3 depth-order bilinear forward warp (+ layered hole fix) for ONE eye. Holes are NOT filled.
    Returns (warped_rgb[B,C,H,W] with holes = 0, hole_mask[bool, B x 1 x H x W]).
    shift_sign=+1 : iw3 "right" eye / shift_sign=-1 : iw3 "left" eye."""
    B, C, H, W = rgb.shape
    device, in_dtype = rgb.device, rgb.dtype
    if depth.shape[-2:] != (H, W):
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)

    # fp32 is required: pixel indices (up to a few thousand) are not representable in fp16.
    c = rgb.float()
    depth = depth[:, 0:1].float()

    base_size = max(H, W)                      # iw3 width_base=False
    iw3_divergence = float(divergence) * 2.0   # iw3 doubles divergence for single-eye views
    padding_size = int(base_size * iw3_divergence * 0.01 + 2)
    shift_size = float(divergence) * 0.01 * base_size * 0.5
    index_shift = (depth[:, 0] * shift_size - shift_size * convergence) * float(shift_sign)  # B,H,W

    if preserve_screen_border:
        border_pix = max(1, round(float(divergence) * 0.75 * 0.01 * W))
        if border_pix > 0 and border_pix * 2 < W:
            border_weight_l = torch.linspace(0.0, 1.0, border_pix, dtype=index_shift.dtype, device=device)
            border_weight_r = torch.linspace(1.0, 0.0, border_pix, dtype=index_shift.dtype, device=device)
            index_shift[..., :border_pix] *= border_weight_l
            index_shift[..., -border_pix:] *= border_weight_r

    # pad (replicate) -> forward warp
    pad = (padding_size, padding_size, 0, 0)
    c = _mi_replication_pad2d(c, pad)
    depth = _mi_replication_pad2d(depth, pad)
    index_shift = _mi_replication_pad2d(index_shift.unsqueeze(1), pad)[:, 0]
    _, _, Hp, Wp = depth.shape

    x_index = torch.arange(0, Wp, device=device).view(1, 1, Wp).expand(B, Hp, Wp)
    src_index = medium_forward_to_flat_index(B, Wp, Hp, x_index)
    # iw3 sorts by its depth (near = large) ascending, last write (= nearest) wins.
    # This app's depth is inverted (near = small) -> sort descending.
    index_order = torch.argsort(depth.reshape(-1), dim=0, descending=True)

    c = torch.cat([c, x_index.view(B, 1, Hp, Wp).to(c.dtype)], dim=1)  # warp width index together
    eye = medium_forward_warp(B, Wp, Hp, c, x_index, index_shift, src_index, index_order)
    eye = _mi_replication_pad2d(eye, (-padding_size, -padding_size, 0, 0))   # unpad
    eye, eye_index = eye[:, :-1, :, :], eye[:, -1:, :, :]

    if shift_sign > 0:   # iw3 "right"
        eye_index = medium_forward_shift_fill(eye_index, 1)
        medium_forward_fix_layered_holes(eye, eye_index, -1)
    else:                # iw3 "left"
        eye_index = medium_forward_shift_fill(eye_index, -1)
        medium_forward_fix_layered_holes(eye, eye_index, 1)

    if _MI_STRETCH_MASK_ON:   # Rowan model: also mark the stretched band (iw3's own mask misses it)
        eye[_mi_stretch_mask(eye_index).expand_as(eye)] = -2
    hole_mask = medium_forward_gen_mask(eye) > 0
    eye = torch.clamp(eye, 0, 1)   # drop undefined values (-1 / -2 -> 0)
    return eye.to(in_dtype).contiguous(), hole_mask


def medium_forward_warp_pair(rgb, depth, divergence, convergence, preserve_screen_border: bool = False):
    """Both eyes at once. Same result as medium_forward_warp_one(-1) + medium_forward_warp_one(+1), but the
    padding, index tables and the (expensive) depth sort are computed once and shared by the two eyes.
    Returns (left_rgb, right_rgb, left_hole_mask, right_hole_mask)."""
    B, C, H, W = rgb.shape
    device, in_dtype = rgb.device, rgb.dtype
    if depth.shape[-2:] != (H, W):
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)

    # fp32 is required: pixel indices (up to a few thousand) are not representable in fp16.
    c = rgb.float()
    depth = depth[:, 0:1].float()

    base_size = max(H, W)                      # iw3 width_base=False
    iw3_divergence = float(divergence) * 2.0   # iw3 doubles divergence for single-eye views
    padding_size = int(base_size * iw3_divergence * 0.01 + 2)
    shift_size = float(divergence) * 0.01 * base_size * 0.5
    base_shift = depth[:, 0] * shift_size - shift_size * convergence   # B,H,W (the eye sign is applied below)

    if preserve_screen_border:
        border_pix = max(1, round(float(divergence) * 0.75 * 0.01 * W))
        if border_pix > 0 and border_pix * 2 < W:
            border_weight_l = torch.linspace(0.0, 1.0, border_pix, dtype=base_shift.dtype, device=device)
            border_weight_r = torch.linspace(1.0, 0.0, border_pix, dtype=base_shift.dtype, device=device)
            base_shift[..., :border_pix] *= border_weight_l
            base_shift[..., -border_pix:] *= border_weight_r

    # pad (replicate) once
    pad = (padding_size, padding_size, 0, 0)
    c = _mi_replication_pad2d(c, pad)
    depth = _mi_replication_pad2d(depth, pad)
    base_shift = _mi_replication_pad2d(base_shift.unsqueeze(1), pad)[:, 0]
    _, _, Hp, Wp = depth.shape

    x_index = torch.arange(0, Wp, device=device).view(1, 1, Wp).expand(B, Hp, Wp)
    src_index = medium_forward_to_flat_index(B, Wp, Hp, x_index)
    # iw3 sorts by its depth (near = large) ascending, last write (= nearest) wins.
    # This app's depth is inverted (near = small) -> sort descending. The order is identical for both eyes.
    index_order = torch.argsort(depth.reshape(-1), dim=0, descending=True)

    c = torch.cat([c, x_index.view(B, 1, Hp, Wp).to(c.dtype)], dim=1)  # warp width index together

    eyes = []
    for sign in (-1.0, +1.0):    # left, right
        eye = medium_forward_warp(B, Wp, Hp, c, x_index, base_shift * sign, src_index, index_order)
        eye = _mi_replication_pad2d(eye, (-padding_size, -padding_size, 0, 0))   # unpad
        eye, eye_index = eye[:, :-1, :, :], eye[:, -1:, :, :]
        if sign > 0:   # iw3 "right"
            eye_index = medium_forward_shift_fill(eye_index, 1)
            medium_forward_fix_layered_holes(eye, eye_index, -1)
        else:          # iw3 "left"
            eye_index = medium_forward_shift_fill(eye_index, -1)
            medium_forward_fix_layered_holes(eye, eye_index, 1)

        if _MI_STRETCH_MASK_ON:   # Rowan model: also mark the stretched band (iw3's own mask misses it)
            eye.masked_fill_(_mi_stretch_mask(eye_index).expand_as(eye), -2)
        hole_mask = medium_forward_gen_mask(eye) > 0
        eye = torch.clamp(eye, 0, 1)   # drop undefined values (-1 / -2 -> 0)
        eyes.append((eye.to(in_dtype).contiguous(), hole_mask))

    (left_t, left_m), (right_t, right_m) = eyes
    return left_t, right_t, left_m, right_m


# =============================================================================
# Light Inpaint (stream): iw3 light_inpaint_v1 = IMAGE inpaint model (ported from live3d.py)
#   - Model file : light_inpaint.pth (downloaded automatically when missing)
#     (NOT the video model 'light_video_inpaint.pth' used by "Light Inpaint")
#   - Warp       : iw3 depth-order bilinear forward warp + layered hole fix (holes are NOT filled,
#                  the hole mask is handed to the model). No stretched-band mask.
#   - Frame-by-frame: one frame in / one frame out. A per-conversion cache (throttled model re-runs,
#                  mask hysteresis, cross-fade) keeps the inpainted area from flickering.
#   - Frames in this file are BGR; the model runs in RGB (channels are flipped around the call).
# =============================================================================
LIGHT_INPAINT_STREAM_URL = "https://github.com/wake-82/DepthLive3D/releases/download/VR/light_inpaint.pth"


def _lis_bchw_to_bnc(x, window_size):
    B, C, H, W = x.shape
    SH, SW = window_size
    oh, ow = H // SH, W // SW
    x = x.reshape(B, C, oh, SH, ow, SW)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B * oh * ow, SH * SW, C)


def _lis_bnc_to_bchw(x, out_shape, window_size):
    _, _, C = x.shape
    OB, OC, OH, OW = out_shape
    SH, SW = window_size
    H, W = OH // SH, OW // SW
    x = x.reshape(OB, H, W, SH, SW, C)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(OB, C, OH, OW)


def _lis_pixel_unshuffle(x, s):
    B, C, H, W = x.shape
    oh, ow = H // s, W // s
    x = x.reshape(B, C, oh, s, ow, s)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(B, C * s * s, oh, ow)


def _lis_pixel_shuffle(x, s):
    B, C, H, W = x.shape
    oc = C // (s * s)
    x = x.reshape(B, oc, s, s, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3)
    return x.reshape(B, oc, H * s, W * s)


def _lis_replication_pad2d(x, padding):
    # padding = (left, right, top, bottom); negative values crop. Same result as nn.ReplicationPad2d.
    left, right, top, bottom = padding
    if left > 0:
        x = torch.cat((*((x[:, :, :, :1],) * left), x), dim=3)
    elif left < 0:
        x = x[:, :, :, -left:]
    if right > 0:
        x = torch.cat((x, *((x[:, :, :, -1:],) * right)), dim=3)
    elif right < 0:
        x = x[:, :, :, :right]
    if top > 0:
        x = torch.cat((*((x[:, :, :1, :],) * top), x), dim=2)
    elif top < 0:
        x = x[:, :, -top:, :]
    if bottom > 0:
        x = torch.cat((x, *((x[:, :, -1:, :],) * bottom)), dim=2)
    elif bottom < 0:
        x = x[:, :, :bottom, :]
    return x.contiguous()


class _LISReplicationPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = tuple(padding)

    def forward(self, x):
        return _lis_replication_pad2d(x, self.padding)


class _LISFastLayerNorm(nn.LayerNorm):
    """LayerNorm that does not upcast to fp32 under autocast (same as nunif FastLayerNorm)."""

    def forward(self, input):
        dev = input.device.type
        try:
            ac_on = torch.is_autocast_enabled(dev)
            ac_dtype = torch.get_autocast_dtype(dev) if ac_on else None
        except (TypeError, AttributeError):   # older torch
            ac_on = torch.is_autocast_enabled() and dev == "cuda"
            ac_dtype = torch.get_autocast_gpu_dtype() if ac_on else None
        if ac_on:
            input = input.to(ac_dtype)
            weight = self.weight.to(ac_dtype) if self.weight is not None else None
            bias = self.bias.to(ac_dtype) if self.bias is not None else None
            with torch.amp.autocast(device_type=dev, enabled=False):
                return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
        return super().forward(input)


class _LISSeparableGaussianFilter2d(nn.Module):
    def __init__(self, kernel_size=15):
        super().__init__()
        sigma = kernel_size * 0.15 + 0.35
        half = (kernel_size - 1) * 0.5
        xs = torch.linspace(-half, half, steps=kernel_size)
        k = torch.exp(-0.5 * (xs / sigma).pow(2))
        k = k / k.sum()
        self.register_buffer("kernel_h", k.reshape(1, 1, 1, kernel_size), persistent=False)
        self.register_buffer("kernel_v", k.reshape(1, 1, kernel_size, 1), persistent=False)
        p = kernel_size // 2
        self.pad = _LISReplicationPad2d((p, p, p, p))

    def forward(self, x):
        x = self.pad(x)
        x = F.conv2d(x, weight=self.kernel_h, bias=None)
        x = F.conv2d(x, weight=self.kernel_v, bias=None)
        return x


class _LISGMLP(nn.Module):
    def __init__(self, embed_dim, seq_len, mlp_ratio=1):
        super().__init__()
        self.proj_in = nn.Linear(embed_dim, int(embed_dim * mlp_ratio * 2))
        self.proj_spatial = nn.Conv1d(seq_len, seq_len, kernel_size=1, stride=1, bias=True)
        self.proj_out = nn.Linear(int(embed_dim * mlp_ratio * 2) // 2, embed_dim)

    def forward(self, x, norm1=None, norm2=None):
        shortcut = x
        if norm1 is not None:
            x = norm1(x)
        x = self.proj_in(x)
        x = F.gelu(x)
        u, v = x.chunk(2, dim=-1)
        if norm2 is not None:
            v = norm2(v)
        v = self.proj_spatial(v)
        x = u * v
        x = self.proj_out(x)
        return x + shortcut


class _LISWindowGMLP2d(nn.Module):
    def __init__(self, in_channels, window_size=(4, 4), mlp_ratio=2, shift=False):
        super().__init__()
        self.window_size = window_size if isinstance(window_size, (tuple, list)) else (window_size, window_size)
        self.shift = shift
        if self.shift:
            self.pad_h = self.window_size[0] // 2
            self.pad_w = self.window_size[1] // 2
        self.seq_len = self.window_size[0] * self.window_size[1]
        self.gmlp = _LISGMLP(in_channels, seq_len=self.seq_len, mlp_ratio=mlp_ratio)

    def forward(self, x, norm1=None, norm2=None):
        if self.shift:
            x = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode="constant", value=0)
        out_shape = x.shape
        x = _lis_bchw_to_bnc(x, self.window_size)
        x = self.gmlp(x, norm1, norm2)
        x = _lis_bnc_to_bchw(x, out_shape, self.window_size)
        if self.shift:
            x = F.pad(x, (-self.pad_w, -self.pad_w, -self.pad_h, -self.pad_h))
        return x


class _LISGLUConvMLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mlp_ratio=1):
        super().__init__()
        mid = int(out_channels * mlp_ratio)
        self.pad = _LISReplicationPad2d(((kernel_size - 1) // 2,) * 4)
        self.w1 = nn.Conv2d(in_channels, mid, kernel_size=1, stride=1, padding=0)
        self.w2 = nn.Conv2d(mid // 2, out_channels, kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        x = self.w1(x)
        x = F.glu(x, dim=1)
        x = self.pad(x)
        return self.w2(x)


class _LISGMLPBlock(nn.Module):
    def __init__(self, in_channels, window_size, mlp_ratio=2, shift=False, kernel_size=3):
        super().__init__()
        self.gmlp = _LISWindowGMLP2d(in_channels, window_size=window_size, shift=shift, mlp_ratio=mlp_ratio)
        self.norm1 = _LISFastLayerNorm(in_channels, bias=False)
        self.norm2 = _LISFastLayerNorm(in_channels * mlp_ratio, bias=False)
        self.glu_conv = _LISGLUConvMLP(in_channels, in_channels, mlp_ratio=1, kernel_size=kernel_size)

    def forward(self, x):
        x = x + self.gmlp(x, self.norm1, self.norm2)
        x = x + self.glu_conv(x)
        return x


class LightInpaintStreamV1(nn.Module):
    """Port of iw3 `inpaint.light_inpaint_v1` (image model, inference only)."""

    def __init__(self):
        super().__init__()
        self.downscaling_factor = 4
        self.mod = 16
        pack = self.downscaling_factor ** 2
        C = 96
        C2 = C * 2
        self.mask_bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.patch = nn.Sequential(
            nn.Conv2d(3 * pack, C, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc1 = _LISGMLPBlock(C, window_size=16, mlp_ratio=2, shift=True)
        self.down = nn.Conv2d(C, C2, kernel_size=2, stride=2, padding=0)
        self.enc2 = nn.Sequential(
            _LISGMLPBlock(C2, window_size=8, mlp_ratio=2, shift=False),
            _LISGMLPBlock(C2, window_size=8, mlp_ratio=2, shift=True),
            _LISGMLPBlock(C2, window_size=8, mlp_ratio=2, shift=False),
            _LISGMLPBlock(C2, window_size=8, mlp_ratio=2, shift=True),
        )
        self.up = nn.Conv2d(C2, C * 4, kernel_size=1, stride=1, padding=0)
        self.dec1 = _LISGMLPBlock(C, window_size=16, mlp_ratio=2, shift=False)
        self.to_image = nn.Sequential(
            _LISReplicationPad2d((1,) * 4),
            nn.Conv2d(C, 3 * pack, kernel_size=3, stride=1, padding=0),
        )
        self.mask_blur = _LISSeparableGaussianFilter2d(kernel_size=15)

    def preprocess(self, x, mask):
        mask = mask.float()
        x = x * (1 - mask)
        mask = torch.clamp(self.mask_blur(mask) + mask, 0, 1)
        return x, mask

    @torch.inference_mode()
    def infer(self, x, mask):
        x, mask = self.preprocess(x, mask)
        return self.forward(x, mask)

    def _forward(self, x, mask):
        x = _lis_pixel_unshuffle(x, self.downscaling_factor)
        x = self.patch(x)

        mask = _lis_pixel_unshuffle(mask, self.downscaling_factor).amax(dim=1, keepdim=True) > 0.99
        x = torch.where(mask, self.mask_bias.to(x.dtype), x)

        x1 = self.enc1(x)
        x2 = self.down(x1)
        x2 = self.enc2(x2)
        x2 = self.up(x2)
        x2 = F.pixel_shuffle(x2, 2)
        x = self.dec1(x1 + x2)
        x = self.to_image(x)
        return _lis_pixel_shuffle(x, self.downscaling_factor)

    def forward(self, x, mask):
        src = x
        x = (x - 0.5) / 0.5

        input_height, input_width = x.shape[2:]
        unit = self.mod * self.downscaling_factor
        pad1 = unit - input_width % unit
        pad2 = unit - input_height % unit
        padding = (0, pad1, 0, pad2)
        x = _lis_replication_pad2d(x, padding)
        mask = _lis_replication_pad2d(mask, padding)

        x = self._forward(x, mask)
        x = F.pad(x, (0, -pad1, 0, -pad2))
        mask = F.pad(mask, (0, -pad1, 0, -pad2))

        mask = mask.expand_as(src)
        src = src * (1 - mask) + x * mask
        return src.clamp(0, 1)


def _light_inpaint_stream_model_path() -> Path:
    # Looked up in the program folder (next to live3d.py, or next to the exe when frozen).
    for base in (APP_DIR, BASE_DIR):
        cand = Path(base) / LIGHT_INPAINT_STREAM_MODEL_FILENAME
        if cand.is_file():
            return cand
    return Path(APP_DIR) / LIGHT_INPAINT_STREAM_MODEL_FILENAME


def light_inpaint_stream_model_available() -> bool:
    return _light_inpaint_stream_model_path().is_file()


# =============================================================================
# Light Inpaint (stream): user-trained nt_inpaint_v2 IMAGE model (iw3-inpaint trainer by Rowan3D)
#   - Same file name (light_inpaint.pth); the architecture is detected from the checkpoint, so the
#     stock iw3 light_inpaint_v1 file still loads exactly as before.
#   - This model was trained on the "stretched band" mask (NOT iw3's tiny gen_mask2 mask), so when it
#     is loaded the warp also marks the stretched runs (source-index gradient detector, same as the
#     iw3-inpaint extras' iw3_mask.py / Medium Inpaint).  Stock models keep iw3's own mask.
# =============================================================================
def _lis_env(name, default, cast=float):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return cast(value)
    except ValueError:
        return default


LIGHT_STREAM_NT_STRETCH_MAX_GRAD = _lis_env("NT_MASK_GRAD", 0.4)       # index advance below this = stretched
LIGHT_STREAM_NT_STRETCH_MIN_RUN  = _lis_env("NT_MASK_RUN", 3, int)     # ignore runs narrower than this
LIGHT_STREAM_NT_STRETCH_GROW     = _lis_env("NT_MASK_GROW", 2, int)    # widen the marked band (px)


def _lis_padb(t, left, right):
    """F.pad for a bool tensor along x (goes through float, always supported)."""
    return F.pad(t.float(), (left, right, 0, 0)) > 0.5


def _lis_stretch_mask(index, max_grad=None, min_run=None, grow=None):
    """Target pixels that the warp stretched or invented. index: (B,1,H,W) source x per pixel."""
    max_grad = LIGHT_STREAM_NT_STRETCH_MAX_GRAD if max_grad is None else max_grad
    min_run = LIGHT_STREAM_NT_STRETCH_MIN_RUN if min_run is None else min_run
    grow = LIGHT_STREAM_NT_STRETCH_GROW if grow is None else grow
    index = index.float()
    grad = index[..., 1:] - index[..., :-1]
    stretched = grad < max_grad
    mask = _lis_padb(stretched, 1, 0) | _lis_padb(stretched, 0, 1)
    mask = mask & ~_lis_padb(grad < -1.0, 1, 0)          # backward jump = occlusion edge, not a stretch
    k = int(min_run)
    if k > 1:                                            # drop runs shorter than k (erode, then dilate along x)
        pad = k // 2
        f = mask.float()
        eroded = -F.max_pool2d(-f, (1, k), stride=1, padding=(0, pad))
        kept = F.max_pool2d(eroded, (1, k), stride=1, padding=(0, pad))
        kept = kept[..., :mask.shape[-1]]
        mask = (kept > 0) & mask
    n = int(grow)
    if n > 0:
        mask = F.max_pool2d(mask.float(), (1, 2 * n + 1), stride=1, padding=(0, n)) > 0
    return mask


class _LISNTGMLPBlock(nn.Module):
    """nt_inpaint_v2 block: the shortcut is added ONCE (inside the gMLP), unlike light_inpaint_v1."""

    def __init__(self, in_channels, window_size, mlp_ratio=2, shift=False, kernel_size=3):
        super().__init__()
        self.gmlp = _LISWindowGMLP2d(in_channels, window_size=window_size, shift=shift, mlp_ratio=mlp_ratio)
        self.norm1 = _LISFastLayerNorm(in_channels, bias=False)
        self.norm2 = _LISFastLayerNorm(in_channels * mlp_ratio, bias=False)
        self.glu_conv = _LISGLUConvMLP(in_channels, in_channels, mlp_ratio=1, kernel_size=kernel_size)

    def forward(self, x):
        x = self.gmlp(x, self.norm1, self.norm2)      # residual is inside
        return x + self.glu_conv(x)


def _lis_nt_stack(n, channels, window, mlp_ratio, shift_first=False):
    return nn.Sequential(*[
        _LISNTGMLPBlock(channels, window_size=window, mlp_ratio=mlp_ratio,
                        shift=((i % 2 == 0) == shift_first))
        for i in range(n)
    ])


class NTInpaintStreamV2(nn.Module):
    """Port of nunif-side `inpaint.nt_inpaint_v2[_s/_b/_l]` (IMAGE model, inference only).
    Trained with iw3-inpaint (Rowan3D) -> needs the stretched-band mask (`uses_stretch_mask`)."""

    # Stretched-band mask: OFF by default (worse results on this image model).
    # Set env NT_LIGHT_STREAM_MASK=1 to turn it back on.
    uses_stretch_mask = bool(int(os.environ.get("NT_LIGHT_STREAM_MASK", "0") or 0))

    def __init__(self, dims=(128, 256, 384), blocks=(1, 4, 6), dec_blocks=(1, 2),
                 windows=(16, 8, 8), mlp_ratio=2, mask_channel=True):
        super().__init__()
        self.downscaling_factor = 4
        pack = self.downscaling_factor ** 2
        C1, C2, C3 = [int(c) for c in dims]
        w1, w2, w3 = [tuple(w) if isinstance(w, (tuple, list)) else (w, w) for w in windows]
        self.mask_channel = bool(mask_channel)
        mod = max(w1[0], w1[1], 2 * w2[0], 2 * w2[1], 4 * w3[0], 4 * w3[1])
        self.align = mod * self.downscaling_factor

        in_ch = 3 * pack + (pack if self.mask_channel else 0)
        self.patch = nn.Sequential(nn.Conv2d(in_ch, C1, kernel_size=1), nn.LeakyReLU(0.2, inplace=True))
        self.mask_bias = nn.Parameter(torch.zeros(1, C1, 1, 1))
        self.enc1 = _lis_nt_stack(blocks[0], C1, w1, mlp_ratio, shift_first=True)
        self.down1 = nn.Conv2d(C1, C2, kernel_size=2, stride=2, padding=0)
        self.enc2 = _lis_nt_stack(blocks[1], C2, w2, mlp_ratio)
        self.down2 = nn.Conv2d(C2, C3, kernel_size=2, stride=2, padding=0)
        self.enc3 = _lis_nt_stack(blocks[2], C3, w3, mlp_ratio)
        self.up2 = nn.Conv2d(C3, C2 * 4, kernel_size=1)
        self.dec2 = _lis_nt_stack(dec_blocks[1], C2, w2, mlp_ratio)
        self.up1 = nn.Conv2d(C2, C1 * 4, kernel_size=1)
        self.dec1 = _lis_nt_stack(dec_blocks[0], C1, w1, mlp_ratio)
        self.to_image = nn.Sequential(_LISReplicationPad2d((1, 1, 1, 1)),
                                      nn.Conv2d(C1, 3 * pack, kernel_size=3, stride=1, padding=0))
        self.mask_blur = _LISSeparableGaussianFilter2d(kernel_size=15)

    @classmethod
    def kwargs_from_state(cls, state, saved=None):
        """Architecture kwargs: from the checkpoint's own `kwargs` when present, else from tensor shapes."""
        def count(prefix):
            return len({int(k[len(prefix):].split(".")[0]) for k in state if k.startswith(prefix)})
        if isinstance(saved, dict) and "dims" in saved:
            keys = ("dims", "blocks", "dec_blocks", "windows", "mlp_ratio", "mask_channel")
            return {k: saved[k] for k in keys if k in saved}
        C1, C2, C3 = (int(state["mask_bias"].shape[1]), int(state["down1.weight"].shape[0]),
                      int(state["down2.weight"].shape[0]))
        win = lambda key: int(round(int(state[key].shape[0]) ** 0.5))
        return dict(
            dims=(C1, C2, C3),
            blocks=(count("enc1."), count("enc2."), count("enc3.")),
            dec_blocks=(count("dec1."), count("dec2.")),
            windows=(win("enc1.0.gmlp.gmlp.proj_spatial.weight"),
                     win("enc2.0.gmlp.gmlp.proj_spatial.weight"),
                     win("enc3.0.gmlp.gmlp.proj_spatial.weight")),
            mlp_ratio=int(state["enc1.0.gmlp.gmlp.proj_in.weight"].shape[0]) // (4 * C1),
            mask_channel=int(state["patch.0.weight"].shape[1]) > 3 * 16,
        )

    def preprocess(self, x, mask):
        mask = mask.float()
        x = x * (1 - mask)
        mask = torch.clamp(self.mask_blur(mask) + mask, 0, 1)
        return x, mask

    @torch.inference_mode()
    def infer(self, x, mask):
        x, mask = self.preprocess(x, mask)
        return self.forward(x, mask)

    def _forward(self, x, mask):
        s = self.downscaling_factor
        x = _lis_pixel_unshuffle(x, s)
        m = _lis_pixel_unshuffle(mask, s)
        if self.mask_channel:
            x = torch.cat([x, m.to(x.dtype)], dim=1)
        x = self.patch(x)
        hole = m.amax(dim=1, keepdim=True) > 0.99
        x = torch.where(hole, self.mask_bias.to(x.dtype), x)

        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        x2 = self.dec2(x2 + F.pixel_shuffle(self.up2(x3), 2))
        x1 = self.dec1(x1 + F.pixel_shuffle(self.up1(x2), 2))
        x = self.to_image(x1)
        return _lis_pixel_shuffle(x, s)

    def forward(self, x, mask):
        src = x
        x = (x - 0.5) / 0.5
        H, W = x.shape[2:]
        pad1 = (-W) % self.align
        pad2 = (-H) % self.align
        if pad1 or pad2:
            padding = (0, pad1, 0, pad2)
            x = _lis_replication_pad2d(x, padding)
            mask = _lis_replication_pad2d(mask, padding)
        x = self._forward(x, mask)
        if pad1 or pad2:
            x = F.pad(x, (0, -pad1, 0, -pad2))
            mask = F.pad(mask, (0, -pad1, 0, -pad2))
        mask = mask.expand_as(src)
        src = src * (1 - mask) + x * mask
        return src.clamp(0, 1)


def _lis_torch_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        # nunif checkpoints carry extra metadata (e.g. a datetime) that the strict loader refuses.
        return torch.load(str(path), map_location="cpu", weights_only=False)


def _load_inpaint_stream_model(device, model_path, url, tag):
    """Loads an image-inpaint checkpoint (Light Inpaint (stream)).
       The architecture is recognised from the tensors themselves:
       - nt_inpaint_v2 image model (trained with iw3-inpaint): has down2./enc3.
       - iw3 light_inpaint_v1 (stock): iw3's own mask, unchanged"""
    path = ensure_inpaint_model(model_path, url, log=lambda m: print(m, flush=True))
    ckpt = _lis_torch_load(path)
    is_ckpt = isinstance(ckpt, dict) and "state_dict" in ckpt
    state = ckpt["state_dict"] if is_ckpt else ckpt
    if "down2.weight" in state and "enc3.0.gmlp.gmlp.proj_in.weight" in state:
        kw = NTInpaintStreamV2.kwargs_from_state(state, ckpt.get("kwargs") if is_ckpt else None)
        model = NTInpaintStreamV2(**kw)
        model.load_state_dict(state, strict=True)
        print(f"{tag} nt_inpaint_v2 model detected (dims={tuple(kw['dims'])}); "
              f"stretched-band mask {'ON' if model.uses_stretch_mask else 'OFF'}", flush=True)
    else:
        model = LightInpaintStreamV1()
        model.load_state_dict(state, strict=True)
        model.uses_stretch_mask = False
    return model.to(device).eval()


def load_light_inpaint_stream_model(device):
    """Light Inpaint (stream): light_inpaint.pth (the small stock model)."""
    return _load_inpaint_stream_model(device, _light_inpaint_stream_model_path(),
                                      LIGHT_INPAINT_STREAM_URL, "[Light Inpaint (stream)]")


_LIGHT_STREAM_MODEL_LOCK = threading.Lock()
_LIGHT_STREAM_MODELS: dict = {}


def get_light_inpaint_stream_model(device):
    key = str(device)
    with _LIGHT_STREAM_MODEL_LOCK:
        model = _LIGHT_STREAM_MODELS.get(key)
        if model is None:
            model = load_light_inpaint_stream_model(device)
            _LIGHT_STREAM_MODELS[key] = model
    return model


# Set while a torch.compile first call is running, so the NVENC stall watchdog
# (conversion3d) does not mistake a long compile for a stalled encoder.
_COMPILE_BUSY = threading.Event()


def set_inpaint_model_compile(model, device, enable, log=print, tag="[Inpaint]", note=""):
    """Turn torch.compile on/off for an inpaint model, in place.
    These models are run through model.infer(), which calls self.forward() directly, so wrapping
    the module with torch.compile() would be bypassed; forward itself is replaced instead.
    Falls back to the original eager forward if the compiled path ever fails (e.g. out of memory)."""
    is_on = bool(getattr(model, "_inp_compiled", False))
    if not enable:
        if is_on:
            model.__dict__.pop("forward", None)
            model._inp_compiled = False
        log(f"{tag} torch.compile: OFF")
        return False
    if is_on:
        log(f"{tag} torch.compile: ON (already compiled in this session)")
        return True
    if getattr(device, "type", str(device)) != "cuda" or not check_compile_support("cuda"):
        log(f"{tag} torch.compile: requested but not supported here -> eager mode")
        return False

    orig = model.forward                      # bound method of the original class
    compiled = torch.compile(orig, dynamic=False)
    state = {"calls": 0}

    def _fallback(err):
        model.__dict__.pop("forward", None)
        model._inp_compiled = False
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        log(f"{tag} torch.compile FAILED -> falling back to eager mode: {err}")

    def _forward(*args, **kwargs):
        if state["calls"] == 0:
            t0 = time.time()
            _COMPILE_BUSY.set()
            try:
                out = compiled(*args, **kwargs)
                torch.cuda.synchronize()
            except Exception as e:
                _fallback(e)
                return orig(*args, **kwargs)
            finally:
                _COMPILE_BUSY.clear()
            state["calls"] = 1
            shp = tuple(args[0].shape) if args and hasattr(args[0], "shape") else "?"
            log(f"{tag} torch.compile ACTIVE: first call (compile) took {time.time() - t0:.1f}s for input {shp}")
            return out
        try:
            return compiled(*args, **kwargs)
        except Exception as e:
            _fallback(e)
            return orig(*args, **kwargs)

    model.forward = _forward
    model._inp_compiled = True
    log(f"{tag} torch.compile: ON (compiles on the first call{note})")
    return True


def set_light_stream_compile(model, device, enable, log=print):
    return set_inpaint_model_compile(model, device, enable, log=log, tag="[Light Inpaint (stream)]",
                                     note=", this takes a while")


# ---- TensorRT for Light Inpaint (stream) ------------------------------------------------------
# Only the iw3 light_inpaint_v1 network is run through TensorRT (an nt_inpaint_v2 checkpoint loaded into the
# same slot keeps running in PyTorch). model.infer() = preprocess (mask blur, stays in PyTorch) + forward();
# forward() is replaced in place, like the torch.compile path. The engine takes ONE packed input
# (N x 4 x H x W = RGB + mask) and returns the finished N x 3 x H x W image (padding, network, crop and the
# hole blend are all inside the engine). Engines are built lazily per input shape (fixed batch of 2 = both
# eyes), cached in trt_cache/, and fall back to the eager forward on any failure.
class _LISTRTExport(nn.Module):
    """Static-shape export graph of LightInpaintStreamV1.forward(). Same maths; the final crop is a slice
    instead of a negative F.pad because slicing is the safer construct for ONNX / TensorRT."""

    def __init__(self, model, height, width):
        super().__init__()
        self.m = model
        self.h = int(height)
        self.w = int(width)

    def forward(self, t):
        m = self.m
        src = t[:, 0:3]
        mask = t[:, 3:4]
        x = (src - 0.5) / 0.5
        unit = m.mod * m.downscaling_factor
        pad1 = unit - self.w % unit
        pad2 = unit - self.h % unit
        padding = (0, pad1, 0, pad2)
        x = _lis_replication_pad2d(x, padding)
        mask = _lis_replication_pad2d(mask, padding)
        x = m._forward(x, mask)
        x = x[:, :, :self.h, :self.w]
        mask = mask[:, :, :self.h, :self.w]
        mask = mask.expand_as(src)
        out = src * (1 - mask) + x * mask
        return out.clamp(0, 1)


def set_light_stream_tensorrt(model, device, enable, log=print, tag="[Light Inpaint (stream)]"):
    """Turn TensorRT on/off for the Light Inpaint (stream) model, in place. Returns True while it is active."""
    is_on = bool(getattr(model, "_lis_trt_on", False))
    if not enable:
        if is_on:
            model.__dict__.pop("forward", None)
            model._lis_trt_on = False
            log(f"{tag} TensorRT: OFF")
        return False
    if is_on:
        log(f"{tag} TensorRT: ON (already active in this session)")
        return True
    if not isinstance(model, LightInpaintStreamV1):
        log(f"{tag} TensorRT: only the light_inpaint_v1 model is supported (this checkpoint uses a different "
            f"architecture) -> PyTorch")
        return False
    if getattr(device, "type", str(device)) != "cuda" or not check_tensorrt_support():
        log(f"{tag} TensorRT: requested but unavailable -> PyTorch")
        return False
    try:
        import copy
        # A torch.compile'd forward from an earlier run in this session must go first (it would be captured as "eager").
        set_inpaint_model_compile(model, device, False, log=lambda m: None)
        orig = model.forward                                    # bound method of the original class
        pristine = copy.deepcopy(model).float().cpu().eval()    # ONNX export source, taken BEFORE forward is replaced

        def _eager(t):
            return orig(t[:, 0:3], t[:, 3:4])

        def _factory(t):
            b, _, h, w = (int(v) for v in t.shape)
            return _LISTRTExport(pristine, h, w).eval(), torch.rand((b, 4, h, w), dtype=torch.float32)

        trt_model = TensorRTDepthModel(model, fallback=_eager, tag="LightInpaintStream", fp16=True,
                                       ckpt_path=_light_inpaint_stream_model_path(), export_factory=_factory)

        def _forward(x, mask):
            # The engines are built for both eyes at once (batch 2); anything else stays in PyTorch.
            if x.dim() != 4 or x.shape[0] != 2 or x.shape[1] != 3 or mask.dim() != 4 or mask.shape[1] != 1 \
                    or mask.shape[-2:] != x.shape[-2:]:
                return orig(x, mask)
            out = trt_model(torch.cat((x.float(), mask.float()), dim=1))
            return out.to(x.dtype)

        model.forward = _forward
        model._lis_trt_on = True
        log(f"{tag} TensorRT: ON (an engine is built on the first call for each input size, then cached)")
        return True
    except Exception as e:
        model.__dict__.pop("forward", None)
        model._lis_trt_on = False
        log(f"{tag} TensorRT setup failed ({e}) -> PyTorch")
        return False


def configure_light_stream_accel(model, device, compile_flag, tensorrt_flag, log=print):
    """TensorRT first; torch.compile only when TensorRT is not active (they are mutually exclusive).
    Returns True when TensorRT is active."""
    trt_on = set_light_stream_tensorrt(model, device, bool(tensorrt_flag), log=log)
    if trt_on and compile_flag:
        log("[Light Inpaint (stream)] torch.compile: OFF (TensorRT takes priority)")
        set_light_stream_compile(model, device, False, log=lambda m: None)
    else:
        set_light_stream_compile(model, device, bool(compile_flag), log=log)
    return trt_on


# ---- TensorRT for Light Inpaint / Medium Inpaint (12-frame video models) -----------------------
# Both architectures that can sit in these two slots are covered: iw3 light_video_inpaint_v1
# (LightVideoInpaintV1) and Rowan nt_video_inpaint_v2 (NTVideoInpaintV2Port). model.infer() = window padding +
# preprocess (mask blur, stays in PyTorch) + forward(); forward() is replaced in place, like torch.compile.
# The engine takes ONE packed input (12 x 4 x H x W = RGB + mask of one eye's 12-frame window) and returns
# the finished 12 x 3 x H x W frames (padding, network incl. the temporal blocks, crop and the hole blend).
# MediumInpaintStream runs the left and the right eye as two calls with the same shape = one engine.
# The graph keeps the eager micro-batching (frames per pass through the per-frame parts) so the VRAM profile
# is not worse than eager; CD3D_TRT_INPAINT_MICRO overrides it (12 = whole window at once: faster to build and
# run, but needs a lot more VRAM at high inpaint resolutions).
TRT_INPAINT_MICRO_BATCH = max(1, _mi_env("CD3D_TRT_INPAINT_MICRO", MEDIUM_INPAINT_MICRO_BATCH, int))


class _MIVideoTRTExport(nn.Module):
    """Static-shape export graph of LightVideoInpaintV1.forward() / NTVideoInpaintV2Port.forward().
    Same maths; the final crop is a slice instead of a negative F.pad (safer for ONNX / TensorRT)."""

    def __init__(self, model, height, width, micro_batch):
        super().__init__()
        self.m = model
        self.h = int(height)
        self.w = int(width)
        self.mb = int(micro_batch)
        for _mod in model.modules():            # export copy only: temporal blocks use the rank<=4 path
            if isinstance(_mod, (_MIGMLP3DBlock, _NTGMLP3DBlock)):
                _mod._flat = True

    def forward(self, t):
        m = self.m
        src = t[:, 0:3]
        mask = t[:, 3:4]
        x = (src - 0.5) / 0.5
        if isinstance(m, NTVideoInpaintV2Port):
            pad1 = (-self.w) % m.align
            pad2 = (-self.h) % m.align
        else:
            unit = m.mod * m.downscaling_factor
            pad1 = unit - self.w % unit
            pad2 = unit - self.h % unit
        padded = bool(pad1 or pad2)
        if padded:
            padding = (0, pad1, 0, pad2)
            x = _mi_replication_pad2d(x, padding)
            mask = _mi_replication_pad2d(mask, padding)
        x = m._forward(x, mask, self.mb)
        if padded:
            x = x[:, :, :self.h, :self.w]
            mask = mask[:, :, :self.h, :self.w]
        mask = mask.expand_as(src)
        out = src * (1 - mask) + x * mask
        return out.clamp(0, 1)


def set_video_inpaint_tensorrt(model, device, enable, fp16=True, log=print, tag="[Medium Inpaint]",
                               engine_tag="MediumInpaint", ckpt_path=None):
    """Turn TensorRT on/off for a Light Inpaint / Medium Inpaint video model, in place.
    Returns True while it is active."""
    is_on = bool(getattr(model, "_vi_trt_on", False))
    if not enable:
        if is_on:
            model.__dict__.pop("forward", None)
            model._vi_trt_on = False
            log(f"{tag} TensorRT: OFF")
        return False
    if is_on:
        log(f"{tag} TensorRT: ON (already active)")
        return True
    if not isinstance(model, (LightVideoInpaintV1, NTVideoInpaintV2Port)):
        log(f"{tag} TensorRT: unsupported model architecture -> PyTorch")
        return False
    if getattr(device, "type", str(device)) != "cuda" or not check_tensorrt_support():
        log(f"{tag} TensorRT: requested but unavailable -> PyTorch")
        return False
    try:
        import copy
        set_inpaint_model_compile(model, device, False, log=lambda m: None)   # drop a compiled forward, if any
        orig = model.forward                                    # bound method of the original class
        pristine = copy.deepcopy(model).float().cpu().eval()    # ONNX export source, taken BEFORE forward is replaced
        mb = int(TRT_INPAINT_MICRO_BATCH)

        def _eager(t):
            return orig(t[:, 0:3], t[:, 3:4], mb)

        def _factory(t):
            n, _, h, w = (int(v) for v in t.shape)
            return _MIVideoTRTExport(pristine, h, w, mb).eval(), torch.rand((n, 4, h, w), dtype=torch.float32)

        trt_model = TensorRTDepthModel(model, fallback=_eager, tag=engine_tag, fp16=bool(fp16),
                                       ckpt_path=ckpt_path, workspace_gb=4, export_factory=_factory)

        def _forward(x, mask, micro_batch_size=None):
            # Engines are built for one full 12-frame window; anything else stays in PyTorch.
            if x.dim() != 4 or x.shape[0] != _MI_SEQ_LEN or x.shape[1] != 3 or mask.dim() != 4 \
                    or mask.shape[0] != x.shape[0] or mask.shape[1] != 1 or mask.shape[-2:] != x.shape[-2:]:
                return orig(x, mask) if micro_batch_size is None else orig(x, mask, micro_batch_size)
            out = trt_model(torch.cat((x.float(), mask.float()), dim=1))
            return out.to(x.dtype)

        model.forward = _forward
        model._vi_trt_on = True
        log(f"{tag} TensorRT: ON (an engine is built on the first window for each input size, then cached; "
            f"the first build of this large model can take several minutes)")
        return True
    except Exception as e:
        model.__dict__.pop("forward", None)
        model._vi_trt_on = False
        log(f"{tag} TensorRT setup failed ({e}) -> PyTorch")
        return False


def configure_video_inpaint_accel(model, device, compile_flag, tensorrt_flag, fp16, label, engine_tag, ckpt_path, log=print):
    """TensorRT first; torch.compile only when TensorRT is not active (mutually exclusive).
    Returns True when TensorRT is active."""
    tag = f"[{label}]"
    trt_on = set_video_inpaint_tensorrt(model, device, bool(tensorrt_flag), fp16=bool(fp16), log=log, tag=tag,
                                        engine_tag=engine_tag, ckpt_path=ckpt_path)
    if trt_on:
        if compile_flag:
            log(f"{tag} torch.compile: OFF (TensorRT takes priority)")
    else:
        set_inpaint_model_compile(model, device, bool(compile_flag), log=log, tag=tag,
                                  note=", large graph: can take several minutes the first time")
    return trt_on


# ---- mask preprocessing (iw3 ForwardInpaintImage.preprocess_mask) ------------------------------
def _lis_mask_closing(mask, kernel_size=3, n_iter=2):
    org = mask.float()
    m = org
    pad = kernel_size // 2
    for _ in range(n_iter):
        m = F.max_pool2d(m, kernel_size=kernel_size, stride=1, padding=pad)
    for _ in range(n_iter):
        m = -F.max_pool2d(-m, kernel_size=kernel_size, stride=1, padding=pad)
    return (m + org).clamp(0, 1)


def _lis_dilate_outer(mask, n_iter):
    # right-eye based (the left eye is flipped before this call)
    if n_iter <= 0:
        return mask
    m = mask.bool()
    for _ in range(n_iter):
        m = m | F.pad(m, (1, 0, 0, 0))[:, :, :, :-1]
    return m.to(mask.dtype)


def _lis_dilate_inner(mask, n_iter):
    if n_iter <= 0:
        return mask
    m = mask.bool()
    for _ in range(n_iter):
        m = m | F.pad(m, (0, 1, 0, 0))[:, :, :, 1:]
    return m.to(mask.dtype)


def _lis_preprocess_mask(mask, outer_dilation=None, inner_dilation=None):
    outer = LIGHT_INPAINT_STREAM_OUTER_DILATION if outer_dilation is None else outer_dilation
    inner = LIGHT_INPAINT_STREAM_INNER_DILATION if inner_dilation is None else inner_dilation
    mask = mask > 0
    mask = _lis_mask_closing(mask)
    mask = _lis_dilate_outer(mask, outer)
    mask = _lis_dilate_inner(mask, inner)
    return mask


# ---- iw3 depth-order bilinear forward warp (holes are NOT filled) -------------------------------
def _lis_shift_fill(x, sign, max_tries=100):
    mask = x < 0
    while mask.any().item() and max_tries > 0:
        if sign > 0:
            x[mask] = F.pad(x[:, :, :, 1:], (0, 1, 0, 0))[mask]
        else:
            x[mask] = F.pad(x[:, :, :, :-1], (1, 0, 0, 0))[mask]
        mask = x < 0
        max_tries -= 1
    return x


def _lis_fix_layered_holes(side_image, index_image, sign, max_tries=100):
    if sign > 0:
        mask = F.pad((index_image[:, :, :, :-1] - index_image[:, :, :, 1:]) > 0, (0, 1, 0, 0))
        while mask.any().item() and max_tries > 0:
            side_image[mask.expand_as(side_image)] = -2  # set undefined value
            index_image[mask] = F.pad(index_image[:, :, :, 1:], (0, 1, 0, 0))[mask]
            mask = F.pad((index_image[:, :, :, :-1] - index_image[:, :, :, 1:]) > 0, (0, 1, 0, 0))
            max_tries -= 1
    else:
        mask = F.pad((index_image[:, :, :, :-1] - index_image[:, :, :, 1:]) > 0, (1, 0, 0, 0))
        while mask.any().item() and max_tries > 0:
            side_image[mask.expand_as(side_image)] = -2
            index_image[mask] = F.pad(index_image[:, :, :, :-1], (1, 0, 0, 0))[mask]
            mask = F.pad((index_image[:, :, :, :-1] - index_image[:, :, :, 1:]) > 0, (1, 0, 0, 0))
            max_tries -= 1


def _lis_to_flat_index(batch, width, height, index):
    index = index + torch.arange(0, height, device=index.device).view(1, height, 1) * width
    index = index + torch.arange(0, batch, device=index.device).view(batch, 1, 1) * height * width
    return index.view(-1)


def _lis_make_bilinear_data(batch, width, height, index, index_shift):
    float_index = torch.clamp(index + index_shift, 0, width - 1)
    floor_index = torch.clamp(float_index.floor(), 0, width - 1)
    ceil_index = torch.clamp(float_index.ceil(), 0, width - 1)
    ceil_weight = (float_index - floor_index).reshape(batch, 1, height, width)
    ceil_weight = torch.clamp(ceil_weight, min=1e-5, max=1.0 - 1e-5)
    floor_weight = 1.0 - ceil_weight
    floor_index = _lis_to_flat_index(batch, width, height, floor_index.long())
    ceil_index = _lis_to_flat_index(batch, width, height, ceil_index.long())
    return floor_index, ceil_index, floor_weight, ceil_weight


def _lis_ordered_index_copy(c, src_index, dest_index, index_order, undefined_value=-1):
    B, _, H, W = c.shape
    c = c.permute(0, 2, 3, 1).reshape(-1, c.shape[1])
    if torch.is_tensor(undefined_value):
        out = undefined_value.view(1, -1).repeat(c.shape[0], 1)
    else:
        out = torch.empty_like(c).fill_(undefined_value)

    # Deterministically resolve "last write wins"
    ordered_dest = dest_index[index_order]
    pos = torch.arange(ordered_dest.numel(), device=c.device)
    best_pos = torch.full((out.shape[0],), -1, dtype=torch.long, device=c.device)
    best_pos.scatter_reduce_(0, ordered_dest, pos, reduce="amax", include_self=False)

    valid = best_pos >= 0
    src_values = c[src_index[index_order]]
    out[valid] = src_values[best_pos[valid]]
    return out.view(B, H, W, -1).permute(0, 3, 1, 2)


def _lis_warp(batch, width, height, c, x_index, index_shift, src_index, index_order):
    floor_index, ceil_index, floor_weight, ceil_weight = _lis_make_bilinear_data(
        batch, width, height, x_index, index_shift)

    floor_data = torch.cat([floor_weight, c], dim=1)
    ceil_data = torch.cat([ceil_weight, c], dim=1)

    # 0 for weight, -1 for pixel
    undefined_value = torch.tensor([0] + [-1] * c.shape[1], dtype=c.dtype, device=c.device)
    floor_warp = _lis_ordered_index_copy(floor_data, src_index, floor_index, index_order, undefined_value)
    ceil_warp = _lis_ordered_index_copy(ceil_data, src_index, ceil_index, index_order, undefined_value)

    floor_weight_warp, floor_warp = floor_warp[:, 0:1, :, :], floor_warp[:, 1:, :, :]
    ceil_weight_warp, ceil_warp = ceil_warp[:, 0:1, :, :], ceil_warp[:, 1:, :, :]

    out = (floor_warp * floor_weight_warp + ceil_warp * ceil_weight_warp) / (floor_weight_warp + ceil_weight_warp)
    return torch.nan_to_num(out, -1)


def _lis_gen_mask(eye):
    eye = eye[:, 0:1]
    return torch.clamp((eye == -1).float() + (eye == -2).float() * 0.5, 0, 1)


def light_stream_forward_warp_pair(rgb, depth, divergence, convergence, preserve_screen_border: bool = False,
                                  stretch_mask: bool = False):
    """iw3 forward warp (+ layered hole fix) for BOTH eyes, holes NOT filled.
    Same result as running the single-eye warp twice, but the depth sort (the expensive part) is shared.
    Returns (left_rgb, right_rgb, left_hole_mask, right_hole_mask); masks are bool, B x 1 x H x W.
    stretch_mask=True (nt_inpaint_v2 models): also marks the stretched band found from the warp index."""
    B, C, H, W = rgb.shape
    device, in_dtype = rgb.device, rgb.dtype
    if depth.shape[-2:] != (H, W):
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)

    # fp32 is required: pixel indices (up to a few thousand) are not representable in fp16.
    c = rgb.float()
    depth = depth[:, 0:1].float()

    base_size = max(H, W)
    padding_size = int(base_size * (float(divergence) * 2.0) * 0.01 + 2)
    shift_size = float(divergence) * 0.01 * base_size * 0.5
    base_shift = depth[:, 0] * shift_size - shift_size * convergence   # B,H,W (sign applied per eye)

    if preserve_screen_border:
        border_pix = max(1, round(float(divergence) * 0.75 * 0.01 * W))
        if border_pix > 0 and border_pix * 2 < W:
            border_weight_l = torch.linspace(0.0, 1.0, border_pix, dtype=base_shift.dtype, device=device)
            border_weight_r = torch.linspace(1.0, 0.0, border_pix, dtype=base_shift.dtype, device=device)
            base_shift[..., :border_pix] *= border_weight_l
            base_shift[..., -border_pix:] *= border_weight_r

    pad = (padding_size, padding_size, 0, 0)
    c = _lis_replication_pad2d(c, pad)
    depth = _lis_replication_pad2d(depth, pad)
    base_shift = _lis_replication_pad2d(base_shift.unsqueeze(1), pad)[:, 0]
    _, _, Hp, Wp = depth.shape

    x_index = torch.arange(0, Wp, device=device).view(1, 1, Wp).expand(B, Hp, Wp)
    src_index = _lis_to_flat_index(B, Wp, Hp, x_index)
    # This app's depth is inverted (near = small) -> sort descending so the nearest pixel is written last.
    index_order = torch.argsort(depth.reshape(-1), dim=0, descending=True)

    c = torch.cat([c, x_index.view(B, 1, Hp, Wp).to(c.dtype)], dim=1)  # warp the width index together

    eyes = []
    for sign in (-1.0, +1.0):    # left, right
        eye = _lis_warp(B, Wp, Hp, c, x_index, base_shift * sign, src_index, index_order)
        eye = _lis_replication_pad2d(eye, (-padding_size, -padding_size, 0, 0))   # unpad
        eye, eye_index = eye[:, :-1, :, :], eye[:, -1:, :, :]
        if sign > 0:   # iw3 "right"
            eye_index = _lis_shift_fill(eye_index, 1)
            _lis_fix_layered_holes(eye, eye_index, -1)
        else:          # iw3 "left"
            eye_index = _lis_shift_fill(eye_index, -1)
            _lis_fix_layered_holes(eye, eye_index, 1)
        if stretch_mask:   # nt_inpaint_v2 model: also mark the stretched band (iw3's own mask misses it)
            eye.masked_fill_(_lis_stretch_mask(eye_index).expand_as(eye), -2)
        hole_mask = _lis_gen_mask(eye) > 0
        eye = torch.clamp(eye, 0, 1)   # drop undefined values (-1 / -2 -> 0)
        eyes.append((eye.to(in_dtype).contiguous(), hole_mask))

    (left_t, left_m), (right_t, right_m) = eyes
    return left_t, right_t, left_m, right_m


class LightInpaintStreamCache:
    """Per-worker state that makes Light Inpaint (stream) temporally stable.

    * Throttling: the model is only re-run when the hole changed a lot (new area uncovered), when the
      divergence/convergence/border settings changed, or every `max_interval` frames. In between, the cached
      texture is re-used for the hole and only brand-new hole pixels are filled with the deterministic fast fill.
    * Hysteresis: a pixel that was inpainted keeps showing the cached texture for `hold_frames` more frames after
      the (sub-pixel jittering) hole mask drops it, so the mask edge no longer flips every frame.
    * Cross-fade: when the model is re-run, the displayed hole texture blends from the previous texture to the
      new one over `fade_frames` frames instead of popping.
    """

    def __init__(self, max_interval=LIGHT_INPAINT_STREAM_CACHE_MAX_INTERVAL,
                 change_thresh=LIGHT_INPAINT_STREAM_CACHE_CHANGE_THRESH,
                 hold_frames=LIGHT_INPAINT_STREAM_HOLD_FRAMES,
                 fade_frames=LIGHT_INPAINT_STREAM_FADE_FRAMES):
        self.max_interval = max(1, int(max_interval))
        self.change_thresh = float(change_thresh)
        self.hold_frames = max(0, int(hold_frames))
        self.fade_frames = max(1, int(fade_frames))
        self.reset()

    def reset(self):
        self.tex = [None, None]    # [left, right] target texture in screen space (1x3xHxW)
        self.old = [None, None]    # [left, right] texture the fade starts from
        self.hold = [None, None]   # [left, right] int16 countdown (>0 -> texture valid / pixel is treated as hole)
        self.fade = 0              # frames elapsed since the last refresh (>= fade_frames -> fade finished)
        self.frames_since_refresh = 0
        self.key = None

    def compatible(self, ref, key):
        t = self.tex[0]
        return (t is not None and self.tex[1] is not None and t.shape == ref.shape
                and t.dtype == ref.dtype and t.device == ref.device and self.key == key)

    def displayed(self, i):
        """Texture currently shown for eye i (cross-fade between old and target)."""
        a = min(1.0, self.fade / float(self.fade_frames))
        if a >= 1.0:
            return self.tex[i]
        return torch.lerp(self.old[i], self.tex[i], a)


def _lis_update_hold(prev_hold, pm, hold_frames):
    # hole pixels -> full countdown, everything else counts down towards 0
    return torch.where(pm, torch.full_like(prev_hold, hold_frames), (prev_hold - 1).clamp_(min=0))


# ---------------------------------------------------------------------------
# Inpaint Resolution (ported from live3d.py): the image/video inpaint models only ever run on a frame
# downscaled to the user's "Inpaint Resolution" setting (get_inpaint_res(), never above the actual output
# height), the full-resolution picture is kept everywhere outside the hole. Used by Light Inpaint (stream)
# (light_inpaint_stream_stereo) and by the video models Light Inpaint / Medium Inpaint (MediumInpaintStream).
# ---------------------------------------------------------------------------
def _inpres_even(n):
    n = int(n)
    return n + 1 if n % 2 else n


def _inpres_target_height(h):
    """Height the inpaint model runs at for a frame `h` tall: the user's Inpaint Resolution setting,
    capped at the frame (= output) height, so a setting above the output is never exceeded."""
    res = get_inpaint_res()
    if res <= 0:                # blank option: no limit -> run at the frame's own resolution
        return int(h)
    return min(int(h), res)


# Frames processed per full-resolution pass in _inpres_downscale/_inpres_compose. The two bicubic
# upsamples back to source resolution in _inpres_compose are the largest tensors in this path (a full
# extra copy of the batch at output resolution each), so this bounds how many of those are ever in
# flight at once for the video models (Light Inpaint / Medium Inpaint), which otherwise hand the whole
# 12-frame window to these functions in one call. Per-frame work, so the result does not change.
# NT_TAIL_CHUNK (same env var as iw3-lowres) overrides it; 0 disables chunking.
def _inpres_chunk():
    try:
        return int(os.environ.get("NT_TAIL_CHUNK", 6))
    except ValueError:
        return 6


def _inpres_downscale(eye, mask, max_height):
    """Downscale `eye`/`mask` so the model only ever runs on a frame `max_height` tall. Divides by the
    valid-pixel weight instead of averaging raw pixels, so the hole's black is not smeared into its
    neighbours, and max-pools the mask instead of sampling it, so a thin hole band cannot vanish on the
    way down. Processed a few frames at a time (see _inpres_chunk) so the source-resolution `eye * valid`
    copy never covers more than a chunk at once (ported from live3d.py / iw3-lowres, Rowan3D)."""
    h, w = eye.shape[-2:]
    new_h = _inpres_even(max_height)
    new_w = _inpres_even(round(w * (new_h / h)))
    size = (max(8, new_h), max(8, new_w))
    mask_f = mask if mask.is_floating_point() else mask.float()

    def _down(e, m):
        valid = 1.0 - m
        low_sum = F.interpolate(e * valid, size=size, mode="bilinear", antialias=True, align_corners=False)
        low_weight = F.interpolate(valid, size=size, mode="bilinear", antialias=True, align_corners=False)
        return (low_sum / torch.clamp(low_weight, min=1e-6),
                F.adaptive_max_pool2d(m, size) > 0)

    n, step = eye.shape[0], _inpres_chunk()
    step = n if step <= 0 or step >= n else step
    if step == n:
        return _down(eye, mask_f)
    parts = [_down(eye[i:i + step], mask_f[i:i + step]) for i in range(0, n, step)]
    low_eye = torch.cat([p[0] for p in parts], dim=0)
    low_mask = torch.cat([p[1] for p in parts], dim=0)
    return low_eye, low_mask


def _inpres_mask_blur(model, eye, mask):
    """The soft mask `model.preprocess(eye, mask)` returns, without the erased picture it returns beside it
    (ported from live3d.py / iw3-lowres, Rowan3D)."""
    m = mask if mask.is_floating_point() else mask.float()
    blur = getattr(model, "mask_blur", None)
    if blur is not None:
        return torch.clamp(blur(m) + m, 0, 1)
    if hasattr(model, "preprocess"):
        return model.preprocess(eye, mask)[1]
    return m


def _inpres_compose(model, eye, low_eye, filled, mask):
    """Put the full-resolution picture back everywhere except inside the hole, where the (upscaled) model
    fill is used. `filled` is at low resolution, `eye`/`low_eye`/`mask` are at full/low resolution
    respectively. Processed a few frames at a time (see _inpres_chunk): the two upsamples back to `eye`'s
    resolution are the largest tensors in this whole path, so how many frames are composed at once here is
    the memory decision that matters most for the video models (Light Inpaint / Medium Inpaint), which
    otherwise pass a whole 12-frame window through in one call (ported from live3d.py / iw3-lowres,
    Rowan3D)."""
    full = eye.shape[-2:]

    def _compose(e, lo, fi, m):
        bm = _inpres_mask_blur(model, e, m)
        low_up = F.interpolate(lo, size=full, mode="bicubic", antialias=True, align_corners=False)
        filled_up = F.interpolate(fi, size=full, mode="bicubic", antialias=True, align_corners=False)
        detail = e - low_up
        return filled_up + detail * (1.0 - bm)

    n, step = eye.shape[0], _inpres_chunk()
    if step <= 0 or step >= n:
        return _compose(eye, low_eye, filled, mask)
    out = None
    for i in range(0, n, step):
        part = _compose(eye[i:i + step], low_eye[i:i + step], filled[i:i + step], mask[i:i + step])
        if out is None:
            out = part.new_empty((n,) + part.shape[1:])
        out[i:i + step] = part
    return out


# ---- Light Inpaint (stream): soft composite (hides the inpaint band edge at high stereo strength) ----
# Full-resolution, float32 (best quality). Speed: the same maths as before, but (a) the horizontal box blur uses a
# running sum (cost independent of the blur radius), (b) all blurs with the same radius are done in ONE call
# (delta x3, margin, 1-mask packed in 5 channels), (c) both eyes are processed as one batch.
LIS_BLEND_ENABLE = True
# Inpaint edge blend is skipped at strength <= 1.0 (it only runs from 1.1 up).
LIS_BLEND_MIN_DIVERGENCE_EXCL = 1.0
LIS_BLEND_FEATHER_MIN = 3         # px (output resolution)
LIS_BLEND_FEATHER_MAX = 24
LIS_BLEND_FEATHER_PER_DIV = 0.6   # feather width = W * 1% * this * divergence
LIS_BLEND_BATCH_EYES = True       # False = one eye at a time (a bit slower, less temporary GPU memory)


def _lis_box_blur(x, r):
    """Horizontal box blur, radius r, replicate padding. Same result as pad + avg_pool2d((1, 2r+1)),
    computed with a running sum so the cost does not grow with r."""
    if r <= 0:
        return x
    k = 2 * r + 1
    c = torch.cumsum(F.pad(x, (r, r, 0, 0), mode="replicate"), dim=-1)
    c = F.pad(c, (1, 0, 0, 0))
    return (c[..., k:] - c[..., :-k]) * (1.0 / k)


def _lis_blur2(x, r):
    # two box blurs = triangle filter (horizontal only, GPU only)
    return _lis_box_blur(_lis_box_blur(x, r), r)


def _lis_soft_composite_group(curs, texs, hole_pms, hole_trues, divergence):
    """curs/texs: lists of 1x3xHxW (warped real image with black holes / inpaint texture), hole_pms: lists of bool
    1x1xHxW (area the model repaints), hole_trues: lists of bool 1x1xHxW (real disocclusion holes). Returns a list of
    composited frames. Instead of a hard torch.where: the colour difference between the real pixels around the band
    and the inpaint texture is diffused into the band (seamless-clone style), and the band edge gets a short alpha
    ramp. Never reads `tex` outside hole_pm (the cache is stale there)."""
    in_dtype = curs[0].dtype
    n = len(curs)
    cur = torch.cat(curs, dim=0).float()
    tex = torch.cat(texs, dim=0).float()
    pm = torch.cat(hole_pms, dim=0)
    ht = torch.cat(hole_trues, dim=0)
    W = cur.shape[-1]
    feather = int(round(W * 0.01 * LIS_BLEND_FEATHER_PER_DIV * max(float(divergence), 0.5)))
    feather = max(LIS_BLEND_FEATHER_MIN, min(LIS_BLEND_FEATHER_MAX, feather))

    m = pm.float()
    margin = (pm & ~ht).float()          # valid real pixels that the model repainted (cur and tex both valid)

    # 1) colour correction: boundary residual (real - inpaint) spread into the band, fading inward
    packed = torch.cat([(cur - tex) * margin, margin, 1.0 - m], dim=1)      # 3 + 1 + 1 channels, one blur call
    bl = _lis_blur2(packed, feather)
    num, den, inv_m = bl[:, 0:3], bl[:, 3:4], bl[:, 4:5]
    corr = num / den.clamp(min=1e-3)
    support = (den * 4.0).clamp(0.0, 1.0)
    w = (2.0 * inv_m).clamp(0.0, 1.0) * m * support
    tex_adj = tex + corr * w

    # 2) short alpha ramp at the band edge (true holes always stay fully inpainted)
    r2 = max(1, feather // 4)
    alpha = _lis_blur2(m, r2) * m
    alpha = torch.maximum(alpha, ht.float()).clamp(0.0, 1.0)
    out = (cur * (1.0 - alpha) + tex_adj * alpha).to(in_dtype)
    return [out[i:i + 1] for i in range(n)]


def light_inpaint_stream_stereo(rgb_t, depth_norm, divergence, convergence, device, preserve_screen_border=False,
                         cache: "LightInpaintStreamCache | None" = None):
    """One frame in, one frame out (no queue, no delay). rgb_t: 1x3xHxW, depth_norm: already normalized.
    With `cache`, the model output is cached/throttled, cross-faded on refresh, and the hole mask gets hysteresis.
    Model: light_inpaint.pth (small stock model or a v2-trained nt_inpaint_v2 checkpoint, detected from the file)."""
    in_dtype = rgb_t.dtype
    model = get_light_inpaint_stream_model(device)
    _outer_dil, _inner_dil = LIGHT_INPAINT_STREAM_OUTER_DILATION, LIGHT_INPAINT_STREAM_INNER_DILATION
    left_t, right_t, left_m, right_m = light_stream_forward_warp_pair(
        rgb_t, depth_norm, divergence, convergence, preserve_screen_border,
        stretch_mask=bool(getattr(model, "uses_stretch_mask", False)))

    # The model works in right-eye geometry: flip the left eye, run both eyes as one batch of 2.
    masks = _lis_preprocess_mask(torch.cat([left_m.flip(-1), right_m], dim=0), _outer_dil, _inner_dil)

    def _run_model():
        eyes = torch.cat([left_t.flip(-1), right_t], dim=0)
        use_amp = device.type == "cuda"
        if not use_amp:
            eyes = eyes.float()
        _target_h = _inpres_target_height(eyes.shape[-2])
        if _target_h < eyes.shape[-2]:
            # Low-res inpainting, full-res output: the model only ever sees a frame capped at the user's
            # Inpaint Resolution setting (never above the output height), the source resolution is kept
            # everywhere outside the hole (ported from live3d.py / iw3-lowres, Rowan3D).
            low_eyes, low_masks = _inpres_downscale(eyes, masks, _target_h)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                filled = model.infer(low_eyes, low_masks)
            filled = filled.to(eyes.dtype)
            out = _inpres_compose(model, eyes, low_eyes, filled, masks)
        else:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model.infer(eyes, masks)
        out = out.to(in_dtype)
        return [out[0:1].flip(-1), out[1:2]]

    if cache is None:
        o = _run_model()
        return o[0], o[1]

    # Pixels the model actually repaints (dilated mask), back in screen space.
    pm_all = masks > 0.5
    pms = [pm_all[0:1].flip(-1), pm_all[1:2]]
    cur = [left_t, right_t]
    key = (round(float(divergence), 4), round(float(convergence), 2), bool(preserve_screen_border))
    hold_frames = cache.hold_frames

    compat = cache.compatible(left_t, key)
    if not compat:
        refresh = True
    else:
        cache.frames_since_refresh += 1
        if cache.frames_since_refresh >= cache.max_interval:
            refresh = True
        else:
            # fraction of the hole that is not covered by the cached texture (= newly uncovered area)
            ratios = torch.stack([
                (pms[i] & ~(cache.hold[i] > 0)).float().sum() / pms[i].float().sum().clamp(min=1.0)
                for i in range(2)
            ])
            refresh = bool((ratios > cache.change_thresh).any())   # single GPU sync

    new_tex, new_old, new_hold = [None, None], [None, None], [None, None]
    if refresh:
        outs = _run_model()
        for i in range(2):
            if compat:
                known = cache.hold[i] > 0
                disp_prev = cache.displayed(i)                       # what is on screen right now
                new_tex[i] = torch.where(pms[i], outs[i], cache.tex[i])
                # fade only where a texture was already visible; newly covered pixels switch directly
                new_old[i] = torch.where(known, disp_prev, new_tex[i])
                prev_hold = cache.hold[i]
            else:
                new_tex[i] = outs[i]
                new_old[i] = outs[i]
                prev_hold = torch.zeros(pms[i].shape, dtype=torch.int16, device=pms[i].device)
            new_hold[i] = _lis_update_hold(prev_hold, pms[i], hold_frames)
        cache.fade = 0 if compat else cache.fade_frames
        cache.frames_since_refresh = 0
        cache.key = key
    else:
        for i in range(2):
            known = cache.hold[i] > 0
            brand_new = pms[i] & ~known
            tex = cache.tex[i]
            disp = cache.displayed(i)
            # newly uncovered pixels: deterministic fast fill (no model call), stored so they stay stable too
            base = torch.where(known, disp, cur[i])
            filled = true_fast_isotropic_inpaint_gpu(base, brand_new, radius=TRUE_FAST_INPAINT_RADIUS)
            new_tex[i] = torch.where(brand_new, filled, tex)
            new_old[i] = torch.where(brand_new, filled, cache.old[i])   # no fade for these pixels
            new_hold[i] = _lis_update_hold(cache.hold[i], pms[i], hold_frames)

    cache.tex, cache.old, cache.hold = new_tex, new_old, new_hold
    cache.fade = min(cache.fade_frames, cache.fade + 1)
    # hysteresis: hole OR still held -> (cross-faded) cached texture, otherwise the real warped pixel
    if LIS_BLEND_ENABLE and float(divergence) > LIS_BLEND_MIN_DIVERGENCE_EXCL + 1e-6:
        _cs = [cur[0], cur[1]]
        _ts = [cache.displayed(0), cache.displayed(1)]
        _ps = [new_hold[0] > 0, new_hold[1] > 0]
        _hs = [left_m, right_m]
        if LIS_BLEND_BATCH_EYES:
            left_out, right_out = _lis_soft_composite_group(_cs, _ts, _ps, _hs, divergence)
        else:
            left_out = _lis_soft_composite_group(_cs[:1], _ts[:1], _ps[:1], _hs[:1], divergence)[0]
            right_out = _lis_soft_composite_group(_cs[1:], _ts[1:], _ps[1:], _hs[1:], divergence)[0]
    else:
        left_out = torch.where(new_hold[0] > 0, cache.displayed(0), cur[0])
        right_out = torch.where(new_hold[1] > 0, cache.displayed(1), cur[1])
    return left_out, right_out


def make_stereo_light_stream(bgr_t, depth_t, divergence, convergence, device, already_normalized=False,
                             preserve_screen_border=False, cache=None):
    """Light Inpaint (stream): BGR in -> BGR out (the model itself works in RGB)."""
    if already_normalized:
        depth_norm = depth_t.to(dtype=bgr_t.dtype)
    else:
        depth_norm = normalize_depth_gpu(depth_t)[0].to(dtype=bgr_t.dtype)
    with torch.inference_mode():
        rgb_t = bgr_t[:, [2, 1, 0], :, :]
        left_t, right_t = light_inpaint_stream_stereo(
            rgb_t, depth_norm, divergence, convergence, device,
            preserve_screen_border=preserve_screen_border, cache=cache)
        return left_t[:, [2, 1, 0], :, :].contiguous(), right_t[:, [2, 1, 0], :, :].contiguous()


def make_stereo_with_masks(bgr_t, depth_t, divergence, convergence, device, already_normalized=False,
                           preserve_screen_border=False):
    if already_normalized:
        depth_norm = depth_t.to(dtype=bgr_t.dtype)
    else:
        depth_norm = normalize_depth_gpu(depth_t)[0].to(dtype=bgr_t.dtype)
    with torch.inference_mode():
        # Medium Inpaint only: iw3 forward warp (NOT the integer warp used by Fast Inpaint)
        left_t, right_t, left_m, right_m = medium_forward_warp_pair(
            bgr_t, depth_norm, divergence, convergence, preserve_screen_border)
    return left_t, right_t, left_m, right_m

# Light Inpaint (video model) only: temporal smoothing INSIDE the inpainted area (anti-flicker), same as live3d.py.
LIGHT_VIDEO_INPAINT_MASK_TEMPORAL = 0   # hole mask = union over +-N neighbouring frames of the window (0 = off). Removes mask #1
                                        # edge jitter (depth noise, stretch-mask threshold flicker) that makes the fill texture boil
LIGHT_VIDEO_INPAINT_TEMPORAL_SMOOTH = 0.40   # 0 = off. Temporal smoothing INSIDE the inpainted (filled) area only: the fill is blended
                                            # with the previous frame's fill (0.5 = half / half; higher = calmer but more ghosting on
                                            # moving backgrounds, ~0.7 max). Reduces the "TV noise" boil of the fill texture.
LIGHT_VIDEO_INPAINT_SMOOTH_RADIUS = 6       # px around the hole mask that count as "fill area" (the model also repaints the dilated rim)
LIGHT_VIDEO_INPAINT_SMOOTH_GATE = (0.20, 0.20)  # (start, width): where the fill really changed by more than `start` (0..1) the blend fades out
                                            # over `width`, so real motion / scene cuts are not smeared


def _lv_temporal_mask_union(masks, k):
    """masks: (T,1,H,W). Per-pixel max over the +-k neighbouring frames (edges replicated)."""
    k = min(int(k), masks.shape[0] - 1)
    if k <= 0:
        return masks
    out = masks
    for d in range(1, k + 1):
        prev = torch.cat([masks[:1].expand(d, -1, -1, -1), masks[:-d]], dim=0)
        nxt = torch.cat([masks[d:], masks[-1:].expand(d, -1, -1, -1)], dim=0)
        out = torch.maximum(out, torch.maximum(prev, nxt))
    return out


class MediumInpaintStream:
    """Frame-by-frame wrapper of the 12-frame video inpaint model (PRE/POST overlap).
    temporal_smooth=True (Light Inpaint only): the filled area is blended with the previous frame's fill (anti-flicker)."""

    def __init__(self, model, pre_padding=MEDIUM_INPAINT_PRE_PADDING, post_padding=MEDIUM_INPAINT_POST_PADDING,
                 temporal_smooth=False):
        self.model = model
        self._temporal_on = bool(temporal_smooth)
        self._sm = {}   # temporal smoothing state per eye: (previous smoothed frame, previous fill-area weight)
        self.seq = _MI_SEQ_LEN
        self.pre = int(pre_padding)
        self.post = int(post_padding)
        self.items = []  # each: (left[3,H,W], right[3,H,W], left_mask[1,H,W], right_mask[1,H,W])

    def _inpaint_eyes(self, eyes, masks, is_left):
        if is_left:
            eyes, masks = eyes.flip(-1), masks.flip(-1)
        masks = _mi_preprocess_mask(masks)
        if self._temporal_on:   # Light Inpaint only (Medium Inpaint shares this class and stays unchanged)
            masks = _lv_temporal_mask_union(masks, LIGHT_VIDEO_INPAINT_MASK_TEMPORAL)
        _target_h = _inpres_target_height(eyes.shape[-2])
        if _target_h < eyes.shape[-2]:
            # Low-res inpainting, full-res output: cap the model's input at the user's Inpaint
            # Resolution setting (never above the output height), same logic as Light Inpaint (stream).
            low_eyes, low_masks = _inpres_downscale(eyes, masks, _target_h)
            filled = self.model.infer(low_eyes, low_masks)
            out = _inpres_compose(self.model, eyes, low_eyes, filled, masks)
        else:
            out = self.model.infer(eyes, masks)
        if is_left:
            out = out.flip(-1)
        return out

    def _run(self, flush):
        lefts = torch.stack([it[0] for it in self.items])
        rights = torch.stack([it[1] for it in self.items])
        lmasks = torch.stack([it[2] for it in self.items])
        rmasks = torch.stack([it[3] for it in self.items])
        use_amp = lefts.device.type == "cuda" and lefts.dtype == torch.float16
        with torch.inference_mode():
            with torch.autocast(device_type=lefts.device.type, dtype=torch.float16, enabled=use_amp):
                left_out = self._inpaint_eyes(lefts, lmasks, is_left=True)
                right_out = self._inpaint_eyes(rights, rmasks, is_left=False)
        n = left_out.shape[0]
        if flush:
            a, b = self.pre, n
        elif self.post > 0:
            a, b = self.pre, n - self.post
        else:
            a, b = self.pre, n
        left_out, right_out = left_out[a:b], right_out[a:b]
        if self._temporal_on and float(LIGHT_VIDEO_INPAINT_TEMPORAL_SMOOTH) > 0.0:
            try:
                with torch.inference_mode():
                    left_out = self._temporal_smooth(left_out, lmasks[a:b], "L")
                    right_out = self._temporal_smooth(right_out, rmasks[a:b], "R")
            except Exception as e:      # never let the optional smoothing break the conversion
                self._sm = {}
                self._temporal_on = False
                print(f"[Light Inpaint] temporal smoothing skipped: {e}", flush=True)
        if flush:
            self.items.clear()
        else:
            del self.items[:self.seq - (self.pre + self.post)]
        return left_out, right_out

    def _temporal_smooth(self, out, mask, key):
        """Blend the filled area of each output frame with the previous frame's (calms the frame-to-frame boil of the
        generated texture). Only pixels that are fill area in BOTH the current and the previous frame are touched, and the
        blend fades out where the fill really changed (motion / cut). `out`: (T,3,H,W), `mask`: raw hole mask (T,1,H,W)."""
        a = float(LIGHT_VIDEO_INPAINT_TEMPORAL_SMOOTH)
        if a <= 0.0:
            return out
        a = min(a, 0.85)
        r = max(1, int(LIGHT_VIDEO_INPAINT_SMOOTH_RADIUS))
        g0, g1 = LIGHT_VIDEO_INPAINT_SMOOTH_GATE
        g1 = max(1e-3, float(g1))
        st = self._sm.get(key)
        res = torch.empty_like(out)
        for i in range(out.shape[0]):
            cur = out[i:i + 1]
            m = mask[i:i + 1].float()
            reg = F.max_pool2d(m, 2 * r + 1, 1, r)
            w = F.avg_pool2d(reg, 5, 1, 2)                           # soft edge of the fill area
            if st is not None and st[0].shape == cur.shape:
                prev_s, prev_w = st
                d = (cur - prev_s).abs().amax(dim=1, keepdim=True).float()
                gate = 1.0 - ((d - float(g0)) / g1).clamp(0.0, 1.0)
                we = (torch.minimum(w, prev_w) * gate * a).to(cur.dtype)
                sm = cur + we * (prev_s - cur)
            else:
                sm = cur
            res[i:i + 1] = sm
            st = (sm, w)
        self._sm[key] = st
        return res

    @staticmethod
    def _split(left_out, right_out):
        return [(left_out[i:i + 1], right_out[i:i + 1]) for i in range(left_out.shape[0])]

    def push(self, left, right, left_mask, right_mask):
        """Add one frame (tensors shaped 1xCxHxW). Returns a list of finished (left, right) pairs."""
        repeat = self.pre + 1 if not self.items else 1
        entry = (left[0], right[0], left_mask[0], right_mask[0])
        for _ in range(repeat):
            self.items.append(entry)
        if len(self.items) == self.seq:
            return self._split(*self._run(flush=False))
        return []

    def flush(self):
        if not self.items:
            return []
        pad = 0
        last = self.items[-1]
        while len(self.items) < self.seq:
            self.items.append(last)
            pad += 1
        left_out, right_out = self._run(flush=True)
        if pad > 0:
            left_out, right_out = left_out[:-pad], right_out[:-pad]
        return self._split(left_out, right_out)


def make_stereo(bgr_t, depth_t, divergence, convergence, device, already_normalized=False, preserve_screen_border=False,
                fill_mode: str = WARP_FILL_MODE_STRETCH,
                inpaint_hole_threshold: float = 1.5):
    if already_normalized:
        depth_norm = depth_t.to(dtype=bgr_t.dtype)
    else:
        depth_norm = normalize_depth_gpu(depth_t)[0].to(dtype=bgr_t.dtype)

    with torch.inference_mode():
        if fill_mode == WARP_FILL_MODE_FAST:
            left_t = monobw_warp_one_fast(
                bgr_t, depth_norm, divergence, convergence, -1.0,
                preserve_screen_border=preserve_screen_border,
                hole_threshold_px=inpaint_hole_threshold,
            )
            right_t = monobw_warp_one_fast(
                bgr_t, depth_norm, divergence, convergence, +1.0,
                preserve_screen_border=preserve_screen_border,
                hole_threshold_px=inpaint_hole_threshold,
            )
        else:
            left_t  = monobw_warp_one(bgr_t, depth_norm, divergence, convergence, -1.0, preserve_screen_border)
            right_t = monobw_warp_one(bgr_t, depth_norm, divergence, convergence, +1.0, preserve_screen_border)

    return left_t, right_t

# ---------------------------------------------------------------------------
# Dubois anaglyph (least-squares optimized red-cyan anaglyph, version 1: clip before sum)
# reference: https://www.site.uottawa.ca/~edubois/anaglyph/LeastSquaresHowToPhotoshop.pdf
# Input/output: (B, 3, H, W) RGB tensors in [0, 1].
# ---------------------------------------------------------------------------
_DUBOIS_L_MAT = (
    (0.437, 0.449, 0.164),
    (-0.062, -0.062, -0.024),
    (-0.048, -0.050, -0.017),
)
_DUBOIS_R_MAT = (
    (-0.011, -0.032, -0.007),
    (0.377, 0.761, 0.009),
    (-0.026, -0.093, 1.234),
)

def _dubois_srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def _dubois_linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)

def dubois_anaglyph_rgb(l_rgb: torch.Tensor, r_rgb: torch.Tensor) -> torch.Tensor:
    in_dtype = l_rgb.dtype
    l_lin = _dubois_srgb_to_linear(l_rgb.float())
    r_lin = _dubois_srgb_to_linear(r_rgb.float())
    l_mat = torch.tensor(_DUBOIS_L_MAT, device=l_lin.device, dtype=torch.float32)
    r_mat = torch.tensor(_DUBOIS_R_MAT, device=r_lin.device, dtype=torch.float32)
    # clip_before=True (dubois v1): each eye's dot product is clamped to [0, 1] before summing
    l_part = torch.einsum("oc,bchw->bohw", l_mat, l_lin).clamp(0.0, 1.0)
    r_part = torch.einsum("oc,bchw->bohw", r_mat, r_lin).clamp(0.0, 1.0)
    out = (l_part + r_part).clamp(0.0, 1.0)
    out = _dubois_linear_to_srgb(out).clamp(0.0, 1.0)
    return out.to(in_dtype)

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
    if fmt == "dubois-anaglyph":
        # conversion3d frames are BGR -> convert to RGB for the Dubois matrices, then back to BGR
        out_rgb = dubois_anaglyph_rgb(left_t.flip(1), right_t.flip(1))
        return out_rgb.flip(1).contiguous()
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

# Filename suffixes read by VR players (same style as nunif/iw3: _LRF_Full_SBS).
#  DeoVR: SBS / LR / TB   HereSphere: _LR / _LRF / _TB / _TBF   Skybox: Full_SBS / Full_TB
_VR_PLAYER_FORMAT_NAMES = {
    "hsbs": "LR_Half_SBS",
    "fsbs": "LRF_Full_SBS",
    "tb": "TB_Half_TB",
    "ftb": "TBF_Full_TB",
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

SHUTDOWN_DELAY_SEC = 300    # "Schedule PC shutdown after conversion": 5 minutes


def _shutdown_exe() -> str:
    return os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "shutdown.exe")


def _run_shutdown(args: list) -> tuple[bool, str]:
    """Runs the Windows shutdown.exe directly (argument list, no shell, no window)."""
    try:
        r = subprocess.run([_shutdown_exe(), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           creationflags=0x08000000, timeout=15)   # CREATE_NO_WINDOW
        msg = (r.stderr or r.stdout or b"").decode("utf-8", errors="ignore").strip()
        return r.returncode == 0, msg
    except Exception as e:
        return False, str(e)


def build_audio_mux_cmd(
    video_only: Path, src_video_for_audio: Path, final_out: Path,
    audio_start_sec: float = None, audio_end_sec: float = None,
    sound_boost: bool = False,
) -> list[str]:
    cmd = [FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_only)]
    if audio_start_sec is not None:
        cmd.extend(["-ss", str(audio_start_sec)])
    if audio_end_sec is not None:
        cmd.extend(["-to", str(audio_end_sec)])
    cmd.extend(["-i", str(src_video_for_audio)])
    cmd.extend(["-map", "0:v:0", "-map", "1:a:0?"])
    if sound_boost:
        cmd.extend(["-af", "volume=4.0,alimiter=limit=0.95"])
    cmd.extend([
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
            # End-of-stream marker must not be dropped: retry until delivered (or Stop is requested).
            while True:
                try:
                    self.out_q.put(None, timeout=0.2)
                    break
                except queue.Full:
                    if self.stop_event.is_set():
                        break

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
        # Never block forever: if the writer thread died (ffmpeg crashed / disk full / broken pipe)
        # nobody drains the queue any more, so re-check the writer state on every short timeout.
        while True:
            if self.error is not None:
                raise self.error
            if self._stopped.is_set():
                raise RuntimeError("ffmpeg pipe writer stopped unexpectedly")
            try:
                self.queue.put(data, timeout=0.5)
                return
            except queue.Full:
                continue

    def signal_stop(self, timeout=None):
        """Queue the end-of-stream marker without risking a permanent block.
        Gives up as soon as the writer thread is gone, or after `timeout` seconds if one is given."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while not self._stopped.is_set():
            try:
                self.queue.put(None, timeout=0.5)
                return True
            except queue.Full:
                if deadline is not None and time.monotonic() >= deadline:
                    return False
        return False

    def close(self):
        self.signal_stop()
        self.join(timeout=10.0)
        if self.error is not None:
            raise self.error

class _NvencStallWatchdog(threading.Thread):

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
                    return                                                  
                if _COMPILE_BUSY.is_set():
                    elapsed = 0.0          # torch.compile first call in progress: not a stall
                    continue
                elapsed += self._poll_interval
                try:
                    if any(t.exists() and t.stat().st_size > 0 for t in self._targets):
                        return                                                      
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

            pass

class ConvertWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(str)
    finished_error = pyqtSignal(str)
    log = pyqtSignal(str)
    auto_edge = pyqtSignal(int)      # Auto Mode: the Edge Fix level the program actually applied to this conversion

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._stop = threading.Event()
        self._temp_files = []
        self._stall_message: str | None = None
        self._last_logged_progress_percent = -1
        self._save_partial_on_stop = True   # user pressed Stop -> keep what has been converted so far
        self.partial_saved = False          # True when the result is a partial video saved after a user stop

    def stop(self, save_partial: bool = True):
        """save_partial=True (Stop button): finish the output with the frames converted so far.
        save_partial=False (window closing): abort and delete the unfinished output."""
        self._save_partial_on_stop = bool(save_partial)
        self._stop.set()

    def _ffmpeg_fail_message(self, label, proc, stderr_thread, stderr_buf, cmd=None, exc=None):
        """ffmpeg 파이프가 깨졌을 때 실제 원인(종료 코드 + ffmpeg stderr)을 수집해 로그창에 출력하고 메시지를 반환."""
        try:
            proc.wait(timeout=3)
        except Exception:
            pass
        try:
            if stderr_thread is not None:
                stderr_thread.join(timeout=2.0)
        except Exception:
            pass
        rc = proc.poll()
        rc_txt = "still running" if rc is None else (f"{rc} (0x{rc & 0xFFFFFFFF:08X})")
        try:
            err_txt = b"".join(stderr_buf or []).decode("utf-8", errors="ignore").strip()
        except Exception:
            err_txt = ""
        if not err_txt:
            err_txt = "(ffmpeg produced no error output - it may have crashed or been killed)"
        self.log.emit(f"[ERROR] {label} pipe broken. ffmpeg exit code: {rc_txt}")
        if exc is not None:
            self.log.emit(f"[ERROR] Pipe write error: {type(exc).__name__}: {exc}")
        if cmd:
            try:
                self.log.emit("[ERROR] ffmpeg command: " + " ".join(str(c) for c in cmd))
            except Exception:
                pass
        self.log.emit("[ERROR] ffmpeg output:")
        for ln in err_txt[-2000:].splitlines():
            self.log.emit("    " + ln)
        return f"{label} pipe broken (ffmpeg exit code: {rc_txt}).\n{err_txt[-600:]}"

    def run(self):
        p = self.params
        try:
            self._run_convert(p)
        except Exception as e:
            self.finished_error.emit(self._stall_message if self._stall_message else str(e))
        finally:
            for tmp in self._temp_files:
                cleanup_temp_path(tmp)
            self._release_gpu_memory()

    @staticmethod
    def _release_gpu_memory():
        # Runs after _run_convert returns (success, error, or user abort), so the depth model
        # and per-conversion tensors are already unreferenced; hand the cached VRAM back.
        try:
            with _LIGHT_STREAM_MODEL_LOCK:
                _LIGHT_STREAM_MODELS.clear()
            _GRID_CACHE.clear()
            _GAUSSIAN_KERNEL_CACHE.clear()
            _DEPTH_AA_KERNEL_CACHE.clear()
            _INPAINT_GRAIN_CACHE.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _run_convert(self, p: dict):
        input_path_orig: Path = p["input_path"]
        output_dir: Path = p["output_dir"]
        divergence = p["divergence"]
        input_size = p["input_size"]
        fmt = p["format"]
        invert_depth = p["invert_depth"]
        fp16 = p["fp16"]
        compile_flag = bool(p.get("compile", False))
        trt_flag = bool(p.get("tensorrt", False))
        # Hybrid: TensorRT only for the depth map model; torch.compile for inpaint models.
        hybrid_flag = bool(p.get("hybrid", False)) and compile_flag and trt_flag
        if hybrid_flag:
            self.log.emit("[Hybrid] ON: TensorRT -> depth map model, torch.compile -> inpaint models "
                          "(VDA keeps both).")
        # cuDNN autotuner: ON by default, OFF while torch.compile is requested
        # (reduce-overhead CUDA-graph capture + cudnnFind stream sync can collide on a new shape).
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = not compile_flag
            self.log.emit(f"[cuDNN] benchmark = {'OFF' if compile_flag else 'ON'} (torch.compile={'ON' if compile_flag else 'OFF'})")
        low_ram = False
        depth_model = p.get("depth_model", "zipdepth") or "zipdepth"
        # Convergence slider 1.1 (shown as "Auto") = Auto Convergence: measured per frame from the depth map.
        _conv_manual = float(p["convergence"])
        auto_conv_on = _conv_manual >= AUTO_CONVERGENCE_THRESHOLD
        auto_conv = AutoConvergence() if auto_conv_on else None
        convergence = 1.0 - (0.5 if auto_conv_on else _conv_manual)   # Auto: neutral value until the first measurement
        edge_dilation = p["edge_dilation"]
        edge_dilation = (
            edge_fix_level_to_xy(edge_dilation, depth_model=depth_model)
            if not isinstance(edge_dilation, (list, tuple)) else edge_dilation
        )
        ema_decay = p["ema_decay"]
        preserve_border = p["preserve_border"]
        fill_mode                = p.get("fill_mode", WARP_FILL_MODE_STRETCH)
        set_inpaint_res(p.get("inpaint_res", INPAINT_RES_DEFAULT))
        inpaint_hole_threshold   = float(p.get("inpaint_hole_threshold", 1.5))
        depth_aa_enabled         = bool(p.get("depth_aa", False))
        depth_comp_k = 1.0
        if bool(p.get("depth_linear_comp", False)):
            depth_comp_k = depth_linear_comp_width(divergence, depth_model)
            if depth_comp_k < 1.0 - 1e-4:
                self.log.emit(f"[Depth Linear Compression] ON (strength {float(divergence):g}, depth range width {depth_comp_k:.3f})")
        _bs = depth_base_scale(depth_model)
        if _bs < 1.0 - 1e-4:
            depth_comp_k *= _bs
            self.log.emit(f"[Strength match] {depth_model}: depth range x{_bs:.2f} (equalized to VDA-S), total width {depth_comp_k:.3f}")

        if auto_conv_on:
            self.log.emit("[Auto Convergence] ON (convergence is measured from the depth map every frame)")

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
        # mp4, mkv 이외의 포맷은 모두 mp4로 통일
        if orig_ext not in {".mp4", ".mkv"}: orig_ext = ".mp4"
        # HDR / 프레임 정규화 옵션 사용 시에는 모든 포맷을 mp4로 저장
        if p.get("hdr_norm"): orig_ext = ".mp4"

        final_video = make_unique_output_path(output_dir, name, vr_player_format_label(fmt), orig_ext)

        self.log.emit(f"Input: {input_path_orig}")
        self.log.emit(f"Output: {final_video}")

        vf_filters = []

        hdr_norm_enabled = bool(p.get("hdr_norm"))
        force_cfr = False
        cfr_fps = None

        if hdr_norm_enabled:

            if check_is_hdr(input_path_orig):
                self.log.emit("HDR video detected: Adding tonemapping filter.")
                vf_filters.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
            else:
                self.log.emit("Not an HDR video: skipping tonemap filter (frame rate normalization will still be applied).")

            force_cfr = True
            cfr_fps = get_video_fps(input_path_orig)
            self.log.emit(f"Frame rate normalization enabled: forcing constant frame rate output at {cfr_fps:.3f} fps.")

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

        if vf_filters or use_start or use_end or force_cfr:
            self.log.emit("Generating temporary preprocessed / trimmed video. Please wait...")
            tmp_pre = Path(tempfile.gettempdir()) / f"stereo_preprocess_{uuid.uuid4().hex}.mp4"
            self._temp_files.append(tmp_pre)

            cmd = [FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "error", "-stats"]

            if use_start: cmd.extend(["-ss", str(start_sec)])
            if use_end: cmd.extend(["-to", str(end_sec)])

            cmd.extend(["-i", str(input_path_orig)])

            if vf_filters:
                cmd.extend(["-vf", ",".join(vf_filters)])

            if force_cfr and cfr_fps:

                cmd.extend(["-fps_mode", "cfr", "-r", f"{cfr_fps:.6f}"])

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
                # The depthmap video can be either
                #   (a) full length  -> same timeline as the source, so it must be seeked to the start time, or
                #   (b) already cut to the selected range (e.g. a depthmap that was extracted with the same
                #       start/end times) -> it starts at 0:00 and must NOT be seeked again.
                # Decide by comparing its duration with the selected range and with the full source length.
                depth_is_trimmed = False
                depth_fps = float(depth_cap.get(cv2.CAP_PROP_FPS) or 0.0) or fps
                if use_start:
                    _d_total_probe = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                    if _d_total_probe > 0 and depth_fps > 0:
                        _depth_dur = _d_total_probe / depth_fps
                        _range_dur = (total / fps) if (total > 0 and fps > 0) else None
                        if is_pre_trimmed:
                            _full_dur = get_video_duration_sec(input_path_orig)
                        else:
                            _full_dur = original_duration if original_duration > 0 else None
                        if _range_dur is not None:
                            if _full_dur is None:
                                depth_is_trimmed = _depth_dur <= _range_dur + 1.0
                            else:
                                depth_is_trimmed = abs(_depth_dur - _range_dur) < abs(_depth_dur - _full_dur)
                        self.log.emit(
                            f"[Depthmap sync] depthmap {_depth_dur:.2f}s, selected range "
                            f"{(_range_dur if _range_dur is not None else 0):.2f}s, source "
                            f"{(_full_dur if _full_dur is not None else 0):.2f}s -> "
                            + ("treated as already trimmed to the selected range (no seek)."
                               if depth_is_trimmed else "treated as full length (seeking to the start time).")
                        )
                depth_start_frame = 0
                if use_start and not depth_is_trimmed:
                    depth_cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
                    depth_start_frame = int(round(start_sec * depth_fps))

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
                        # When the source was pre-processed (letterbox crop / trim), working_input_path is the
                        # temporary file that already starts at the selected start time, so the audio must be
                        # taken from 0 s. Seeking to start_sec there runs past the end of the short temp file,
                        # yields an empty audio stream and `-shortest` then cuts the output to ~2 KB.
                        if is_pre_trimmed:
                            base_start = 0.0
                        else:
                            base_start = start_sec if use_start else 0.0
                        audio_start_sec = base_start
                        audio_end_sec = base_start + (clamped / fps if fps > 0 else 0.0)
                    elif main_frames_available > 0 and depth_frames_available > main_frames_available:
                        extra_sec = (depth_frames_available - main_frames_available) / fps if fps > 0 else 0.0
                        self.log.emit(
                            f"[Info] Depthmap video is longer than the source video by ~{extra_sec:.2f}s; "
                            f"the extra depthmap tail will simply be unused."
                        )

                elif depth_total_raw == 0:
                    self.log.emit(
                        "[Warning] Could not determine the depthmap video's frame count in advance; "
                        "falling back to a per-frame sync check during encoding."
                    )
            else:
                self.log.emit(f"[torch.compile] requested={compile_flag}")
                self.log.emit(f"[TensorRT] requested={trt_flag}")
                if depth_model in ("vda_s", "vda_s_metric"):
                    _is_metric = depth_model == "vda_s_metric"
                    self.log.emit(f"Loading Video Depth Anything (VDA-S{' Metric' if _is_metric else ''}, streaming)...")
                    predictor = load_vda_streaming(encoder="vits", input_size=input_size, fp16=fp16, metric=_is_metric,
                                                   compile_model_flag=compile_flag, tensorrt_flag=trt_flag)
                elif depth_model == "distill_any_s":
                    self.log.emit("Loading Distill-Any-Depth Small...")
                    predictor = load_distill_any_s(input_size=input_size, device=str(device), fp16=fp16,
                                                   compile_model_flag=compile_flag, tensorrt_flag=trt_flag)
                elif depth_model == "dav2_s":
                    self.log.emit("Loading Depth-Anything V2 Small (DA-V2-S)...")
                    predictor = load_dav2_s(input_size=input_size, device=str(device), fp16=fp16,
                                            compile_model_flag=compile_flag, tensorrt_flag=trt_flag)
                elif depth_model == "da3_mono_l":
                    self.log.emit("Loading Depth Anything 3 Mono-Large (DA3-Mono-L)...")
                    predictor = load_da3_mono_l(input_size=input_size, device=str(device), fp16=fp16,
                                                compile_model_flag=compile_flag, tensorrt_flag=trt_flag)
                else:
                    self.log.emit("Loading ZipDepth...")
                    predictor = load_zipdepth(input_size=input_size, device=str(device), fp16=fp16,
                                              compile_model_flag=compile_flag, tensorrt_flag=trt_flag)

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
            _ema_decay_clamped = float(max(0.0, min(EMA_MAX, float(ema_decay))))

            _ef_applied = None
            if bool(p.get("auto_mode", False)):
                try:
                    _ef_applied = int(p.get("edge_dilation"))      # built-in table value (used when nothing is measured)
                except Exception:
                    _ef_applied = None

            if use_external_depthmap:
                depth_ext0 = depthmap_frame_to_tensor(depth_frame0, frame0.shape[1], frame0.shape[0], device, dtype_tensor)
                depth_ext0 = 1.0 - depth_ext0
                with torch.inference_mode():
                    depth_norm0, _, _ = normalize_depth_gpu(depth_ext0.float(), decay=0.0)
                try:
                    depth_cap.set(cv2.CAP_PROP_POS_FRAMES, int(depth_start_frame))
                except Exception:
                    try:
                        depth_cap.release()
                        depth_cap = cv2.VideoCapture(str(depth_read_path))
                        if use_start and not depth_is_trimmed:
                            depth_cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
                    except Exception:
                        pass
            else:
                depth_raw0 = estimate_depth_raw(predictor, frame0)
                if (
                    isinstance(predictor, VDAStreamingAdapter)
                    and predictor.streaming_method is not None
                ):

                    _native_h, _native_w = vda_native_hw(
                        frame0.shape[0], frame0.shape[1],
                        predictor.input_size - (VDA_METRIC_PADDING * 2 if getattr(predictor, "metric", False) else 0)
                    )
                    depth_native_wh = (_native_w, _native_h)
                else:
                    depth_native_wh = (int(depth_raw0.shape[-1]), int(depth_raw0.shape[-2]))
                depth_raw0 = depth_raw0.max() - depth_raw0
                depth0 = upsample_depth(depth_raw0.float(), frame0.shape[0], frame0.shape[1])
                with torch.inference_mode():
                    depth_norm0, _, _ = normalize_depth_gpu(depth0, decay=0.0)
                del depth_raw0, depth0

                if EDGEFIX_AUTO_MEASURE and bool(p.get("auto_mode", False)):
                    # Auto Mode: measure the best edge fix with this depth model / input size / frame size
                    # right before the conversion starts (compression ON and OFF come from the same measurement).
                    edgefix_reset_profile()
                    try:
                        _ef_t0 = time.perf_counter()
                        _ef_hw = (int(frame0.shape[0]), int(frame0.shape[1]))
                        edgefix_measure_profile(predictor, depth_model, _ef_hw, _ef_hw, input_size, log=self.log.emit)
                        _ef_edge, _ef_ema = compute_auto_edge_ema(
                            divergence, depth_model=depth_model, warp_fill_mode=fill_mode,
                            depth_linear_comp=bool(p.get("depth_linear_comp", False)))
                        edge_dilation = edge_fix_level_to_xy(_ef_edge, depth_model=depth_model)
                        _ef_applied = int(_ef_edge)
                        self.log.emit(f"[Auto Edge Fix] strength {float(divergence):g} -> Edge Fix {int(_ef_edge)} "
                                      f"({time.perf_counter() - _ef_t0:.1f}s)")
                    except Exception as _ef_err:
                        self.log.emit(f"[Auto Edge Fix] measurement failed ({type(_ef_err).__name__}: {_ef_err}); "
                                      f"using the built-in table value {p.get('edge_dilation')}")

                _reset_fn = getattr(predictor, "reset_state", None)
                if callable(_reset_fn):
                    try:
                        _reset_fn()
                    except Exception as _reset_err:
                        self.log.emit(f"[Warning] predictor.reset_state() after warm-up frame failed: {_reset_err}")

            if _ef_applied is not None:
                self.auto_edge.emit(int(_ef_applied))      # UI: show the applied Auto Edge Fix number instead of AUTO
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

            if depth_aa_enabled:
                with torch.inference_mode():
                    depth_for_warp = apply_depth_aa_gpu(depth_for_warp, _DEPTH_AA_SIGMA)

            use_medium_inpaint = fill_mode in (WARP_FILL_MODE_MEDIUM, WARP_FILL_MODE_LIGHT)
            medium_stream = None
            if use_medium_inpaint:
                _is_light = (fill_mode == WARP_FILL_MODE_LIGHT)
                _mi_label = "Light Inpaint" if _is_light else "Medium Inpaint"
                _mi_path = _light_inpaint_model_path() if _is_light else _medium_inpaint_model_path()
                ensure_inpaint_model(_mi_path, LIGHT_INPAINT_URL if _is_light else MEDIUM_INPAINT_URL,
                                     log=self.log.emit)
                self.log.emit(f"[{_mi_label}] Loading model: {_mi_path}")
                _mi_model = load_medium_inpaint_model(device, path=_mi_path, disable_stretch_mask=_is_light)
                configure_video_inpaint_accel(
                    _mi_model, device, bool(compile_flag), bool(trt_flag and not hybrid_flag), bool(fp16), _mi_label,
                    "LightInpaint" if _is_light else "MediumInpaint", _mi_path, log=self.log.emit)
                medium_stream = MediumInpaintStream(_mi_model, temporal_smooth=_is_light)
                if _is_light:
                    _mi_set_stretch_params(LIGHT_STRETCH_MAX_GRAD, LIGHT_STRETCH_MIN_RUN, LIGHT_STRETCH_GROW)
                else:
                    _mi_set_stretch_params(MEDIUM_STRETCH_MAX_GRAD, MEDIUM_STRETCH_MIN_RUN, MEDIUM_STRETCH_GROW)
                _mi_stretch_txt = (
                    f", stretch mask grad{_MI_STRETCH_ACTIVE['grad']}/run{_MI_STRETCH_ACTIVE['run']}/grow{_MI_STRETCH_ACTIVE['grow']}"
                    if _MI_STRETCH_MASK_ON else "")
                self.log.emit(
                    f"[{_mi_label}] Ready (medium forward warp, overlap PRE{MEDIUM_INPAINT_PRE_PADDING}/POST{MEDIUM_INPAINT_POST_PADDING}, "
                    f"mask dilation inner{MEDIUM_INPAINT_INNER_DILATION}/outer{MEDIUM_INPAINT_OUTER_DILATION}{_mi_stretch_txt})"
                )

            use_light_stream = (fill_mode == WARP_FILL_MODE_LIGHT_STREAM)
            light_stream_cache = None
            if use_light_stream:
                _ls_path = _light_inpaint_stream_model_path()
                ensure_inpaint_model(_ls_path, LIGHT_INPAINT_STREAM_URL, log=self.log.emit)
                self.log.emit(f"[Light Inpaint (stream)] Loading model: {_ls_path}")
                _ls_model = get_light_inpaint_stream_model(device)
                configure_light_stream_accel(_ls_model, device, bool(compile_flag), bool(trt_flag and not hybrid_flag),
                                             log=self.log.emit)
                light_stream_cache = LightInpaintStreamCache()   # fresh cache per conversion
                self.log.emit(
                    f"[Light Inpaint (stream)] Ready (iw3 forward warp, mask dilation "
                    f"inner{LIGHT_INPAINT_STREAM_INNER_DILATION}/outer{LIGHT_INPAINT_STREAM_OUTER_DILATION}, "
                    f"cache: re-run <= every {LIGHT_INPAINT_STREAM_CACHE_MAX_INTERVAL} frames or hole change > "
                    f"{int(LIGHT_INPAINT_STREAM_CACHE_CHANGE_THRESH * 100)}%, hold {LIGHT_INPAINT_STREAM_HOLD_FRAMES}, "
                    f"fade {LIGHT_INPAINT_STREAM_FADE_FRAMES})"
                )

            l0, r0 = make_stereo(
                bgr_t0, depth_for_warp, divergence, convergence, device,
                already_normalized=True, preserve_screen_border=preserve_border,
                fill_mode=(WARP_FILL_MODE_STRETCH if (use_medium_inpaint or use_light_stream) else fill_mode),
                inpaint_hole_threshold=inpaint_hole_threshold,
            )
            packed0 = pack_frame_gpu(l0, r0, fmt)
            oh, ow = packed0.shape[2], packed0.shape[3]
            ow = ow - (ow % 2)
            oh = oh - (oh % 2)

            if low_ram:
                del l0, r0, packed0, depth_norm0, depth_for_warp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if _HAS_EMA_SCALER:
                depth_scaler = EMAMinMaxScaler(decay=_ema_decay_clamped, buffer_size=1, mode="minmax")
                self.log.emit(f"[EMA] depth_scaler active (EMAMinMaxScaler), decay={_ema_decay_clamped}"
                              + (f", abs-gate ON (max {DEPTH_EMA_ABS_MAX}, gate {DEPTH_EMA_GATE_PX[0]}-{DEPTH_EMA_GATE_PX[1]} px)"
                                 if DEPTH_EMA_ABS_GATE else ", abs-gate OFF (motion-map blend)"))
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
            try:
                self.log.emit("[FFmpeg] " + " ".join(str(c) for c in ffmpeg_cmd))
            except Exception:
                pass

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
            partial_stop = False

            try:
                while True:
                    if self._stop.is_set():
                        if self._stall_message or not self._save_partial_on_stop or idx <= 0:
                            self.log.emit("Aborted by user")
                            aborted = True
                        else:
                            # Stop button: do NOT discard the work. Leave the loop and let the normal
                            # finalize path (flush -> close encoder -> mux audio) save the part that has
                            # been converted so far. Audio is cut to the video length by "-shortest".
                            self.log.emit(
                                f"Stopped by user at frame {idx}. Saving the part converted so far..."
                            )
                            partial_stop = True
                        break

                    # Never block forever: poll so Stop and a finished reader are always noticed.
                    frame = None
                    _end_of_stream = False
                    while True:
                        try:
                            frame = frame_q.get(timeout=0.5)
                            _end_of_stream = frame is None
                            break
                        except queue.Empty:
                            if self._stop.is_set():
                                break
                            if not reader.is_alive():
                                try:
                                    frame = frame_q.get_nowait()
                                    _end_of_stream = frame is None
                                except queue.Empty:
                                    _end_of_stream = True
                                break
                    if frame is None:
                        if _end_of_stream: break
                        continue  # Stop requested while waiting: let the check at the top handle it

                    dtype_tensor = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
                    frame_cpu_t = torch.from_numpy(frame)
                    if device.type == "cuda":
                        frame_cpu_t = frame_cpu_t.pin_memory()
                    bgr_t = frame_cpu_t.to(device=device, dtype=dtype_tensor, non_blocking=(device.type == "cuda")).permute(2, 0, 1).unsqueeze(0).div_(255.0)

                    decay = float(max(0.0, min(EMA_MAX, float(ema_decay))))
                    if depth_scaler is not None and abs(depth_scaler.decay - decay) > 1e-9:
                        depth_scaler.set_decay(decay)
                        _ema_decay_clamped = decay
                    if depth_scaler is not None:
                        # disparity pixels per 1.0 of normalized depth (same formula as the forward warp's shift_size)
                        depth_scaler.set_px_per_unit(
                            float(divergence) * 0.01 * max(frame.shape[0], frame.shape[1]) * 0.5)

                    raw_depth_snapshot = None
                    depth_norm = None
                    depth_norm_raw = None
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

                        self.log.emit(
                            f"[Warning] Depthmap video ran out of frames at frame {idx}. "
                            f"Stopping here and finalizing the output so video/audio stay in sync."
                        )
                        break

                    if auto_conv_on:
                        _ac_src = depth_norm_raw if depth_norm_raw is not None else depth_norm
                        _conv_slider = auto_conv.update(_ac_src, frame, float(divergence) * 0.005)
                        convergence = 1.0 - _conv_slider
                        if depth_comp_k < 1.0 - 1e-4:
                            # the subject level was measured on the uncompressed depth, the warp uses the linearly compressed depth
                            convergence = convergence * depth_comp_k + 0.5 * (1.0 - depth_comp_k)

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

                    if depth_aa_enabled:
                        with torch.inference_mode():
                            depth_for_warp = apply_depth_aa_gpu(depth_for_warp, _DEPTH_AA_SIGMA)

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
                        except Exception as _pe:
                            raise RuntimeError(self._ffmpeg_fail_message(
                                "ffmpeg depthmap encoding", depthmap_proc, depthmap_stderr_thread,
                                depthmap_stderr_buf, depthmap_ffmpeg_cmd if 'depthmap_ffmpeg_cmd' in locals() else None, _pe))

                    # Depth Linear Compression is applied BEFORE the corrected depthmap is saved, so the saved
                    # corrected depthmap is the same depth that is actually used for warping (only models that
                    # have compression params, i.e. VDA-S / Distill-Any-S, are affected). The plain / raw
                    # depthmap output above is saved before this step and stays uncompressed.
                    if depth_comp_k < 1.0 - 1e-4:
                        with torch.inference_mode():
                            depth_for_warp = apply_depth_linear_comp_gpu(depth_for_warp, depth_comp_k)

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
                        except Exception as _pe:
                            raise RuntimeError(self._ffmpeg_fail_message(
                                "ffmpeg corrected depthmap encoding", corrected_proc, corrected_stderr_thread,
                                corrected_stderr_buf, corrected_ffmpeg_cmd if 'corrected_ffmpeg_cmd' in locals() else None, _pe))

                    frame_bytes_list = []
                    if use_medium_inpaint:
                        left_t, right_t, left_m, right_m = make_stereo_with_masks(
                            bgr_t, depth_for_warp, divergence, convergence, device,
                            already_normalized=True, preserve_screen_border=preserve_border,
                        )
                        # The video model works on a 12-frame window, so frames come out in bursts (and delayed).
                        for _el, _er in medium_stream.push(left_t, right_t, left_m, right_m):
                            frame_bytes_list.append(tensor_to_bgr_bytes(pack_frame_gpu(_el, _er, fmt)))
                        del left_m, right_m
                    elif use_light_stream:
                        # image model, strictly one frame in / one frame out (cache keeps the hole texture stable)
                        left_t, right_t = make_stereo_light_stream(
                            bgr_t, depth_for_warp, divergence, convergence, device,
                            already_normalized=True, preserve_screen_border=preserve_border,
                            cache=light_stream_cache,
                        )
                        packed_t = pack_frame_gpu(left_t, right_t, fmt)
                        frame_bytes_list.append(tensor_to_bgr_bytes(packed_t))
                        del packed_t
                    else:
                        left_t, right_t = make_stereo(
                            bgr_t, depth_for_warp, divergence, convergence, device,
                            already_normalized=True, preserve_screen_border=preserve_border,
                            fill_mode=fill_mode,
                            inpaint_hole_threshold=inpaint_hole_threshold,
                        )
                        packed_t = pack_frame_gpu(left_t, right_t, fmt)
                        frame_bytes_list.append(tensor_to_bgr_bytes(packed_t))
                        del packed_t

                    del left_t, right_t, depth_norm, depth_for_warp, bgr_t, frame
                    if raw_depth_snapshot is not None:
                        del raw_depth_snapshot

                    for frame_bytes in frame_bytes_list:
                        try:
                            pipe_writer.submit(frame_bytes)
                        except Exception as _pe:
                            raise RuntimeError(self._ffmpeg_fail_message(
                                "ffmpeg encoding", proc, stderr_thread, stderr_buf, ffmpeg_cmd, _pe))
                    del frame_bytes_list

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

                        if total > 0:
                            percent = min(100, int(idx / total * 100))
                            if percent != self._last_logged_progress_percent or idx == total:
                                self.log.emit(f"Progress: {msg}")
                                self._last_logged_progress_percent = percent
                        else:
                            if idx % 200 == 0:
                                self.log.emit(f"Progress: {msg}")

                if use_medium_inpaint and not aborted and (partial_stop or not self._stop.is_set()):
                    # Emit the frames still waiting in the 12-frame window (keeps frame count identical).
                    for _el, _er in medium_stream.flush():
                        try:
                            pipe_writer.submit(tensor_to_bgr_bytes(pack_frame_gpu(_el, _er, fmt)))
                        except Exception as _pe:
                            raise RuntimeError(self._ffmpeg_fail_message(
                                "ffmpeg encoding", proc, stderr_thread, stderr_buf, ffmpeg_cmd, _pe))
            finally:
                self._stop.set()
                if nvenc_watchdog is not None:
                    nvenc_watchdog.mark_done()
                reader.join(timeout=2.0)
                try:
                    if not aborted:
                        pipe_writer.close()
                    else:
                        pipe_writer.signal_stop(3.0)
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
                                depthmap_pipe_writer.signal_stop(3.0)
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
                                corrected_pipe_writer.signal_stop(3.0)
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
                sound_boost=bool(p.get("sound_boost", False)),
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
                    sound_boost=bool(p.get("sound_boost", False)),
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
                    sound_boost=bool(p.get("sound_boost", False)),
                )
                corr_mux_ret = subprocess.run(corr_mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, startupinfo=si)
                if corr_mux_ret.returncode != 0:
                    self.log.emit(f"[Warning] Corrected depthmap audio muxing failed: {corr_mux_ret.stderr.decode('utf-8', errors='ignore')[-1000:]}")
                else:
                    self.log.emit(f"Corrected depthmap video completed: {corrected_final_video}")

            if partial_stop:
                self.partial_saved = True
                self.log.emit(f"Partial video saved ({idx} frames): {final_video}")

            self.finished_ok.emit(str(final_video))

        finally:
            cap.release()

# ---------------------------------------------------------------------------
# Image conversion (single still image -> stereo image)
# Reuses the same depth / edge fix / warp / inpaint / pack functions as the video path,
# without ffmpeg, frame queues or the audio mux.
# ---------------------------------------------------------------------------

def _jpeg_exif_orientation(data: bytes) -> int:
    """EXIF orientation (1-8) of a JPEG, parsed by hand (used when PIL is not installed)."""
    try:
        if data[:2] != b"\xff\xd8":
            return 1
        i, n = 2, len(data)
        while i + 4 <= n:
            if data[i] != 0xFF:
                break
            marker = data[i + 1]
            if marker == 0xFF:
                i += 1
                continue
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg_len = int.from_bytes(data[i + 2:i + 4], "big")
            if marker == 0xE1 and data[i + 4:i + 10] == b"Exif\x00\x00":
                t = i + 10
                if data[t:t + 2] == b"II":
                    order = "little"
                elif data[t:t + 2] == b"MM":
                    order = "big"
                else:
                    return 1
                ifd = t + int.from_bytes(data[t + 4:t + 8], order)
                cnt = int.from_bytes(data[ifd:ifd + 2], order)
                for k in range(cnt):
                    e = ifd + 2 + k * 12
                    if int.from_bytes(data[e:e + 2], order) == 0x0112:
                        v = int.from_bytes(data[e + 8:e + 10], order)
                        return v if 1 <= v <= 8 else 1
                return 1
            if marker == 0xDA:
                break
            i += 2 + seg_len
    except Exception:
        pass
    return 1

def _image_exif_orientation(data: bytes) -> int:
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            o = int(im.getexif().get(0x0112, 1))
            return o if 1 <= o <= 8 else 1
    except Exception:
        return _jpeg_exif_orientation(data)

def _apply_exif_orientation(arr: np.ndarray, o: int) -> np.ndarray:
    if o == 2:
        arr = arr[:, ::-1]
    elif o == 3:
        arr = arr[::-1, ::-1]
    elif o == 4:
        arr = arr[::-1, :]
    elif o == 5:
        arr = np.swapaxes(arr, 0, 1)
    elif o == 6:
        arr = np.rot90(arr, k=-1)
    elif o == 7:
        arr = np.swapaxes(arr[::-1, ::-1], 0, 1)
    elif o == 8:
        arr = np.rot90(arr, k=1)
    return np.ascontiguousarray(arr)

def read_image_bgr(path: Path) -> np.ndarray:
    """Read any still image as 8-bit BGR (HxWx3). Works with non-ASCII paths (no cv2.imread on the path).
    Handles 16-bit / float images, grayscale, alpha (composited on white) and EXIF rotation."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise RuntimeError(f"Image file is empty or unreadable: {path}")
    arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    if arr.dtype == np.uint16:
        arr = ((arr.astype(np.uint32) + 128) // 257).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 1:
        arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        a = arr[:, :, 3:4].astype(np.float32) / 255.0
        arr = (arr[:, :, :3].astype(np.float32) * a + 255.0 * (1.0 - a)).round().astype(np.uint8)
    orient = _image_exif_orientation(data.tobytes())
    if orient != 1:
        arr = _apply_exif_orientation(arr, orient)
    return np.ascontiguousarray(arr)

def write_image_bgr(path: Path, bgr: np.ndarray, jpeg_quality: int = 95):
    """Write a BGR uint8 image; works with non-ASCII paths."""
    ext = path.suffix.lower()
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    elif ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        raise RuntimeError(f"Failed to encode image as {ext}")
    buf.tofile(str(path))

def resolve_image_format(image_format: str, input_path: Path) -> str:
    """UI choice -> "png" or "jpg". Auto keeps the input's family: JPG/JPEG -> JPG, everything else -> PNG."""
    f = str(image_format).lower()
    if f in ("jpg", "jpeg"):
        return "jpg"
    if f == "png":
        return "png"
    return "jpg" if input_path.suffix.lower() in (".jpg", ".jpeg") else "png"

def pick_image_output_ext(fmt: str, out_w: int, out_h: int) -> str:
    """"jpg" -> .jpg (JPG cannot exceed 65535 px per side, so a bigger result falls back to PNG), otherwise .png."""
    if fmt == "jpg" and max(out_w, out_h) <= 65500:
        return ".jpg"
    return ".png"


def detect_letterbox_image(bgr: np.ndarray, threshold: int = 16, min_black_ratio: float = 0.92):
    """Black bars of a still image -> (top, bottom, left, right) crop box, or None when there is nothing to crop.
    Same rule as detect_letterbox() for videos (a row counts as black when >= 92% of its pixels are below
    gray 16, bars are searched in the outer half only, a result under 55% of the size is rejected as a
    false alarm), but evaluated on this one image and for all four sides (letterbox + pillarbox)."""
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        h, w = gray.shape[:2]
        if h < 8 or w < 8:
            return None
        black = gray < threshold
        row_ratio = black.mean(axis=1)
        col_ratio = black.mean(axis=0)

        def _edges(ratio, n):
            lo = 0
            for i in range(n // 2):
                if ratio[i] < min_black_ratio:
                    lo = i
                    break
            hi = n
            for i in range(n - 1, n // 2, -1):
                if ratio[i] < min_black_ratio:
                    hi = i + 1
                    break
            return lo, hi

        top, bottom = _edges(row_ratio, h)
        left, right = _edges(col_ratio, w)
        if (bottom - top) < h * 0.55:
            top, bottom = 0, h
        if (right - left) < w * 0.55:
            left, right = 0, w
        if top < 2 and (h - bottom) < 2 and left < 2 and (w - right) < 2:
            return None
        return int(top), int(bottom), int(left), int(right)
    except Exception:
        return None

def pad_image_to_ratio(bgr: np.ndarray, stereo_format: str):
    """Black padding up to 16:9 (32:9 for full SBS, 8:9 for full TB) - the same target ratios as the
    video encoder's pad filter - but applied to each eye separately: for SBS / TB the packed image is
    split into its two halves, every half is padded and centered on its own, and the halves are joined
    again, so the picture sits exactly in the middle of the left and of the right (or top and bottom)
    area. Anaglyph (a single image) is centered as a whole. Returns the input itself if nothing changes."""
    f = str(stereo_format).lower()
    dar = 32 / 9 if f == "fsbs" else (8 / 9 if f == "ftb" else 16 / 9)

    def _center(img, tw, th):
        ih, iw = int(img.shape[0]), int(img.shape[1])
        if tw == iw and th == ih:
            return img
        canvas = np.zeros((th, tw) + img.shape[2:], dtype=img.dtype)
        x0, y0 = (tw - iw) // 2, (th - ih) // 2
        canvas[y0:y0 + ih, x0:x0 + iw] = img
        return canvas

    def _target(ih, iw, ratio):
        return (int(round(max(iw, ih * ratio))), int(round(max(ih, iw / ratio))))

    h, w = int(bgr.shape[0]), int(bgr.shape[1])
    if f in ("hsbs", "fsbs") and w >= 2:
        half_ratio = dar / 2.0                    # each half of a side-by-side frame is half as wide
        mid = w // 2
        left, right = bgr[:, :mid], bgr[:, mid:]
        tw, th = _target(h, max(left.shape[1], right.shape[1]), half_ratio)
        out = np.concatenate([_center(left, tw, th), _center(right, tw, th)], axis=1)
    elif f in ("tb", "ftb") and h >= 2:
        half_ratio = dar * 2.0                    # each half of a top-bottom frame is half as tall
        mid = h // 2
        top, bottom = bgr[:mid], bgr[mid:]
        tw, th = _target(max(top.shape[0], bottom.shape[0]), w, half_ratio)
        out = np.concatenate([_center(top, tw, th), _center(bottom, tw, th)], axis=0)
    else:
        tw, th = _target(h, w, dar)
        out = _center(bgr, tw, th)
    return bgr if (out.shape[0] == h and out.shape[1] == w) else np.ascontiguousarray(out)


DEPTHMAP_IMAGE_BITS = 16    # bit depth of saved depthmap PNGs: 16 (default) or 8. The loader reads both.

def read_depthmap_image(path: Path) -> np.ndarray:
    """Read a depthmap image as float32 HxW in 0..1 (8-bit and 16-bit both supported).
    Like the video path, the first channel is used (gray depthmaps have identical channels).
    Convention: bright = near (same as the depthmaps this program saves)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise RuntimeError(f"Depthmap image is empty or unreadable: {path}")
    arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Failed to decode depthmap image: {path}")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.dtype == np.uint16:
        g = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        g = arr.astype(np.float32) / 255.0
    else:
        g = arr.astype(np.float32)
        gmax = float(g.max()) if g.size else 1.0
        g = np.clip(g / gmax if gmax > 1.0 else g, 0.0, 1.0)
    orient = _image_exif_orientation(data.tobytes())
    if orient != 1:
        g = _apply_exif_orientation(g, orient)
    return np.ascontiguousarray(g)

def write_depthmap_image(path: Path, depth_t: torch.Tensor, size_wh=None, bits: int = DEPTHMAP_IMAGE_BITS):
    """Save a normalized depth tensor (1,1,H,W; 0..1) as a grayscale PNG. Saved as (1 - depth): bright = near,
    the same convention as the depthmap videos. size_wh=(w,h) resizes first (bilinear)."""
    with torch.inference_mode():
        d = depth_t.float()
        if size_wh is not None and tuple(d.shape[-2:]) != (int(size_wh[1]), int(size_wh[0])):
            d = F.interpolate(d, size=(int(size_wh[1]), int(size_wh[0])), mode="bilinear", align_corners=False)
        v = (1.0 - d).clamp(0.0, 1.0)
        if int(bits) == 8:
            arr = v.mul(255.0).round().to(torch.int32)[0, 0].cpu().numpy().astype(np.uint8)
        else:
            arr = v.mul(65535.0).round().to(torch.int32)[0, 0].cpu().numpy().astype(np.uint16)
    write_image_bgr(path, arr)      # imencode handles 2-D gray arrays and 16-bit PNG

def unique_numbered_path(output_dir: Path, stem: str, ext: str) -> Path:
    """name.ext, then name_(1).ext, name_(2).ext ... (same numbering as the depthmap videos)."""
    cand = output_dir / f"{stem}{ext}"
    i = 1
    while cand.exists():
        cand = output_dir / f"{stem}_({i}){ext}"
        i += 1
    return cand


class ImageConvertWorker(QThread):
    """Converts still images to stereo images: params["input_paths"] is a list (or just "input_path"); the models are loaded once for the whole list."""
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(str)
    finished_error = pyqtSignal(str)
    log = pyqtSignal(str)
    auto_edge = pyqtSignal(int)
    file_started = pyqtSignal(int, int, str)    # (n, total, file name): the batch starts file n

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._stop = threading.Event()
        self.error_count = 0            # images skipped because of an error (read by the GUI when the batch ends)
        self.partial_saved = False      # images have no partial result (kept for GUI compatibility)

    def stop(self, save_partial: bool = True):
        self._stop.set()

    def run(self):
        try:
            self._run_image(self.params)
        except Exception as e:
            import traceback
            self.log.emit("[ERROR] " + traceback.format_exc())
            self.finished_error.emit(str(e))
        finally:
            ConvertWorker._release_gpu_memory()

    def _aborted(self) -> bool:
        if self._stop.is_set():
            self.log.emit("Aborted by user")
            self.finished_error.emit("Conversion was aborted.")
            return True
        return False

    def _run_image(self, p: dict):
        input_paths = [Path(x) for x in (p.get("input_paths") or [p["input_path"]])]
        n_files = len(input_paths)
        output_dir: Path = p["output_dir"]
        image_format = str(p.get("image_format", "auto") or "auto").lower()     # auto -> PNG
        try:
            image_quality = max(1, min(100, int(p.get("image_quality", 95))))
        except Exception:
            image_quality = 95
        divergence = p["divergence"]
        input_size = p["input_size"]
        fmt = p["format"]
        fp16 = p["fp16"]
        compile_flag = bool(p.get("compile", False))
        trt_flag = bool(p.get("tensorrt", False))
        hybrid_flag = bool(p.get("hybrid", False)) and compile_flag and trt_flag
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = not compile_flag
        depth_model = p.get("depth_model", "zipdepth") or "zipdepth"

        # Convergence slider 1.1 ("Auto") = measured from the depth map. A single image has no history,
        # so AutoConvergence.update() returns the measured value right away (no smoothing).
        _conv_manual = float(p["convergence"])
        auto_conv_on = _conv_manual >= AUTO_CONVERGENCE_THRESHOLD
        convergence = 1.0 - (0.5 if auto_conv_on else _conv_manual)
        edge_dilation = p["edge_dilation"]
        edge_dilation = (
            edge_fix_level_to_xy(edge_dilation, depth_model=depth_model)
            if not isinstance(edge_dilation, (list, tuple)) else edge_dilation
        )
        preserve_border = p["preserve_border"]
        fill_mode = p.get("fill_mode", WARP_FILL_MODE_STRETCH)
        set_inpaint_res(p.get("inpaint_res", INPAINT_RES_DEFAULT))
        inpaint_hole_threshold = float(p.get("inpaint_hole_threshold", 1.5))
        depth_aa_enabled = bool(p.get("depth_aa", False))
        depth_comp_k = 1.0
        if bool(p.get("depth_linear_comp", False)):
            depth_comp_k = depth_linear_comp_width(divergence, depth_model)
            if depth_comp_k < 1.0 - 1e-4:
                self.log.emit(f"[Depth Linear Compression] ON (strength {float(divergence):g}, depth range width {depth_comp_k:.3f})")
        _bs = depth_base_scale(depth_model)
        if _bs < 1.0 - 1e-4:
            depth_comp_k *= _bs
            self.log.emit(f"[Strength match] {depth_model}: depth range x{_bs:.2f} (equalized to VDA-S), total width {depth_comp_k:.3f}")
        if auto_conv_on:
            self.log.emit("[Auto Convergence] ON (measured from this image's depth map)")

        # Depthmap options (same meaning as the video path): extract_depthmap = raw or corrected checked
        extract_depthmap = bool(p.get("extract_depthmap", False))
        raw_depthmap = bool(p.get("raw_depthmap", False))
        corrected_depthmap = bool(p.get("corrected_depthmap", False))
        depthmap_input_str = str(p.get("depthmap_input_path", "") or "").strip()
        depthmap_input_path = Path(depthmap_input_str) if depthmap_input_str else None
        use_external_depthmap = False
        if depthmap_input_path is not None:
            if not depthmap_input_path.is_file():
                self.log.emit(f"[Warning] Depthmap input file not found, ignoring: {depthmap_input_path}")
            elif depthmap_input_path.suffix.lower() not in IMAGE_EXTS:
                self.log.emit(f"[Warning] Depthmap input '{depthmap_input_path.name}' is not an image "
                              f"(a depthmap video can only be used with a video input); ignoring it.")
            else:
                use_external_depthmap = True
        if use_external_depthmap and n_files > 1:
            self.log.emit("[Warning] An external depthmap image can only be used with a single image; ignoring it for this batch.")
            use_external_depthmap = False
        if use_external_depthmap and extract_depthmap:
            self.log.emit("[Info] A depthmap input file was provided, so depthmap extraction will be skipped (already have one).")
        do_extract_depthmap = extract_depthmap and not use_external_depthmap
        do_extract_corrected = do_extract_depthmap and corrected_depthmap

        # Video-only options stay in the UI; they are simply not applied to images (listed here only if turned on).
        _skipped = []
        if p.get("use_start_time") or p.get("use_end_time"):
            _skipped.append("start/end time")
        if p.get("resize"):
            _skipped.append("resize")
        if p.get("hdr_norm"):
            _skipped.append("HDR normalization")
        if p.get("sound_boost"):
            _skipped.append("sound boost")
        if _skipped:
            self.log.emit("[Info] Image input: skipped video-only options: " + ", ".join(_skipped))

        output_dir.mkdir(parents=True, exist_ok=True)
        if n_files > 1:
            self.log.emit(f"Image batch: {n_files} file(s) (models are loaded once)")
        STEPS = 5
        def _prog(i, step, msg):
            self.progress.emit(i * STEPS + step, n_files * STEPS, msg)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype_tensor = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
        self.file_started.emit(1, n_files, input_paths[0].name)
        idx = 0
        _prog(0, 1, "Loading depth model")
        predictor = None
        predictor = None
        if use_external_depthmap:
            self.log.emit(f"Using external depthmap image (skipping the depth model): {depthmap_input_path}")
        else:
            self.log.emit(f"[torch.compile] requested={compile_flag}")
            self.log.emit(f"[TensorRT] requested={trt_flag}")
            if depth_model in ("vda_s", "vda_s_metric"):
                _is_metric = depth_model == "vda_s_metric"
                self.log.emit(f"Loading Video Depth Anything (VDA-S{' Metric' if _is_metric else ''}, streaming)...")
                predictor = load_vda_streaming(encoder="vits", input_size=input_size, fp16=fp16, metric=_is_metric,
                                               compile_model_flag=compile_flag, tensorrt_flag=trt_flag)
            elif depth_model == "distill_any_s":
                self.log.emit("Loading Distill-Any-Depth Small...")
                predictor = load_distill_any_s(input_size=input_size, device=str(device), fp16=fp16,
                                               compile_model_flag=compile_flag, tensorrt_flag=trt_flag)
            elif depth_model == "dav2_s":
                self.log.emit("Loading Depth-Anything V2 Small (DA-V2-S)...")
                predictor = load_dav2_s(input_size=input_size, device=str(device), fp16=fp16,
                                        compile_model_flag=compile_flag, tensorrt_flag=trt_flag)
            elif depth_model == "da3_mono_l":
                self.log.emit("Loading Depth Anything 3 Mono-Large (DA3-Mono-L)...")
                predictor = load_da3_mono_l(input_size=input_size, device=str(device), fp16=fp16,
                                            compile_model_flag=compile_flag, tensorrt_flag=trt_flag)
            else:
                self.log.emit("Loading ZipDepth...")
                predictor = load_zipdepth(input_size=input_size, device=str(device), fp16=fp16,
                                          compile_model_flag=compile_flag, tensorrt_flag=trt_flag)
        if self._aborted():
            return

        # Inpaint model: loaded once and shared by every image of the batch.
        use_medium_inpaint = fill_mode in (WARP_FILL_MODE_MEDIUM, WARP_FILL_MODE_LIGHT)
        use_light_stream = (fill_mode == WARP_FILL_MODE_LIGHT_STREAM)
        _is_light = (fill_mode == WARP_FILL_MODE_LIGHT)
        _mi_model = None
        if use_medium_inpaint:
            _mi_label = "Light Inpaint" if _is_light else "Medium Inpaint"
            _mi_path = _light_inpaint_model_path() if _is_light else _medium_inpaint_model_path()
            ensure_inpaint_model(_mi_path, LIGHT_INPAINT_URL if _is_light else MEDIUM_INPAINT_URL, log=self.log.emit)
            self.log.emit(f"[{_mi_label}] Loading model: {_mi_path}")
            _mi_model = load_medium_inpaint_model(device, path=_mi_path, disable_stretch_mask=_is_light)
            configure_video_inpaint_accel(
                _mi_model, device, bool(compile_flag), bool(trt_flag and not hybrid_flag), bool(fp16), _mi_label,
                "LightInpaint" if _is_light else "MediumInpaint", _mi_path, log=self.log.emit)
            if _is_light:
                _mi_set_stretch_params(LIGHT_STRETCH_MAX_GRAD, LIGHT_STRETCH_MIN_RUN, LIGHT_STRETCH_GROW)
            else:
                _mi_set_stretch_params(MEDIUM_STRETCH_MAX_GRAD, MEDIUM_STRETCH_MIN_RUN, MEDIUM_STRETCH_GROW)
        elif use_light_stream:
            _ls_path = _light_inpaint_stream_model_path()
            ensure_inpaint_model(_ls_path, LIGHT_INPAINT_STREAM_URL, log=self.log.emit)
            self.log.emit(f"[Light Inpaint (stream)] Loading model: {_ls_path}")
            _ls_model = get_light_inpaint_stream_model(device)
            configure_light_stream_accel(_ls_model, device, bool(compile_flag), bool(trt_flag and not hybrid_flag),
                                         log=self.log.emit)
        if self._aborted():
            return

        _ef_cache = {}                       # Auto Edge Fix is measured once per image size
        edge_dilation_base = edge_dilation
        convergence_base = convergence
        done_paths = []
        self.error_count = 0
        _last_err = ""
        for idx, input_path in enumerate(input_paths):
            if self._aborted():
                return
            if idx > 0:
                self.file_started.emit(idx + 1, n_files, input_path.name)
            edge_dilation = edge_dilation_base
            convergence = convergence_base
            try:
                name = safe_stem(input_path)
                self.log.emit(f"Input (image): {input_path}")
                _prog(idx, 0, "Reading image")
                frame = read_image_bgr(input_path)
                h, w = int(frame.shape[0]), int(frame.shape[1])
                self.log.emit(f"Image resolution: {w}x{h}")
                full_hw = (h, w)
                crop_box = None
                if p.get("auto_crop"):
                    crop_box = detect_letterbox_image(frame)
                    if crop_box:
                        _ct, _cb, _cl, _cr = crop_box
                        frame = np.ascontiguousarray(frame[_ct:_cb, _cl:_cr])
                        h, w = int(frame.shape[0]), int(frame.shape[1])
                        self.log.emit(f"Letterbox detected: cropped to {w}x{h} "
                                      f"(top {_ct}, bottom {full_hw[0] - _cb}, left {_cl}, right {full_hw[1] - _cr}).")
                    else:
                        self.log.emit("Auto crop: no black bars found.")
                if self._aborted():
                    return
                _ef_applied = None
                if bool(p.get("auto_mode", False)):
                    _ef_key = (h, w)
                    if _ef_key in _ef_cache:
                        edge_dilation, _ef_applied = _ef_cache[_ef_key]
                    else:
                        try:
                            _ef_applied = int(p.get("edge_dilation"))      # table value, used when nothing is measured
                        except Exception:
                            _ef_applied = None
                        if EDGEFIX_AUTO_MEASURE and predictor is not None:
                            edgefix_reset_profile()
                            try:
                                _t0 = time.perf_counter()
                                edgefix_measure_profile(predictor, depth_model, (h, w), (h, w), input_size, log=self.log.emit)
                                _ef_edge, _ef_ema = compute_auto_edge_ema(
                                    divergence, depth_model=depth_model, warp_fill_mode=fill_mode,
                                    depth_linear_comp=bool(p.get("depth_linear_comp", False)))
                                edge_dilation = edge_fix_level_to_xy(_ef_edge, depth_model=depth_model)
                                _ef_applied = int(_ef_edge)
                                self.log.emit(f"[Auto Edge Fix] strength {float(divergence):g} -> Edge Fix {int(_ef_edge)} "
                                              f"({time.perf_counter() - _t0:.1f}s)")
                            except Exception as _ef_err:
                                self.log.emit(f"[Auto Edge Fix] measurement failed ({type(_ef_err).__name__}: {_ef_err}); "
                                              f"using the built-in table value {p.get('edge_dilation')}")
                        _ef_cache[_ef_key] = (edge_dilation, _ef_applied)
                if _ef_applied is not None:
                    self.auto_edge.emit(int(_ef_applied))
                _reset_fn = getattr(predictor, "reset_state", None) if predictor is not None else None
                if callable(_reset_fn):
                    try:
                        _reset_fn()
                    except Exception as _reset_err:
                        self.log.emit(f"[Warning] predictor.reset_state() failed: {_reset_err}")

                # 3) depth: estimated by the model, or read from the external depthmap image
                #    (EMA off: a single image has no previous frame)
                _prog(idx, 2, "Reading depthmap" if use_external_depthmap else "Estimating depth")
                depth_norm_raw = None
                raw_depth_snapshot = None
                depth_native_wh = None
                if use_external_depthmap:
                    _g = read_depthmap_image(depthmap_input_path)
                    if crop_box:
                        # the depthmap belongs to the uncropped image: cut the same region (scaled to its resolution)
                        _ct, _cb, _cl, _cr = crop_box
                        _g0h, _g0w = int(_g.shape[0]), int(_g.shape[1])
                        _y0 = int(round(_ct * _g0h / full_hw[0])); _y1 = max(_y0 + 1, int(round(_cb * _g0h / full_hw[0])))
                        _x0 = int(round(_cl * _g0w / full_hw[1])); _x1 = max(_x0 + 1, int(round(_cr * _g0w / full_hw[1])))
                        _g = np.ascontiguousarray(_g[_y0:_y1, _x0:_x1])
                        self.log.emit("Auto crop: the external depthmap was cropped to the same region.")
                    _gh, _gw = int(_g.shape[0]), int(_g.shape[1])
                    self.log.emit(f"Depthmap image resolution: {_gw}x{_gh}")
                    if abs((_gw / _gh) - (w / h)) > 0.02:
                        self.log.emit(f"[Warning] Depthmap aspect ratio ({_gw}x{_gh}) differs from the image ({w}x{h}); "
                                      f"it will be stretched to the image size.")
                    depth_ext = torch.from_numpy(_g).to(device=device, dtype=dtype_tensor).unsqueeze(0).unsqueeze(0)
                    if tuple(depth_ext.shape[-2:]) != (h, w):
                        depth_ext = F.interpolate(depth_ext, size=(h, w), mode="bilinear", align_corners=True)
                    depth_ext = (1.0 - depth_ext).float()
                    if _HAS_EMA_SCALER:
                        _scaler = EMAMinMaxScaler(decay=0.0, buffer_size=1, mode="minmax")
                        with torch.inference_mode():
                            depth_norm = _scaler.update(depth_ext)
                    else:
                        with torch.inference_mode():
                            depth_norm, _, _ = normalize_depth_gpu(depth_ext, decay=0.0)
                    del depth_ext, _g
                else:
                    depth_raw = estimate_depth_raw(predictor, frame)
                    # native depth resolution (the saved depthmaps use it, like the video path)
                    if isinstance(predictor, VDAStreamingAdapter) and predictor.streaming_method is not None:
                        _nh, _nw = vda_native_hw(
                            h, w, predictor.input_size - (VDA_METRIC_PADDING * 2 if getattr(predictor, "metric", False) else 0))
                        depth_native_wh = (_nw, _nh)
                    else:
                        depth_native_wh = (int(depth_raw.shape[-1]), int(depth_raw.shape[-2]))
                    depth_raw = (depth_raw.max() - depth_raw).float()
                    if do_extract_depthmap and raw_depthmap:
                        with torch.inference_mode():
                            _raw_snap, _, _ = normalize_depth_gpu(depth_raw, decay=0.0)
                        raw_depth_snapshot = upsample_depth(_raw_snap, h, w)
                        del _raw_snap
                    if _HAS_EMA_SCALER:
                        _scaler = EMAMinMaxScaler(decay=0.0, buffer_size=1, mode="minmax")
                        with torch.inference_mode():
                            depth_norm_raw = _scaler.update(depth_raw)
                        depth_norm = upsample_depth(depth_norm_raw, h, w)
                    else:
                        depth_up = upsample_depth(depth_raw, h, w)
                        with torch.inference_mode():
                            depth_norm, _, _ = normalize_depth_gpu(depth_up, decay=0.0)
                        del depth_up
                    del depth_raw
                if self._aborted():
                    return

                if auto_conv_on:
                    _ac_src = depth_norm_raw if depth_norm_raw is not None else depth_norm
                    _conv_slider = AutoConvergence().update(_ac_src, frame, float(divergence) * 0.005)
                    convergence = 1.0 - _conv_slider
                    if depth_comp_k < 1.0 - 1e-4:
                        convergence = convergence * depth_comp_k + 0.5 * (1.0 - depth_comp_k)
                    self.log.emit(f"[Auto Convergence] measured convergence slider = {_conv_slider:.3f}")

                # 4) depth post-processing (same order as the video path)
                depth_for_warp = depth_norm
                if edge_dilation_is_enabled(edge_dilation):
                    _dh, _dw = compute_dilation_target_dims(h, w)
                    with torch.inference_mode():
                        depth_for_dilate = upsample_depth(depth_norm, _dh, _dw)
                        inv = 1.0 - depth_for_dilate
                        inv = dilate_edge(inv, edge_dilation)
                        depth_dilated = 1.0 - inv
                        depth_for_warp = (
                            upsample_depth(depth_dilated, h, w) if (_dh, _dw) != (h, w) else depth_dilated
                        )
                if depth_aa_enabled:
                    with torch.inference_mode():
                        depth_for_warp = apply_depth_aa_gpu(depth_for_warp, _DEPTH_AA_SIGMA)

                # Depthmap image(s): same order as the video path (plain/raw before the linear compression,
                # corrected after it, so the corrected map is exactly the depth used for the warp).
                if do_extract_depthmap:
                    _dm_src = raw_depth_snapshot if (raw_depthmap and raw_depth_snapshot is not None) else depth_for_warp
                    _dm_path = unique_numbered_path(output_dir, f"{name}_depthmap", ".png")
                    write_depthmap_image(_dm_path, _dm_src, depth_native_wh)
                    self.log.emit(f"Depthmap output ({DEPTHMAP_IMAGE_BITS}-bit PNG): {_dm_path}")
                raw_depth_snapshot = None

                if depth_comp_k < 1.0 - 1e-4:
                    with torch.inference_mode():
                        depth_for_warp = apply_depth_linear_comp_gpu(depth_for_warp, depth_comp_k)

                if do_extract_corrected:
                    _cm_path = unique_numbered_path(output_dir, f"{name}_depthmap_corrected", ".png")
                    write_depthmap_image(_cm_path, depth_for_warp, depth_native_wh)
                    self.log.emit(f"Corrected depthmap output ({DEPTHMAP_IMAGE_BITS}-bit PNG): {_cm_path}")

                # 5) stereo warp + hole filling
                _prog(idx, 3, "Generating stereo image")
                bgr_t = (torch.from_numpy(frame).to(device=device, dtype=dtype_tensor)
                         .permute(2, 0, 1).unsqueeze(0).div_(255.0))

                if use_medium_inpaint:
                    medium_stream = MediumInpaintStream(_mi_model, temporal_smooth=_is_light)
                    left_t, right_t, left_m, right_m = make_stereo_with_masks(
                        bgr_t, depth_for_warp, divergence, convergence, device,
                        already_normalized=True, preserve_screen_border=preserve_border,
                    )
                    # The video model works on a 12-frame window: the single frame is repeated / padded by the stream
                    # wrapper, so push() + flush() returns exactly one finished frame.
                    outs = list(medium_stream.push(left_t, right_t, left_m, right_m))
                    outs += list(medium_stream.flush())
                    if not outs:
                        raise RuntimeError("Inpaint model returned no frame.")
                    left_t, right_t = outs[0]
                    del left_m, right_m, outs
                elif use_light_stream:
                    left_t, right_t = make_stereo_light_stream(
                        bgr_t, depth_for_warp, divergence, convergence, device,
                        already_normalized=True, preserve_screen_border=preserve_border,
                        cache=LightInpaintStreamCache(),
                    )
                else:
                    left_t, right_t = make_stereo(
                        bgr_t, depth_for_warp, divergence, convergence, device,
                        already_normalized=True, preserve_screen_border=preserve_border,
                        fill_mode=fill_mode, inpaint_hole_threshold=inpaint_hole_threshold,
                    )
                del depth_norm, depth_for_warp, bgr_t
                if self._aborted():
                    return

                # 6) pack + save (no even-size restriction for images)
                _prog(idx, 4, "Saving image")
                packed_t = pack_frame_gpu(left_t, right_t, fmt)
                del left_t, right_t
                out_bgr = (packed_t[0].clamp(0.0, 1.0).mul(255.0).byte()
                           .permute(1, 2, 0).contiguous().cpu().numpy())
                del packed_t
                if p.get("pad_169"):
                    _pre_h, _pre_w = int(out_bgr.shape[0]), int(out_bgr.shape[1])
                    out_bgr = pad_image_to_ratio(out_bgr, fmt)
                    if out_bgr.shape[0] != _pre_h or out_bgr.shape[1] != _pre_w:
                        self.log.emit(f"Pad to 16:9: {_pre_w}x{_pre_h} -> {int(out_bgr.shape[1])}x{int(out_bgr.shape[0])}")
                oh, ow = int(out_bgr.shape[0]), int(out_bgr.shape[1])

                _save_fmt = resolve_image_format(image_format, input_path)
                ext = pick_image_output_ext(_save_fmt, ow, oh)
                if _save_fmt == "jpg" and ext != ".jpg":
                    self.log.emit(f"[Info] {ow}x{oh} is too large for JPG; saving as PNG instead.")
                out_path = make_unique_output_path(output_dir, name, vr_player_format_label(fmt), ext)
                write_image_bgr(out_path, out_bgr, jpeg_quality=image_quality)
                self.log.emit(f"Output ({ow}x{oh}, {fmt}): {out_path}")
                _prog(idx, STEPS, "Done")
                done_paths.append(out_path)
            except Exception as _img_err:
                if n_files == 1:
                    raise
                import traceback
                self.error_count += 1
                _last_err = f"{type(_img_err).__name__}: {_img_err}"
                self.log.emit(f"[Skip] {input_path.name}: {_last_err}")
                self.log.emit("[ERROR] " + traceback.format_exc())
                try:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        if not done_paths:
            raise RuntimeError(f"All {n_files} image(s) failed. Last error: {_last_err}")
        if self.error_count:
            self.log.emit(f"[Info] {len(done_paths)} of {n_files} image(s) converted, {self.error_count} skipped due to errors.")
        self.finished_ok.emit(str(done_paths[-1]))


_LAUNCHER_SUPPORTED_LANGUAGES = ("en", "ko")
_LAUNCHER_DEFAULT_LANGUAGE = "en"

def _launcher_config_path() -> Path:
    return APP_DIR / "depthlive3d_config.json"

def _load_ui_language() -> str:

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
        "input_placeholder": "Select an input video or image file (or a folder)...",
        "btn_input": "Select File...",
        "depthmap_input_label": "Depthmap Input File",
        "depthmap_input_placeholder": "(Optional) Select a pre-made depthmap video or image...",
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
        "depth_aa": "Depth AA",
        "depth_aa_restart_title": "Notice",
        "depth_aa_restart_msg": "The Depth AA setting will take effect on the next conversion.\nPlease stop and restart the conversion to apply this change.",
        "depth_aa_auto_disable_title": "Auto Mode Active",
        "depth_aa_auto_disable_msg": "Depth AA is automatically enabled by Auto Mode when using a VDA or DA3 model.\nTo disable Depth AA, please turn off Auto Mode first.",
        "depth_linear_comp": "Depth Linear Compression",
        "depth_linear_comp_tooltip": "Compresses the depth range to reduce inpaint holes at foreground/background edges.\nAvailable for all depth map models; turn this option off to use the original stereo strength.",
        "depth_linear_comp_restart_msg": "The Depth Linear Compression setting will take effect on the next conversion.\nPlease stop and restart the conversion to apply this change.",
        "warp_grp": "Warping Options",
        "warp_mode_label": "Method",
        "inpaint_res": "Inpaint Resolution",
        "inpaint_res_tooltip": "Resolution (height) the inpaint model runs at for Light Inpaint, Medium Inpaint and Light Inpaint (stream). Pick 512 / 720 / 1080, type a custom value (64\u20134320), or choose MAX for no limit (MAX = the model runs at the original video resolution, without downscaling). Default: MAX for all of them. It is never higher than the output resolution. No restart needed \u2014 it is applied the next time a conversion starts.",
        "compile_tooltip": "Uses Torch.compile acceleration for the depth model, Light (including stream) and Medium Inpaint models.\nFor each input size, the model is compiled and warmed up before the conversion starts.\nAutomatically disabled if Torch.compile is not supported on this PC.",
        "trt_tooltip": "Uses TensorRT acceleration for the depth model, Light (including stream) and Medium Inpaint models.\nThe first run for each input size builds an engine, which can take a few minutes; it is cached afterwards.\nIf Torch.compile is also enabled, TensorRT takes priority.\nAutomatically disabled if TensorRT is not installed.",
        "err_medium_model_missing": "Medium Inpaint requires the model file 'medium_inpaint.pth'.\nPlease place it in this folder:\n{0}",
        "err_light_model_missing": "Light Inpaint requires the model file 'light_video_inpaint.pth'.\nPlease place it in this folder:\n{0}",
        "err_conversion_aborted": "The conversion was stopped by the user or terminated due to an error.",
        "chk_extract_depthmap": "Extract Depthmap Video",
        "chk_raw_depthmap": "Extract Raw Depthmap",
        "chk_corrected_depthmap": "Extract Corrected Depthmap",
        "chk_preserve": "Screen Border Protection",
        "chk_fp16": "Use FP16",
        "chk_auto_mode": "Auto Mode",
        "compile_unsupported": "Torch.compile is not supported on this PC (compiler toolchain missing or broken). The option has been turned off automatically.",
        "trt_unsupported": "TensorRT is not available on this PC (tensorrt 10.x is not installed or CUDA is unavailable). The option has been turned off automatically.",
        "hybrid_tooltip": "Hybrid mode: TensorRT accelerates the depth map model, while Torch.compile accelerates the inpaint models.\nRequires both Torch.compile and TensorRT.",
        "hybrid_unsupported_compile": "Torch.compile is not supported on this PC, so Hybrid mode has been disabled.",
        "hybrid_unsupported_trt": "TensorRT is not available on this PC, so Hybrid mode has been disabled.",
        "hybrid_unsupported_both": "Torch.compile and TensorRT are not supported on this PC, so Hybrid mode has been disabled.",
        "chk_low_ram": "Use Memory Saver",
        "vcodec_label": "Video Codec:",
        "preset_label": "Preset:",
        "quality_crf": "Quality (CRF):",
        "quality_cq": "Quality (CQ):",
        "image_format_label": "Image Format:",
        "image_format_auto": "Auto",
        "image_quality_label": "Image Quality (JPG):",
        "image_format_tip": "Output format for still images. Auto: JPG/JPEG input is saved as JPG, PNG input as PNG, any other image format as PNG.",
        "image_quality_tip": "JPG quality (1-100). Used only when the image is saved as JPG (JPG format, or Auto with a JPG input).",
        "chk_resize": "Resize Resolution",
        "chk_hdr_norm": "HDR / Frame Rate Normalize (Auto Tonemapping + CFR, fixes MKV/TS sync drift)",
        "chk_auto_crop": "Auto Letterbox Crop (Remove Black Bars)",
        "chk_pad_169": "Pad to 16:9 (Auto Aspect Ratio Fill)",
        "chk_sound_boost": "Sound Booster (400% Volume)",
        "chk_shutdown": "Schedule PC Shutdown After Conversion",
        "chk_shutdown_tip": "When all conversions have completed, the computer will automatically shut down after 5 minutes.\nIf you uncheck this option within those 5 minutes, the scheduled shutdown is canceled. You can turn this option ON or OFF at any time: before or during a conversion (it is applied when everything finishes), and also after it has finished (checking it then schedules a shutdown 5 minutes from that moment).\nNo shutdown is scheduled automatically if the conversion is stopped or fails.",
        "status_done_shutdown": "Completed - PC shuts down in 5 min (uncheck the option to cancel)",
        "status_shutdown_canceled": "Completed - scheduled shutdown canceled",
        "done_msg_shutdown": "Conversion completed.\n\nThe computer will shut down in 5 minutes.\nTo cancel, uncheck \'Schedule PC Shutdown After Conversion\'.",
        "chk_start_time": "Start Time",
        "chk_end_time": "End Time",
        "btn_start": "Start",
        "btn_stop": "Stop",
        "status_idle": "Idle",
        "status_starting": "Starting conversion...",
        "status_done": "Completed",
        "status_error": "Error / Aborted",
        "status_stopped_saved": "Stopped (converted part saved)",
        "stopped_saved_msg": "The conversion was stopped.\nThe video converted so far was saved:\n{}",
        "dialog_input_title": "Select Input Video or Image",
        "dialog_depthmap_title": "Select Depthmap Video or Image",
        "dialog_input_folder_title": "Select Input Folder",
        "filter_video_image": "Video and Image Files",
        "filter_video": "Video Files",
        "filter_image": "Image Files",
        "filter_depthmap": "Depthmap Files",
        "filter_all": "All Files",
        "all_files_processed": "All files processed.",
        "dialog_output_title": "Select Output Folder",
        "btn_input_folder": "Select Folder...",
        "err_no_videos_in_folder": "No supported video or image files found in the selected folder.",
        "batch_progress": "Batch Progress: {} / {} - {}",
        "eta_text": "Time left: {}",
        "eta_calculating": "calculating...",
        "time_hms": "{h}h {m:02d}m {s:02d}s",
        "time_ms": "{m}m {s:02d}s",
        "time_s": "{s}s",
        "err_title": "Input Error",
        "err_no_input": "Please select an input file.",
        "err_no_output": "Please select an output folder.",
        "err_file_not_found": "Input file does not exist: {}",
        "err_unsupported_ext": "Unsupported file extension: {}",
        "err_invalid_size_range": "Input Size must be between 64 and 630.",
        "err_invalid_size_num": "Input Size is not a valid number.",
        "err_invalid_inpaint_res_range": "Inpaint Resolution must be between 64 and 4320.",
        "err_invalid_inpaint_res_num": "Inpaint Resolution is not a valid number.",
        "err_invalid_resize": "Resize Resolution must be in WIDTHxHEIGHT format (e.g. 1920x1080).",
        "notice_title": "Notice",
        "notice_already_running": "Conversion is already in progress.",
        "btn_reset_settings": "Reset Settings",
        "reset_confirm_title": "Reset Settings",
        "reset_confirm_msg": "Do you want to restore all settings to their defaults?\n\nYour saved presets will not be deleted (the 'Best Quality' preset will be recreated).",
        "reset_yes": "Yes",
        "reset_no": "No",
        "reset_done": "Settings have been reset to defaults.",
        "preset_auto_mode_off": "Auto Mode was turned off to apply the preset.",
        "time_err_title": "Time Input Error",
        "time_err_start_exceed": "Start time ({}s) exceeds total video length ({:.1f}s).",
        "time_err_end_exceed": "End time ({}s) exceeds total video length ({:.1f}s).",
        "time_err_order": "Start time cannot be greater than or equal to end time.",
        "done_title": "Completed",
        "done_msg": "Conversion completed.\n\n{}",
        "err_dialog_title": "Error",
        "err_depthmap_folder_conflict": "A depthmap input file is selected, but the input is set to a folder.\nPlease select a single input video file instead of a folder.",
        "ffmpeg_license_title": "ffmpeg Not Found",
        "ffmpeg_license_msg": (
            "ffmpeg was not found in the program folder or system PATH.\n\n"
            "Conversion3D can automatically download the latest official FFmpeg build\n"
            "(FFmpeg, licensed under GPL v3, https://ffmpeg.org) from the\n"
            "FFmpeg-Builds release page (https://github.com/BtbN/FFmpeg-Builds) and\n"
            "install it in the program folder.\n\n"
            "Do you want to download and install ffmpeg now?"
        ),
        "ffmpeg_missing_msg": (
            "ffmpeg is required to run this feature.\n"
            "Please download ffmpeg/ffprobe manually and place them in:\n{dir}"
        ),
    },
    "ko": {
        "title": "Depth Conversion 3D  |  © 2026 Wake-82",
        "file_grp": "입력 / 출력",
        "input_label": "입력 파일",
        "input_placeholder": "변환할 영상 또는 이미지 파일(또는 폴더)을 선택하세요...",
        "btn_input": "파일 선택...",
        "depthmap_input_label": "뎁스맵 입력 파일",
        "depthmap_input_placeholder": "(선택) 미리 만들어진 뎁스맵 영상 또는 이미지를 선택하세요...",
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
        "depth_aa": "뎁스 AA",
        "depth_aa_restart_title": "알림",
        "depth_aa_restart_msg": "뎁스 AA 설정은 다음 변환부터 적용됩니다.\n변환을 중지하고 다시 시작해 주세요.",
        "depth_aa_auto_disable_title": "자동 모드 활성 중",
        "depth_aa_auto_disable_msg": "VDA / DA3 모델 사용 시 자동 모드에 의해 뎁스 AA가 자동으로 켜집니다.\n뎁스 AA를 끄려면 먼저 자동 모드를 꺼주세요.",
        "depth_linear_comp": "뎁스 선형 압축",
        "depth_linear_comp_tooltip": "뎁스 범위를 압축하여 전경/배경 경계에서 생기는 인페인트 홀을 줄입니다.\n모든 뎁스맵 모델에서 사용할 수 있으며, 원래 입체강도 그대로 사용하려면 이 옵션을 끄세요.",
        "depth_linear_comp_restart_msg": "뎁스 선형 압축 설정은 다음 변환부터 적용됩니다.\n변환을 중지하고 다시 시작해 주세요.",
        "warp_grp": "워핑 옵션",
        "warp_mode_label": "메서드",
        "inpaint_res": "인페인트 해상도",
        "inpaint_res_tooltip": "라이트 인페인트, 미디움 인페인트, 라이트 인페인트 스트림에서 인페인트 모델이 작동하는 해상도(세로)입니다. 512 / 720 / 1080 중에서 선택하거나 직접 입력(64~4320)할 수 있으며, MAX는 해상도 제한 없이(다운스케일 없이) 원본 영상의 해상도 그대로 인페인트 모델이 작동하는 옵션입니다. 기본값: 모두 MAX. 출력 해상도보다 높게 적용되지는 않습니다. 재시작이 필요하지 않으며, 다음 변환 시작부터 적용됩니다.",
        "compile_tooltip": "뎁스 모델, 라이트(스트림 포함), 미디움 인페인트 모델에 Torch.compile 가속을 사용합니다.\n각 입력 크기별로 컴파일 및 워밍업을 마친 후 변환이 시작됩니다.\n이 PC에서 Torch.compile을 지원하지 않으면 자동으로 해제됩니다.",
        "trt_tooltip": "뎁스 모델, 라이트(스트림 포함), 미디움 인페인트 모델에 TensorRT 가속을 사용합니다.\n각 입력 크기별 첫 실행 때 엔진을 빌드하므로 몇 분 걸릴 수 있으며, 이후에는 캐시되어 바로 사용됩니다.\nTorch.compile 같이 켜져있으면, TensorRT가 우선 적용됩니다.\nTensorRT가 설치되어 있지 않으면 자동으로 해제됩니다.",
        "err_medium_model_missing": "미디움 인페인트에는 모델 파일 'medium_inpaint.pth'가 필요합니다.\n아래 폴더에 파일을 넣어 주세요:\n{0}",
        "err_light_model_missing": "Light Inpaint에는 모델 파일 'light_video_inpaint.pth'가 필요합니다.\n아래 폴더에 파일을 넣어 주세요:\n{0}",
        "err_conversion_aborted": "사용자가 변환을 중단 했거나, 오류로 인해서 종료되었습니다.",
        "chk_extract_depthmap": "뎁스맵 영상 추출",
        "chk_raw_depthmap": "원본 뎁스맵 추출",
        "chk_corrected_depthmap": "보정 뎁스맵 추출",
        "chk_preserve": "화면 테두리 보호",
        "chk_fp16": "FP16 사용",
        "chk_auto_mode": "자동 모드",
        "compile_unsupported": "이 PC에서는 Torch.compile을 지원하지 않습니다 (컴파일러 도구가 없거나 문제가 있습니다). 옵션이 자동으로 꺼졌습니다.",
        "trt_unsupported": "이 PC에서는 TensorRT를 사용할 수 없습니다 (tensorrt 10.x가 설치되지 않았거나 CUDA를 사용할 수 없습니다). 옵션이 자동으로 비활성화되었습니다.",
        "hybrid_tooltip": "하이브리드 모드: TensorRT는 뎁스맵 모델을 가속하고, Torch.compile은 인페인트 모델을 가속합니다.\nTorch.compile과 TensorRT가 모두 지원되어야 합니다.",
        "hybrid_unsupported_compile": "이 PC에서는 Torch.compile이 지원되지 않아 Hybrid 모드가 비활성화됩니다.",
        "hybrid_unsupported_trt": "이 PC에서는 TensorRT가 지원되지 않아 Hybrid 모드가 비활성화됩니다.",
        "hybrid_unsupported_both": "이 PC에서는 Torch.compile과 TensorRT가 지원되지 않아 Hybrid 모드가 비활성화됩니다.",
        "chk_low_ram": "메모리 절약 모드 사용",
        "vcodec_label": "비디오 코덱:",
        "preset_label": "프리셋:",
        "quality_crf": "품질 (CRF):",
        "quality_cq": "품질 (CQ):",
        "image_format_label": "이미지 포맷:",
        "image_format_auto": "Auto",
        "image_quality_label": "이미지 품질 (JPG):",
        "image_format_tip": "이미지 저장 형식입니다. Auto: JPG/JPEG 입력은 JPG, PNG 입력은 PNG, 그 외 이미지 형식은 PNG로 저장합니다.",
        "image_quality_tip": "JPG 품질(1~100)입니다. JPG로 저장될 때만 적용됩니다(포맷이 JPG이거나, Auto에서 입력이 JPG인 경우).",
        "chk_resize": "해상도 조정",
        "chk_hdr_norm": "HDR / 프레임 정규화 (자동 톤매핑 + 가변프레임 보정, MKV/TS 싱크 어긋남 해결)",
        "chk_auto_crop": "자동 레터박스 크롭 (검은 여백 제거)",
        "chk_pad_169": "16:9 비율로 패딩 (자동 화면 비율 채우기)",
        "chk_sound_boost": "사운드 부스터 (볼륨 400%)",
        "chk_shutdown": "변환 후 컴퓨터 종료 예약",
        "chk_shutdown_tip": "변환이 모두 완료되면 5분 뒤에 컴퓨터가 자동으로 종료됩니다.\n5분 안에 이 옵션의 체크를 해제하면 종료 예약이 취소됩니다. 이 옵션은 언제든지 켜고 끌 수 있습니다: 변환 전이나 변환 중에도 가능하며 (모든 변환이 끝났을 때 적용됨), 변환이 끝난 뒤에 체크하면 그 시점부터 5분 뒤에 종료가 예약됩니다.\n변환을 중지했거나 오류가 발생한 경우에는 종료가 자동으로 예약되지 않습니다.",
        "status_done_shutdown": "완료 - 5분 뒤 컴퓨터가 종료됩니다 (취소하려면 옵션 체크 해제)",
        "status_shutdown_canceled": "완료 - 종료 예약이 취소되었습니다",
        "done_msg_shutdown": "변환이 완료되었습니다.\n\n5분 뒤에 컴퓨터가 종료됩니다.\n취소하려면 \'변환 후 컴퓨터 종료 예약\' 체크를 해제하세요.",
        "chk_start_time": "시작 시간",
        "chk_end_time": "종료 시간",
        "btn_start": "시작",
        "btn_stop": "중지",
        "status_idle": "대기 중",
        "status_starting": "변환을 시작합니다...",
        "status_done": "완료",
        "status_error": "오류 / 중단됨",
        "status_stopped_saved": "중지됨 (진행된 부분까지 저장됨)",
        "stopped_saved_msg": "변환을 중지했습니다.\n현재까지 변환된 영상이 저장되었습니다:\n{}",
        "dialog_input_title": "입력 영상 또는 이미지 선택",
        "dialog_depthmap_title": "뎁스맵 영상 또는 이미지 선택",
        "dialog_input_folder_title": "입력 폴더 선택",
        "filter_video_image": "영상 및 이미지 파일",
        "filter_video": "영상 파일",
        "filter_image": "이미지 파일",
        "filter_depthmap": "뎁스맵 파일",
        "filter_all": "모든 파일",
        "all_files_processed": "모든 파일 처리가 완료되었습니다.",
        "dialog_output_title": "출력 폴더 선택",
        "btn_input_folder": "폴더 선택...",
        "err_no_videos_in_folder": "선택한 폴더에 지원되는 영상 또는 이미지 파일이 없습니다.",
        "batch_progress": "일괄 진행: {} / {} - {}",
        "eta_text": "남은 예상 시간: {}",
        "eta_calculating": "계산 중...",
        "time_hms": "{h}시간 {m:02d}분 {s:02d}초",
        "time_ms": "{m}분 {s:02d}초",
        "time_s": "{s}초",
        "err_title": "입력 오류",
        "err_no_input": "입력 파일을 선택해 주세요.",
        "err_no_output": "출력 폴더를 선택해 주세요.",
        "err_file_not_found": "입력 파일이 존재하지 않습니다: {}",
        "err_unsupported_ext": "지원하지 않는 파일 확장자입니다: {}",
        "err_invalid_size_range": "입력 크기는 64~630 사이여야 합니다.",
        "err_invalid_size_num": "입력 크기가 올바른 숫자가 아닙니다.",
        "err_invalid_inpaint_res_range": "인페인트 해상도는 64~4320 사이여야 합니다.",
        "err_invalid_inpaint_res_num": "인페인트 해상도가 올바른 숫자가 아닙니다.",
        "err_invalid_resize": "해상도 조정 값은 '가로x세로' 형식이어야 합니다 (예: 1920x1080).",
        "notice_title": "알림",
        "notice_already_running": "이미 변환이 진행 중입니다.",
        "btn_reset_settings": "설정 초기화",
        "reset_confirm_title": "설정 초기화",
        "reset_confirm_msg": "모든 설정을 기본값으로 되돌리시겠습니까?\n\n저장해 둔 프리셋은 삭제되지 않습니다 ('Best Quality' 프리셋은 다시 생성됩니다).",
        "reset_yes": "예",
        "reset_no": "아니요",
        "reset_done": "설정이 기본값으로 초기화되었습니다.",
        "preset_auto_mode_off": "프리셋 적용을 위해 자동 모드를 껐습니다.",
        "time_err_title": "시간 입력 오류",
        "time_err_start_exceed": "시작 시간({}초)이 전체 영상 길이({:.1f}초)를 초과합니다.",
        "time_err_end_exceed": "종료 시간({}초)이 전체 영상 길이({:.1f}초)를 초과합니다.",
        "time_err_order": "시작 시간은 종료 시간보다 크거나 같을 수 없습니다.",
        "done_title": "완료",
        "done_msg": "변환이 완료되었습니다.\n\n{}",
        "err_dialog_title": "오류",
        "err_depthmap_folder_conflict": "뎁스맵 입력 파일이 선택되어 있지만, 입력이 폴더로 설정되어 있습니다.\n폴더 대신 단일 입력 영상 파일을 선택해 주세요.",
        "ffmpeg_license_title": "ffmpeg를 찾을 수 없음",
        "ffmpeg_license_msg": (
            "프로그램 폴더와 시스템 PATH에서 ffmpeg를 찾을 수 없습니다.\n\n"
            "Conversion3D는 공식 FFmpeg-Builds 릴리스 페이지\n"
            "(https://github.com/BtbN/FFmpeg-Builds)에서 최신 FFmpeg\n"
            "(GPL v3 라이센스, https://ffmpeg.org)를 자동으로 다운로드하여\n"
            "프로그램 폴더에 설치할 수 있습니다.\n\n"
            "지금 ffmpeg를 다운로드하여 설치하시겠습니까?"
        ),
        "ffmpeg_missing_msg": (
            "이 기능을 사용하려면 ffmpeg가 필요합니다.\n"
            "ffmpeg/ffprobe를 직접 다운로드하여 아래 폴더에 넣어주세요:\n{dir}"
        ),
    },
}

# ---------------------------------------------------------------------------------------------
# Auto Convergence (no AI model): picks the slider value (= preview brightness of the subject) from the depth map.
# Convergence slider 1.1 = AUTO, 1.0 or lower = manual.
# ---------------------------------------------------------------------------------------------
AUTO_CONVERGENCE_THRESHOLD = 1.05   # slider value at/above this -> AUTO (the slider offers 1.1)
AUTO_CONV_INTERVAL = 2              # measure every N frames (1 small GPU->CPU copy per measurement)
AUTO_CONV_OFFSET = 0.03             # screen plane slightly behind the subject (slider value minus this)
AUTO_CONV_EMA = 0.30                # smoothing per measurement
AUTO_CONV_THR_MUL = 1.3             # near/far split = Otsu threshold x this (keeps floor / far people out of the subject blob)
AUTO_CONV_CENTER_BAND = 0.20        # several subjects: level is taken from the centre band (+-this fraction of the width) of the chosen blob
AUTO_CONV_PCT = 60                  # percentile of the chosen blob used as the subject level
AUTO_CONV_STEP = 0.05               # output quantisation / hysteresis (keeps caches stable)
AUTO_CONV_SLEW = 0.0007             # max output change per measurement, as a fraction of the frame width (~1.3 px at 1920)
AUTO_CONV_SLEW_CUT = 0.0030         # same, right after a scene cut (~5.8 px at 1920)
AUTO_CONV_CUT_THRESH = 0.80         # gray histogram L1 distance (0..2) treated as a scene cut
# Subject level picking. "center" (default): sharpness-weighted percentile of the depth around the screen centre,
# tuned on 19 hand-picked scenes (mean error 0.34 -> 0.10, leave-one-out 0.12). "blob": the old blob / Otsu picker.
AUTO_CONV_MODE = "center"
AUTO_CONV_C_SIGMA = 0.08            # centre weight: gaussian sigma as a fraction of the frame width
AUTO_CONV_C_SHARP = 1.0             # sharpness weight exponent (0 = off): in-focus areas count more
AUTO_CONV_C_PCT = 60                # weighted percentile of the depth used as the subject level
AUTO_CONV_C_OFFSET = 0.06           # slider value = subject level + this (screen plane slightly in front, as picked by eye)

class AutoConvergence:
    W, H = 154, 90

    def __init__(self):
        self.n = 0
        self.ema = None
        self.out = None
        self.prev_hist = None

    def _measure(self, depth_norm_raw, frame):
        W, H = self.W, self.H
        d = depth_norm_raw.detach().float()
        d = d.reshape(1, 1, d.shape[-2], d.shape[-1])
        bright = F.adaptive_avg_pool2d(1.0 - d, (H, W))
        b = (bright.clamp(0, 1) * 255.0).round().to(torch.uint8)[0, 0].cpu().numpy()

        g = None
        if isinstance(frame, np.ndarray) and frame.ndim == 3:
            small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            code = cv2.COLOR_BGRA2GRAY if small.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            g = cv2.cvtColor(small, code)

        # scene cut (gray histogram, falls back to the depth histogram)
        src = g if g is not None else b
        hist = cv2.calcHist([src], [0], None, [32], [0, 256]).ravel()
        hist = hist / max(1.0, float(hist.sum()))
        cut = self.prev_hist is not None and float(np.abs(hist - self.prev_hist).sum()) > AUTO_CONV_CUT_THRESH
        self.prev_hist = hist

        if int(b.max()) - int(b.min()) < 12:
            return None, cut                      # flat depth: no reliable subject -> keep the previous value

        if AUTO_CONV_MODE == "blob":
            return self._pick_blob(b, g), cut
        return self._pick_center(b, g), cut

    @staticmethod
    def _pick_center(b, g):
        # Weighted percentile of the depth: weight = gaussian around the screen centre x sharpness (in-focus = subject).
        W, H = AutoConvergence.W, AutoConvergence.H
        xs = (np.arange(W, dtype=np.float32) + 0.5) / W
        w = np.exp(-((xs - 0.5) ** 2) / (2.0 * AUTO_CONV_C_SIGMA ** 2))[None, :] * np.ones((H, 1), np.float32)
        if g is not None and AUTO_CONV_C_SHARP > 0:
            lap = cv2.GaussianBlur(np.abs(cv2.Laplacian(g, cv2.CV_32F)), (0, 0), 2)
            w = w * (lap / (float(lap.max()) + 1e-6) + 0.05) ** AUTO_CONV_C_SHARP
        v = b.astype(np.float32).ravel()
        order = np.argsort(v, kind="stable")
        cs = np.cumsum(w.ravel()[order])
        cs /= max(float(cs[-1]), 1e-12)
        level = float(np.interp(AUTO_CONV_C_PCT / 100.0, cs, v[order])) / 255.0
        return float(np.clip(level + AUTO_CONV_C_OFFSET, 0.0, 1.0))

    @staticmethod
    def _pick_blob(b, g):
        # Old picker (AUTO_CONV_MODE = "blob"): Otsu near/far split, biggest central sharp blob.
        W, H = AutoConvergence.W, AutoConvergence.H
        thr, mask = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(b, int(thr * AUTO_CONV_THR_MUL), 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return None

        sharp = None
        if g is not None:
            lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
        best, best_score = -1, -1.0
        cands = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 0.005 * W * H:
                continue
            cols = (labels == i).sum(0).astype(np.float32)          # blob mass per column -> how central the blob is
            cw = 0.2 + 0.8 * float((cols * np.exp(-((np.arange(W) / W - 0.5) ** 2) / (2 * 0.15 ** 2))).sum() / max(1.0, cols.sum()))
            ep = 0.3 if (x <= 0 or x + w >= W) else 1.0           # blobs touching the left/right border: likely foreground blur
            sh = float(lap[labels == i].mean()) if g is not None else 1.0
            cands.append((i, float(area) * cw * ep, sh))
        if not cands:
            return None
        max_sh = max(c[2] for c in cands) + 1e-6
        for i, base, sh in cands:
            score = base * (sh / max_sh + 0.05)
            if score > best_score:
                best, best_score = i, score
        sel = labels == best
        _c = sel.copy()
        _c[:, :int(W * (0.5 - AUTO_CONV_CENTER_BAND))] = False
        _c[:, int(W * (0.5 + AUTO_CONV_CENTER_BAND)):] = False
        if int(_c.sum()) >= 0.01 * W * H:
            sel = _c                                                # centre person(s) only
        val = float(np.percentile(b[sel], AUTO_CONV_PCT)) / 255.0
        return float(np.clip(val - AUTO_CONV_OFFSET, 0.0, 1.0))

    def update(self, depth_norm_raw, frame, shift_per_unit=0.0125):
        # shift_per_unit: warp shift (fraction of frame width) per 1.0 of slider value = divergence * 0.005
        _spu = max(1e-4, float(shift_per_unit))
        self.n += 1
        if self.out is not None and (self.n % AUTO_CONV_INTERVAL) != 0:
            return self.out
        try:
            target, cut = self._measure(depth_norm_raw, frame)
        except Exception as _e:
            if self.n <= AUTO_CONV_INTERVAL:
                print(f"[Auto Convergence] measurement failed ({type(_e).__name__}: {_e})", flush=True)
            return self.out if self.out is not None else 0.5
        if target is None:
            return self.out if self.out is not None else 0.5
        if self.ema is None:
            self.ema = target
            self.out = round(self.ema / AUTO_CONV_STEP) * AUTO_CONV_STEP
        else:
            if cut:
                self.ema = target
            else:
                self.ema += AUTO_CONV_EMA * (target - self.ema)
            if cut or abs(self.ema - self.out) >= AUTO_CONV_STEP:
                new = round(self.ema / AUTO_CONV_STEP) * AUTO_CONV_STEP
                # slew limit in screen pixels: at high 3D strength the same slider change moves the image further
                lim = (AUTO_CONV_SLEW_CUT if cut else AUTO_CONV_SLEW) / _spu
                self.out = self.out + min(lim, max(-lim, new - self.out))
        self.out = float(min(1.0, max(0.0, self.out)))
        return self.out


class _MaxOrIntValidator(QIntValidator):
    """Combo input: digits in [lo, hi] or the label 'MAX' (= no limit); typing is case-insensitive."""

    def __init__(self, lo, hi, max_label, parent=None):
        super().__init__(lo, hi, parent)
        self._max_label = max_label.lower()

    def validate(self, text, pos):
        s_ = text.strip().lower()
        if s_ == "":
            return (QValidator.Acceptable, text, pos)
        if s_ == self._max_label:
            return (QValidator.Acceptable, text, pos)
        if self._max_label.startswith(s_):
            return (QValidator.Intermediate, text, pos)
        return super().validate(text, pos)


class ConvergenceSpinBox(QDoubleSpinBox):
    """Convergence spin box: 0.0 ~ 1.0 manual, and the top step (internal value 1.1) is shown as 'Auto'."""

    def textFromValue(self, value: float) -> str:
        if value >= AUTO_CONVERGENCE_THRESHOLD:
            return "Auto"
        return super().textFromValue(value)

    def valueFromText(self, text: str) -> float:
        if text.strip().lower().startswith("a"):
            return self.maximum()
        return super().valueFromText(text)

    def validate(self, text, pos):
        s_ = text.strip().lower()
        if s_ and s_[0].isalpha():
            if s_ == "auto":
                return (QValidator.Acceptable, text, pos)
            if "auto".startswith(s_):
                return (QValidator.Intermediate, text, pos)
            return (QValidator.Invalid, text, pos)
        return super().validate(text, pos)


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

        # Preset row now lives inside the first parameter box (same border), as its top row.
        self.preset_grp = QWidget()
        preset_lay = QHBoxLayout(self.preset_grp)
        preset_lay.setContentsMargins(0, 0, 0, 0)
        self.cmb_preset_name = QComboBox()
        self.cmb_preset_name.setEditable(True)      # one widget: type a name to save, or pick an existing preset
        self.cmb_preset_name.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.cmb_preset_name.setMinimumWidth(240)
        self.btn_preset_save = QPushButton()
        self.btn_preset_save.clicked.connect(self._preset_save)
        self.btn_preset_load = QPushButton()
        self.btn_preset_load.clicked.connect(self._preset_load)
        self.btn_preset_delete = QPushButton()
        self.btn_preset_delete.clicked.connect(self._preset_delete)
        preset_lay.addWidget(self.cmb_preset_name, 1)
        preset_lay.addWidget(self.btn_preset_save)
        preset_lay.addWidget(self.btn_preset_load)
        preset_lay.addWidget(self.btn_preset_delete)

        self.param_grp = QGroupBox()
        param_lay = QGridLayout(self.param_grp)

        self.lbl_format = QLabel()
        self.cmb_format = QComboBox()
        self.cmb_format.addItems(["hsbs", "fsbs", "tb", "ftb", "anaglyph", "half-anaglyph", "dubois-anaglyph"])
        self.cmb_format.setCurrentText("hsbs")

        self.lbl_div = QLabel()
        self.spin_div = QDoubleSpinBox(); self.spin_div.setRange(0.5, DIVERGENCE_MAX); self.spin_div.setSingleStep(0.5); self.spin_div.setValue(1.0); self.spin_div.setDecimals(1)

        self.lbl_conv = QLabel()
        self.spin_conv = ConvergenceSpinBox(); self.spin_conv.setRange(0.0, 1.1); self.spin_conv.setSingleStep(0.1); self.spin_conv.setValue(0.5); self.spin_conv.setDecimals(1)

        self.lbl_edge = QLabel()
        self.spin_edge = QSpinBox(); self.spin_edge.setRange(0, EDGE_FIX_MAX); self.spin_edge.setValue(2); self._edge_manual_value = 2
        # Auto Mode display: "AUTO" only until the first conversion after launch / reset to defaults;
        # afterwards the box shows the number the program applies (_edge_auto_value).
        self._edge_auto_numeric = False
        self._edge_auto_value = 2
        self._edge_in_auto_display = False

        self.lbl_ema = QLabel()
        self._EMA_LEVEL_VALUES = [0.0, 0.30, 0.40, 0.50, 0.60]
        self.spin_ema = QComboBox()
        self.spin_ema.addItems(["OFF", "Low", "Medium", "High", "Ultra"])
        self.spin_ema.setCurrentIndex(0)

        self.lbl_depth_model = QLabel()

        self._depth_model_input_sizes = {
            "zipdepth": ["256", "384", "512", "576"],
            "distill_any_s": ["252", "350", "392", "518", "630"],
            "dav2_s": ["252", "350", "392", "518", "630"],
            "vda_s": ["252", "350", "392", "518", "630"],
            "vda_s_metric": ["252", "350", "392", "518", "630"],
            "da3_mono_l": ["252", "350", "392", "504", "630"],
        }
        self._depth_model_default_size = {
            "zipdepth": "512",
            "distill_any_s": "518",
            "dav2_s": "518",
            "vda_s": "518",
            "vda_s_metric": "518",
            "da3_mono_l": "504",
        }
        self._input_size_by_model = {}      # model key -> Depth Resolution the user chose (only if != default)
        self._cur_model_key = "zipdepth"    # model whose value is currently shown in cmb_input_size
        self.cmb_depth_model = QComboBox()
        self.cmb_depth_model.setEditable(False)
        self.cmb_depth_model.addItem("ZipDepth", "zipdepth")
        self.cmb_depth_model.addItem("DA-V2-S", "dav2_s")
        self.cmb_depth_model.addItem("Distill-Any-S", "distill_any_s")
        self.cmb_depth_model.addItem("VDA-S", "vda_s")
        self.cmb_depth_model.addItem("VDA-S (Metric)", "vda_s_metric")
        self.cmb_depth_model.addItem("DA3-Mono-L", "da3_mono_l")
        self.cmb_depth_model.setCurrentIndex(0)

        self.lbl_input_size = QLabel()
        self.cmb_input_size = QComboBox()
        self.cmb_input_size.addItems(self._depth_model_input_sizes["zipdepth"])
        self.cmb_input_size.setCurrentText(self._depth_model_default_size["zipdepth"])
        self.cmb_input_size.setEditable(True)
        self.cmb_input_size.lineEdit().setValidator(QIntValidator(64, 630, self.cmb_input_size))

        self.cmb_depth_model.currentIndexChanged.connect(self._on_depth_model_changed)
        self._update_depth_model_tooltip()

        self.chk_depth_aa = QCheckBox()
        self.chk_depth_aa.setChecked(False)
        self.chk_depth_aa.toggled.connect(self._on_depth_aa_toggled)

        self.chk_depth_comp = QCheckBox()
        self.chk_depth_comp.setChecked(True)
        self.chk_depth_comp.toggled.connect(self._on_depth_comp_toggled)

        self.lbl_warp_mode = QLabel()
        self._inpaint_res_by_mode = {}      # method key -> Inpaint Resolution the user chose (only if != default)
        self._cur_warp_mode = WARP_FILL_MODE_STRETCH   # method currently selected
        self.cmb_warp_mode = QComboBox()
        self.cmb_warp_mode.setEditable(False)
        self.cmb_warp_mode.addItem("Normal",      WARP_FILL_MODE_STRETCH)
        self.cmb_warp_mode.addItem("Fast Inpaint", WARP_FILL_MODE_FAST)
        self.cmb_warp_mode.addItem("Light Inpaint", WARP_FILL_MODE_LIGHT)
        self.cmb_warp_mode.addItem("Light Inpaint (stream)", WARP_FILL_MODE_LIGHT_STREAM)
        self.cmb_warp_mode.addItem("Medium Inpaint", WARP_FILL_MODE_MEDIUM)
        self.cmb_warp_mode.setCurrentIndex(0)
        self.cmb_warp_mode.currentIndexChanged.connect(self._on_warp_mode_changed)

        # Inpaint resolution (Light Inpaint / Medium Inpaint / Light Inpaint (stream)); read fresh when a
        # conversion job starts. Presets: MAX (= no limit, stored as blank) / 512 / 720 / 1080; a custom value (64-4320) can be typed.
        self.lbl_inpaint_res = QLabel()
        self.cmb_inpaint_res = QComboBox()
        self.cmb_inpaint_res.setEditable(True)
        self.cmb_inpaint_res.setValidator(_MaxOrIntValidator(INPAINT_RES_MIN, INPAINT_RES_MAX, INPAINT_RES_MAX_LABEL, self.cmb_inpaint_res))
        self.cmb_inpaint_res.addItem(INPAINT_RES_MAX_LABEL)
        for _v in INPAINT_RES_PRESETS:
            self.cmb_inpaint_res.addItem(str(_v))
        self._set_inpaint_res(INPAINT_RES_DEFAULT)
        self.cmb_inpaint_res.setEnabled(False)      # starts in Normal: blank + disabled
        self.cmb_inpaint_res.setCurrentText("")
        self.cmb_inpaint_res.activated.connect(self._on_inpaint_res_changed)
        self.cmb_inpaint_res.lineEdit().editingFinished.connect(self._on_inpaint_res_changed)

        self.chk_raw_depthmap = QCheckBox()
        self.chk_corrected_depthmap = QCheckBox()

        self.chk_preserve = QCheckBox()
        self.chk_preserve.setChecked(True)
        self.chk_fp16 = QCheckBox()
        self.chk_fp16.setChecked(True)

        # torch.compile (iw3-style, same as live3d.py): label stays
        # English-only by request. Toggling it runs a real tiny
        # torch.compile() test on this machine; if it fails (missing/broken
        # compiler toolchain), the checkbox is automatically unchecked and
        # the app keeps running normally in eager mode.
        self.chk_compile = QCheckBox("Torch.compile")
        self.chk_compile.setChecked(False)
        self.chk_compile.setToolTip(
            "Uses Torch.compile acceleration for the depth model, Light (including stream) and Medium Inpaint models.\n"
            "For each input size, the model is compiled and warmed up before the conversion starts.\n"
            "Automatically disabled if Torch.compile is not supported on this PC."
        )
        self.chk_compile.toggled.connect(self._on_compile_toggled)

        # TensorRT (same as live3d.py; all depth models: ZipDepth / Distill-Any-S / VDA-S / VDA-S Metric).
        # Toggling runs a quick availability check (tensorrt 10.x importable + CUDA); if it fails
        # the box is unchecked again. Label stays English-only like torch.compile.
        self.chk_trt = QCheckBox("TensorRT")
        self.chk_trt.setChecked(False)
        self.chk_trt.setToolTip(
            "Uses TensorRT acceleration for the depth model, Light (including stream) and Medium Inpaint models.\n"
            "The first run for each input size builds an engine, which can take a few minutes; it is cached afterwards.\n"
            "If Torch.compile is also enabled, TensorRT takes priority.\n"
            "Automatically disabled if TensorRT is not installed."
        )
        self.chk_trt.toggled.connect(self._on_trt_toggled)

        # Hybrid: TensorRT -> depth map model only, torch.compile -> inpaint models.
        self.chk_hybrid = QCheckBox("Hybrid")
        self.chk_hybrid.setChecked(False)
        self.chk_hybrid.setToolTip(
            "Hybrid mode: TensorRT accelerates the depth map model, while Torch.compile accelerates the inpaint models.\n"
            "Requires both Torch.compile and TensorRT."
        )
        self.chk_hybrid.toggled.connect(self._on_hybrid_toggled)

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

        # Image output (still images only): format and JPG quality
        self.lbl_image_format = QLabel()
        self.cmb_image_format = QComboBox()
        self.cmb_image_format.addItems(["", "PNG", "JPG"])      # item 0 text ("Auto") is set per language
        self.cmb_image_format.setItemData(0, "auto")
        self.cmb_image_format.setItemData(1, "png")
        self.cmb_image_format.setItemData(2, "jpg")
        self.cmb_image_format.setCurrentIndex(0)
        self.lbl_image_quality = QLabel()
        self.spin_image_quality = QSpinBox()
        self.spin_image_quality.setRange(1, 100)
        self.spin_image_quality.setValue(95)

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
        self.chk_sound_boost = QCheckBox()
        # Schedule a PC shutdown (5 min) when ALL conversions have completed. Never saved to the config: it always
        # starts unchecked. Unchecking it (even during a conversion) cancels a pending shutdown.
        self.chk_shutdown = QCheckBox()
        self.chk_shutdown.setChecked(False)
        self._shutdown_scheduled = False
        self._shutdown_box = None
        self._batch_had_error = False
        self._last_batch_ok = False     # True after a batch finished normally (until the next start)
        if sys.platform != "win32":
            self.chk_shutdown.setEnabled(False)
        self.chk_shutdown.toggled.connect(self._on_shutdown_toggled)

        # Make the numeric spin boxes (3D strength / convergence / edge) and the encoding row (codec / preset / quality) as tall as the combo boxes (e.g. depth resolution).
        _row_h = max(self.cmb_input_size.sizeHint().height(), self.cmb_depth_model.sizeHint().height())
        for _w in (self.spin_div, self.spin_conv, self.spin_edge,
                   self.cmb_vcodec, self.cmb_preset, self.spin_quality,
                   self.cmb_image_format, self.spin_image_quality):
            _w.setMinimumHeight(_row_h)

        r = 0
        param_lay.addWidget(self.preset_grp, r, 0, 1, 2); r += 1
        param_lay.setRowMinimumHeight(r, 10)  # small gap between the preset row and Auto Mode
        r += 1
        param_lay.addWidget(self.chk_auto_mode, r, 0, 1, 2); r += 1
        param_lay.addWidget(self.lbl_format, r, 0); param_lay.addWidget(self.cmb_format, r, 1); r += 1
        param_lay.addWidget(self.lbl_div, r, 0); param_lay.addWidget(self.spin_div, r, 1); r += 1
        param_lay.addWidget(self.lbl_conv, r, 0); param_lay.addWidget(self.spin_conv, r, 1); r += 1
        param_lay.addWidget(self.lbl_edge, r, 0); param_lay.addWidget(self.spin_edge, r, 1); r += 1
        param_lay.addWidget(self.lbl_ema, r, 0); param_lay.addWidget(self.spin_ema, r, 1); r += 1
        param_lay.addWidget(self.lbl_depth_model, r, 0); param_lay.addWidget(self.cmb_depth_model, r, 1); r += 1
        param_lay.addWidget(self.lbl_input_size, r, 0); param_lay.addWidget(self.cmb_input_size, r, 1); r += 1
        param_lay.addWidget(self.lbl_warp_mode, r, 0); param_lay.addWidget(self.cmb_warp_mode, r, 1); r += 1
        param_lay.addWidget(self.lbl_inpaint_res, r, 0); param_lay.addWidget(self.cmb_inpaint_res, r, 1); r += 1
        # One row (same order as live3d): [Screen Border Protection] [Depth Linear Compression] [Depth AA]
        aa_row_lay = QHBoxLayout()
        aa_row_lay.setSpacing(12)
        aa_row_lay.addWidget(self.chk_preserve)
        aa_row_lay.addWidget(self.chk_depth_comp)
        aa_row_lay.addWidget(self.chk_depth_aa)
        aa_row_lay.addStretch(1)
        param_lay.addLayout(aa_row_lay, r, 0, 1, 2); r += 1
        # Second untitled bordered group box (same look as the Preset box): everything below Depth AA
        self.param_grp2 = QGroupBox()
        param_lay2 = QGridLayout(self.param_grp2)
        r = 0
        depthmap_opt_lay = QHBoxLayout()
        depthmap_opt_lay.addWidget(self.chk_raw_depthmap)
        depthmap_opt_lay.addWidget(self.chk_corrected_depthmap)
        depthmap_opt_lay.addStretch(1)
        param_lay2.addLayout(depthmap_opt_lay, r, 0, 1, 2); r += 1

        preserve_lay = QHBoxLayout()
        preserve_lay.addWidget(self.chk_fp16)
        preserve_lay.addWidget(self.chk_compile)
        preserve_lay.addWidget(self.chk_trt)
        preserve_lay.addWidget(self.chk_hybrid)
        preserve_lay.addStretch(1)
        param_lay2.addLayout(preserve_lay, r, 0, 1, 2); r += 1

        # Codec row and image row share one grid so their columns line up:
        # the image format box ends where the preset box ends, and the image quality box sits
        # exactly under the codec quality box (same left edge, same width).
        encode_lay = QGridLayout()
        encode_lay.setContentsMargins(0, 0, 0, 0)
        encode_lay.addWidget(self.lbl_vcodec, 0, 0)
        encode_lay.addWidget(self.cmb_vcodec, 0, 1)
        encode_lay.addWidget(self.lbl_preset, 0, 2)
        encode_lay.addWidget(self.cmb_preset, 0, 3)
        encode_lay.addWidget(self.lbl_quality, 0, 4)
        encode_lay.addWidget(self.spin_quality, 0, 5)
        encode_lay.addWidget(self.lbl_image_format, 1, 0)
        encode_lay.addWidget(self.cmb_image_format, 1, 1, 1, 3)
        encode_lay.addWidget(self.lbl_image_quality, 1, 4)
        encode_lay.addWidget(self.spin_image_quality, 1, 5)
        encode_lay.setColumnStretch(3, 1)
        encode_lay.setColumnStretch(5, 1)
        param_lay2.addLayout(encode_lay, r, 0, 1, 2); r += 1

        resize_lay = QHBoxLayout()
        resize_lay.addWidget(self.chk_resize)
        resize_lay.addWidget(self.edt_resize)
        resize_lay.addWidget(self.chk_hdr_norm)
        resize_lay.addStretch(1)
        param_lay2.addLayout(resize_lay, r, 0, 1, 2); r += 1

        crop_lay = QHBoxLayout()
        crop_lay.addWidget(self.chk_auto_crop)
        crop_lay.addWidget(self.chk_pad_169)
        crop_lay.addWidget(self.chk_sound_boost)
        crop_lay.addStretch(1)
        param_lay2.addLayout(crop_lay, r, 0, 1, 2); r += 1

        shutdown_lay = QHBoxLayout()
        shutdown_lay.addWidget(self.chk_shutdown)
        shutdown_lay.addStretch(1)
        param_lay2.addLayout(shutdown_lay, r, 0, 1, 2); r += 1

        self.btn_reset_settings = QPushButton()
        self.btn_reset_settings.clicked.connect(self._reset_settings)
        # Separate bordered group box (same look as the other two) so the reset button is visually split off.
        self.param_grp3 = QGroupBox()
        reset_lay = QHBoxLayout(self.param_grp3)
        reset_lay.addWidget(self.btn_reset_settings)
        reset_lay.addStretch(1)

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
        content_lay.addWidget(self.param_grp2)
        content_lay.addWidget(self.param_grp3)

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

        self.param_grp.setTitle("")
        self.cmb_preset_name.lineEdit().setPlaceholderText(t["preset_name"])
        self.btn_preset_save.setText(t["preset_save"])
        self.btn_preset_load.setText(t["preset_load"])
        self.btn_preset_delete.setText(t["preset_delete"])

        self.lbl_format.setText(t["fmt_label"])
        self.lbl_div.setText(t["div_label"])
        self.lbl_conv.setText(t["conv_label"])
        self.lbl_edge.setText(t["edge_label"])
        self.lbl_ema.setText(t["ema_label"])
        self.lbl_depth_model.setText(t["depth_model_label"])
        self._update_depth_model_tooltip()
        self.lbl_input_size.setText(t["size_label"])
        self.chk_depth_aa.setText(t["depth_aa"])
        self.chk_depth_comp.setText(t["depth_linear_comp"])
        self.chk_depth_comp.setToolTip(t.get("depth_linear_comp_tooltip", ""))
        self.lbl_warp_mode.setText(t.get("warp_mode_label", "Method"))
        self.cmb_warp_mode.setToolTip("")
        self.lbl_inpaint_res.setText(t.get("inpaint_res", "Inpaint Resolution"))
        self.cmb_inpaint_res.setToolTip(t.get("inpaint_res_tooltip", ""))
        self.chk_compile.setToolTip(t.get("compile_tooltip", ""))
        self.chk_trt.setToolTip(t.get("trt_tooltip", ""))
        self.chk_hybrid.setToolTip(t.get("hybrid_tooltip", ""))
        self.chk_raw_depthmap.setText(t["chk_raw_depthmap"])
        self.chk_corrected_depthmap.setText(t["chk_corrected_depthmap"])

        self.chk_preserve.setText(t["chk_preserve"])
        self.chk_fp16.setText(t["chk_fp16"])
        self.chk_auto_mode.setText(t["chk_auto_mode"])

        self.lbl_vcodec.setText(t["vcodec_label"])
        self.lbl_preset.setText(t["preset_label"])
        self.lbl_image_format.setText(t["image_format_label"])
        self.lbl_image_quality.setText(t["image_quality_label"])
        self.cmb_image_format.setItemText(0, t["image_format_auto"])
        self.cmb_image_format.setToolTip(t["image_format_tip"])
        self.spin_image_quality.setToolTip(t["image_quality_tip"])

        codec = self.cmb_vcodec.currentText()
        self.lbl_quality.setText(t["quality_cq"] if "nvenc" in codec else t["quality_crf"])

        self.chk_resize.setText(t["chk_resize"])
        self.chk_hdr_norm.setText(t["chk_hdr_norm"])
        self.chk_auto_crop.setText(t["chk_auto_crop"])
        self.chk_pad_169.setText(t["chk_pad_169"])
        self.chk_sound_boost.setText(t["chk_sound_boost"])
        self.chk_shutdown.setText(t["chk_shutdown"])
        self.chk_shutdown.setToolTip(t["chk_shutdown_tip"])
        self.btn_reset_settings.setText(t["btn_reset_settings"])

        self.chk_start_time.setText(t["chk_start_time"])
        self.chk_end_time.setText(t["chk_end_time"])

        self.btn_start.setText(t["btn_start"])
        self.btn_stop.setText(t["btn_stop"])

        if self.worker is None or not self.worker.isRunning():
            cur = self.lbl_status.text()
            if cur in ["Idle", ""]:
                self.lbl_status.setText(t["status_idle"])

    def _edge_display_auto(self, auto: bool):
        """Auto Mode: the Edge Fix box shows AUTO until the first conversion after launch / reset to defaults,
        then the Edge Fix number the program applies. Manual: it shows the user's number."""
        sp = self.spin_edge
        sp.blockSignals(True)
        try:
            if auto:
                # Remember the manual number only when switching over from the manual display
                # (never overwrite it with an auto number that is already shown).
                if not getattr(self, "_edge_in_auto_display", False) and sp.value() >= 0:
                    self._edge_manual_value = int(sp.value())
                if getattr(self, "_edge_auto_numeric", False):
                    sp.setSpecialValueText("")
                    sp.setRange(0, EDGE_FIX_MAX)
                    sp.setValue(int(getattr(self, "_edge_auto_value", 0)))
                else:
                    sp.setRange(-1, EDGE_FIX_MAX)
                    sp.setSpecialValueText("Auto")
                    sp.setValue(-1)
                self._edge_in_auto_display = True
            else:
                # Leaving Auto Mode after an Edge Fix number was measured/applied: that number becomes the manual value.
                if getattr(self, "_edge_in_auto_display", False) and getattr(self, "_edge_auto_numeric", False):
                    self._edge_manual_value = max(0, min(EDGE_FIX_MAX, int(getattr(self, "_edge_auto_value", 0))))
                sp.setSpecialValueText("")
                sp.setRange(0, EDGE_FIX_MAX)
                sp.setValue(int(getattr(self, "_edge_manual_value", 2)))
                self._edge_in_auto_display = False
        finally:
            sp.blockSignals(False)

    def _edge_manual(self) -> int:
        """The manual Edge Fix number (also while the box shows AUTO or the auto number)."""
        if getattr(self, "_edge_in_auto_display", False):
            return int(getattr(self, "_edge_manual_value", 2))
        v = int(self.spin_edge.value())
        return v if v >= 0 else int(getattr(self, "_edge_manual_value", 2))

    def _on_auto_edge_applied(self, level: int):
        """From the worker: the Auto Edge Fix level applied to the conversion that just started."""
        self._edge_auto_numeric = True
        self._edge_auto_value = max(0, min(EDGE_FIX_MAX, int(level)))
        if self.chk_auto_mode.isChecked():
            self._edge_display_auto(True)

    def _update_auto_mode_state(self):
        auto_enabled = self.chk_auto_mode.isChecked()
        self._edge_display_auto(auto_enabled)

        self.spin_edge.setEnabled(not auto_enabled)
        self.spin_ema.setEnabled(not auto_enabled)

        if auto_enabled:
            self._apply_auto_mode_values()
        elif hasattr(self, "chk_depth_aa"):
            self.chk_depth_aa.setEnabled(True)
            if hasattr(self, "chk_depth_comp"):
                self.chk_depth_comp.setEnabled(True)

    def _hybrid_follow(self, checked: bool):
        # Hybrid needs both torch.compile and TensorRT: turning either off turns Hybrid off too.
        if (not checked) and hasattr(self, "chk_hybrid") and self.chk_hybrid.isChecked():
            self.chk_hybrid.blockSignals(True)
            self.chk_hybrid.setChecked(False)
            self.chk_hybrid.blockSignals(False)

    def _on_hybrid_toggled(self, checked: bool):
        # Hybrid = TensorRT for the depth map model only; torch.compile for inpaint models.
        # Both must be supported here, otherwise the box is unchecked again and the user is told which one is missing.
        if checked:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ok_compile = bool(check_compile_support(device))
            ok_trt = bool(check_tensorrt_support())
            if not (ok_compile and ok_trt):
                self.chk_hybrid.blockSignals(True)
                self.chk_hybrid.setChecked(False)
                self.chk_hybrid.blockSignals(False)
                key = ("hybrid_unsupported_both" if (not ok_compile and not ok_trt)
                       else ("hybrid_unsupported_compile" if not ok_compile else "hybrid_unsupported_trt"))
                QMessageBox.information(self, "DepthLive3D", self._tr(key))
                return
            for _chk in (self.chk_compile, self.chk_trt):
                if not _chk.isChecked():
                    _chk.blockSignals(True)
                    _chk.setChecked(True)
                    _chk.blockSignals(False)

    def _on_compile_toggled(self, checked: bool):
        self._hybrid_follow(checked)
        # IW3-style behavior (same as live3d.py): when the user checks the
        # box, actually try a tiny real torch.compile() on this machine
        # right now. If it works, keep the box checked (ON). If it fails
        # for any reason (no triton-windows, broken MSVC toolchain, etc.),
        # uncheck it again so conversion simply runs in normal (eager) mode
        # without crashing.
        if checked:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if not check_compile_support(device):
                self.chk_compile.blockSignals(True)
                self.chk_compile.setChecked(False)
                self.chk_compile.blockSignals(False)
                QMessageBox.information(self, "DepthLive3D", self._tr("compile_unsupported"))

    def _on_trt_toggled(self, checked: bool):
        self._hybrid_follow(checked)
        # Same behavior as live3d.py: when checked, verify that TensorRT 10.x + CUDA are usable.
        # If not, uncheck again so conversion simply runs on PyTorch without crashing.
        if checked:
            if not check_tensorrt_support():
                self.chk_trt.blockSignals(True)
                self.chk_trt.setChecked(False)
                self.chk_trt.blockSignals(False)
                QMessageBox.information(self, "DepthLive3D", self._tr("trt_unsupported"))

    def _is_vda_model(self) -> bool:
        key = self.cmb_depth_model.currentData() if hasattr(self, "cmb_depth_model") else ""
        return key in ("vda_s", "vda_s_metric")

    def _apply_auto_mode_values(self):
        div = self.spin_div.value()
        model_key = self.cmb_depth_model.currentData() if hasattr(self, "cmb_depth_model") else None
        warp_mode_key = self.cmb_warp_mode.currentData() if hasattr(self, "cmb_warp_mode") else None
        _comp_on = bool(self.chk_depth_comp.isChecked()) if hasattr(self, "chk_depth_comp") else False
        auto_edge, auto_ema = compute_auto_edge_ema(div, depth_model=model_key, warp_fill_mode=warp_mode_key,
                                                    depth_linear_comp=_comp_on)
        if getattr(self, "_edge_auto_numeric", False):
            self._edge_auto_value = max(0, min(EDGE_FIX_MAX, int(auto_edge)))   # measured profile (else table) value
        self._edge_display_auto(True)      # AUTO before the first conversion, the number afterwards
        self.spin_ema.blockSignals(True)
        self._set_ema_value(auto_ema)
        self.spin_ema.blockSignals(False)

        if hasattr(self, "chk_depth_aa"):
            # Auto Mode: Depth AA is ON for VDA-S / VDA-S (Metric) / DA-V2-S / DA3-Mono-L, OFF otherwise (ZipDepth, Distill-Any-S).
            want_aa = model_key in ("vda_s", "vda_s_metric", "dav2_s", "da3_mono_l")
            self.chk_depth_aa.blockSignals(True)
            self.chk_depth_aa.setChecked(want_aa)
            self.chk_depth_aa.setEnabled(False)
            self.chk_depth_aa.blockSignals(False)

        if hasattr(self, "chk_depth_comp"):
            # Depth Linear Compression is a free user option (like Screen Border Protection): Auto Mode neither
            # checks/unchecks it nor disables it. It only changes the Edge Fix / EMA values computed above.
            self.chk_depth_comp.setEnabled(True)

    def _on_div_changed(self):
        if self.chk_auto_mode.isChecked():
            self._apply_auto_mode_values()

    def _set_inpaint_res(self, value):
        self.cmb_inpaint_res.setCurrentText(inpaint_res_display(value))

    def _inpaint_res_text(self) -> str:
        return inpaint_res_text(self.cmb_inpaint_res.currentText())

    def _on_inpaint_res_changed(self, *_args):
        if not self.cmb_inpaint_res.isEnabled():
            return          # Normal / Fast Inpaint: blank + disabled
        # Normalise typed input (blank / MAX = no limit, otherwise 64..4320). Read fresh when a conversion starts.
        self._set_inpaint_res(self.cmb_inpaint_res.currentText())

    def _apply_inpaint_res_mode_default(self):
        if not hasattr(self, "cmb_inpaint_res"):
            return
        _mode = self.cmb_warp_mode.currentData()
        d = inpaint_res_default_for_mode(_mode)
        if d is not None:
            # the value the user set earlier for this method, or the default if never changed
            self._set_inpaint_res(self._inpaint_res_by_mode.get(_mode, d))

    def _on_warp_mode_changed(self, _index=None):
        # every inpaint method defaults to MAX (no limit) unless the user changed it for that method
        self._remember_inpaint_res(getattr(self, "_cur_warp_mode", None))
        self._apply_inpaint_res_mode_default()
        self._cur_warp_mode = self.cmb_warp_mode.currentData()
        self._update_inpaint_res_enabled()
        if self.chk_auto_mode.isChecked():
            self._apply_auto_mode_values()

    def _on_depth_aa_toggled(self, checked: bool):
        _model_key = self.cmb_depth_model.currentData() if hasattr(self, "cmb_depth_model") else None
        if (
            not checked
            and hasattr(self, "chk_auto_mode")
            and self.chk_auto_mode.isChecked()
            and _model_key in ("vda_s", "vda_s_metric", "dav2_s", "da3_mono_l")
        ):
            self.chk_depth_aa.blockSignals(True)
            self.chk_depth_aa.setChecked(True)
            self.chk_depth_aa.blockSignals(False)
            self._show_message_box(
                QMessageBox.Information,
                self._tr("depth_aa_auto_disable_title"),
                self._tr("depth_aa_auto_disable_msg"),
            )
            return

        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    self._show_message_box(
                        QMessageBox.Information,
                        self._tr("depth_aa_restart_title"),
                        self._tr("depth_aa_restart_msg"),
                    )
            except RuntimeError:
                pass

    def _on_depth_comp_toggled(self, checked: bool):
        if hasattr(self, "chk_auto_mode") and self.chk_auto_mode.isChecked():
            self._apply_auto_mode_values()

    # ---- Remember user-changed Depth Resolution (per depth model) / Inpaint Resolution (per method) ----
    # Only values the user changed away from the program default are stored (and saved to the config on exit),
    # so a model / method that was never touched keeps showing its default.
    @staticmethod
    def _sanitize_input_size_map(d, valid_keys):
        out = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if k in valid_keys:
                    try:
                        out[k] = str(max(64, min(630, int(str(v).strip()))))
                    except (TypeError, ValueError):
                        pass
        return out

    def _remember_input_size(self, model_key=None):
        key = model_key if model_key is not None else getattr(self, "_cur_model_key", None)
        if not key or not hasattr(self, "_input_size_by_model"):
            return
        try:
            v = max(64, min(630, int(self.cmb_input_size.currentText().strip())))
        except (TypeError, ValueError):
            return
        if str(v) != str(self._depth_model_default_size.get(key, "")):
            self._input_size_by_model[key] = str(v)
        else:
            self._input_size_by_model.pop(key, None)

    def _input_size_map_snapshot(self) -> dict:
        self._remember_input_size()
        return dict(self._input_size_by_model)

    def _remember_inpaint_res(self, mode=None):
        mode = mode if mode is not None else getattr(self, "_cur_warp_mode", None)
        if mode is None or not hasattr(self, "_inpaint_res_by_mode"):
            return
        d = self._inpaint_default_for_mode(mode)
        if d is None:
            return                                   # this method does not use the setting
        txt = inpaint_res_text(self.cmb_inpaint_res.currentText())
        if txt != inpaint_res_text(d):
            self._inpaint_res_by_mode[mode] = txt
        else:
            self._inpaint_res_by_mode.pop(mode, None)

    def _sanitize_inpaint_res_map(self, d) -> dict:
        out = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if self._inpaint_default_for_mode(k) is None:
                    continue
                s_ = "" if v is None else str(v).strip()
                if s_ == "" or s_.isdigit():
                    out[k] = inpaint_res_text(s_)
        return out

    def _inpaint_res_map_snapshot(self) -> dict:
        self._remember_inpaint_res()
        return dict(self._inpaint_res_by_mode)

    @staticmethod
    def _inpaint_default_for_mode(mode):
        return inpaint_res_default_for_mode(mode)

    def _update_inpaint_res_enabled(self):
        # Normal / Fast Inpaint do not use the Inpaint Resolution: show it blank and disabled.
        # Every other inpaint method: enabled.
        if not hasattr(self, "cmb_inpaint_res") or not hasattr(self, "cmb_warp_mode"):
            return
        _active = self._inpaint_default_for_mode(self.cmb_warp_mode.currentData()) is not None
        self.cmb_inpaint_res.setEnabled(_active)
        if hasattr(self, "lbl_inpaint_res"):
            self.lbl_inpaint_res.setEnabled(_active)
        if not _active:
            self.cmb_inpaint_res.setCurrentText("")

    def _round_input_size(self, val: int) -> int:
        model_key = self.cmb_depth_model.currentData() if hasattr(self, "cmb_depth_model") else "zipdepth"
        if model_key in ("vda_s", "vda_s_metric", "distill_any_s", "dav2_s", "da3_mono_l"):

            return max(70, int(round(val / 14.0) * 14))
        _r = max(64, int(round(val / 32.0) * 32))
        if _r > 630:
            _r = 608
        return _r

    def _update_depth_model_tooltip(self):
        _tip = ""  # tooltip removed: nothing is shown on the depth model combo / its list items
        _cur = self.cmb_depth_model.currentData()
        self.cmb_depth_model.setToolTip(_tip if _cur == "vda_s_metric" else "")
        _i = self.cmb_depth_model.findData("vda_s_metric")
        if _i >= 0:
            self.cmb_depth_model.setItemData(_i, _tip, Qt.ItemDataRole.ToolTipRole)

    def _on_depth_model_changed(self, _index=None):
        self._update_depth_model_tooltip()

        model_key = self.cmb_depth_model.currentData() or "zipdepth"
        sizes = self._depth_model_input_sizes.get(model_key, self._depth_model_input_sizes["zipdepth"])
        default_size = self._depth_model_default_size.get(model_key, sizes[0])

        # Keep the value the user had set for the model we are leaving, and show the remembered value
        # (or the default, if the user never changed it) for the model we are entering.
        self._remember_input_size(getattr(self, "_cur_model_key", None))
        self.cmb_input_size.blockSignals(True)
        self.cmb_input_size.clear()
        self.cmb_input_size.addItems(sizes)
        self.cmb_input_size.setCurrentText(self._input_size_by_model.get(model_key, default_size))
        self.cmb_input_size.blockSignals(False)
        self._cur_model_key = model_key

        if hasattr(self, "chk_auto_mode") and self.chk_auto_mode.isChecked():
            self._apply_auto_mode_values()
        elif hasattr(self, "chk_depth_aa"):
            self.chk_depth_aa.setEnabled(True)
            if hasattr(self, "chk_depth_comp"):
                self.chk_depth_comp.setEnabled(True)

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
            "edge_dilation": self._edge_manual(),
            "ema_decay": self._ema_value(),
            "depth_model": self.cmb_depth_model.currentData(),
            "input_size": self.cmb_input_size.currentText(),
            "input_size_by_model": self._input_size_map_snapshot(),
            "inpaint_res_by_mode": self._inpaint_res_map_snapshot(),
            "depth_aa": self.chk_depth_aa.isChecked(),
            "depth_linear_comp": self.chk_depth_comp.isChecked(),
            "fill_mode": self.cmb_warp_mode.currentData() or WARP_FILL_MODE_STRETCH,
            "inpaint_res": self._inpaint_res_text(),
            "inpaint_hole_threshold": 1.5,
            "preserve_border": self.chk_preserve.isChecked(),
            "invert_depth": True,
            "fp16": self.chk_fp16.isChecked(),
            "compile": self.chk_compile.isChecked(),
            "tensorrt": self.chk_trt.isChecked(),
            "hybrid": self.chk_hybrid.isChecked(),
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
            "image_format": self.cmb_image_format.currentData() or "auto",
            "image_quality": self.spin_image_quality.value(),
            "resize": self.chk_resize.isChecked(),
            "resize_value": self.edt_resize.text().strip(),
            "hdr_norm": self.chk_hdr_norm.isChecked(),
            "auto_crop": self.chk_auto_crop.isChecked(),
            "pad_169": self.chk_pad_169.isChecked(),
            "sound_boost": self.chk_sound_boost.isChecked(),
            "preset_name": self.cmb_preset_name.currentText(),
        }

    def _apply_config(self, cfg: dict):
        # Restore the auto-mode checkbox FIRST (signals blocked). Otherwise the default
        # (checked) state stays active while divergence / model / warp-mode are restored,
        # and their change handlers overwrite the saved manual edge/EMA values.
        self.chk_auto_mode.blockSignals(True)
        self.chk_auto_mode.setChecked(cfg.get("auto_mode", True))
        self.chk_auto_mode.blockSignals(False)

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

        _saved_depth_model = cfg.get("depth_model", "zipdepth")
        _dm_idx = self.cmb_depth_model.findData(_saved_depth_model)
        self._input_size_by_model = self._sanitize_input_size_map(
            cfg.get("input_size_by_model"), self._depth_model_default_size.keys())
        self._cur_model_key = None          # do not let the old model's text leak into the new map
        self.cmb_depth_model.blockSignals(True)
        self.cmb_depth_model.setCurrentIndex(_dm_idx if _dm_idx >= 0 else 0)
        self.cmb_depth_model.blockSignals(False)
        self._on_depth_model_changed()

        try:
            _isz_fallback = self._depth_model_default_size.get(
                _saved_depth_model if _dm_idx >= 0 else "zipdepth", "384"
            )
            _loaded_isz = max(64, min(630, int(cfg.get("input_size", _isz_fallback))))
        except (TypeError, ValueError):
            _loaded_isz = int(self._depth_model_default_size.get(
                _saved_depth_model if _dm_idx >= 0 else "zipdepth", "384"
            ))
        self.cmb_input_size.setCurrentText(str(_loaded_isz))
        self.chk_depth_aa.blockSignals(True)
        self.chk_depth_aa.setChecked(cfg.get("depth_aa", False))
        self.chk_depth_aa.blockSignals(False)
        self.chk_depth_comp.blockSignals(True)
        self.chk_depth_comp.setChecked(bool(cfg.get("depth_linear_comp", True)))
        self.chk_depth_comp.blockSignals(False)
        _saved_warp_mode = cfg.get("fill_mode", WARP_FILL_MODE_STRETCH)
        _wm_idx = self.cmb_warp_mode.findData(_saved_warp_mode)
        self._inpaint_res_by_mode = self._sanitize_inpaint_res_map(cfg.get("inpaint_res_by_mode"))
        self._cur_warp_mode = None          # do not let the old method's text leak into the new map
        self.cmb_warp_mode.setCurrentIndex(_wm_idx if _wm_idx >= 0 else 0)
        self._cur_warp_mode = self.cmb_warp_mode.currentData()
        _ipr_cfg = cfg.get("inpaint_res", None)
        if _ipr_cfg is not None and str(_ipr_cfg).strip() == "" or (
                _ipr_cfg is not None and str(_ipr_cfg).strip().isdigit()):
            self._set_inpaint_res(str(_ipr_cfg).strip())
        else:   # missing / old value: use the selected mode's default
            self._apply_inpaint_res_mode_default()
        self._update_inpaint_res_enabled()

        self.chk_preserve.setChecked(cfg.get("preserve_border", True))
        self.chk_fp16.setChecked(cfg.get("fp16", True))
        self.chk_compile.setChecked(cfg.get("compile", False))
        self.chk_trt.blockSignals(True)
        self.chk_trt.setChecked(bool(cfg.get("tensorrt", False)))
        self.chk_trt.blockSignals(False)
        self.chk_hybrid.blockSignals(True)
        self.chk_hybrid.setChecked(bool(cfg.get("hybrid", False)) and self.chk_compile.isChecked()
                                   and self.chk_trt.isChecked())
        self.chk_hybrid.blockSignals(False)

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
        _imgf = str(cfg.get("image_format", "auto") or "auto").lower()
        _imgi = self.cmb_image_format.findData(_imgf)
        self.cmb_image_format.setCurrentIndex(_imgi if _imgi >= 0 else 0)
        try:
            self.spin_image_quality.setValue(int(cfg.get("image_quality", 95)))
        except (TypeError, ValueError):
            self.spin_image_quality.setValue(95)
        self.edt_resize.setText(cfg.get("resize_value", "1920x1080") or "1920x1080")
        self.chk_resize.setChecked(cfg.get("resize", False))
        self.edt_resize.setEnabled(self.chk_resize.isChecked())
        self.chk_hdr_norm.setChecked(cfg.get("hdr_norm", False))
        self.chk_auto_crop.setChecked(cfg.get("auto_crop", False))
        self.chk_pad_169.setChecked(cfg.get("pad_169", False))
        self.chk_sound_boost.setChecked(cfg.get("sound_boost", False))

        # Restore manual edge dilation / EMA last. In auto mode they are recomputed
        # by _apply_auto_mode_values(), so only restore the saved values when it's off.
        self._edge_manual_value = int(cfg.get("edge_dilation", 2))
        if not self.chk_auto_mode.isChecked():
            self.spin_edge.setValue(cfg.get("edge_dilation", 2))
            self._set_ema_value(cfg.get("ema_decay", 0.0))

        # Restore the preset selection only (the values themselves are restored above). Blank / deleted preset -> blank.
        _pn = cfg.get("preset_name", "")
        if _pn and _pn in self.presets:
            self._select_preset_name(_pn)
        else:
            self._clear_preset_name()

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

    def _default_presets(self) -> dict:
        return {
            "Best Quality": {
                "divergence": 3.0,
                "convergence": 0.5,
                "edge_dilation": 1,
                "ema_decay": 0.0,
                "depth_model": "vda_s",
                "input_size": "630",
                "depth_aa": False,
                "depth_linear_comp": True,
                "preserve_border": True,
                "fill_mode": WARP_FILL_MODE_MEDIUM,
                "inpaint_res": "",
            },
        }

    def _load_presets(self):
        self.presets = {}
        if not PRESET_FILE.exists():
            self.presets = self._default_presets()
            self._save_presets_quiet()
            return
        if PRESET_FILE.exists():
            try:
                data = json.loads(PRESET_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.presets = data
            except Exception:
                self.presets = {}

    def _save_presets_quiet(self):
        try:
            PRESET_FILE.write_text(json.dumps(self.presets, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _save_presets(self):
        try:
            PRESET_FILE.write_text(json.dumps(self.presets, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self._append_log(f"[Warning] Failed to save preset: {e}")

    def _refresh_preset_combo(self):
        current = self._preset_name_text()
        self.cmb_preset_name.blockSignals(True)
        self.cmb_preset_name.clear()
        self.cmb_preset_name.addItem("")    # blank entry = no preset selected (default)
        self.cmb_preset_name.addItems(sorted(self.presets.keys()))
        if current in self.presets:
            self._select_preset_name(current)
        else:
            self._clear_preset_name()
        self.cmb_preset_name.blockSignals(False)

    def _preset_name_text(self) -> str:
        return self.cmb_preset_name.currentText().strip()

    def _select_preset_name(self, name: str):
        i = self.cmb_preset_name.findText(name)
        if i >= 0:
            self.cmb_preset_name.setCurrentIndex(i)
        else:
            self.cmb_preset_name.setEditText(name)

    def _clear_preset_name(self):
        # Back to the blank entry (index 0) = no preset selected
        self.cmb_preset_name.setCurrentIndex(0 if self.cmb_preset_name.count() > 0 else -1)
        self.cmb_preset_name.setEditText("")

    def _preset_save(self):
        name = self._preset_name_text()
        if not name:
            return
        self.presets[name] = {
            "divergence": self.spin_div.value(),
            "convergence": self.spin_conv.value(),
            "edge_dilation": self._edge_manual(),
            "ema_decay": self._ema_value(),
            "depth_model": self.cmb_depth_model.currentData(),
            "input_size": self.cmb_input_size.currentText(),
            "depth_aa": self.chk_depth_aa.isChecked(),
            "depth_linear_comp": self.chk_depth_comp.isChecked(),
            "preserve_border": self.chk_preserve.isChecked(),
            "fill_mode": self.cmb_warp_mode.currentData() or WARP_FILL_MODE_STRETCH,
            "inpaint_res": self._inpaint_res_text(),
        }
        self._save_presets()
        self._refresh_preset_combo()
        self._select_preset_name(name)
        self._append_log(self._tr("preset_saved") + f" : {name}")

    def _preset_load(self):
        name = self._preset_name_text()
        if not name:
            return
        if name not in self.presets:
            return
        # Loading a preset applies its saved values as-is, so Auto Mode is switched off first
        # (this also re-enables the edge / EMA / depth AA / linear-compression widgets).
        if self.chk_auto_mode.isChecked():
            self.chk_auto_mode.setChecked(False)
            self._append_log(self._tr("preset_auto_mode_off"))
        p = self.presets[name]
        self.spin_div.setValue(p.get("divergence", self.spin_div.value()))
        self.spin_conv.setValue(p.get("convergence", self.spin_conv.value()))
        self.spin_edge.setValue(p.get("edge_dilation", self._edge_manual()))
        self._set_ema_value(p.get("ema_decay", self._ema_value()))

        # Depth model (+ its input size). Changing the model resets the input-size list, so restore it right after.
        if "depth_model" in p:
            _dm_idx = self.cmb_depth_model.findData(p.get("depth_model"))
            if _dm_idx >= 0:
                self.cmb_depth_model.blockSignals(True)
                self.cmb_depth_model.setCurrentIndex(_dm_idx)
                self.cmb_depth_model.blockSignals(False)
                self._on_depth_model_changed()
        if "input_size" in p:
            try:
                _isz = max(64, min(630, int(p.get("input_size"))))
                self.cmb_input_size.setCurrentText(str(_isz))
            except (TypeError, ValueError):
                pass
        if "depth_aa" in p:
            self.chk_depth_aa.blockSignals(True)
            self.chk_depth_aa.setChecked(bool(p.get("depth_aa")))
            self.chk_depth_aa.blockSignals(False)
        if "depth_linear_comp" in p:
            self.chk_depth_comp.blockSignals(True)
            self.chk_depth_comp.setChecked(bool(p.get("depth_linear_comp")))
            self.chk_depth_comp.blockSignals(False)
        if "preserve_border" in p:
            self.chk_preserve.setChecked(bool(p.get("preserve_border")))
        # Method (warp fill mode); its change handler resets the inpaint resolution, so restore that afterwards.
        if "fill_mode" in p:
            _wm_idx = self.cmb_warp_mode.findData(p.get("fill_mode"))
            if _wm_idx >= 0:
                self.cmb_warp_mode.setCurrentIndex(_wm_idx)
        if "inpaint_res" in p:
            _ipr = str(p.get("inpaint_res", "")).strip()
            if _ipr == "" or _ipr.isdigit():
                self._set_inpaint_res(_ipr)
        self._update_inpaint_res_enabled()
        self._append_log(self._tr("preset_loaded") + f" : {name}")

    def _reset_settings(self):
        """Restore every setting to the program's first-run defaults.
        User presets are kept; only the built-in 'Best Quality' preset is recreated.
        Input / output paths are left as they are."""
        box = QMessageBox(QMessageBox.Question, self._tr("reset_confirm_title"),
                          self._tr("reset_confirm_msg"), QMessageBox.NoButton, self)
        btn_yes = box.addButton(self._tr("reset_yes"), QMessageBox.ButtonRole.YesRole)
        btn_no = box.addButton(self._tr("reset_no"), QMessageBox.ButtonRole.NoRole)
        box.setDefaultButton(btn_no)
        box.setEscapeButton(btn_no)
        box.setWindowModality(Qt.WindowModal)
        geo = box.frameGeometry()
        geo.moveCenter(self.frameGeometry().center())
        box.move(geo.topLeft())
        box.exec()
        if box.clickedButton() is not btn_yes:
            return

        # Keep file selections; everything else falls back to the built-in defaults of _apply_config().
        keep = {
            "input_path": self.edt_input.text(),
            "depthmap_input_path": self.edt_depthmap_input.text(),
            "output_dir": self.edt_output.text(),
        }
        self.chk_shutdown.setChecked(False)     # never saved; also cancels a pending shutdown
        self._edge_auto_numeric = False         # defaults: Edge Fix shows AUTO again
        self._apply_config(keep)

        # Recreate the built-in preset (user presets stay untouched).
        self.presets.update(self._default_presets())
        self._save_presets()
        self._refresh_preset_combo()
        self._clear_preset_name()

        self._update_auto_mode_state()
        if self.chk_auto_mode.isChecked():
            self._apply_auto_mode_values()

        self._save_config()
        self._append_log(self._tr("reset_done"))

    def _preset_delete(self):
        name = self._preset_name_text()
        if name in self.presets:
            del self.presets[name]
            self._save_presets()
            self._refresh_preset_combo()
            self._append_log(self._tr("preset_deleted") + f" : {name}")

    def _select_input(self):
        _vid = " ".join("*" + e for e in sorted(VIDEO_EXTS))
        _img = " ".join("*" + e for e in sorted(IMAGE_EXTS))
        filters = (f"{self._tr('filter_video_image')} ({_vid} {_img});;{self._tr('filter_video')} ({_vid});;"
                   f"{self._tr('filter_image')} ({_img});;{self._tr('filter_all')} (*)")
        path, _ = QFileDialog.getOpenFileName(self, self._tr("dialog_input_title"), "", filters)
        if path: self.edt_input.setText(path)

    def _select_depthmap_input(self):
        _vid = " ".join("*" + e for e in sorted(VIDEO_EXTS))
        _img = " ".join("*" + e for e in sorted(IMAGE_EXTS))
        filters = (f"{self._tr('filter_depthmap')} ({_vid} {_img});;{self._tr('filter_video')} ({_vid});;"
                   f"{self._tr('filter_image')} ({_img});;{self._tr('filter_all')} (*)")
        path, _ = QFileDialog.getOpenFileName(self, self._tr("dialog_depthmap_title"), "", filters)
        if path: self.edt_depthmap_input.setText(path)

    def _clear_depthmap_input(self):
        self.edt_depthmap_input.setText("")

    def _select_output(self):
        path = QFileDialog.getExistingDirectory(self, self._tr("dialog_output_title"))
        if path: self.edt_output.setText(path)

    def _select_input_folder(self):
        path = QFileDialog.getExistingDirectory(self, self._tr("dialog_input_folder_title"))
        if path: self.edt_input.setText(path)

    def _validate(self) -> str | None:
        # Missing Medium / Light / Light (stream) inpaint model files are downloaded automatically when conversion starts.
        if not self.edt_input.text().strip(): return self._tr("err_no_input")
        if not self.edt_output.text().strip(): return self._tr("err_no_output")
        p = Path(self.edt_input.text().strip())
        if not p.exists(): return self._tr("err_file_not_found").format(p)

        if p.is_file():
            if p.suffix.lower() not in VIDEO_EXTS and p.suffix.lower() not in IMAGE_EXTS:
                return self._tr("err_unsupported_ext").format(p.suffix)
        elif p.is_dir():
            media = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in (VIDEO_EXTS | IMAGE_EXTS)]
            if not media: return self._tr("err_no_videos_in_folder")
            if self.edt_depthmap_input.text().strip():
                return self._tr("err_depthmap_folder_conflict")

        try:
            isz = int(self.cmb_input_size.currentText().strip())
            if isz < 64 or isz > 630: return self._tr("err_invalid_size_range")
        except ValueError: return self._tr("err_invalid_size_num")
        _is_image_input = p.is_file() and p.suffix.lower() in IMAGE_EXTS    # resize is a video-only option
        if self.chk_resize.isChecked() and not _is_image_input:
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

        _inp_path = Path(self.edt_input.text().strip())
        if _inp_path.is_file():
            _image_only = _inp_path.suffix.lower() in IMAGE_EXTS   # images do not need ffmpeg
        else:
            _image_only = not any(f.is_file() and f.suffix.lower() in VIDEO_EXTS for f in _inp_path.iterdir())
        if FFMPEG_MISSING and not _image_only:
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
        if self.current_params.get("auto_mode"):
            # Auto Mode: the worker measures the best edge fix right before the conversion starts.
            # This table value is only the fallback if the measurement fails.
            edgefix_reset_profile()
            _fb_edge, _fb_ema = compute_auto_edge_ema(
                self.current_params["divergence"], depth_model=self.current_params.get("depth_model"),
                warp_fill_mode=self.current_params.get("fill_mode"),
                depth_linear_comp=bool(self.current_params.get("depth_linear_comp", False)))
            self.current_params["edge_dilation"] = int(_fb_edge)

        input_path = Path(self.current_params["input_path"])
        self.batch_queue = []

        if input_path.is_file():
            self.batch_queue.append(input_path)
        elif input_path.is_dir():
            # Images first (they are converted together so the models are loaded once), then the videos.
            images = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
            images.sort(key=lambda x: x.name)
            videos = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
            videos.sort(key=lambda x: x.name)
            self.batch_queue.extend(images)
            self.batch_queue.extend(videos)

        self.batch_total = len(self.batch_queue)

        # A pending shutdown must never fire in the middle of a new conversion.
        self._cancel_scheduled_shutdown()
        self._batch_had_error = False
        self._last_batch_ok = False

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._append_log("=" * 50)
        self._append_log(f"Starting batch conversion. Total files: {self.batch_total}")

        self._process_next_in_batch()

    def _schedule_shutdown(self) -> bool:
        ok, msg = _run_shutdown(["/s", "/t", str(SHUTDOWN_DELAY_SEC), "/c",
                                 "Depth Conversion 3D: scheduled shutdown after conversion"])
        if ok:
            self._shutdown_scheduled = True
            self._append_log(f"[Shutdown] PC shutdown scheduled in {SHUTDOWN_DELAY_SEC // 60} minutes. "
                             f"Uncheck the option to cancel.")
        else:
            self._append_log(f"[Shutdown] Could not schedule the shutdown: {msg}")
        return ok

    def _cancel_scheduled_shutdown(self) -> bool:
        if not self._shutdown_scheduled:
            return False
        ok, msg = _run_shutdown(["/a"])
        self._shutdown_scheduled = False
        self._append_log("[Shutdown] Scheduled shutdown canceled." if ok
                         else f"[Shutdown] Cancel request returned: {msg or 'no pending shutdown'}")
        if self._shutdown_box is not None:
            try: self._shutdown_box.close()
            except Exception: pass
            self._shutdown_box = None
        return True

    def _show_shutdown_notice(self):
        # Non-modal notice: a modal box would block the checkbox that cancels the shutdown.
        self.lbl_status.setText(self._tr("status_done_shutdown"))
        box = QMessageBox(QMessageBox.Information, self._tr("done_title"), self._tr("done_msg_shutdown"),
                          QMessageBox.Ok, self)
        box.setWindowModality(Qt.NonModal)
        box.setAttribute(Qt.WA_DeleteOnClose, True)
        self._shutdown_box = box
        box.destroyed.connect(lambda *_: setattr(self, "_shutdown_box", None))
        box.show()

    def _on_shutdown_toggled(self, checked: bool):
        # ON/OFF at any time:
        #  - before / during a conversion: nothing to do now, the state is read when everything finishes;
        #  - after a batch finished normally: checking schedules the shutdown now, unchecking cancels it.
        if not checked:
            if self._cancel_scheduled_shutdown():
                self.lbl_status.setText(self._tr("status_shutdown_canceled"))
            return
        converting = not self.btn_start.isEnabled()
        if not converting and self._last_batch_ok and not self._shutdown_scheduled:
            if self._schedule_shutdown():
                self._show_shutdown_notice()

    def _process_next_in_batch(self):
        if not self.batch_queue:
            self.progress.setValue(100)
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._append_log("All batch conversions completed.")
            self._last_batch_ok = not self._batch_had_error
            if self.chk_shutdown.isChecked() and self._last_batch_ok and self._schedule_shutdown():
                self._show_shutdown_notice()
                return
            self.lbl_status.setText(self._tr("status_done"))
            self._show_message_box(QMessageBox.Information, self._tr("done_title"), self._tr("done_msg").format(self._tr("all_files_processed")))
            return

        next_file = self.batch_queue.pop(0)
        # Consecutive images are handed to one image worker (models loaded once for the whole group).
        image_group = [next_file]
        if next_file.suffix.lower() in IMAGE_EXTS:
            while self.batch_queue and self.batch_queue[0].suffix.lower() in IMAGE_EXTS:
                image_group.append(self.batch_queue.pop(0))
        self._img_group_offset = self.batch_total - len(self.batch_queue) - len(image_group)
        current_index = self.batch_total - len(self.batch_queue) - (len(image_group) - 1)

        batch_msg = self._tr("batch_progress").format(current_index, self.batch_total, next_file.name)
        self.lbl_status.setText(batch_msg)
        self._batch_status_text = batch_msg   # "n / total - file name", ETA is appended in _on_progress
        self._eta_ref = None
        self._append_log(f"\n[{current_index}/{self.batch_total}] Processing: {next_file.name}")
        self.progress.setValue(0)

        params = self.current_params.copy()
        params["input_path"] = next_file
        if len(image_group) > 1 or next_file.suffix.lower() in IMAGE_EXTS:
            params["input_paths"] = list(image_group)

        use_start = params.get("use_start_time", False)
        start_sec = params.get("start_time_sec", 0.0)
        use_end = params.get("use_end_time", False)
        end_sec = params.get("end_time_sec", 0.0)

        is_image = next_file.suffix.lower() in IMAGE_EXTS

        if (use_start or use_end) and not is_image:
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

        self.worker = ImageConvertWorker(params) if is_image else ConvertWorker(params)
        _w = self.worker
        _w.finished.connect(_w.deleteLater)
        _w.finished.connect(lambda: setattr(self, 'worker', None) if self.worker is _w else None)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._append_log)
        self.worker.auto_edge.connect(self._on_auto_edge_applied)
        if is_image:
            self.worker.file_started.connect(self._on_image_file_started)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished_error.connect(self._on_finished_error)
        self.worker.start()

    def _on_image_file_started(self, n: int, total: int, file_name: str):
        # The image worker moved on to its n-th file: update the "n / total - file name" status and the log.
        # (The progress bar and ETA cover the whole image group, so the ETA reference is not reset here.)
        idx = int(getattr(self, "_img_group_offset", 0)) + int(n)
        batch_msg = self._tr("batch_progress").format(idx, self.batch_total, file_name)
        self.lbl_status.setText(batch_msg)
        self._batch_status_text = batch_msg
        if n > 1:
            self._append_log(f"\n[{idx}/{self.batch_total}] Processing: {file_name}")

    def _stop(self):
        self.batch_queue.clear()
        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    if isinstance(self.worker, ImageConvertWorker):
                        self._append_log("Requesting stop...")
                    else:
                        self._append_log("Requesting stop... (the part converted so far will be saved)")
                    self.worker.stop()
            except RuntimeError:
                self.worker = None
            self.btn_stop.setEnabled(False)

    def _format_eta(self, seconds: float) -> str:
        sec = max(0, int(round(seconds)))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return self._tr("time_hms").format(h=h, m=m, s=s)
        if m > 0:
            return self._tr("time_ms").format(m=m, s=s)
        return self._tr("time_s").format(s=s)

    def _on_progress(self, current: int, total: int, msg: str):
        if total > 0:
            pct = int(current / total * 100)
            self.progress.setValue(min(100, pct))

            # Remaining-time estimate for the current file, from the speed measured between progress
            # updates (the first update is only the reference point, so model loading is not counted).
            base = getattr(self, "_batch_status_text", None)
            if base:
                now = time.perf_counter()
                ref = getattr(self, "_eta_ref", None)
                eta_txt = None
                if ref is None:
                    self._eta_ref = (now, current)
                    eta_txt = self._tr("eta_calculating")
                else:
                    dt = now - ref[0]
                    df = current - ref[1]
                    if dt > 0.5 and df > 0:
                        eta_txt = self._format_eta((total - current) / (df / dt))
                    else:
                        eta_txt = self._tr("eta_calculating")
                self.lbl_status.setText(f"{base}  |  {self._tr('eta_text').format(eta_txt)}")

    def _on_finished_ok(self, path: str):
        partial = bool(getattr(self.worker, "partial_saved", False)) if self.worker is not None else False
        if self.worker is not None:
            self.worker.wait()
            if int(getattr(self.worker, "error_count", 0) or 0) > 0:
                self._batch_had_error = True      # some images of the group were skipped
        if partial:
            # The user pressed Stop: the video converted so far was saved (the batch queue was cleared in _stop).
            self._append_log(f"Conversion stopped. Partial video saved: {path}")
            self.batch_queue.clear()
            self._last_batch_ok = False
            self.lbl_status.setText(self._tr("status_stopped_saved"))
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._show_message_box(QMessageBox.Information, self._tr("done_title"),
                                   self._tr("stopped_saved_msg").format(path))
            return
        self._append_log(f"Conversion completed: {path}")
        self._process_next_in_batch()

    def _on_finished_error(self, err_msg):
        if self.worker is not None:
            self.worker.wait()
        if hasattr(self, 'batch_queue') and len(self.batch_queue) > 0:
            self._append_log(f"[Skip] Skipping file due to an error. Reason: {err_msg}")
            self._batch_had_error = True

            self._process_next_in_batch()

        else:
            if str(err_msg).strip() == "Conversion was aborted.":
                self._show_message_box(QMessageBox.Information, self._tr("err_dialog_title"), self._tr("err_conversion_aborted"))
            else:
                self._show_message_box(QMessageBox.Critical, self._tr("err_dialog_title"), f"An error occurred during conversion:\n{err_msg}")

            self._last_batch_ok = False
            self.lbl_status.setText(self._tr("status_error"))
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def closeEvent(self, event):
        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    self.worker.stop(save_partial=False)   # closing the window: abort quickly
                    self.worker.wait(3000)
            except RuntimeError:
                self.worker = None
        self._save_config()
        event.accept()

def _build_splash_pixmap() -> "QPixmap":

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

    QTimer.singleShot(0, _init_and_show)

    sys.exit(app.exec())

# ---------------------------------------------------------------------------
# Embedded reference image for the Auto Mode edge fix measurement (JPEG, base64).
# ---------------------------------------------------------------------------
_EDGEFIX_REF_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAQDAwMDAgQDAwMEBAQFBgoGBgUFBgwICQcKDgwPDg4MDQ0PERYTDxAVEQ0NExoTFRcYGRkZDxIbHRsYHRYY"
    "GRj/2wBDAQQEBAYFBgsGBgsYEA0QGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBj/wAARCAMABVADASIAAhEB"
    "AxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS"
    "0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKz"
    "tLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgEC"
    "BAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpj"
    "ZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6"
    "/9oADAMBAAIRAxEAPwD45vtWvdRObqUt7dKoEU40lWklohttu7G4yaMc4xS47UDpzQxCY9qbgVJ6mm0gGEcYIphAqQjg0w00AsMXmyBMda0bjRbm2sUu"
    "pImWJ8hWIwGx1rOicxyBx2NbV/4o1bUNBsdHu72WWysRILWBvuw723Pj6nmnd9ASVtTF2gZ9qaaUnPWmZoEhf1pKKBQMKMCkx/8ArpaADvR+FFFAkBoA"
    "zxTasWiB5vm6UXH5CLbyMMhTio2Qq2GFdPDHGIgCuaxtRjRZDjis4zu7FyhZFCjtRSfpWhmLSfUUDtR34FAy7pP/ACFo69WsogbIeuK8p0k41eL6163p"
    "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"
)

if __name__ == "__main__":
    main()
