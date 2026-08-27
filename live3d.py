

from __future__ import annotations
import os
import sys 


def _early_get_cpu_headroom_ratio(default: float = 0.5) -> float:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--cpu-headroom-ratio" and i + 1 < len(argv):
            try:
                return float(argv[i + 1])
            except ValueError:
                return default
        if a.startswith("--cpu-headroom-ratio="):
            try:
                return float(a.split("=", 1)[1])
            except ValueError:
                return default
    return default


_CPU_COUNT = os.cpu_count() or 4
_RESERVED_THREADS = 3


_CPU_HEADROOM_RATIO = _early_get_cpu_headroom_ratio(0.5)
_CPU_THREADS = max(2, min(int((_CPU_COUNT - _RESERVED_THREADS) * _CPU_HEADROOM_RATIO), 12))

os.environ["OMP_NUM_THREADS"] = str(_CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(_CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(_CPU_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(_CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(_CPU_THREADS)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import atexit
import signal
import ctypes
import queue
import re
import io
import threading
import time
import json
import sys
from pathlib import Path
from ctypes import wintypes


try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

cv2.setNumThreads(_CPU_THREADS)
torch.set_num_threads(_CPU_THREADS)


try:
    from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, Signal, QTimer
    from PySide6.QtGui import QFont, QPixmap, QPainter, QColor
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QComboBox, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
        QCheckBox, QGroupBox, QScrollArea, QTextEdit, QMessageBox, QSplashScreen
    )
except ImportError:
    print("[Warning] PySide6 package is not installed. (pip install PySide6)")
    sys.exit(1)


try:
    from OpenGL.GL import *
    from OpenGL.GL import shaders
    _HAS_OPENGL = True
except ImportError:
    _HAS_OPENGL = False


_HAS_DILATION = True

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
def dilate_edge(
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

try:
    import win32api
    import win32con
    import win32gui
    import win32process
except ImportError:
    print("[Error] pywin32 is required: pip install pywin32")
    sys.exit(1)


if getattr(sys, 'frozen', False):

    BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    APP_DIR = Path(sys.executable).resolve().parent
else:

    BASE_DIR = Path(__file__).resolve().parent
    APP_DIR = BASE_DIR

CONFIG_FILE = APP_DIR / "live3d_gui_config.json"


def _force_letterbox_off_in_config():
    try:
        if CONFIG_FILE.exists():
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        else:
            cfg = {}
        if cfg.get("auto_crop", False):
            cfg["auto_crop"] = False
            CONFIG_FILE.write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
    except Exception:
        pass
ZIPDEPTH_ROOT = BASE_DIR / "ZipDepth"
sys.path.insert(0, str(ZIPDEPTH_ROOT))

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
opengl32 = ctypes.windll.opengl32

WDA_EXCLUDEFROMCAPTURE = 0x00000011
WDA_NONE = 0x00000000

WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WS_EX_NOREDIRECTIONBITMAP = 0x00200000

HWND_TOPMOST = -1
SWP_SHOWWINDOW = 0x0040
SWP_NOACTIVATE = 0x0010
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001

BI_RGB = 0
DIB_RGB_COLORS = 0

WINDOW_CLASS = "Live3DOverlayClass"
WINDOW_TITLE = "DepthLive3D Overlay"


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

class PIXELFORMATDESCRIPTOR(ctypes.Structure):
    _fields_ = [
        ("nSize", wintypes.WORD),
        ("nVersion", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("iPixelType", ctypes.c_byte),
        ("cColorBits", ctypes.c_byte),
        ("cRedBits", ctypes.c_byte),
        ("cRedShift", ctypes.c_byte),
        ("cGreenBits", ctypes.c_byte),
        ("cGreenShift", ctypes.c_byte),
        ("cBlueBits", ctypes.c_byte),
        ("cBlueShift", ctypes.c_byte),
        ("cAlphaBits", ctypes.c_byte),
        ("cAlphaShift", ctypes.c_byte),
        ("cAccumBits", ctypes.c_byte),
        ("cAccumRedBits", ctypes.c_byte),
        ("cAccumGreenBits", ctypes.c_byte),
        ("cAccumBlueBits", ctypes.c_byte),
        ("cAccumAlphaBits", ctypes.c_byte),
        ("cDepthBits", ctypes.c_byte),
        ("cStencilBits", ctypes.c_byte),
        ("cAuxBuffers", ctypes.c_byte),
        ("iLayerType", ctypes.c_byte),
        ("bReserved", ctypes.c_byte),
        ("dwLayerMask", wintypes.DWORD),
        ("dwVisibleMask", wintypes.DWORD),
        ("dwDamageMask", wintypes.DWORD),
    ]


VK_MAP = {
    "esc": 0x1B, "escape": 0x1B,
    "space": 0x20, "tab": 0x09, "enter": 0x0D, "return": 0x0D,
    "backspace": 0x08, "bs": 0x08, "delete": 0x2E, "del": 0x2E,
    "insert": 0x2D, "ins": 0x2D, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pgup": 0x21, "pagedown": 0x22, "pgdn": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
    "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    **{str(i): 0x30 + i for i in range(10)},
    **{chr(ord("a") + i): 0x41 + i for i in range(26)},
    "[": 0xDB, "]": 0xDD, ";": 0xBA, "'": 0xDE, "`": 0xC0,
    "\\": 0xDC, "/": 0xBF, "-": 0xBD, "=": 0xBB, "+": 0xBB, ",": 0xBC, ".": 0xBE,
}

MOD_NAMES = {"ctrl": "ctrl", "control": "ctrl", "shift": "shift", "alt": "alt", "menu": "alt"}

def parse_hotkey(spec: str):
    if not spec or not str(spec).strip():
        return None
    s = str(spec).strip().lower()


    if s in ("+", "-"):

        key_name = "=" if s == "+" else "-"
        if key_name not in VK_MAP:
            raise argparse.ArgumentTypeError(f"Unknown key: '{s}'")
        return {"mods": set(), "vk": VK_MAP[key_name]}

    parts = [p.strip() for p in re.split(r"[+\s]+", s) if p.strip()]
    mods = set()
    key_vk = None
    for p in parts:
        if p in MOD_NAMES:
            mods.add(MOD_NAMES[p])
        elif p in VK_MAP:
            key_vk = VK_MAP[p]
        else:
            raise argparse.ArgumentTypeError(f"Unknown key: '{p}'")
    if key_vk is None:
        raise argparse.ArgumentTypeError(f"No main key specified: '{spec}'")
    return {"mods": mods, "vk": key_vk}

def is_hotkey_down(hk, edge: bool = False) -> bool:
    if hk is None:
        return False
    if "ctrl" in hk["mods"] and not (user32.GetAsyncKeyState(0x11) & 0x8000):
        return False
    if "shift" in hk["mods"] and not (user32.GetAsyncKeyState(0x10) & 0x8000):
        return False
    if "alt" in hk["mods"] and not (user32.GetAsyncKeyState(0x12) & 0x8000):
        return False
    state = user32.GetAsyncKeyState(hk["vk"])
    if edge:
        return bool(state & 0x0001)
    return bool(state & 0x8000)

def parse_size(s: str) -> tuple[int, int]:
    a, b = s.lower().split("x")
    return int(a), int(b)

def parse_bgr(s: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in s.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Color must be three values R,G,B")
    r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
    for v in (r, g, b):
        if not 0 <= v <= 255:
            raise argparse.ArgumentTypeError("Color components must be 0-255")
    return (b, g, r)

def parse_edge_dilation(s: str):
    s = s.strip()
    if not s or s == "0":
        return 0
    parts = [p.strip() for p in s.replace(" ", "").split(",")]
    vals = [int(p) for p in parts]
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return vals
    raise argparse.ArgumentTypeError("edge-dilation accepts only 1 or 2 values")


EMA_MAX = 0.50


EMA_LEVELS = [
    ("Off", 0.00),
    ("Low", 0.15),
    ("Medium", 0.30),
    ("High", 0.50),
]
EMA_STEP_GRID = [v for _, v in EMA_LEVELS]


def ema_value_to_label(value: float) -> str:
    v = float(value)
    best_label, best_val = EMA_LEVELS[0]
    best_diff = abs(v - best_val)
    for label, lv in EMA_LEVELS:
        diff = abs(v - lv)
        if diff < best_diff:
            best_diff = diff
            best_label, best_val = label, lv
    return best_label


def ema_label_to_value(label: str) -> float:
    for l, v in EMA_LEVELS:
        if l == label:
            return v
    return 0.00


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


def ema_step_value(current: float, increase: bool) -> float:
    cur = round(float(current), 2)
    if increase:
        for v in EMA_STEP_GRID:
            if v > cur + 1e-9:
                return v
        return EMA_STEP_GRID[-1]
    else:
        for v in reversed(EMA_STEP_GRID):
            if v < cur - 1e-9:
                return v
        return EMA_STEP_GRID[0]


def compute_auto_edge_ema(divergence: float) -> tuple[int, float]:
    div = float(divergence)
    if div <= 0.5 + 1e-9:
        edge, ema = 0, 0.00
    elif div <= 1.0 + 1e-9:
        edge, ema = 1, 0.00
    elif div <= 1.5 + 1e-9:
        edge, ema = 2, 0.15
    elif div <= 2.0 + 1e-9:
        edge, ema = 3, 0.30
    elif div <= 2.5 + 1e-9:
        edge, ema = 4, 0.50
    else:
        edge, ema = 4, 0.50
    edge = max(0, min(5, edge))
    return edge, clamp_ema_value(ema)


def detect_letterbox_frame(frame, threshold=16, min_black_ratio=0.92):
    if frame is None or frame.size == 0:
        return None
    try:
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

        crop_h = bottom - top

        if crop_h < h * 0.55 or (top < 2 and (h - bottom) < 2):
            return None


        top = max(0, top - (top % 2))
        crop_h = crop_h - (crop_h % 2)
        bottom = min(h, top + crop_h)
        if bottom - top < h * 0.55:
            return None
        return top, bottom
    except Exception:
        return None


def pad_to_target_size(t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if t.dim() == 3 and t.shape[0] in (1, 3):
        c, h, w = t.shape
        if h == target_h and w == target_w:
            return t
        out = torch.zeros((c, target_h, target_w), dtype=t.dtype, device=t.device)
        y0 = max(0, (target_h - h) // 2)
        x0 = max(0, (target_w - w) // 2)
        y1 = min(target_h, y0 + h)
        x1 = min(target_w, x0 + w)
        out[:, y0:y1, x0:x1] = t[:, :y1 - y0, :x1 - x0]
        return out
    if t.dim() == 4:
        b, c, h, w = t.shape
        if h == target_h and w == target_w:
            return t
        out = torch.zeros((b, c, target_h, target_w), dtype=t.dtype, device=t.device)
        y0 = max(0, (target_h - h) // 2)
        x0 = max(0, (target_w - w) // 2)
        y1 = min(target_h, y0 + h)
        x1 = min(target_w, x0 + w)
        out[:, :, y0:y1, x0:x1] = t[:, :, :y1 - y0, :x1 - x0]
        return out
    if t.dim() == 3 and t.shape[2] in (1, 3):
        h, w, c = t.shape
        if h == target_h and w == target_w:
            return t
        out = torch.zeros((target_h, target_w, c), dtype=t.dtype, device=t.device)
        y0 = max(0, (target_h - h) // 2)
        x0 = max(0, (target_w - w) // 2)
        y1 = min(target_h, y0 + h)
        x1 = min(target_w, x0 + w)
        out[y0:y1, x0:x1] = t[:y1 - y0, :x1 - x0]
        return out
    return t


class CUDART_GL:
    CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2
    CUDA_MEMCPY_DEVICE_TO_DEVICE = 3

    def __init__(self, device_id: int = 0):
        import os
        torch_dir = os.path.dirname(torch.__file__)
        candidates = [
            os.path.join(torch_dir, "lib", "cudart64_12.dll"),
            os.path.join(torch_dir, "lib", "cudart64_11.dll"),
            os.path.join(torch_dir, "bin", "cudart64_12.dll"),
            os.path.join(torch_dir, "bin", "cudart64_11.dll"),
        ]
        cudart_path = next((p for p in candidates if os.path.exists(p)), "cudart64_12.dll")
        if sys.platform == "win32":
            self.lib = ctypes.WinDLL(cudart_path)
        else:
            self.lib = ctypes.CDLL(cudart_path)

        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaSetDevice.restype = ctypes.c_int
        self.lib.cudaGraphicsGLRegisterBuffer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint]
        self.lib.cudaGraphicsGLRegisterBuffer.restype = ctypes.c_int
        self.lib.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
        self.lib.cudaGraphicsUnregisterResource.restype = ctypes.c_int
        self.lib.cudaGraphicsMapResources.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.lib.cudaGraphicsMapResources.restype = ctypes.c_int
        self.lib.cudaGraphicsUnmapResources.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.lib.cudaGraphicsUnmapResources.restype = ctypes.c_int
        self.lib.cudaGraphicsResourceGetMappedPointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
        self.lib.cudaGraphicsResourceGetMappedPointer.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaSetDevice(device_id)

    def register_buffer(self, pbo_id: int):
        resource = ctypes.c_void_p()
        res = self.lib.cudaGraphicsGLRegisterBuffer(ctypes.byref(resource), pbo_id, self.CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD)
        if res != 0: raise RuntimeError(f"cudaGraphicsGLRegisterBuffer failed: {res}")
        return resource

    def unregister(self, resource):
        if resource: self.lib.cudaGraphicsUnregisterResource(resource)

    def map(self, resource):
        res = self.lib.cudaGraphicsMapResources(1, ctypes.byref(resource), None)
        if res != 0:
            raise RuntimeError(f"cudaGraphicsMapResources failed: {res}")

    def unmap(self, resource):
        res = self.lib.cudaGraphicsUnmapResources(1, ctypes.byref(resource), None)
        if res != 0:
            raise RuntimeError(f"cudaGraphicsUnmapResources failed: {res}")

    def get_mapped_pointer(self, resource):
        ptr = ctypes.c_void_p()
        size = ctypes.c_size_t()
        res = self.lib.cudaGraphicsResourceGetMappedPointer(ctypes.byref(ptr), ctypes.byref(size), resource)
        if res != 0: raise RuntimeError(f"cudaGraphicsResourceGetMappedPointer failed: {res}")
        return ptr.value, size.value

    def memcpy_d2d(self, dst, src, size):
        res = self.lib.cudaMemcpy(dst, src, size, self.CUDA_MEMCPY_DEVICE_TO_DEVICE)
        if res != 0: raise RuntimeError(f"cudaMemcpy D2D failed: {res}")


try:
    import xr
    from xr.utils.gl import ContextObject
    from xr.utils.gl.glfw_util import GLFWOffscreenContextProvider
    from OpenGL import GL as GL_XR
    _HAS_OPENXR = True
except ImportError:
    _HAS_OPENXR = False


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", _MOUSEINPUT),
    ]

_INPUT_MOUSE = 0

def _send_mouse_input(dx: int = 0, dy: int = 0, mouse_data: int = 0, flags: int = 0):
    extra = ctypes.pointer(ctypes.c_ulong(0))
    mi = _MOUSEINPUT(dx, dy, mouse_data, flags, 0, extra)
    inp = _INPUT(_INPUT_MOUSE, mi)
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


_real_mouse_suppress_until = [0.0]


def _send_mouse_button_at_virtual_cursor(u: float, v: float, flags: int):
    old_x, old_y = None, None
    try:
        old_x, old_y = win32api.GetCursorPos()


        _real_mouse_suppress_until[0] = time.perf_counter() + 0.3
        abs_x = int(max(0, min(65535, round(float(u) * 65535.0))))
        abs_y = int(max(0, min(65535, round(float(v) * 65535.0))))
        _send_mouse_input(
            dx=abs_x,
            dy=abs_y,
            flags=win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE
        )
        _send_mouse_input(flags=flags)
    finally:
        if old_x is not None:
            try:
                win32api.SetCursorPos(old_x, old_y)
            except Exception:
                pass
        _real_mouse_suppress_until[0] = time.perf_counter() + 0.3


def _send_wheel_at_virtual_cursor(u: float, v: float, wheel_delta: int):
    old_x, old_y = None, None
    try:
        old_x, old_y = win32api.GetCursorPos()
        _real_mouse_suppress_until[0] = time.perf_counter() + 0.3
        abs_x = int(max(0, min(65535, round(float(u) * 65535.0))))
        abs_y = int(max(0, min(65535, round(float(v) * 65535.0))))
        _send_mouse_input(
            dx=abs_x,
            dy=abs_y,
            flags=win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE
        )
        _send_mouse_input(
            mouse_data=wheel_delta,
            flags=win32con.MOUSEEVENTF_WHEEL
        )
    finally:
        if old_x is not None:
            try:
                win32api.SetCursorPos(old_x, old_y)
            except Exception:
                pass
        _real_mouse_suppress_until[0] = time.perf_counter() + 0.3


def _release_injected_mouse_buttons():
    try:
        _send_mouse_input(flags=win32con.MOUSEEVENTF_LEFTUP)
    except Exception:
        pass
    try:
        _send_mouse_input(flags=win32con.MOUSEEVENTF_RIGHTUP)
    except Exception:
        pass
    try:
        _send_mouse_input(flags=win32con.MOUSEEVENTF_MIDDLEUP)
    except Exception:
        pass


class OpenXRStereoViewer:
    def __init__(self):
        if not _HAS_OPENXR:
            raise RuntimeError("pyopenxr is not installed. (pip install pyopenxr)")

        self.context = None
        self.left_tex = None
        self.right_tex = None
        self.tex_w = 0
        self.tex_h = 0
        self.program = None
        self.vao = None
        self.vbo = None
        self.loc_pos = -1
        self.loc_uv = -1

        self._running = True
        self._lock = threading.Lock()
        self._latest_left = None
        self._latest_right = None
        self._thread = None
        self._gl_ready = False
        self._has_new_frame = False
        self._render_ready = threading.Event()
        self._render_error = None


        self.panel_distance = 2.0
        self.panel_height = 1.5
        self.panel_y_offset = 0.0
        self.panel_curve = 0.0
        self.near_z = 0.05
        self.far_z = 100.0
        self._eye_to_mouth_offset = 0.12


        self._panel_lock = threading.Lock()
        self._default_panel_distance = self.panel_distance
        self._default_panel_height = self.panel_height
        self._default_panel_y_offset = self.panel_y_offset
        self._default_panel_curve = self.panel_curve
        self._panel_distance_range = (0.3, 20.0)
        self._panel_height_range = (0.2, 10.0)
        self._panel_y_offset_range = (-5.0, 5.0)
        self._panel_curve_range = (0.0, 100.0)

        # Curved-screen mesh state: the flat quad is subdivided into
        # _CURVE_SEGMENTS columns so it can be bent like a cylindrical
        # (Virtual-Desktop-style) curved VR screen. Rebuilt/re-uploaded
        # whenever panel_curve changes so adjustments show up live.
        self._CURVE_SEGMENTS = 32
        self._CURVE_MAX_SAG = 0.6
        self._last_uploaded_curve = None


        if CONFIG_FILE.exists():
            try:
                cfg_data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                self.panel_distance = float(cfg_data.get("openxr_panel_distance", self.panel_distance))
                self.panel_height = float(cfg_data.get("openxr_panel_height", self.panel_height))
                self.panel_y_offset = float(cfg_data.get("openxr_panel_y_offset", self.panel_y_offset))
                self.panel_curve = float(cfg_data.get("openxr_panel_curve", self.panel_curve))
            except Exception:
                pass


        self._recenter_requested = True
        self._center_pos = np.zeros(3, dtype=np.float64)
        self._center_yaw = 0.0


        self._actions_ready = False
        self._active_hand = None
        self._prev_trigger = {"left": False, "right": False}
        self._prev_menu = {"left": False, "right": False}
        self._prev_letterbox = {"left": False, "right": False}
        self.shared_auto_crop = None
        self._letterbox_notify_seq = 0
        self._last_wheel_time = 0.0
        self._wheel_repeat_interval = 0.06
        self._wheel_deadzone = 0.5


        self.shared_divergence = None
        self.shared_auto_mode = None
        self.shared_edge_dilation = None
        self.args = None
        self._div_step = 0.5
        self._div_min = 0.5
        self._div_max = 2.5
        self._stereo_x_trigger = 0.7
        self._stereo_x_release = 0.3
        self._prev_stereo_dir = {"left": 0, "right": 0}


        self._prev_grip = {"left": False, "right": False}
        self._active_grip_hand = None
        self._grip_start_x = 0.0
        self._grip_start_y = 0.0
        self._grip_start_height = 0.0
        self._grip_start_distance = 0.0
        self._grip_axis_lock = None
        self._grip_deadzone = 0.02
        self._grip_size_sensitivity = 8.0
        self._grip_dist_sensitivity = 16.0
        self._grip_threshold = 0.5


        self._smooth_u = 0.5
        self._smooth_v = 0.5
        self._pointer_margin = 0.10
        self._pointer_deadzone = 0.0100
        self._virtual_cursor_u = 0.5
        self._virtual_cursor_v = 0.5
        self._virtual_cursor_valid = False
        self.shared_cursor = None
        self._prev_recenter = {"left": False, "right": False}
        self._injected_left_down = False
        self._injected_right_down = False


        self._thread = threading.Thread(target=self._render_loop, name="OpenXRRender", daemon=True)
        self._thread.start()
        print("[OpenXR] Session thread started successfully", flush=True)

    def _init_gl_resources(self):
        from OpenGL import GL
        from OpenGL.GL import shaders

        vert = """
        #version 130
        in vec3 pos;
        in vec2 uv;
        out vec2 v_uv;
        uniform mat4 u_mvp;
        void main() {
            // pos.xy is a "unit quad" local coordinate between -0.5 and 0.5.
            // pos.z carries the (optional) curve sag for a curved VR screen,
            // expressed in the same local units as x so it scales with the
            // panel's width via u_mvp exactly like x/y do.
            // The actual position/size/projection in world space is fully handled by u_mvp (model*view*projection).
            // -> This screen stays fixed at the same place in space even if the headset rotates/moves.
            gl_Position = u_mvp * vec4(pos, 1.0);
            v_uv = uv;
        }
        """
        frag = """
        #version 130
        in vec2 v_uv;
        out vec4 fragColor;

        uniform sampler2D u_tex;

        void main()
        {
            // The screen (quad) geometry itself is built to match the video aspect ratio,
            // so no separate letterbox handling is needed here; sample it directly.
            fragColor = texture(u_tex, v_uv);
        }
        """
        self.program = shaders.compileProgram(
            shaders.compileShader(vert, GL.GL_VERTEX_SHADER),
            shaders.compileShader(frag, GL.GL_FRAGMENT_SHADER),
        )


        vertices = self._build_panel_mesh_vertices(self.panel_curve)

        self.loc_pos = GL.glGetAttribLocation(self.program, "pos")
        self.loc_uv = GL.glGetAttribLocation(self.program, "uv")

        self.loc_mvp = GL.glGetUniformLocation(
            self.program,
            "u_mvp"
        )


        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)


        # Stride is 5 floats (pos.xyz + uv) per vertex now that the panel can
        # be a curved multi-segment strip instead of a single flat quad.
        # GL_DYNAMIC_DRAW because the curve amount (and thus the mesh) can be
        # re-uploaded every frame while the user drags the curve slider/hotkeys.
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_DYNAMIC_DRAW)
        self._panel_vertex_count = (self._CURVE_SEGMENTS + 1) * 2
        self._last_uploaded_curve = self.panel_curve


        stride = 5 * 4
        GL.glEnableVertexAttribArray(self.loc_pos)
        GL.glVertexAttribPointer(self.loc_pos, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))

        GL.glEnableVertexAttribArray(self.loc_uv)
        GL.glVertexAttribPointer(self.loc_uv, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(12))


        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        self._gl_ready = True
        print("[OpenXR] GL resources initialized on render thread", flush=True)

    def _build_panel_mesh_vertices(self, curve_value: float) -> np.ndarray:
        """Build a triangle-strip mesh for the VR screen quad, bent along the
        horizontal axis to approximate a curved (cylindrical) monitor, like
        Virtual Desktop's curve option. curve_value is the 0..100 UI value;
        0 stays perfectly flat (identical to the original single quad)."""
        segments = self._CURVE_SEGMENTS
        lo, hi = self._panel_curve_range
        c = max(lo, min(hi, curve_value)) / (hi - lo if hi != lo else 1.0)
        sag_amount = c * self._CURVE_MAX_SAG

        xs = np.linspace(-0.5, 0.5, segments + 1)
        t = xs / 0.5  # -1..1
        # Parabolic sag: 0 at the edges, deepest (curving away from the
        # viewer, i.e. wrapping toward them) at the center.
        sag = -sag_amount * (1.0 - t * t)
        us = xs + 0.5

        verts = np.empty((2 * (segments + 1), 5), dtype=np.float32)
        verts[0::2, 0] = xs
        verts[0::2, 1] = -0.5
        verts[0::2, 2] = sag
        verts[0::2, 3] = us
        verts[0::2, 4] = 1.0

        verts[1::2, 0] = xs
        verts[1::2, 1] = 0.5
        verts[1::2, 2] = sag
        verts[1::2, 3] = us
        verts[1::2, 4] = 0.0

        return verts.flatten()

    def _sync_panel_mesh(self, curve_value: float):
        """Re-upload the panel mesh if the curve value changed since the last
        upload, so curve adjustments (UI slider, hotkeys, reset) are reflected
        in the OpenXR output in real time."""
        from OpenGL import GL

        if self._last_uploaded_curve is not None and abs(curve_value - self._last_uploaded_curve) < 1e-6:
            return

        vertices = self._build_panel_mesh_vertices(curve_value)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self._last_uploaded_curve = curve_value

    def _ensure_textures(self, w: int, h: int):
        from OpenGL import GL

        if self.tex_w == w and self.tex_h == h and self.left_tex is not None:
            return

        if self.left_tex is not None:
            try:
                GL.glDeleteTextures([self.left_tex, self.right_tex])
            except Exception:
                pass

        self.tex_w, self.tex_h = w, h

        def create_tex():
            tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_SRGB8,
                w,
                h,
                0,
                GL.GL_RGB,
                GL.GL_UNSIGNED_BYTE,
                None
            )
            return tex

        self.left_tex  = create_tex()
        self.right_tex = create_tex()
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        print(f"[OpenXR] textures created {w}x{h}", flush=True)

    def _upload_tensor(self, tex_id: int, tensor: torch.Tensor):
        from OpenGL import GL

        if tensor is None:
            return

        t = tensor
        if t.dim() == 3 and t.shape[0] == 3:
            t = t.permute(1, 2, 0).contiguous()
        if t.dtype != torch.uint8:
            t = t.clamp(0, 255).byte()
        t = t.contiguous()

        h, w = t.shape[:2]
        data = t.cpu().numpy()

        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)


        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
            GL.GL_RGB, GL.GL_UNSIGNED_BYTE, data
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    @staticmethod
    def _quat_to_mat3(x: float, y: float, z: float, w: float) -> np.ndarray:
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    @classmethod
    def _view_matrix_from_pose(cls, pose) -> np.ndarray:
        q = pose.orientation
        p = pose.position
        R = cls._quat_to_mat3(q.x, q.y, q.z, q.w)
        Rt = R.T
        pos = np.array([p.x, p.y, p.z], dtype=np.float64)

        view = np.eye(4, dtype=np.float64)
        view[0:3, 0:3] = Rt
        view[0:3, 3] = -Rt @ pos
        return view

    def _projection_matrix_from_fov(self, fov) -> np.ndarray:
        tan_left = math.tan(fov.angle_left)
        tan_right = math.tan(fov.angle_right)
        tan_down = math.tan(fov.angle_down)
        tan_up = math.tan(fov.angle_up)

        tan_w = tan_right - tan_left
        tan_h = tan_up - tan_down
        near, far = self.near_z, self.far_z

        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = 2.0 / tan_w
        proj[0, 2] = (tan_right + tan_left) / tan_w
        proj[1, 1] = 2.0 / tan_h
        proj[1, 2] = (tan_up + tan_down) / tan_h
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -(far * 2.0 * near) / (far - near)
        proj[3, 2] = -1.0
        return proj

    def _model_matrix_for_panel(self, tex_w: int, tex_h: int, view_pose=None) -> np.ndarray:
        with self._panel_lock:
            panel_height = self.panel_height
            panel_y_offset = self.panel_y_offset
            panel_distance = self.panel_distance


            if self._recenter_requested and view_pose is not None:
                p = view_pose.position
                q = view_pose.orientation
                siny_cosp = 2 * (q.w * q.y + q.x * q.z)
                cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
                yaw = math.atan2(siny_cosp, cosy_cosp)


                self._center_pos[0] = p.x
                self._center_pos[1] = p.y - self._eye_to_mouth_offset
                self._center_pos[2] = p.z

                self._center_yaw = yaw
                self._recenter_requested = False

        aspect = float(tex_w) / float(tex_h) if tex_h else 16.0 / 9.0
        width = panel_height * aspect


        M_local = np.eye(4, dtype=np.float64)
        M_local[0, 0] = width
        M_local[1, 1] = panel_height
        # Scale the curve-sag (local z) by width too, since it's expressed in
        # the same local units as x -- this keeps the curvature looking like
        # a consistent cylindrical bend regardless of screen size.
        M_local[2, 2] = width
        M_local[1, 3] = panel_y_offset
        M_local[2, 3] = -panel_distance


        c_x, c_y, c_z = self._center_pos
        cy = math.cos(self._center_yaw)
        sy = math.sin(self._center_yaw)

        M_center = np.array([
            [cy,  0, sy, c_x],
            [0,   1,  0, c_y],
            [-sy, 0, cy, c_z],
            [0,   0,  0,   1]
        ], dtype=np.float64)

        return M_center @ M_local


    def _save_panel_config(self):
        try:
            cfg_data = {}
            if CONFIG_FILE.exists():
                cfg_data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            with self._panel_lock:
                cfg_data["openxr_panel_distance"] = self.panel_distance
                cfg_data["openxr_panel_height"] = self.panel_height
                cfg_data["openxr_panel_y_offset"] = self.panel_y_offset
                cfg_data["openxr_panel_curve"] = self.panel_curve
            CONFIG_FILE.write_text(json.dumps(cfg_data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[OpenXR] Config save failed: {e}", flush=True)

    def adjust_panel_distance(self, delta: float):
        lo, hi = self._panel_distance_range
        with self._panel_lock:
            self.panel_distance = max(lo, min(hi, self.panel_distance + delta))
        self._save_panel_config()

    def adjust_panel_size(self, delta: float):
        lo, hi = self._panel_height_range
        with self._panel_lock:
            self.panel_height = max(lo, min(hi, self.panel_height + delta))
        self._save_panel_config()

    def adjust_panel_height_offset(self, delta: float):
        lo, hi = self._panel_y_offset_range
        with self._panel_lock:
            self.panel_y_offset = max(lo, min(hi, self.panel_y_offset + delta))
        self._save_panel_config()

    def adjust_panel_curve(self, delta: float):
        lo, hi = self._panel_curve_range
        with self._panel_lock:
            self.panel_curve = max(lo, min(hi, self.panel_curve + delta))
        self._save_panel_config()

    def _apply_auto_mode_if_enabled(self):
        try:
            if not self.shared_auto_mode or not self.shared_auto_mode[0]:
                return
            if self.shared_divergence is None:
                return
            auto_edge, auto_ema = compute_auto_edge_ema(self.shared_divergence[0])
            if self.shared_edge_dilation is not None:
                self.shared_edge_dilation[0] = auto_edge
            if self.args is not None:
                self.args.ema_decay = auto_ema
        except Exception:
            pass

    def recenter_panel(self):
        with self._panel_lock:
            self._recenter_requested = True

    def reset_panel(self):
        with self._panel_lock:
            self.panel_distance = self._default_panel_distance
            self.panel_height = self._default_panel_height
            self.panel_y_offset = self._default_panel_y_offset
            self.panel_curve = self._default_panel_curve
            self._center_pos = np.zeros(3, dtype=np.float64)
            self._center_yaw = 0.0
            self._recenter_requested = True
        self._save_panel_config()

    def _draw_quad(
        self,
        tex_id: int,
        tex_w: int,
        tex_h: int,
        view_w: int,
        view_h: int,
        view_pose=None,
        view_fov=None,
    ):
        from OpenGL import GL


        try:
            encoding = GL.glGetFramebufferAttachmentParameteriv(
                GL.GL_DRAW_FRAMEBUFFER,
                GL.GL_COLOR_ATTACHMENT0,
                GL.GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING
            )

            if encoding == GL.GL_SRGB:
                GL.glEnable(GL.GL_FRAMEBUFFER_SRGB)
            else:
                GL.glDisable(GL.GL_FRAMEBUFFER_SRGB)

        except Exception:
            GL.glDisable(GL.GL_FRAMEBUFFER_SRGB)

        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(self.program)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)

        GL.glUniform1i(
            GL.glGetUniformLocation(self.program, "u_tex"),
            0
        )

        if view_pose is not None and view_fov is not None:
            proj = self._projection_matrix_from_fov(view_fov)
            view = self._view_matrix_from_pose(view_pose)
            model = self._model_matrix_for_panel(tex_w, tex_h, view_pose=view_pose)
            mvp = proj @ view @ model
        else:

            mvp = np.eye(4, dtype=np.float64)


        GL.glUniformMatrix4fv(
            self.loc_mvp, 1, GL.GL_TRUE, mvp.astype(np.float32)
        )


        GL.glDisable(GL.GL_CULL_FACE)

        with self._panel_lock:
            curve_value = self.panel_curve
        self._sync_panel_mesh(curve_value)

        GL.glBindVertexArray(self.vao)

        GL.glDrawArrays(
            GL.GL_TRIANGLE_STRIP,
            0,
            self._panel_vertex_count
        )

        GL.glBindVertexArray(0)

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)

    def _init_controller_actions_and_attach(self):
        if getattr(self, "_actions_ready", False):
            return
        instance = self.context.instance
        session = self.context.session
        action_set = self.context.default_action_set

        def p(path):
            return xr.string_to_path(instance, path)

        self._left_hand_path = p("/user/hand/left")
        self._right_hand_path = p("/user/hand/right")

        self._aim_pose_action = xr.create_action(
            action_set,
            xr.ActionCreateInfo(action_type=xr.ActionType.POSE_INPUT,
                                 action_name="pointer_aim_pose",
                                 localized_action_name="Pointer Aim Pose",
                                 subaction_paths=[self._left_hand_path, self._right_hand_path]),
        )
        self._trigger_action = xr.create_action(
            action_set,
            xr.ActionCreateInfo(action_type=xr.ActionType.BOOLEAN_INPUT,
                                 action_name="pointer_trigger_click",
                                 localized_action_name="Pointer Trigger (Left Click)",
                                 subaction_paths=[self._left_hand_path, self._right_hand_path]),
        )
        self._menu_action = xr.create_action(
            action_set,
            xr.ActionCreateInfo(action_type=xr.ActionType.BOOLEAN_INPUT,
                                 action_name="pointer_menu_click",
                                 localized_action_name="Pointer Menu (Right Click)",
                                 subaction_paths=[self._left_hand_path, self._right_hand_path]),
        )

        self._letterbox_action = xr.create_action(
            action_set,
            xr.ActionCreateInfo(action_type=xr.ActionType.BOOLEAN_INPUT,
                                 action_name="pointer_letterbox_toggle",
                                 localized_action_name="Pointer Letterbox Toggle",
                                 subaction_paths=[self._left_hand_path, self._right_hand_path]),
        )

        self._scroll_action = xr.create_action(
            action_set,
            xr.ActionCreateInfo(action_type=xr.ActionType.FLOAT_INPUT,
                                 action_name="pointer_scroll_y",
                                 localized_action_name="Pointer Scroll Y",
                                 subaction_paths=[self._left_hand_path, self._right_hand_path]),
        )

        self._recenter_action = xr.create_action(
            action_set,
            xr.ActionCreateInfo(action_type=xr.ActionType.BOOLEAN_INPUT,
                                 action_name="pointer_stick_click_recenter",
                                 localized_action_name="Pointer Stick Click (Recenter)",
                                 subaction_paths=[self._left_hand_path, self._right_hand_path]),
        )


        self._grip_action = xr.create_action(
            action_set,
            xr.ActionCreateInfo(action_type=xr.ActionType.FLOAT_INPUT,
                                 action_name="pointer_grip_value",
                                 localized_action_name="Pointer Grip (Screen Size/Distance Adjust)",
                                 subaction_paths=[self._left_hand_path, self._right_hand_path]),
        )


        self._stereo_x_action = xr.create_action(
            action_set,
            xr.ActionCreateInfo(action_type=xr.ActionType.FLOAT_INPUT,
                                 action_name="pointer_stereo_strength_x",
                                 localized_action_name="Pointer Stereo Strength (Stick X)",
                                 subaction_paths=[self._left_hand_path, self._right_hand_path]),
        )


        xr.suggest_interaction_profile_bindings(
            instance,
            xr.InteractionProfileSuggestedBinding(
                interaction_profile=p("/interaction_profiles/khr/simple_controller"),
                suggested_bindings=[
                    xr.ActionSuggestedBinding(self._aim_pose_action, p("/user/hand/left/input/aim/pose")),
                    xr.ActionSuggestedBinding(self._aim_pose_action, p("/user/hand/right/input/aim/pose")),
                    xr.ActionSuggestedBinding(self._trigger_action, p("/user/hand/left/input/select/click")),
                    xr.ActionSuggestedBinding(self._trigger_action, p("/user/hand/right/input/select/click")),
                    xr.ActionSuggestedBinding(self._menu_action, p("/user/hand/left/input/menu/click")),
                    xr.ActionSuggestedBinding(self._menu_action, p("/user/hand/right/input/menu/click")),
                ],
            ),
        )


        try:
            xr.suggest_interaction_profile_bindings(
                instance,
                xr.InteractionProfileSuggestedBinding(
                    interaction_profile=p("/interaction_profiles/oculus/touch_controller"),
                    suggested_bindings=[
                        xr.ActionSuggestedBinding(self._aim_pose_action, p("/user/hand/left/input/aim/pose")),
                        xr.ActionSuggestedBinding(self._aim_pose_action, p("/user/hand/right/input/aim/pose")),

                        xr.ActionSuggestedBinding(self._trigger_action, p("/user/hand/left/input/trigger/click")),
                        xr.ActionSuggestedBinding(self._trigger_action, p("/user/hand/right/input/trigger/click")),

                        xr.ActionSuggestedBinding(self._menu_action, p("/user/hand/right/input/a/click")),
                        xr.ActionSuggestedBinding(self._menu_action, p("/user/hand/left/input/x/click")),


                        xr.ActionSuggestedBinding(self._letterbox_action, p("/user/hand/right/input/b/click")),
                        xr.ActionSuggestedBinding(self._letterbox_action, p("/user/hand/left/input/y/click")),

                        xr.ActionSuggestedBinding(self._scroll_action, p("/user/hand/left/input/thumbstick/y")),
                        xr.ActionSuggestedBinding(self._scroll_action, p("/user/hand/right/input/thumbstick/y")),


                        xr.ActionSuggestedBinding(self._stereo_x_action, p("/user/hand/left/input/thumbstick/x")),
                        xr.ActionSuggestedBinding(self._stereo_x_action, p("/user/hand/right/input/thumbstick/x")),

                        xr.ActionSuggestedBinding(self._recenter_action, p("/user/hand/left/input/thumbstick/click")),
                        xr.ActionSuggestedBinding(self._recenter_action, p("/user/hand/right/input/thumbstick/click")),


                        xr.ActionSuggestedBinding(self._grip_action, p("/user/hand/left/input/squeeze/value")),
                        xr.ActionSuggestedBinding(self._grip_action, p("/user/hand/right/input/squeeze/value")),
                    ],
                ),
            )
            print("[OpenXR] oculus/touch_controller bindings OK", flush=True)
        except Exception as e:
            print(f"[OpenXR] oculus/touch_controller binding failed: {e}", flush=True)


        _other_profiles = [

            ("/interaction_profiles/htc/vive_controller",        "input/trackpad/y",   "input/trackpad/click", None),
            ("/interaction_profiles/valve/index_controller",     "input/thumbstick/y", "input/thumbstick/click", "input/squeeze/value"),
            ("/interaction_profiles/microsoft/motion_controller","input/thumbstick/y", "input/thumbstick/click", "input/squeeze/value"),
            ("/interaction_profiles/htc/vive_focus3_controller", "input/thumbstick/y", "input/thumbstick/click", "input/squeeze/value"),
        ]
        for profile_path, stick_y, stick_click, grip_path in _other_profiles:
            try:
                bindings = [
                    xr.ActionSuggestedBinding(self._aim_pose_action, p("/user/hand/left/input/aim/pose")),
                    xr.ActionSuggestedBinding(self._aim_pose_action, p("/user/hand/right/input/aim/pose")),
                    xr.ActionSuggestedBinding(self._trigger_action, p("/user/hand/left/input/trigger/click")),
                    xr.ActionSuggestedBinding(self._trigger_action, p("/user/hand/right/input/trigger/click")),
                    xr.ActionSuggestedBinding(self._menu_action, p("/user/hand/left/input/menu/click")),
                    xr.ActionSuggestedBinding(self._scroll_action, p(f"/user/hand/left/{stick_y}")),
                    xr.ActionSuggestedBinding(self._scroll_action, p(f"/user/hand/right/{stick_y}")),
                    xr.ActionSuggestedBinding(self._recenter_action, p(f"/user/hand/left/{stick_click}")),
                    xr.ActionSuggestedBinding(self._recenter_action, p(f"/user/hand/right/{stick_click}")),

                    xr.ActionSuggestedBinding(self._stereo_x_action, p(f"/user/hand/left/{stick_y[:-1]}x")),
                    xr.ActionSuggestedBinding(self._stereo_x_action, p(f"/user/hand/right/{stick_y[:-1]}x")),
                ]
                if grip_path is not None:
                    bindings.append(xr.ActionSuggestedBinding(self._grip_action, p(f"/user/hand/left/{grip_path}")))
                    bindings.append(xr.ActionSuggestedBinding(self._grip_action, p(f"/user/hand/right/{grip_path}")))

                xr.suggest_interaction_profile_bindings(
                    instance,
                    xr.InteractionProfileSuggestedBinding(
                        interaction_profile=p(profile_path),
                        suggested_bindings=bindings,
                    ),
                )
            except Exception as e:
                print(f"[OpenXR] profile binding skipped ({profile_path}): {e}", flush=True)

        self._aim_space_left = xr.create_action_space(
            session, xr.ActionSpaceCreateInfo(action=self._aim_pose_action, subaction_path=self._left_hand_path))
        self._aim_space_right = xr.create_action_space(
            session, xr.ActionSpaceCreateInfo(action=self._aim_pose_action, subaction_path=self._right_hand_path))


        self._actions_ready = True
        print("[OpenXR] Controller pointer actions ready (attach will happen inside frame_loop)", flush=True)

    def _update_controller_pointer(self, frame_state):
        if not self._actions_ready:
            return

        session = self.context.session
        base_space = self.context.space


        try:
            xr.sync_actions(
                session,
                xr.ActionsSyncInfo(
                    active_action_sets=[
                        xr.ActiveActionSet(
                            action_set=aset,
                            subaction_path=xr.NULL_PATH,
                        )
                        for aset in self.context.action_sets
                    ]
                ),
            )
            self._sync_warned = False
        except Exception as e:


            if not getattr(self, "_sync_warned", False):
                print(f"[OpenXR] sync_actions warning: {e}", flush=True)
                self._sync_warned = True
            return

        hands = {
            "left": self._left_hand_path,
            "right": self._right_hand_path,
        }
        spaces = {
            "left": self._aim_space_left,
            "right": self._aim_space_right,
        }

        trigger_state = {"left": False, "right": False}
        menu_state = {"left": False, "right": False}
        letterbox_state = {"left": False, "right": False}
        grip_state = {"left": False, "right": False}
        scroll_state = {"left": 0.0, "right": 0.0}
        recenter_state = {"left": False, "right": False}
        stereo_x_state = {"left": 0.0, "right": 0.0}


        for hand, path in hands.items():
            try:
                ts = xr.get_action_state_boolean(
                    session,
                    xr.ActionStateGetInfo(
                        action=self._trigger_action,
                        subaction_path=path,
                    ),
                )
                trigger_state[hand] = bool(ts.is_active and ts.current_state)
            except Exception:
                trigger_state[hand] = self._prev_trigger.get(hand, False)

            try:
                ms = xr.get_action_state_boolean(
                    session,
                    xr.ActionStateGetInfo(
                        action=self._menu_action,
                        subaction_path=path,
                    ),
                )
                menu_state[hand] = bool(ms.is_active and ms.current_state)
            except Exception:
                menu_state[hand] = self._prev_menu.get(hand, False)

            try:
                ls = xr.get_action_state_boolean(
                    session,
                    xr.ActionStateGetInfo(
                        action=self._letterbox_action,
                        subaction_path=path,
                    ),
                )
                letterbox_state[hand] = bool(ls.is_active and ls.current_state)
            except Exception:
                letterbox_state[hand] = self._prev_letterbox.get(hand, False)

            try:
                gs = xr.get_action_state_float(
                    session,
                    xr.ActionStateGetInfo(
                        action=self._grip_action,
                        subaction_path=path,
                    ),
                )
                grip_state[hand] = bool(
                    gs.is_active and float(gs.current_state) >= self._grip_threshold
                )
            except Exception:
                grip_state[hand] = self._prev_grip.get(hand, False)

            try:
                ss = xr.get_action_state_float(
                    session,
                    xr.ActionStateGetInfo(
                        action=self._scroll_action,
                        subaction_path=path,
                    ),
                )
                scroll_state[hand] = float(ss.current_state) if ss.is_active else 0.0
            except Exception:
                scroll_state[hand] = 0.0

            try:
                sxs = xr.get_action_state_float(
                    session,
                    xr.ActionStateGetInfo(
                        action=self._stereo_x_action,
                        subaction_path=path,
                    ),
                )
                stereo_x_state[hand] = float(sxs.current_state) if sxs.is_active else 0.0
            except Exception:
                stereo_x_state[hand] = 0.0

            try:
                rs = xr.get_action_state_boolean(
                    session,
                    xr.ActionStateGetInfo(
                        action=self._recenter_action,
                        subaction_path=path,
                    ),
                )
                recenter_state[hand] = bool(rs.is_active and rs.current_state)
            except Exception:
                recenter_state[hand] = self._prev_recenter.get(hand, False)


        for hand in ("left", "right"):
            if (trigger_state[hand] and not self._prev_trigger.get(hand, False)) or\
               (menu_state[hand] and not self._prev_menu.get(hand, False)):
                self._active_hand = hand


        for hand in ("left", "right"):
            if letterbox_state[hand] and not self._prev_letterbox.get(hand, False):
                if self.shared_auto_crop is not None:
                    self.shared_auto_crop[0] = not self.shared_auto_crop[0]
                    self._letterbox_notify_seq += 1
                    print(
                        f"[OpenXR] Letterbox toggled via controller → "
                        f"{'ON' if self.shared_auto_crop[0] else 'OFF'}",
                        flush=True,
                    )


        if self.shared_divergence is not None:
            for hand in ("left", "right"):
                x = stereo_x_state.get(hand, 0.0)
                prev_dir = self._prev_stereo_dir.get(hand, 0)

                if x >= self._stereo_x_trigger and prev_dir != 1:
                    self.shared_divergence[0] = round(
                        min(self._div_max, self.shared_divergence[0] + self._div_step), 2
                    )
                    self._prev_stereo_dir[hand] = 1
                    self._apply_auto_mode_if_enabled()
                    print(f"[OpenXR] Stereo strength +step via controller({hand}) → {self.shared_divergence[0]:.1f}", flush=True)
                elif x <= -self._stereo_x_trigger and prev_dir != -1:
                    self.shared_divergence[0] = round(
                        max(self._div_min, self.shared_divergence[0] - self._div_step), 2
                    )
                    self._prev_stereo_dir[hand] = -1
                    self._apply_auto_mode_if_enabled()
                    print(f"[OpenXR] Stereo strength -step via controller({hand}) → {self.shared_divergence[0]:.1f}", flush=True)
                elif abs(x) <= self._stereo_x_release and prev_dir != 0:
                    self._prev_stereo_dir[hand] = 0


        for hand in ("left", "right"):
            held = grip_state[hand]
            prev = self._prev_grip.get(hand, False)

            if held and not prev:
                try:
                    loc = xr.locate_space(
                        spaces[hand],
                        base_space,
                        frame_state.predicted_display_time,
                    )
                    if loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                        self._active_grip_hand = hand
                        self._grip_start_x = float(loc.pose.position.x)
                        self._grip_start_y = float(loc.pose.position.y)
                        self._grip_axis_lock = None
                        with self._panel_lock:
                            self._grip_start_height = self.panel_height
                            self._grip_start_distance = self.panel_distance
                except Exception:
                    pass

            elif held and self._active_grip_hand == hand:
                try:
                    loc = xr.locate_space(
                        spaces[hand],
                        base_space,
                        frame_state.predicted_display_time,
                    )
                    if loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                        current_x = float(loc.pose.position.x)
                        current_y = float(loc.pose.position.y)
                        delta_x = current_x - self._grip_start_x
                        delta_y = current_y - self._grip_start_y

                        # Deadzone-based axis lock so left/right (size) and
                        # up/down (distance) movements never interfere.
                        if self._grip_axis_lock is None:
                            if abs(delta_x) >= self._grip_deadzone and abs(delta_x) > abs(delta_y):
                                self._grip_axis_lock = "size"
                            elif abs(delta_y) >= self._grip_deadzone and abs(delta_y) > abs(delta_x):
                                self._grip_axis_lock = "dist"

                        if self._grip_axis_lock == "size":
                            sensitivity = getattr(self, "_grip_size_sensitivity", 3.0)
                            lo, hi = self._panel_height_range
                            with self._panel_lock:
                                self.panel_height = max(
                                    lo,
                                    min(
                                        hi,
                                        self._grip_start_height + delta_x * sensitivity,
                                    ),
                                )
                        elif self._grip_axis_lock == "dist":
                            sensitivity = getattr(self, "_grip_dist_sensitivity", 4.0)
                            lo, hi = self._panel_distance_range
                            with self._panel_lock:
                                self.panel_distance = max(
                                    lo,
                                    min(
                                        hi,
                                        self._grip_start_distance + delta_y * sensitivity,
                                    ),
                                )
                except Exception:
                    pass

            elif (not held) and prev and self._active_grip_hand == hand:
                self._active_grip_hand = None
                self._grip_axis_lock = None
                self._save_panel_config()
                with self._panel_lock:
                    d = self.panel_distance
                    h = self.panel_height
                    y = self.panel_y_offset
                    c = self.panel_curve
                print(f"SYNC_VR_PANEL:{d:.2f}:{h:.2f}:{y:.2f}:{c:.1f}", flush=True)
                print("[OpenXR] Grip released → panel size/distance locked", flush=True)

        self._prev_grip = grip_state.copy()


        if self._active_hand is not None:
            hand = self._active_hand
            try:
                loc = xr.locate_space(
                    spaces[hand],
                    base_space,
                    frame_state.predicted_display_time,
                )
                if (loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT) and\
                   (loc.location_flags & xr.SPACE_LOCATION_ORIENTATION_VALID_BIT):
                    pos = loc.pose.position
                    q = loc.pose.orientation
                    rot = self._quat_to_mat3(q.x, q.y, q.z, q.w)
                    ray_o = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
                    ray_d = rot @ np.array([0.0, 0.0, -1.0])

                    M = self._model_matrix_for_panel(self.tex_w, self.tex_h)
                    M_inv = np.linalg.inv(M)
                    o_local = (M_inv @ np.array([*ray_o, 1.0]))[:3]
                    d_local = M_inv[:3, :3] @ ray_d

                    if abs(d_local[2]) > 1e-6:
                        t_hit = -o_local[2] / d_local[2]
                        if t_hit > 0:
                            hit = o_local + d_local * t_hit
                            if -0.5 <= hit[0] <= 0.5 and -0.5 <= hit[1] <= 0.5:
                                m = self._pointer_margin
                                norm_u = (hit[0] - (-0.5 + m)) / (1.0 - 2.0 * m)
                                norm_v = (hit[1] - (-0.5 + m)) / (1.0 - 2.0 * m)
                                u = max(0.0, min(1.0, norm_u))
                                v = max(0.0, min(1.0, 1.0 - norm_v))

                                dist = math.hypot(
                                    u - self._smooth_u,
                                    v - self._smooth_v,
                                )
                                if dist >= self._pointer_deadzone:
                                    alpha = 0.15 + min(0.45, dist * 6.0)
                                    self._smooth_u += alpha * (u - self._smooth_u)
                                    self._smooth_v += alpha * (v - self._smooth_v)

                                self._virtual_cursor_u = self._smooth_u
                                self._virtual_cursor_v = self._smooth_v
                                self._virtual_cursor_valid = True

                                if self.shared_cursor is not None:
                                    self.shared_cursor[0] = self._smooth_u
                                    self.shared_cursor[1] = self._smooth_v
                                    self.shared_cursor[2] = True
                                    self.shared_cursor[3] = time.perf_counter()

                                    if len(self.shared_cursor) > 4:
                                        self.shared_cursor[4] = 0
            except Exception:
                pass


        if self._virtual_cursor_valid:
            for hand in ("left", "right"):
                prev_t = self._prev_trigger.get(hand, False)
                prev_m = self._prev_menu.get(hand, False)

                if trigger_state[hand] and not prev_t and not self._injected_left_down:
                    _send_mouse_button_at_virtual_cursor(
                        self._virtual_cursor_u,
                        self._virtual_cursor_v,
                        win32con.MOUSEEVENTF_LEFTDOWN,
                    )
                    self._injected_left_down = True

                if (not trigger_state[hand]) and prev_t and self._injected_left_down:
                    _send_mouse_button_at_virtual_cursor(
                        self._virtual_cursor_u,
                        self._virtual_cursor_v,
                        win32con.MOUSEEVENTF_LEFTUP,
                    )
                    self._injected_left_down = False

                if menu_state[hand] and not prev_m and not self._injected_right_down:
                    _send_mouse_button_at_virtual_cursor(
                        self._virtual_cursor_u,
                        self._virtual_cursor_v,
                        win32con.MOUSEEVENTF_RIGHTDOWN,
                    )
                    self._injected_right_down = True

                if (not menu_state[hand]) and prev_m and self._injected_right_down:
                    _send_mouse_button_at_virtual_cursor(
                        self._virtual_cursor_u,
                        self._virtual_cursor_v,
                        win32con.MOUSEEVENTF_RIGHTUP,
                    )
                    self._injected_right_down = False


        now = time.perf_counter()
        scroll_value = 0.0
        if abs(scroll_state["left"]) >= abs(scroll_state["right"]):
            scroll_value = scroll_state["left"]
        else:
            scroll_value = scroll_state["right"]

        if abs(scroll_value) >= self._wheel_deadzone and\
           (now - self._last_wheel_time) >= self._wheel_repeat_interval:
            wheel_delta = 120 if scroll_value > 0 else -120
            if self._virtual_cursor_valid:
                _send_wheel_at_virtual_cursor(
                    self._virtual_cursor_u,
                    self._virtual_cursor_v,
                    wheel_delta,
                )
            self._last_wheel_time = now


        for hand in ("left", "right"):
            if recenter_state[hand] and not self._prev_recenter.get(hand, False):
                self.recenter_panel()
                print(
                    f"[OpenXR] {hand} stick click → panel recentered",
                    flush=True,
                )


        self._prev_trigger = trigger_state.copy()
        self._prev_menu = menu_state.copy()
        self._prev_letterbox = letterbox_state.copy()
        self._prev_recenter = recenter_state.copy()
    def _render_loop(self):
        from OpenGL import GL

        print("[OpenXR] render_loop entered", flush=True)

        try:

            context_provider = GLFWOffscreenContextProvider()
            self.context = ContextObject(
                context_provider=context_provider,
                instance_create_info=xr.InstanceCreateInfo(
                    enabled_extension_names=[
                        xr.KHR_OPENGL_ENABLE_EXTENSION_NAME,
                    ],
                ),
            )

            with self.context:
                try:
                    self._init_controller_actions_and_attach()
                except Exception as e:
                    print(f"[OpenXR] controller action init failed: {e}", flush=True)

                for frame_state in self.context.frame_loop():
                    if not self._running:
                        break

                    if not self._gl_ready:
                        try:
                            self._init_gl_resources()
                            self._ensure_textures(2, 2)
                            black = torch.zeros((2, 2, 3), dtype=torch.uint8)
                            self._upload_tensor(self.left_tex, black)
                            self._upload_tensor(self.right_tex, black)
                            self._render_ready.set()
                            print("[OpenXR] GL resources initialized; compositor rendering is ready", flush=True)
                        except Exception as e:
                            self._render_error = e
                            self._render_ready.set()
                            print(f"[OpenXR] GL init failed: {e}", flush=True)
                            break

                    with self._lock:
                        left = self._latest_left
                        right = self._latest_right
                        is_new = self._has_new_frame
                        self._has_new_frame = False


                    if left is None or right is None:
                        draw_w, draw_h = 2, 2
                    else:
                        if left.dim() == 3 and left.shape[0] == 3:
                            draw_h, draw_w = left.shape[1], left.shape[2]
                        else:
                            draw_h, draw_w = left.shape[0], left.shape[1]
                        self._ensure_textures(draw_w, draw_h)
                        if is_new:
                            self._upload_tensor(self.left_tex, left)
                            self._upload_tensor(self.right_tex, right)

                        if not hasattr(self, "_draw_count"):
                            self._draw_count = 0
                        self._draw_count += 1
                        if self._draw_count % 30 == 1:
                            print(f"[OpenXR] drawing frame #{self._draw_count}", flush=True)

                    for view_index, view in enumerate(self.context.view_loop(frame_state)):
                        tex = self.left_tex if view_index == 0 else self.right_tex
                        try:
                            viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
                            view_w = int(viewport[2])
                            view_h = int(viewport[3])
                        except Exception:
                            view_w = draw_w
                            view_h = draw_h

                        self._draw_quad(
                            tex,
                            draw_w,
                            draw_h,
                            view_w,
                            view_h,
                            view_pose=view.pose,
                            view_fov=view.fov,
                        )

                    self._update_controller_pointer(frame_state)

        except Exception as e:
            self._render_error = e
            self._render_ready.set()
            print(f"[OpenXR] render_loop error: {e}", flush=True)
        finally:
            _release_injected_mouse_buttons()
            self._injected_left_down = False
            self._injected_right_down = False
            print("[OpenXR] render_loop exited", flush=True)


    def submit(self, left_t: torch.Tensor, right_t: torch.Tensor):
        if not self._running:
            return
        with self._lock:
            self._latest_left = left_t
            self._latest_right = right_t
            self._has_new_frame = True

    def destroy(self):
        print("[OpenXR] destroy requested", flush=True)
        self._save_panel_config()
        self._running = False
        _release_injected_mouse_buttons()
        self._injected_left_down = False
        self._injected_right_down = False


        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        print("[OpenXR] Session destroyed", flush=True)

class Win32GLOverlay:
    def __init__(self, width: int, height: int, left: int = 0, top: int = 0,
                 click_through: bool = True, vr_mode: bool = False,
                 capture_visible: bool = False, preserve_aspect: bool = False):
        if not _HAS_OPENGL:
            raise RuntimeError("PyOpenGL is required.")
        self.width = width
        self.height = height
        self.left = left
        self.top = top
        self.click_through = click_through
        self.vr_mode = vr_mode
        # When True (Full SBS format), the packed frame keeps its original
        # (non-16:9) aspect ratio on screen (letterboxed/pillarboxed) instead
        # of being stretched to fill the monitor like the half-SBS look.
        self.preserve_aspect = preserve_aspect
        # Optional reference to a live-updating [float] list (shared_fsbs_scale).
        # 0.0 = full Full-SBS screen size; higher values shrink the displayed
        # image in real time (checked every frame in blit_tensor).
        self.fsbs_scale_ref = None
        # PC mode always uses capture_visible=True (WDA_NONE + Mag HWND exclude)
        self.capture_visible = bool(vr_mode or capture_visible)
        self.hwnd = None
        self.hdc = None
        self.hglrc = None
        self.tex_id = None
        self.pbo_id = None
        self.cuda_resource = None
        self.cudart = None
        self.program = None
        self.vbo = None
        self.vbo_left = None
        self.vbo_right = None
        self.loc_pos = -1
        self.loc_uv = -1
        self.tex_w = 0
        self.tex_h = 0

        try:
            self._create_window()
            self._create_gl_context()
            self._init_gl_resources()
            self._apply_exclude_capture()
        except Exception:
            self.destroy()
            raise

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        if msg == win32con.WM_DESTROY:
            win32gui.PostQuitMessage(0)
            return 0
        if msg == win32con.WM_ERASEBKGND:
            return 1
        if msg == win32con.WM_PAINT:
            hdc, ps = win32gui.BeginPaint(hwnd)
            win32gui.EndPaint(hwnd, ps)
            return 0
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def set_click_through(self, enable: bool):
        self.click_through = enable
        hwnd = self.hwnd
        if not hwnd: return
        ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        if enable: ex_style |= WS_EX_TRANSPARENT
        else: ex_style &= ~WS_EX_TRANSPARENT
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style)

    def _create_window(self):
        hinst = win32api.GetModuleHandle(None)
        wc = win32gui.WNDCLASS()
        wc.hInstance = hinst
        wc.lpszClassName = "Live3D_Win32GL_Class"
        wc.lpfnWndProc = self._wnd_proc
        wc.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
        wc.hbrBackground = 0
        wc.style = win32con.CS_OWNDC | win32con.CS_HREDRAW | win32con.CS_VREDRAW

        try: win32gui.RegisterClass(wc)
        except win32gui.error: pass

        ex = WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        if self.click_through:
            ex |= WS_EX_TRANSPARENT

        self.hwnd = win32gui.CreateWindowEx(
            ex, "Live3D_Win32GL_Class", "DepthLive3D Overlay",
            WS_POPUP | WS_VISIBLE,
            self.left, self.top, self.width, self.height,
            0, 0, hinst, None,
        )
        win32gui.SetWindowPos(
            self.hwnd, HWND_TOPMOST,
            self.left, self.top, self.width, self.height,
            SWP_SHOWWINDOW | SWP_NOACTIVATE,
        )
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
        win32gui.UpdateWindow(self.hwnd)

        for _ in range(8):
            win32gui.PumpWaitingMessages()
            time.sleep(0.008)

    def _create_gl_context(self):
        try: opengl32.wglMakeCurrent(None, None)
        except Exception: pass

        self.hdc = win32gui.GetDC(self.hwnd)
        PFD_DRAW_TO_WINDOW = 0x00000004
        PFD_SUPPORT_OPENGL = 0x00000020
        PFD_DOUBLEBUFFER = 0x00000001
        PFD_TYPE_RGBA = 0
        PFD_MAIN_PLANE = 0

        pfd = PIXELFORMATDESCRIPTOR()
        pfd.nSize = ctypes.sizeof(PIXELFORMATDESCRIPTOR)
        pfd.nVersion = 1
        pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER
        pfd.iPixelType = PFD_TYPE_RGBA
        pfd.cColorBits = 24
        pfd.cDepthBits = 16
        pfd.iLayerType = PFD_MAIN_PLANE

        fmt = gdi32.ChoosePixelFormat(self.hdc, ctypes.byref(pfd))
        if not fmt: raise RuntimeError(f"ChoosePixelFormat failed (GetLastError={ctypes.GetLastError()})")
        if not gdi32.SetPixelFormat(self.hdc, fmt, ctypes.byref(pfd)):
            raise RuntimeError(f"SetPixelFormat failed (GetLastError={ctypes.GetLastError()})")

        self.hglrc = opengl32.wglCreateContext(self.hdc)
        if not self.hglrc: raise RuntimeError(f"wglCreateContext failed (GetLastError={ctypes.GetLastError()})")

        if not opengl32.wglMakeCurrent(self.hdc, self.hglrc):
            err = ctypes.GetLastError()
            opengl32.wglDeleteContext(self.hglrc)
            self.hglrc = None
            raise RuntimeError(f"wglMakeCurrent failed (GetLastError={err})")

        if not opengl32.wglGetCurrentContext():
            raise RuntimeError("Failed to activate OpenGL context")

        if not self.vr_mode:
            try:
                ex_style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE)
                ex_style |= WS_EX_LAYERED
                if self.click_through:
                    ex_style |= WS_EX_TRANSPARENT
                else:
                    ex_style &= ~WS_EX_TRANSPARENT
                win32gui.SetWindowLong(self.hwnd, win32con.GWL_EXSTYLE, ex_style)
                win32gui.SetLayeredWindowAttributes(self.hwnd, 0, 255, win32con.LWA_ALPHA)
                win32gui.SetWindowPos(
                    self.hwnd, HWND_TOPMOST,
                    self.left, self.top, self.width, self.height,
                    SWP_SHOWWINDOW | SWP_NOACTIVATE | 0x0020
                )
            except Exception as e:
                print(f"[Warning] Failed to apply LAYERED/click-through style: {e}")
            # Style change can drop affinity — re-apply for PC VD-compat
            try:
                self._apply_exclude_capture()
            except Exception:
                pass

        if self.vr_mode or getattr(self, "capture_visible", False):
            mode_str = "PC/VR Mode (WDA_NONE, Virtual Desktop compatible)"
        else:
            mode_str = "PC Mode (WDA_EXCLUDEFROMCAPTURE)"
        print(f"[Win32GL] OpenGL context created successfully ({mode_str})")

    def _init_gl_resources(self):
        vert = """
        #version 130
        in vec2 pos;
        in vec2 uv;
        out vec2 v_uv;
        void main() {
            gl_Position = vec4(pos, 0.0, 1.0);
            v_uv = uv;
        }
        """
        frag = """
        #version 130
        in vec2 v_uv;
        out vec4 fragColor;
        uniform sampler2D u_tex;
        void main() {
            fragColor = texture(u_tex, v_uv);
        }
        """
        self.program = shaders.compileProgram(
            shaders.compileShader(vert, GL_VERTEX_SHADER),
            shaders.compileShader(frag, GL_FRAGMENT_SHADER),
        )

        vertices = np.array([
            -1.0, -1.0,  0.0, 1.0,
             1.0, -1.0,  1.0, 1.0,
             1.0,  1.0,  1.0, 0.0,
            -1.0,  1.0,  0.0, 0.0,
        ], dtype=np.float32)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # Separate full -1..1 NDC quads that each sample only one half (one
        # eye) of the packed SBS texture. Used so each eye can be fitted /
        # shrunk independently within its own half of the screen (via
        # glViewport) instead of the whole packed image being scaled as one
        # unit around the screen's overall center.
        left_vertices = np.array([
            -1.0, -1.0,  0.0, 1.0,
             1.0, -1.0,  0.5, 1.0,
             1.0,  1.0,  0.5, 0.0,
            -1.0,  1.0,  0.0, 0.0,
        ], dtype=np.float32)
        right_vertices = np.array([
            -1.0, -1.0,  0.5, 1.0,
             1.0, -1.0,  1.0, 1.0,
             1.0,  1.0,  1.0, 0.0,
            -1.0,  1.0,  0.5, 0.0,
        ], dtype=np.float32)

        self.vbo_left = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_left)
        glBufferData(GL_ARRAY_BUFFER, left_vertices.nbytes, left_vertices, GL_STATIC_DRAW)

        self.vbo_right = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_right)
        glBufferData(GL_ARRAY_BUFFER, right_vertices.nbytes, right_vertices, GL_STATIC_DRAW)

        self.loc_pos = glGetAttribLocation(self.program, "pos")
        self.loc_uv = glGetAttribLocation(self.program, "uv")

        self.cudart = CUDART_GL(0)
        print("[Win32GL] GL resources + CUDA interop initialization complete")

    def _apply_exclude_capture(self):
        """PC mode (capture_visible) and VR mode → always WDA_NONE for Virtual Desktop."""
        if not self.hwnd:
            return
        if self.vr_mode or getattr(self, "capture_visible", False):
            ok = False
            for i in range(5):
                ok = bool(user32.SetWindowDisplayAffinity(int(self.hwnd), WDA_NONE))
                if ok:
                    break
                time.sleep(0.05)
            if ok:
                print("[Win32GL] Virtual Desktop compat applied (WDA_NONE)", flush=True)
            else:
                print(f"[Warning] SetWindowDisplayAffinity(WDA_NONE) failed err={ctypes.GetLastError()}", flush=True)
        else:
            for i in range(5):
                ok = user32.SetWindowDisplayAffinity(int(self.hwnd), WDA_EXCLUDEFROMCAPTURE)
                if ok:
                    print(f"[Win32GL] Capture exclusion applied successfully ({i+1} times)", flush=True)
                    return
                time.sleep(0.05)
            print("[Warning] Final failure to apply capture exclusion", flush=True)

    def _ensure_texture(self, w: int, h: int):
        if self.tex_w == w and self.tex_h == h and self.tex_id is not None:
            return
        if self.cuda_resource is not None:
            self.cudart.unregister(self.cuda_resource)
            self.cuda_resource = None
        if self.pbo_id is not None:
            glDeleteBuffers(1, [self.pbo_id])
            self.pbo_id = None
        if self.tex_id is not None:
            glDeleteTextures(1, [self.tex_id])
            self.tex_id = None

        self.tex_w, self.tex_h = w, h
        self.tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.pbo_id = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * 3, None, GL_STREAM_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        self.cuda_resource = self.cudart.register_buffer(self.pbo_id)

    def blit_tensor(self, tensor: torch.Tensor):
        if tensor is None or self.hwnd is None: return
        opengl32.wglMakeCurrent(self.hdc, self.hglrc)

        if tensor.dim() == 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0).contiguous()
        h, w = tensor.shape[:2]
        self._ensure_texture(w, h)

        if tensor.dtype != torch.uint8:
            tensor = tensor.clamp(0, 255).byte()
        tensor = tensor.contiguous()
        size = w * h * 3
        src_ptr = tensor.data_ptr()

        self.cudart.map(self.cuda_resource)
        try:
            dst_ptr, _ = self.cudart.get_mapped_pointer(self.cuda_resource)
            self.cudart.memcpy_d2d(dst_ptr, src_ptr, size)
        finally:
            self.cudart.unmap(self.cuda_resource)

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glUniform1i(glGetUniformLocation(self.program, "u_tex"), 0)

        def _draw_quad(vbo):
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glEnableVertexAttribArray(self.loc_pos)
            glVertexAttribPointer(self.loc_pos, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            glEnableVertexAttribArray(self.loc_uv)
            glVertexAttribPointer(self.loc_uv, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
            glDisableVertexAttribArray(self.loc_pos)
            glDisableVertexAttribArray(self.loc_uv)

        if self.preserve_aspect:
            # Full SBS: the packed frame is two eye images side by side
            # (2*eye_w × eye_h). Each eye keeps its own true (non-16:9)
            # aspect ratio and is fitted/centered independently within its
            # own half of the screen (left eye in the left half, right eye
            # in the right half), instead of the whole packed image being
            # treated as one block and scaled around the screen's overall
            # center. That way shrinking the "Full SBS Screen Size" makes
            # each eye's image shrink toward the center of its own half,
            # not toward the middle seam of the screen.
            eye_w = max(1, w // 2)
            eye_h = max(1, h)
            half_w = self.width / 2.0
            half_h = float(self.height)

            box_w = half_w
            box_h = box_w * (eye_h / eye_w)
            if box_h > half_h:
                box_h = half_h
                box_w = box_h * (eye_w / eye_h)

            # Full SBS Screen Size: shrink each eye's fitted box further, in
            # real time, around its own center (within its own half of the
            # screen). 0 = no change (current full size); each step shrinks
            # the displayed image, growing the letterbox/pillarbox border
            # around it. The HUD/FPS text is already baked into the frame
            # content before this scaling, so it shrinks along with the
            # image and always stays inside the visible screen area.
            fsbs_scale = 0.0
            if self.fsbs_scale_ref is not None:
                try:
                    fsbs_scale = max(0.0, float(self.fsbs_scale_ref[0]))
                except Exception:
                    fsbs_scale = 0.0
            shrink = 1.0 / (1.0 + fsbs_scale * 0.1) if fsbs_scale > 0.0 else 1.0
            new_w = max(2, int(round(box_w * shrink)))
            new_h = max(2, int(round(box_h * shrink)))

            # Left eye: centered within [0, half_w)
            lvp_x = int(round((half_w - new_w) / 2.0))
            lvp_y = int(round((half_h - new_h) / 2.0))
            glViewport(lvp_x, lvp_y, new_w, new_h)
            _draw_quad(self.vbo_left)

            # Right eye: centered within [half_w, width)
            rvp_x = int(round(half_w)) + int(round((half_w - new_w) / 2.0))
            rvp_y = lvp_y
            glViewport(rvp_x, rvp_y, new_w, new_h)
            _draw_quad(self.vbo_right)
        else:
            # Always stretch the packed frame to fill the entire overlay window.
            # Full-SBS (2W×H) and full-TB (W×2H) have non-16:9 aspect ratios;
            # stretching them onto the monitor produces the classic half-SBS /
            # half-TB visual with no letterbox or pillarbox black bars.
            glViewport(0, 0, self.width, self.height)
            _draw_quad(self.vbo)

        gdi32.SwapBuffers(self.hdc)

    def keep_topmost(self):
        if not self.hwnd: return
        try:
            if not self.vr_mode:
                ex_style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE)
                ex_style |= WS_EX_LAYERED
                if self.click_through:
                    ex_style |= WS_EX_TRANSPARENT
                else:
                    ex_style &= ~WS_EX_TRANSPARENT
                win32gui.SetWindowLong(self.hwnd, win32con.GWL_EXSTYLE, ex_style)
            win32gui.SetWindowPos(
                self.hwnd, HWND_TOPMOST,
                self.left, self.top, self.width, self.height,
                SWP_NOACTIVATE | SWP_SHOWWINDOW | 0x0020,
            )
            # PC mode always uses capture_visible → WDA_NONE (Virtual Desktop compatible)
            if self.vr_mode or getattr(self, "capture_visible", False):
                user32.SetWindowDisplayAffinity(int(self.hwnd), WDA_NONE)
            else:
                user32.SetWindowDisplayAffinity(int(self.hwnd), WDA_EXCLUDEFROMCAPTURE)
        except Exception: pass

    def pump(self): win32gui.PumpWaitingMessages()
    def should_close(self): return False

    def destroy(self):
        if self.hglrc and self.hdc:
            try: opengl32.wglMakeCurrent(self.hdc, self.hglrc)
            except Exception: pass
        if self.cuda_resource is not None:
            try: self.cudart.unregister(self.cuda_resource)
            except Exception: pass
            self.cuda_resource = None
        try:
            if self.tex_id is not None: glDeleteTextures(1, [self.tex_id]); self.tex_id = None
            if self.pbo_id is not None: glDeleteBuffers(1, [self.pbo_id]); self.pbo_id = None
            if self.vbo is not None: glDeleteBuffers(1, [self.vbo]); self.vbo = None
            if getattr(self, "vbo_left", None) is not None: glDeleteBuffers(1, [self.vbo_left]); self.vbo_left = None
            if getattr(self, "vbo_right", None) is not None: glDeleteBuffers(1, [self.vbo_right]); self.vbo_right = None
            if self.program is not None: glDeleteProgram(self.program); self.program = None
        except Exception: pass
        if self.hglrc:
            try: opengl32.wglMakeCurrent(None, None); opengl32.wglDeleteContext(self.hglrc)
            except Exception: pass
            self.hglrc = None
        if self.hdc and self.hwnd:
            try: win32gui.ReleaseDC(self.hwnd, self.hdc)
            except Exception: pass
            self.hdc = None
        if self.hwnd:
            try:
                user32.SetWindowDisplayAffinity(int(self.hwnd), WDA_NONE)
                win32gui.DestroyWindow(self.hwnd)
            except Exception: pass
            self.hwnd = None
        try:
            hinst = win32api.GetModuleHandle(None)
            win32gui.UnregisterClass("Live3D_Win32GL_Class", hinst)
        except Exception: pass


def _get_monitor_friendly_name(hmon):
    """Best-effort lookup of a human-readable monitor/device name for hmon."""
    try:
        info = win32api.GetMonitorInfo(hmon)
        device_name = info.get("Device", "")
        # First try EnumDisplayDevices on the adapter to get the attached
        # monitor's friendly name (e.g. "PnP Monitor", "Virtual Display").
        try:
            monitor_dev = win32api.EnumDisplayDevices(device_name, 0)
            name = getattr(monitor_dev, "DeviceString", "") or ""
            if name:
                return name
        except Exception:
            pass
        # Fall back to the adapter's own device string.
        try:
            adapter_dev = win32api.EnumDisplayDevices(None, 0)
            name = getattr(adapter_dev, "DeviceString", "") or ""
            if name:
                return name
        except Exception:
            pass
    except Exception:
        pass
    return "Unknown Display"


def get_monitor_list():
    monitors = []
    try:
        for i, mon in enumerate(win32api.EnumDisplayMonitors(None, None)):
            hmon, hdc, rect = mon
            left, top, right, bottom = rect
            name = _get_monitor_friendly_name(hmon)
            monitors.append({
                "index": i, "left": left, "top": top,
                "width": right - left, "height": bottom - top,
                "name": name,
            })
    except Exception as e:
        print(f"[Warning] Monitor enumeration failed: {e}")
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        monitors.append({"index": 0, "left": 0, "top": 0, "width": w, "height": h, "name": "Unknown Display"})
    return monitors

def get_current_display_mode(device_name: str = None):
    import win32api
    import win32con
    if device_name is None:
        device_name = win32api.EnumDisplayDevices(None, 0).DeviceName
    mode = win32api.EnumDisplaySettings(device_name, win32con.ENUM_CURRENT_SETTINGS)
    return {
        "width": mode.PelsWidth,
        "height": mode.PelsHeight,
        "bpp": mode.BitsPerPel,
        "freq": mode.DisplayFrequency,
        "device_name": device_name,
    }

def change_display_resolution(width: int, height: int, device_name: str = None, freq: int = 0):
    import win32api
    import win32con
    if device_name is None:
        device_name = win32api.EnumDisplayDevices(None, 0).DeviceName


    i = 0
    best = None
    while True:
        try:
            mode = win32api.EnumDisplaySettings(device_name, i)
        except Exception:
            break
        if mode.PelsWidth == width and mode.PelsHeight == height:
            if best is None:
                best = mode
            elif freq > 0 and mode.DisplayFrequency == freq:
                best = mode
                break
            elif mode.DisplayFrequency > best.DisplayFrequency:
                best = mode
        i += 1

    if best is None:
        mode = win32api.EnumDisplaySettings(device_name, win32con.ENUM_CURRENT_SETTINGS)
        mode.PelsWidth = width
        mode.PelsHeight = height
        if freq > 0:
            mode.DisplayFrequency = freq

        mode.Fields = (
            win32con.DM_PELSWIDTH |
            win32con.DM_PELSHEIGHT |
            win32con.DM_BITSPERPEL |
            win32con.DM_DISPLAYFREQUENCY
        )
        result = win32api.ChangeDisplaySettingsEx(device_name, mode, 0)
    else:
        result = win32api.ChangeDisplaySettingsEx(device_name, best, 0)

    return result == win32con.DISP_CHANGE_SUCCESSFUL

def restore_display_mode(saved_mode: dict):
    if not saved_mode:
        return False
    ok = change_display_resolution(
        saved_mode["width"],
        saved_mode["height"],
        device_name=saved_mode.get("device_name"),
        freq=saved_mode.get("freq", 0),
    )
    if not ok:

        try:
            ok = change_display_resolution(
                saved_mode["width"],
                saved_mode["height"],
                device_name=None,
                freq=saved_mode.get("freq", 0),
            )
        except Exception:
            pass
    return ok

def load_zipdepth(input_size=384, fp16=True):
    from zipdepth.inference.predictor import DepthInference
    import urllib.request

    ckpt = ZIPDEPTH_ROOT / "checkpoints" / "zipdepth_base.pth"


    if not ckpt.exists():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/fabiotosi92/ZipDepth/raw/main/checkpoints/zipdepth_base.pth"
        print(f"[Info] ZipDepth checkpoint file not found. Downloading from: {url}")
        urllib.request.urlretrieve(url, ckpt)
        print("[Info] ZipDepth checkpoint downloaded successfully.")

    use_cuda = torch.cuda.is_available()
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

def estimate_depth_raw(predictor, bgr: np.ndarray) -> torch.Tensor:
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


DILATION_MAX_SHORT_SIDE = 720


def compute_dilation_target_dims(frame_h: int, frame_w: int, cap: int = DILATION_MAX_SHORT_SIDE) -> tuple[int, int]:
    """Caps the resolution used for edge dilation at `cap` px on the short side.

    At 720p/1080p (short side already <= cap) this returns (frame_h, frame_w)
    unchanged, so behaviour matches the original full-resolution dilation
    exactly. At 2K/4K it returns a smaller size so dilate_edge() runs on a
    1080p-scale depth map instead of the full render resolution, then the
    caller upsamples the diluted result back up once.
    """
    short_side = min(frame_h, frame_w)
    if short_side <= cap or cap <= 0:
        return frame_h, frame_w
    scale = cap / float(short_side)
    dh = max(2, int(round(frame_h * scale)))
    dw = max(2, int(round(frame_w * scale)))
    dh -= dh % 2
    dw -= dw % 2
    return max(2, dh), max(2, dw)

def estimate_depth(predictor, bgr: np.ndarray) -> torch.Tensor:
    h, w = bgr.shape[:2]
    depth = estimate_depth_raw(predictor, bgr)
    depth = upsample_depth(depth, h, w)
    return depth


def normalize_depth_gpu(depth: torch.Tensor, ema_lo: torch.Tensor | None = None, ema_hi: torch.Tensor | None = None, decay: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = depth.float().view(-1)
    sample = flat[::8] if flat.numel() > 100_000 else flat
    lo = torch.quantile(sample, 0.01)
    hi = torch.quantile(sample, 0.99)

    if decay > 0.0 and ema_lo is not None and ema_hi is not None:

        lo = ema_lo.mul_(decay).add_(lo, alpha=1.0 - decay)
        hi = ema_hi.mul_(decay).add_(hi, alpha=1.0 - decay)

    d = depth.to(dtype=depth.dtype)
    eps = torch.tensor(1e-8, device=d.device, dtype=d.dtype)
    normalized = (d - lo) / (hi - lo + eps)
    normalized.clamp_(0.0, 1.0)
    return normalized, lo.detach(), hi.detach()


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

def monobw_warp_one(rgb, depth, divergence, convergence, shift_sign=1.0, preserve_screen_border: bool = False):
    B, _, H, W = rgb.shape
    device, dtype = rgb.device, rgb.dtype
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

    src_index = torch.arange(0, W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)
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
    mesh_y = torch.linspace(-1, 1, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    grid = torch.cat([grid_x, mesh_y], dim=1).permute(0, 2, 3, 1)
    return F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=True)


_PINNED_BUFFERS: dict[tuple[int, int, int], torch.Tensor] = {}


def _get_pinned_uint8_buffer(h: int, w: int, c: int) -> torch.Tensor:
    key = (h, w, c)
    buf = _PINNED_BUFFERS.get(key)
    if buf is None:

        try:
            buf = torch.empty((h, w, c), dtype=torch.uint8, pin_memory=True)
        except RuntimeError:
            buf = torch.empty((h, w, c), dtype=torch.uint8)
        _PINNED_BUFFERS[key] = buf
    return buf


def gpu_resize_bgr(raw: np.ndarray, dst_w: int, dst_h: int, device: torch.device) -> np.ndarray:
    h, w = raw.shape[:2]
    if (w, h) == (dst_w, dst_h):
        return raw
    if device.type != "cuda":
        interp = _pick_interp(w, h, dst_w, dst_h)
        return cv2.resize(raw, (dst_w, dst_h), interpolation=interp)

    c = raw.shape[2] if raw.ndim == 3 else 1
    try:
        pinned = _get_pinned_uint8_buffer(h, w, c)
        pinned.copy_(torch.from_numpy(raw))
        gpu_u8 = pinned.to(device=device, non_blocking=True)
        gpu_f = gpu_u8.permute(2, 0, 1).unsqueeze(0).float()
        return _gpu_downsample_from_float(gpu_f, dst_w, dst_h)
    except Exception as e:
        print(f"[GPU Resize] Fallback to cv2 due to error: {e}")
        interp = _pick_interp(w, h, dst_w, dst_h)
        return cv2.resize(raw, (dst_w, dst_h), interpolation=interp)


def _gpu_downsample_from_float(gpu_f: torch.Tensor, dst_w: int, dst_h: int) -> np.ndarray:
    src_h, src_w = gpu_f.shape[-2:]
    if (src_w, src_h) == (dst_w, dst_h):
        out_u8 = gpu_f.clamp(0, 255).round().to(torch.uint8)
        return out_u8[0].permute(1, 2, 0).contiguous().cpu().numpy()

    scale = max(src_w / float(dst_w), src_h / float(dst_h)) if dst_w > 0 and dst_h > 0 else 1.0
    try:
        resized = F.interpolate(gpu_f, size=(dst_h, dst_w), mode="bilinear",
                                 align_corners=False, antialias=(scale >= 1.5))
    except TypeError:

        resized = F.interpolate(gpu_f, size=(dst_h, dst_w), mode="bilinear", align_corners=False)

    resized_u8 = resized.clamp_(0, 255).round_().to(torch.uint8)
    return resized_u8[0].permute(1, 2, 0).contiguous().cpu().numpy()


def compute_depth_target_dims(raw_w: int, raw_h: int, input_size: int) -> tuple[int, int]:
    if raw_w <= 0 or raw_h <= 0 or input_size <= 0:
        return raw_w, raw_h
    short_side = min(raw_w, raw_h)
    scale = input_size / float(short_side)
    dw = max(2, int(round(raw_w * scale)))
    dh = max(2, int(round(raw_h * scale)))
    dw -= dw % 2
    dh -= dh % 2
    return max(2, dw), max(2, dh)


def gpu_resize_dual(raw: np.ndarray, proc_w: int, proc_h: int, depth_w: int, depth_h: int,
                     device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    h, w = raw.shape[:2]
    if device.type != "cuda":
        interp_p = _pick_interp(w, h, proc_w, proc_h)
        frame_p = raw if (w, h) == (proc_w, proc_h) else cv2.resize(raw, (proc_w, proc_h), interpolation=interp_p)
        interp_d = _pick_interp(w, h, depth_w, depth_h)
        frame_d = raw if (w, h) == (depth_w, depth_h) else cv2.resize(raw, (depth_w, depth_h), interpolation=interp_d)
        return frame_p, frame_d

    try:
        c = raw.shape[2] if raw.ndim == 3 else 1
        pinned = _get_pinned_uint8_buffer(h, w, c)
        pinned.copy_(torch.from_numpy(raw))
        gpu_u8 = pinned.to(device=device, non_blocking=True)
        gpu_f = gpu_u8.permute(2, 0, 1).unsqueeze(0).float()

        frame_p = _gpu_downsample_from_float(gpu_f, proc_w, proc_h)
        frame_d = _gpu_downsample_from_float(gpu_f, depth_w, depth_h)
        return frame_p, frame_d
    except Exception as e:
        print(f"[GPU Resize] Fallback to cv2 due to error: {e}")
        interp_p = _pick_interp(w, h, proc_w, proc_h)
        frame_p = raw if (w, h) == (proc_w, proc_h) else cv2.resize(raw, (proc_w, proc_h), interpolation=interp_p)
        interp_d = _pick_interp(w, h, depth_w, depth_h)
        frame_d = raw if (w, h) == (depth_w, depth_h) else cv2.resize(raw, (depth_w, depth_h), interpolation=interp_d)
        return frame_p, frame_d



def pack_frame_gpu(left_t: torch.Tensor, right_t: torch.Tensor, fmt: str) -> torch.Tensor:
    """Pack left/right BCHW tensors into a single stereo frame for PC overlay output.

    hsbs / tb (labelled "half SBS" / "half TB" in the UI) now pack at *full*
    eye resolution (true full-SBS / full-TB). The Win32GL overlay then
    stretches the packed frame to fill the 16:9 monitor with no letterbox /
    pillarbox, so the on-screen result looks like classic half-SBS / half-TB
    while each eye keeps full process resolution quality.

    fsbs ("Full SBS") packs identically to hsbs (side-by-side at full eye
    resolution), but the overlay does NOT stretch it to fill the monitor —
    it is displayed at its true 2W×H aspect ratio (letterboxed/pillarboxed
    if needed), i.e. a normal, correctly-proportioned full-SBS output.
    """
    if fmt in ("hsbs", "fsbs"):
        # Full SBS: side-by-side at original eye resolution → 2W × H
        return torch.cat([left_t, right_t], dim=3)
    if fmt in ("htb", "tb"):
        # Full TB: top-bottom at original eye resolution → W × 2H
        return torch.cat([left_t, right_t], dim=2)
    if fmt == "anaglyph":
        l, r = left_t, right_t
        out = torch.zeros_like(l)
        out[:, 0:1] = l[:, 0:1]
        out[:, 1:2] = r[:, 1:2]
        out[:, 2:3] = r[:, 2:3]
        return out
    if fmt == "half-anaglyph":
        l, r = left_t, right_t
        gray_l = 0.299 * l[:, 0:1] + 0.587 * l[:, 1:2] + 0.114 * l[:, 2:3]
        out = torch.zeros_like(l)
        out[:, 0:1] = gray_l
        out[:, 1:2] = r[:, 1:2]
        out[:, 2:3] = r[:, 2:3]
        return out
    raise ValueError(f"Unsupported PC format: {fmt}")


def make_stereo(bgr: np.ndarray, depth_t: torch.Tensor, divergence: float, convergence: float, device: torch.device, already_normalized: bool = False, preserve_screen_border: bool = False):
    dtype = torch.float16 if depth_t.dtype == torch.float16 else torch.float32

    h, w, c = bgr.shape
    if device.type == "cuda":
        pinned = _get_pinned_uint8_buffer(h, w, c)


        pinned.copy_(torch.from_numpy(bgr))
        bgr_u8_gpu = pinned.to(device=device, non_blocking=True)
        bgr_t = bgr_u8_gpu.to(dtype=dtype)
    else:
        bgr_t = torch.from_numpy(bgr).to(device=device, dtype=dtype)
    bgr_t = bgr_t.permute(2, 0, 1).unsqueeze(0).div_(255.0)
    rgb_t = bgr_t[:, [2, 1, 0], :, :]
    if already_normalized: depth_norm = depth_t.to(dtype=rgb_t.dtype)
    else: depth_norm = normalize_depth_gpu(depth_t)[0].to(dtype=rgb_t.dtype)
    with torch.inference_mode():
        left_t = monobw_warp_one(rgb_t, depth_norm, divergence, convergence, -1.0, preserve_screen_border=preserve_screen_border)
        right_t = monobw_warp_one(rgb_t, depth_norm, divergence, convergence, +1.0, preserve_screen_border=preserve_screen_border)
    return left_t, right_t

# ---------------------------------------------------------------------------
# GPU-resident HUD/FPS text overlay
#
# The old approach pulled the *entire* output frame (up to 4K) down to the
# CPU every time the HUD or FPS counter needed to be drawn, ran cv2.putText
# on the CPU numpy array, then uploaded the whole frame back to the GPU.
# That GPU->CPU->GPU round trip on a multi-megapixel tensor is the dominant
# cost, not the text rendering itself.
#
# Instead, cv2.putText only ever runs on a tiny cached tile (just the text's
# bounding box, a few hundred px) whenever the *text content* changes - not
# every frame. That tile is uploaded to the GPU once and cached. Compositing
# it onto the live frame each frame is then a small in-place GPU tensor blend
# over that tiny region, with no CPU round trip of the full frame at all.
# ---------------------------------------------------------------------------

from collections import OrderedDict as _OrderedDict

_HUD_TILE_CACHE_MAXSIZE = 24
_hud_tile_cache_cpu: "_OrderedDict" = _OrderedDict()  # key -> (canvas_u8 HWC, alpha_f32 HW)
_hud_tile_cache_gpu: dict = {}  # (key, device_str) -> (canvas_gpu HWC float32, alpha_gpu HW1 float32)


def _hud_cache_put(cache: "_OrderedDict", key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _HUD_TILE_CACHE_MAXSIZE:
        cache.popitem(last=False)


def _hud_cache_touch(cache: "_OrderedDict", key):
    """Refresh LRU recency for `key` without altering its stored value."""
    if key in cache:
        cache.move_to_end(key)


def _render_boxed_text_tile(text: str, font_scale: float = 0.85, thickness: int = 2, pad: int = 8):
    """Renders white text over a semi-transparent black box, e.g. the corner HUD label."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    tile_w, tile_h = tw + pad * 2, th + baseline + pad * 2
    canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    alpha = np.full((tile_h, tile_w), 0.55, dtype=np.float32)
    text_mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
    cv2.putText(text_mask, text, (pad, th + pad), font, font_scale, 255, thickness, cv2.LINE_AA)
    m = text_mask > 0
    canvas[m] = (255, 255, 255)
    alpha[m] = 1.0
    return canvas, alpha, tile_w, tile_h, th, pad


def _render_plain_text_tile(text: str, color_bgr, font_scale: float = 1.0, thickness: int = 2, pad: int = 4):
    """Renders colored text with no background box, e.g. the FPS counter."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    tile_w, tile_h = tw + pad * 2, th + baseline + pad * 2
    canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    text_mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
    cv2.putText(text_mask, text, (pad, th + pad), font, font_scale, 255, thickness, cv2.LINE_AA)
    canvas[text_mask > 0] = color_bgr
    alpha = (text_mask.astype(np.float32) / 255.0)
    return canvas, alpha, tile_w, tile_h, th, pad


def _get_gpu_text_tile(key, render_fn, device: torch.device):
    """Returns (canvas_gpu, alpha_gpu, tile_w, tile_h, text_h, pad), building/uploading only on cache miss."""
    gpu_key = (key, str(device))
    cached = _hud_tile_cache_gpu.get(gpu_key)
    if cached is not None:
        _hud_cache_touch(_hud_tile_cache_cpu, key)  # keep LRU order fresh without corrupting the stored tuple
        return cached

    cpu_cached = _hud_tile_cache_cpu.get(key)
    if cpu_cached is None:
        canvas, alpha, tile_w, tile_h, text_h, pad = render_fn()
        cpu_cached = (canvas, alpha, tile_w, tile_h, text_h, pad)
        _hud_cache_put(_hud_tile_cache_cpu, key, cpu_cached)
    canvas, alpha, tile_w, tile_h, text_h, pad = cpu_cached

    canvas_gpu = torch.from_numpy(canvas).to(device=device, dtype=torch.float32)
    alpha_gpu = torch.from_numpy(alpha).to(device=device, dtype=torch.float32).unsqueeze(-1)
    result = (canvas_gpu, alpha_gpu, tile_w, tile_h, text_h, pad)

    if len(_hud_tile_cache_gpu) >= _HUD_TILE_CACHE_MAXSIZE:
        _hud_tile_cache_gpu.pop(next(iter(_hud_tile_cache_gpu)))
    _hud_tile_cache_gpu[gpu_key] = result
    return result


def blit_text_tile_gpu(frame_u8_hwc: torch.Tensor, canvas_gpu: torch.Tensor, alpha_gpu: torch.Tensor, x: int, y: int):
    """In-place alpha composite of a small GPU tile onto a larger GPU frame tensor (HWC, uint8). No CPU sync."""
    H, W = frame_u8_hwc.shape[:2]
    th, tw = canvas_gpu.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + tw), min(H, y + th)
    if x1 <= x0 or y1 <= y0:
        return
    tx0, ty0 = x0 - x, y0 - y
    tx1, ty1 = tx0 + (x1 - x0), ty0 + (y1 - y0)

    region = frame_u8_hwc[y0:y1, x0:x1].float()
    t_rgb = canvas_gpu[ty0:ty1, tx0:tx1]
    t_a = alpha_gpu[ty0:ty1, tx0:tx1]
    blended = region * (1.0 - t_a) + t_rgb * t_a
    frame_u8_hwc[y0:y1, x0:x1] = blended.round_().clamp_(0, 255).to(torch.uint8)


def draw_hud_overlay_gpu(frame_u8_hwc: torch.Tensor, hud_text: str = "", show_fps: bool = False, fps_value: float = 0.0):
    """Composites the corner HUD label and/or the FPS counter directly onto a GPU frame tensor (HWC uint8)."""
    H, W = frame_u8_hwc.shape[:2]

    if hud_text:
        canvas_gpu, alpha_gpu, tile_w, tile_h, text_h, pad = _get_gpu_text_tile(
            ("hud", hud_text), lambda: _render_boxed_text_tile(hud_text), frame_u8_hwc.device
        )
        # original placement: text baseline at (w - tw - 16, th + 24); box padded by `pad` around it
        fx = W - (tile_w - 2 * pad) - 16
        fy = text_h + 24
        blit_text_tile_gpu(frame_u8_hwc, canvas_gpu, alpha_gpu, fx - pad, fy - text_h - pad)

    if show_fps:
        fps_str = f"{fps_value:.1f} FPS"
        canvas_gpu, alpha_gpu, tile_w, tile_h, text_h, pad = _get_gpu_text_tile(
            ("fps", fps_str), lambda: _render_plain_text_tile(fps_str, (0, 255, 0)), frame_u8_hwc.device
        )
        # original placement: text baseline at (20, 40)
        fx, fy = 20, 40
        blit_text_tile_gpu(frame_u8_hwc, canvas_gpu, alpha_gpu, fx - pad, fy - text_h - pad)


def draw_cursor_arrow(img, x, y, scale=1.0, fill_bgr=(255, 255, 255), outline_bgr=(15, 15, 15), outline_thickness=1):
    h, w = img.shape[:2]
    if not (0 <= x < w and 0 <= y < h): return
    s = float(np.clip(scale, 0.45, 3.5))
    pts = np.array([[0.0, 0.0], [0.0, 17.0], [4.0, 13.0], [7.0, 22.0], [10.0, 20.5], [6.5, 12.0], [13.0, 12.0]], dtype=np.float32)
    pts *= s; pts[:, 0] += x; pts[:, 1] += y
    pts_i = np.round(pts).astype(np.int32)
    cv2.fillConvexPoly(img, pts_i, fill_bgr, lineType=cv2.LINE_AA)
    th = max(2, int(round(0.5 * s)))
    cv2.polylines(img, [pts_i], isClosed=True, color=outline_bgr, thickness=th, lineType=cv2.LINE_AA)

def screen_to_frame_xy(mx, my, raw_w, raw_h, process_w, process_h, cursor_ox, cursor_oy):
    fx = int(mx * process_w / max(raw_w, 1)) + cursor_ox
    fy = int(my * process_h / max(raw_h, 1)) + cursor_oy
    return fx, fy

_cursor_last_pos = None
_cursor_last_move_time = 0.0
_CURSOR_HIDE_SECONDS = 3.0

_video_playing_cache = False
_video_playing_cache_time = 0.0
_VIDEO_PLAYING_CACHE_SECONDS = 0.5
_smtc_available = None


def _is_video_playing_via_smtc() -> "bool | None":
    global _smtc_available
    if _smtc_available is False:
        return None
    try:
        import asyncio
        from winsdk.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as _MediaManager,
            GlobalSystemMediaTransportControlsSessionPlaybackStatus as _PlaybackStatus,
        )
    except Exception:
        _smtc_available = False
        return None
    _smtc_available = True

    async def _query():
        mgr = await _MediaManager.request_async()
        sessions = mgr.get_sessions()

        for session in sessions:
            try:
                info = session.get_playback_info()
                if info is not None and info.playback_status == _PlaybackStatus.PLAYING:
                    return True
            except Exception:
                continue
        return False

    try:
        return asyncio.run(_query())
    except Exception:
        return None


def is_video_playing() -> bool:
    global _video_playing_cache, _video_playing_cache_time
    now = time.perf_counter()
    if now - _video_playing_cache_time < _VIDEO_PLAYING_CACHE_SECONDS:
        return _video_playing_cache
    _video_playing_cache_time = now


    smtc_result = _is_video_playing_via_smtc()
    if smtc_result is not None:
        _video_playing_cache = smtc_result
        return smtc_result


    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            _video_playing_cache = False
            return False
        title = win32gui.GetWindowText(hwnd).lower()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        proc_name = ""
        try:
            import psutil
            proc_name = psutil.Process(pid).name().lower()
        except Exception: pass
        video_players = ["potplayermini", "potplayermini64", "vlc", "kmplayer", "gom", "mpv", "kmplayer64"]
        if any(vp in proc_name for vp in video_players):
            _video_playing_cache = True
            return True
        video_keywords = [
            "youtube", "netflix", "twitch", "disney", "player", "video", "watch", "playing",
            "동영상", "재생", "시청",
            "動画", "再生", "視聴",
            "视频", "播放",
        ]
        if any(kw in title for kw in video_keywords):
            _video_playing_cache = True
            return True
    except Exception: pass
    _video_playing_cache = False
    return False


def composite_cursor_before_warp(*args, **kwargs):


    return


_real_mouse_state = {"pos": None, "buttons": (False, False, False)}
_REAL_MOUSE_MOVE_THRESHOLD_PX = 2


def detect_real_mouse_activity() -> bool:
    global _real_mouse_state
    try:
        now = time.perf_counter()
        x, y = win32api.GetCursorPos()


        l_down = bool(user32.GetAsyncKeyState(0x01) & 0x8000)
        r_down = bool(user32.GetAsyncKeyState(0x02) & 0x8000)
        m_down = bool(user32.GetAsyncKeyState(0x04) & 0x8000)

        suppressed = now < _real_mouse_suppress_until[0]

        prev_pos = _real_mouse_state["pos"]
        prev_buttons = _real_mouse_state["buttons"]

        moved = False
        if not suppressed and prev_pos is not None:
            dx = x - prev_pos[0]
            dy = y - prev_pos[1]
            if (dx * dx + dy * dy) >= (_REAL_MOUSE_MOVE_THRESHOLD_PX * _REAL_MOUSE_MOVE_THRESHOLD_PX):
                moved = True

        button_pressed = False
        if not suppressed:
            if (l_down and not prev_buttons[0]) or (r_down and not prev_buttons[1]) or (m_down and not prev_buttons[2]):
                button_pressed = True


        if not suppressed:
            _real_mouse_state["pos"] = (x, y)
            _real_mouse_state["buttons"] = (l_down, r_down, m_down)

        return moved or button_pressed
    except Exception:
        return False


def capture_windows_cursor_bgra():
    # While the real Windows system cursor is force-hidden (SetSystemCursor
    # replaced it with a 1x1 transparent cursor), GetCursorInfo() would only
    # ever return that transparent cursor's own bitmap — drawing it would make
    # the app's own converted cursor invisible too. So while hidden, reuse the
    # last real cursor image captured before we hid it, instead of re-querying.
    if _winmouse_autohide_enabled and _real_cursor_cache["arr"] is not None:
        return _real_cursor_cache["arr"], _real_cursor_cache["hot_x"], _real_cursor_cache["hot_y"]
    try:
        import win32ui
        flags, hcursor, _pos = win32gui.GetCursorInfo()
        if flags == 0 or not hcursor:
            return None

        f_icon, hotspot_x, hotspot_y, hbm_mask, hbm_color = win32gui.GetIconInfo(hcursor)
        try:
            size = 32
            hdc_screen = win32gui.GetDC(0)
            hdc_mem = win32ui.CreateDCFromHandle(hdc_screen)
            hdc_compat = hdc_mem.CreateCompatibleDC()
            try:
                bmp = win32ui.CreateBitmap()
                bmp.CreateCompatibleBitmap(hdc_mem, size, size)
                hdc_compat.SelectObject(bmp)
                hdc_compat.FillSolidRect((0, 0, size, size), 0x00000000)
                win32gui.DrawIconEx(hdc_compat.GetHandleOutput(), 0, 0, hcursor, size, size, 0, None, win32con.DI_NORMAL)

                bmp_bits = bmp.GetBitmapBits(True)
                arr = np.frombuffer(bmp_bits, dtype=np.uint8).reshape(size, size, 4).copy()


                if int(arr[:, :, 3].max()) == 0:
                    mask_dc = hdc_mem.CreateCompatibleDC()
                    mask_bmp = win32ui.CreateBitmapFromHandle(hbm_mask)
                    mask_dc.SelectObject(mask_bmp)
                    mask_bits = mask_bmp.GetBitmapBits(True)
                    mask_h = mask_bmp.GetInfo()["bmHeight"]
                    mask_w = mask_bmp.GetInfo()["bmWidth"]

                    mask_arr = np.frombuffer(mask_bits, dtype=np.uint8)
                    try:
                        mask_arr = mask_arr.reshape(mask_h, -1)[:size, :size]
                        opaque = (mask_arr == 0)
                        arr[:, :, 3] = np.where(opaque, 255, 0).astype(np.uint8)

                        if int(arr[:, :, :3].max()) == 0:
                            arr[:, :, 0:3] = np.where(opaque[:, :, None], 255, 0).astype(np.uint8)
                    except Exception:
                        pass
                    mask_dc.DeleteDC()

                # Cache the last known-good real cursor bitmap so it can still
                # be drawn while the real system cursor is force-hidden later.
                if not _winmouse_autohide_enabled:
                    _real_cursor_cache["arr"] = arr
                    _real_cursor_cache["hot_x"] = int(hotspot_x)
                    _real_cursor_cache["hot_y"] = int(hotspot_y)

                return arr, int(hotspot_x), int(hotspot_y)
            finally:
                hdc_compat.DeleteDC()
                hdc_mem.DeleteDC()
                win32gui.ReleaseDC(0, hdc_screen)
        finally:
            try:
                win32gui.DeleteObject(hbm_mask)
            except Exception:
                pass
            try:
                win32gui.DeleteObject(hbm_color)
            except Exception:
                pass
    except Exception:
        return None


def draw_real_mouse_cursor(img, x, y, scale=1.0, fill_bgr=(255, 255, 255), outline_bgr=(15, 15, 15), outline_thickness=1):
    h, w = img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return
    captured = capture_windows_cursor_bgra()
    if captured is None:

        draw_cursor_arrow(img, x, y, scale=scale, fill_bgr=fill_bgr, outline_bgr=outline_bgr, outline_thickness=outline_thickness)
        return
    try:
        cur_bgra, hot_x, hot_y = captured
        cur_h, cur_w = cur_bgra.shape[:2]
        s = float(np.clip(scale, 0.5, 4.0))
        if s != 1.0:
            new_w, new_h = max(1, int(cur_w * s)), max(1, int(cur_h * s))
            cur_bgra = cv2.resize(cur_bgra, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            hot_x = int(hot_x * s)
            hot_y = int(hot_y * s)
            cur_h, cur_w = cur_bgra.shape[:2]

        top_left_x = x - hot_x
        top_left_y = y - hot_y

        src_x0, src_y0 = 0, 0
        dst_x0, dst_y0 = top_left_x, top_left_y
        if dst_x0 < 0:
            src_x0 = -dst_x0
            dst_x0 = 0
        if dst_y0 < 0:
            src_y0 = -dst_y0
            dst_y0 = 0
        dst_x1 = min(w, top_left_x + cur_w)
        dst_y1 = min(h, top_left_y + cur_h)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        region = cur_bgra[src_y0:src_y1, src_x0:src_x1]
        alpha = (region[:, :, 3:4].astype(np.float32)) / 255.0
        roi = img[dst_y0:dst_y1, dst_x0:dst_x1]
        blended = roi.astype(np.float32) * (1.0 - alpha) + region[:, :, :3].astype(np.float32) * alpha
        img[dst_y0:dst_y1, dst_x0:dst_x1] = blended.astype(np.uint8)
    except Exception:
        draw_cursor_arrow(img, x, y, scale=scale, fill_bgr=fill_bgr, outline_bgr=outline_bgr, outline_thickness=outline_thickness)


def _pick_interp(src_w: int, src_h: int, dst_w: int, dst_h: int) -> int:
    if dst_w <= 0 or dst_h <= 0:
        return cv2.INTER_AREA
    scale = max(src_w / float(dst_w), src_h / float(dst_h))
    return cv2.INTER_AREA if scale >= 1.5 else cv2.INTER_LINEAR


# ============================================================
# Magnification API Capture (PC + Virtual Desktop compat only)
# DXGI/dxcam cannot exclude a specific HWND. Magnification API can:
#   MagSetWindowFilterList(..., MW_FILTERMODE_EXCLUDE, overlay_hwnd)
# → same monitor + WDA_NONE without feedback loop (Owl3D-style).
# Virtual Desktop / Win+PrtSc still see the overlay via normal DWM composition.
# ============================================================
MW_FILTERMODE_EXCLUDE = 0
WC_MAGNIFIER = "Magnifier"

try:
    _mag_dll = ctypes.WinDLL("Magnification.dll")
    _HAS_MAGNIFICATION = True
except OSError:
    _mag_dll = None
    _HAS_MAGNIFICATION = False


class _MAG_GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class MAGIMAGEHEADER(ctypes.Structure):
    # MSDN: width, height, GUID format, stride, offset, cbSize
    _fields_ = [
        ("width", ctypes.c_uint),
        ("height", ctypes.c_uint),
        ("format", _MAG_GUID),
        ("stride", ctypes.c_uint),
        ("offset", ctypes.c_uint),
        ("cbSize", ctypes.c_size_t),
    ]


MagImageScalingCallback = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    wintypes.HWND,
    ctypes.c_void_p,
    MAGIMAGEHEADER,
    ctypes.c_void_p,
    MAGIMAGEHEADER,
    wintypes.RECT,
    wintypes.RECT,
    wintypes.HRGN,
)


def _bind_mag_functions():
    if not _HAS_MAGNIFICATION:
        return None
    lib = _mag_dll
    lib.MagInitialize.restype = wintypes.BOOL
    lib.MagUninitialize.restype = wintypes.BOOL
    lib.MagSetWindowSource.argtypes = [wintypes.HWND, wintypes.RECT]
    lib.MagSetWindowSource.restype = wintypes.BOOL
    lib.MagSetWindowFilterList.argtypes = [
        wintypes.HWND, wintypes.DWORD, ctypes.c_int, ctypes.POINTER(wintypes.HWND)
    ]
    lib.MagSetWindowFilterList.restype = wintypes.BOOL
    lib.MagSetImageScalingCallback.argtypes = [wintypes.HWND, MagImageScalingCallback]
    lib.MagSetImageScalingCallback.restype = wintypes.BOOL
    lib.MagSetWindowTransform.argtypes = [wintypes.HWND, ctypes.c_void_p]
    lib.MagSetWindowTransform.restype = wintypes.BOOL
    try:
        lib.MagShowSystemCursor.argtypes = [wintypes.BOOL]
        lib.MagShowSystemCursor.restype = wintypes.BOOL
    except Exception:
        pass
    return lib



# --- System cursor hide/restore (PC overlay mode, live3d-style) ---
_OCR_CURSOR_IDS = (32512, 32513, 32514, 32515, 32516, 32640, 32641, 32642, 32643, 32644, 32645, 32646, 32648, 32649, 32650, 32651)
_winmouse_autohide_enabled = False
# Last real cursor bitmap captured while the system cursor was still visible —
# reused for drawing while the system cursor is force-hidden (see
# capture_windows_cursor_bgra above).
_real_cursor_cache = {"arr": None, "hot_x": 0, "hot_y": 0}

def hide_windows_system_cursor():
    """Replace system cursors with a fully transparent cursor (process-global while running)."""
    global _winmouse_autohide_enabled
    try:
        # Grab whatever the real cursor currently looks like before blanking
        # it out, so the app's own drawn cursor still has a real shape to show.
        if not _winmouse_autohide_enabled:
            try:
                captured = capture_windows_cursor_bgra()
                if captured is not None:
                    arr, hot_x, hot_y = captured
                    _real_cursor_cache["arr"] = arr
                    _real_cursor_cache["hot_x"] = hot_x
                    _real_cursor_cache["hot_y"] = hot_y
            except Exception:
                pass
        # 1x1 transparent cursor.
        # IMPORTANT: SetSystemCursor() destroys the hcursor handle it's given
        # after copying its bits in, so a single handle reused across every
        # cursor ID would only actually replace the *first* one in the loop
        # (plain arrow) — every other cursor (hand/pointer over links,
        # I-beam over text, resize arrows, etc.) would silently stay the
        # real system cursor, which is exactly why it "pops back up" over
        # clickable/focusable elements. Create a fresh handle per ID instead.
        and_mask = (ctypes.c_ubyte * 1)(0xFF)
        xor_mask = (ctypes.c_ubyte * 1)(0x00)
        any_ok = False
        for cid in _OCR_CURSOR_IDS:
            try:
                hcur = ctypes.windll.user32.CreateCursor(0, 0, 0, 1, 1, and_mask, xor_mask)
                if not hcur:
                    continue
                if ctypes.windll.user32.SetSystemCursor(hcur, cid):
                    any_ok = True
            except Exception:
                pass
        if not any_ok:
            return
        _winmouse_autohide_enabled = True
    except Exception as e:
        print(f"[Cursor] hide failed: {e}", flush=True)

def restore_windows_system_cursor():
    global _winmouse_autohide_enabled
    try:
        # SPI_SETCURSORS = 0x0057 — reload system default cursors
        ctypes.windll.user32.SystemParametersInfoW(0x0057, 0, None, 0)
    except Exception as e:
        print(f"[Cursor] restore failed: {e}", flush=True)
    _winmouse_autohide_enabled = False


class MagCaptureWorker(threading.Thread):
    """
    Windows Magnification API capture worker (same interface as CaptureWorker).
    Puts BGR uint8 numpy frames into frame_q. Excludes overlay HWNDs from capture.
    """

    def __init__(self, monitor_idx: int, target_fps: float, frame_q, running, exclude_hwnds=None):
        super().__init__(daemon=True, name="MagCaptureWorker")
        self.monitor_idx = int(monitor_idx)
        self.target_fps = float(target_fps) if target_fps and target_fps > 0 else 60.0
        self.frame_q = frame_q
        self.running = running
        self._exclude_lock = threading.Lock()
        self._exclude_hwnds = [int(h) for h in (exclude_hwnds or []) if h]
        self._lib = None
        self._host_hwnd = None
        self._mag_hwnd = None
        self._callback_ref = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._ready = threading.Event()
        self._init_error = None
        self._cb_err_logged = False

    def set_exclude_hwnds(self, hwnds):
        with self._exclude_lock:
            self._exclude_hwnds = [int(h) for h in (hwnds or []) if h]
        self._apply_filter_list()

    def _apply_filter_list(self):
        if not self._lib or not self._mag_hwnd:
            return
        with self._exclude_lock:
            hwnds = list(self._exclude_hwnds)
        if not hwnds:
            self._lib.MagSetWindowFilterList(self._mag_hwnd, MW_FILTERMODE_EXCLUDE, 0, None)
            return
        arr = (wintypes.HWND * len(hwnds))(*hwnds)
        ok = self._lib.MagSetWindowFilterList(
            self._mag_hwnd, MW_FILTERMODE_EXCLUDE, len(hwnds), arr
        )
        if not ok:
            print(f"[MagCapture] MagSetWindowFilterList failed: {ctypes.GetLastError()}", flush=True)
        else:
            print(f"[MagCapture] Excluding {len(hwnds)} HWND(s) from capture: {hwnds}", flush=True)

    def _on_mag_callback(self, hwnd, srcdata, srcheader, destdata, destheader, unclipped, clipped, dirty):
        try:
            w = int(srcheader.width)
            h = int(srcheader.height)
            stride = int(srcheader.stride) if srcheader.stride else (w * 4)
            if w <= 0 or h <= 0 or not srcdata:
                return True
            nbytes = stride * h
            buf = (ctypes.c_ubyte * nbytes).from_address(srcdata)
            arr = np.frombuffer(buf, dtype=np.uint8, count=nbytes).reshape(h, stride)
            bgr = arr[:, : w * 4].reshape(h, w, 4)[:, :, :3].copy()
            with self._frame_lock:
                self._latest_frame = bgr
            self._frame_event.set()
        except Exception as e:
            if not self._cb_err_logged:
                print(f"[MagCapture] callback error: {e}", flush=True)
                self._cb_err_logged = True
        return True

    def _get_monitor_rect(self):
        monitors = get_monitor_list()
        idx = self.monitor_idx if 0 <= self.monitor_idx < len(monitors) else 0
        m = monitors[idx]
        left, top = int(m["left"]), int(m["top"])
        right = left + int(m["width"])
        bottom = top + int(m["height"])
        return left, top, right, bottom

    def _create_magnifier(self) -> bool:
        self._lib = _bind_mag_functions()
        if not self._lib:
            self._init_error = "Magnification.dll not available"
            return False
        if not self._lib.MagInitialize():
            self._init_error = f"MagInitialize failed: {ctypes.GetLastError()}"
            return False

        hinst = win32api.GetModuleHandle(None)
        wc = win32gui.WNDCLASS()
        wc.hInstance = hinst
        wc.lpszClassName = "Live3D_MagHost"
        wc.lpfnWndProc = win32gui.DefWindowProc
        try:
            win32gui.RegisterClass(wc)
        except win32gui.error:
            pass

        left, top, right, bottom = self._get_monitor_rect()
        width, height = max(1, right - left), max(1, bottom - top)

        self._host_hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_LAYERED | win32con.WS_EX_TOOLWINDOW | win32con.WS_EX_NOACTIVATE,
            "Live3D_MagHost",
            "Live3D Mag Host",
            win32con.WS_POPUP,
            left, top, width, height,
            0, 0, hinst, None,
        )
        win32gui.SetLayeredWindowAttributes(self._host_hwnd, 0, 0, win32con.LWA_ALPHA)
        win32gui.ShowWindow(self._host_hwnd, win32con.SW_SHOWNOACTIVATE)

        self._mag_hwnd = win32gui.CreateWindow(
            WC_MAGNIFIER,
            "Live3DMagnifier",
            win32con.WS_CHILD | win32con.WS_VISIBLE,
            0, 0, width, height,
            self._host_hwnd,
            0, hinst, None,
        )
        if not self._mag_hwnd:
            self._init_error = f"CreateWindow(Magnifier) failed: {ctypes.GetLastError()}"
            return False

        mat = (ctypes.c_float * 9)(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )
        if not self._lib.MagSetWindowTransform(self._mag_hwnd, ctypes.byref(mat)):
            print(f"[MagCapture] MagSetWindowTransform warning: {ctypes.GetLastError()}", flush=True)

        self._callback_ref = MagImageScalingCallback(self._on_mag_callback)
        if not self._lib.MagSetImageScalingCallback(self._mag_hwnd, self._callback_ref):
            self._init_error = f"MagSetImageScalingCallback failed: {ctypes.GetLastError()}"
            return False

        self._apply_filter_list()
        # Exclude the real OS mouse cursor from the Magnification-captured
        # frame (this is the "Windows-style cursor in the center" baked into
        # the converted output). This only affects what Magnification API
        # captures — it does NOT hide the real system cursor on the desktop,
        # and does NOT affect the app's own drawn output-cursor overlay.
        try:
            if hasattr(self._lib, "MagShowSystemCursor"):
                if self._lib.MagShowSystemCursor(False):
                    print("[MagCapture] System cursor excluded from captured frame", flush=True)
                else:
                    print(f"[MagCapture] MagShowSystemCursor(False) failed: {ctypes.GetLastError()}", flush=True)
        except Exception as e:
            print(f"[MagCapture] MagShowSystemCursor warning: {e}", flush=True)
        print(
            f"[MagCapture] Initialized monitor={self.monitor_idx} "
            f"rect=({left},{top})-({right},{bottom}) {width}x{height}",
            flush=True,
        )
        return True

    def _destroy_magnifier(self):
        try:
            if self._lib:
                try:
                    if hasattr(self._lib, "MagShowSystemCursor"):
                        self._lib.MagShowSystemCursor(True)
                except Exception:
                    pass
            if self._lib and self._mag_hwnd:
                try:
                    # Clear callback (NULL)
                    self._lib.MagSetImageScalingCallback(self._mag_hwnd, ctypes.cast(None, MagImageScalingCallback))
                except Exception:
                    pass
            if self._host_hwnd:
                try:
                    win32gui.DestroyWindow(self._host_hwnd)
                except Exception:
                    pass
            self._host_hwnd = None
            self._mag_hwnd = None
            if self._lib:
                try:
                    self._lib.MagUninitialize()
                except Exception:
                    pass
        except Exception as e:
            print(f"[MagCapture] cleanup warning: {e}", flush=True)

    def _trigger_capture(self) -> bool:
        if not self._lib or not self._mag_hwnd:
            return False
        left, top, right, bottom = self._get_monitor_rect()
        rc = wintypes.RECT(left, top, right, bottom)
        return bool(self._lib.MagSetWindowSource(self._mag_hwnd, rc))

    def run(self):
        try:
            if not self._create_magnifier():
                print(f"[MagCapture] init failed: {self._init_error}", flush=True)
                self._ready.set()
                return
            self._ready.set()
            interval = 1.0 / self.target_fps
            next_t = time.perf_counter()
            print(f"[MagCapture] Started (monitor {self.monitor_idx}, FPS: {self.target_fps})", flush=True)

            while self.running.is_set():
                win32gui.PumpWaitingMessages()
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(min(0.001, next_t - now))
                    continue
                next_t = now + interval
                if time.perf_counter() - next_t > interval * 2:
                    next_t = time.perf_counter() + interval

                self._frame_event.clear()
                if not self._trigger_capture():
                    time.sleep(0.01)
                    continue

                deadline = time.perf_counter() + 0.05
                while not self._frame_event.is_set() and time.perf_counter() < deadline:
                    win32gui.PumpWaitingMessages()
                    time.sleep(0.0005)

                with self._frame_lock:
                    frame = self._latest_frame
                    self._latest_frame = None
                if frame is None:
                    continue

                try:
                    while True:
                        self.frame_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_q.put_nowait(frame)
                except queue.Full:
                    pass
        except Exception as e:
            print(f"[MagCapture] run error: {e}", flush=True)
        finally:
            self._destroy_magnifier()
            print("[MagCapture] Stopped", flush=True)

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready.wait(timeout=timeout)




class CaptureWorker(threading.Thread):
    def __init__(self, monitor_idx: int, target_fps: float, frame_q, running):
        super().__init__(daemon=True)
        self.monitor_idx = monitor_idx
        self.target_fps = target_fps
        self.frame_q = frame_q
        self.running = running
        self.camera = None

    def _push_frame(self, raw):
        try: self.frame_q.put_nowait(raw)
        except queue.Full:
            try: self.frame_q.get_nowait()
            except queue.Empty: pass
            try: self.frame_q.put_nowait(raw)
            except queue.Full: pass

    def run(self):
        try: import dxcam
        except ImportError:
            print("pip install dxcam")
            self.running.clear()
            return
        self.camera = dxcam.create(output_idx=self.monitor_idx, output_color="BGR")
        if self.camera is None:
            print("Capture failed")
            self.running.clear()
            return
        capture_fps = self.target_fps if self.target_fps > 0 else 60
        self.camera.start(target_fps=capture_fps, video_mode=True)
        print(f"[Capture] Started via dxcam (monitor {self.monitor_idx}, FPS: {capture_fps})")
        try:
            while self.running.is_set():
                try: raw = self.camera.get_latest_frame()
                except Exception as e:
                    print(f"[Capture] Error: {e}"); time.sleep(0.05); continue
                if raw is None: time.sleep(0.001); continue
                self._push_frame(raw)


                time.sleep(0.0002)
        finally:
            try:
                if self.camera is not None: self.camera.stop()
            except Exception: pass
            print("[Capture] Stopped (dxcam)")

class InferenceWorker(threading.Thread):
    def __init__(self, args, predictor, device, frame_q, result_q, running, shared_divergence, shared_convergence, shared_edge_dilation, use_gl=True, shared_fps=None,
                 shared_auto_crop=None, shared_preserve_border=None, shared_cursor=None, monitor_left=0, monitor_top=0):
        super().__init__(daemon=True)
        self.args = args
        self.predictor = predictor
        self.device = device
        self.frame_q = frame_q
        self.result_q = result_q
        self.running = running
        self.process_w, self.process_h = parse_size(args.process_size)
        self.shared_divergence = shared_divergence
        self.shared_convergence = shared_convergence
        self.shared_edge_dilation = shared_edge_dilation
        self.use_gl = use_gl
        self.shared_fps = shared_fps

        self.shared_cursor = shared_cursor if shared_cursor is not None else [0.5, 0.5, False, 0.0, 0]
        if len(self.shared_cursor) < 5:
            self.shared_cursor.extend([0] * (5 - len(self.shared_cursor)))

        self.monitor_left = int(monitor_left)
        self.monitor_top = int(monitor_top)


        self._ema_decay_clamped = self._clamp_ema_decay(getattr(args, "ema_decay", 0.0))
        if _HAS_IW3_EMA:
            self.depth_scaler = EMAMinMaxScaler(decay=self._ema_decay_clamped, buffer_size=1, mode="minmax")
        else:
            self.depth_scaler = None
        self._frame_count = 0


        self.shared_auto_crop = shared_auto_crop if shared_auto_crop is not None else [bool(getattr(args, "auto_crop", False))]

        self.shared_preserve_border = shared_preserve_border if shared_preserve_border is not None else [bool(getattr(args, "preserve_screen_border", True))]
        self.crop_top = 0
        self.crop_bottom = None
        self._crop_detected = False
        self._crop_samples = []
        self._crop_sample_cap = 300
        self._crop_detect_start_time = None
        self._crop_min_detect_time = 10.0


        self._pad_top_ratio = 0.0
        self._pad_bottom_ratio = 0.0
        self._orig_h = None
        self._content_process_h = None

    @staticmethod
    def _clamp_ema_decay(decay):
        return clamp_ema_value(decay)

    def set_ema_decay(self, decay):
        decay = self._clamp_ema_decay(decay)
        self._ema_decay_clamped = decay
        if self.depth_scaler is not None:
            self.depth_scaler.decay = decay
        return decay

    def reset_ema(self):
        if self.depth_scaler is not None:
            self.depth_scaler.reset(decay=self._ema_decay_clamped, buffer_size=1)

    def run(self):
        print("[Inference] Started")
        print(f"[EMA] depth_scaler active: {self.depth_scaler is not None} "
              f"(mode={'IW3 EMAMinMaxScaler' if self.depth_scaler is not None else 'FALLBACK plain min/max, decay ignored!'}), "
              f"initial decay={self._ema_decay_clamped}")
        _ema_debug_last_print = 0.0

        consecutive_errors = 0
        try:
            while self.running.is_set():
                try:
                    try:

                        raw = self.frame_q.get(timeout=0.1)


                        if self.frame_q.qsize() > 1:
                            while not self.frame_q.empty():
                                try:
                                    raw = self.frame_q.get_nowait()
                                except queue.Empty:
                                    break
                    except queue.Empty:
                        continue

                    t_start = time.perf_counter()


                    _capture_raw_h, _capture_raw_w = raw.shape[:2]

                    auto_crop_on = bool(self.shared_auto_crop[0])


                    if not auto_crop_on:
                        if self._crop_detected or self.crop_bottom is not None:
                            self._crop_detected = False
                            self.crop_top = 0
                            self.crop_bottom = None
                            self._crop_samples.clear()
                            self._crop_detect_start_time = None
                            self._crop_redetect_counter = 0
                            self._pad_top_ratio = 0.0
                            self._pad_bottom_ratio = 0.0
                            self._orig_h = None
                            self._content_process_h = None

                    orig_h, orig_w = raw.shape[:2]
                    self._orig_h = orig_h

                    if auto_crop_on:

                        if not self._crop_detected:
                            res = detect_letterbox_frame(raw)
                            if res is not None:
                                if self._crop_detect_start_time is None:
                                    self._crop_detect_start_time = time.perf_counter()
                                self._crop_samples.append(res)
                                if len(self._crop_samples) > self._crop_sample_cap:
                                    self._crop_samples.pop(0)

                            elapsed = (time.perf_counter() - self._crop_detect_start_time)\
                                if self._crop_detect_start_time is not None else 0.0


                            if self._crop_samples and elapsed >= self._crop_min_detect_time:
                                tops = [t for t, b in self._crop_samples]
                                bottoms = [b for t, b in self._crop_samples]
                                self.crop_top = int(np.median(tops))
                                self.crop_bottom = int(np.median(bottoms))
                                self._crop_detected = True
                                self._crop_samples.clear()
                                self._crop_detect_start_time = None
                                content_h = self.crop_bottom - self.crop_top
                                if content_h >= int(orig_h * 0.55):
                                    self._pad_top_ratio = self.crop_top / float(orig_h)
                                    self._pad_bottom_ratio = (orig_h - self.crop_bottom) / float(orig_h)
                                    print(f"[Letterbox] ON – crop top={self.crop_top}, bottom={self.crop_bottom} "
                                          f"(content {content_h}px / {orig_h}px)", flush=True)
                                else:
                                    self.crop_top = 0
                                    self.crop_bottom = None
                                    self._pad_top_ratio = 0.0
                                    self._pad_bottom_ratio = 0.0
                                    print("[Letterbox] Detection failed → full frame", flush=True)


                        if self.crop_bottom is not None:
                            t = max(0, min(self.crop_top, orig_h - 2))
                            b = max(t + 2, min(self.crop_bottom, orig_h))
                            raw = raw[t:b, :]

                    raw_h, raw_w = raw.shape[:2]


                    depth_input_size = int(getattr(self.args, "input_size", 256) or 256)
                    depth_target_w, depth_target_h = compute_depth_target_dims(raw_w, raw_h, depth_input_size)

                    if auto_crop_on and self.crop_bottom is not None and raw_h > 0 and raw_w > 0:
                        content_aspect = raw_h / float(raw_w)
                        content_h = int(round(self.process_w * content_aspect))
                        content_h = max(2, content_h - (content_h % 2))
                        self._content_process_h = content_h
                        if (raw_w, raw_h) != (self.process_w, content_h):
                            frame, depth_frame = gpu_resize_dual(raw, self.process_w, content_h,
                                                                  depth_target_w, depth_target_h, self.device)
                        else:
                            frame = raw.copy()
                            depth_frame = gpu_resize_bgr(raw, depth_target_w, depth_target_h, self.device)
                    else:
                        self._content_process_h = None
                        if (raw_w, raw_h) != (self.process_w, self.process_h):
                            frame, depth_frame = gpu_resize_dual(raw, self.process_w, self.process_h,
                                                                  depth_target_w, depth_target_h, self.device)
                        else:
                            frame = raw.copy()
                            depth_frame = gpu_resize_bgr(raw, depth_target_w, depth_target_h, self.device)


                    try:
                        mode = str(getattr(self.args, "mode", "openxr")).lower()
                        autohide_on = bool(getattr(self.args, "cursor_autohide", False))

                        # Real mouse movement → refresh activity timestamp
                        real_mouse_active = detect_real_mouse_activity()
                        if real_mouse_active:
                            self.shared_cursor[4] = 1
                            self.shared_cursor[3] = time.perf_counter()

                        cursor_source = self.shared_cursor[4] if len(self.shared_cursor) > 4 else 0

                        # PC mode: system cursor may be hidden (anaglyph: always;
                        # hsbs/fsbs/tb: never, real cursor stays visible via
                        # Magnification capture, which never captures a cursor at
                        # all) — always use the real-mouse draw path so the overlay
                        # still shows a synthetic cursor when needed.
                        # Dual-Monitor mode intentionally does NOT force this path:
                        # its dxcam capture already bakes the real, natively-visible
                        # system cursor into the raw frame, so drawing a second,
                        # synthetic one here would double it up. Autohide for
                        # Dual-Monitor mode instead hides the real system cursor
                        # itself (see the main loop), which dxcam then simply
                        # doesn't capture.
                        if mode == "pc":
                            cursor_source = 1

                        if cursor_source == 1:
                            # If autohide is OFF, never hide the drawn cursor.
                            # (System cursor visibility is managed separately in the main loop.)
                            cursor_hidden = False
                            if autohide_on:
                                cursor_hidden = (
                                    is_video_playing()
                                    and len(self.shared_cursor) >= 4
                                    and self.shared_cursor[3] > 0.0
                                    and (time.perf_counter() - self.shared_cursor[3]) >= _CURSOR_HIDE_SECONDS
                                )
                            if not cursor_hidden:
                                mx, my = win32api.GetCursorPos()
                                local_x = mx - self.monitor_left
                                local_y = my - self.monitor_top
                                if 0 <= local_x < _capture_raw_w and 0 <= local_y < _capture_raw_h:
                                    fx, fy = screen_to_frame_xy(
                                        local_x, local_y,
                                        _capture_raw_w, _capture_raw_h,
                                        self.process_w, frame.shape[0],
                                        int(getattr(self.args, "cursor_ox", 0)),
                                        int(getattr(self.args, "cursor_oy", 0)),
                                    )
                                    draw_real_mouse_cursor(
                                        frame, fx, fy,
                                        scale=float(self.args.cursor_scale),
                                        fill_bgr=self.args.cursor_color,
                                        outline_bgr=self.args.cursor_outline,
                                        outline_thickness=self.args.cursor_outline_width,
                                    )
                        elif self.shared_cursor[2]:
                            # OpenXR virtual cursor
                            cursor_hidden = False
                            if autohide_on:
                                cursor_hidden = (
                                    is_video_playing()
                                    and len(self.shared_cursor) >= 4
                                    and self.shared_cursor[3] > 0.0
                                    and (time.perf_counter() - self.shared_cursor[3]) >= _CURSOR_HIDE_SECONDS
                                )
                            if not cursor_hidden:
                                cx = int(np.clip(self.shared_cursor[0], 0.0, 1.0) * max(0, self.process_w - 1))
                                cy = int(np.clip(self.shared_cursor[1], 0.0, 1.0) * max(0, frame.shape[0] - 1))
                                draw_cursor_arrow(
                                    frame, cx, cy,
                                    scale=float(self.args.cursor_scale),
                                    fill_bgr=self.args.cursor_color,
                                    outline_bgr=self.args.cursor_outline,
                                    outline_thickness=self.args.cursor_outline_width,
                                )
                    except Exception:
                        pass


                    with torch.no_grad():
                        frame_h, frame_w = frame.shape[:2]


                        depth_raw = estimate_depth_raw(self.predictor, depth_frame)
                        if self.args.invert_depth:
                            depth_raw = depth_raw.max() - depth_raw

                        decay = self.set_ema_decay(getattr(self.args, "ema_decay", 0.0))


                        if self.depth_scaler is not None:
                            self.depth_scaler.decay = decay
                            depth_norm_raw = self.depth_scaler.update(depth_raw)


                            _now = time.perf_counter()
                            if _now - _ema_debug_last_print >= 1.0:
                                _ema_debug_last_print = _now
                                _adaptive_info = ""
                                if self.depth_scaler._is_adaptive_active():
                                    _adaptive_info = (
                                        f" | adaptive motion={float(self.depth_scaler.last_motion_score):.2f} "
                                        f"eff_decay={float(self.depth_scaler.last_effective_decay):.3f}"
                                    )
                                print(f"[EMA] decay={decay:.2f} "
                                      f"min={float(self.depth_scaler.min_value):.4f} "
                                      f"max={float(self.depth_scaler.max_value):.4f}"
                                      f"{_adaptive_info}")
                        else:

                            depth_norm_raw = normalize_depth_gpu(depth_raw)[0]

                        # Edge dilation is run at a resolution capped to 1080p (short side),
                        # not at the full render resolution. At 720p/1080p this is identical
                        # to the original behaviour (cap >= actual size, so it's a no-op).
                        # At 2K/4K, the depth map is upsampled only as far as 1080p, dilated
                        # there (so the dilation kernel's pixel-scale behaves the same as it
                        # always did at 1080p), and only the final result is upsampled once
                        # more to the full 2K/4K render resolution.
                        divergence = float(self.shared_divergence[0])

                        convergence = 1.0 - float(self.shared_convergence[0])
                        edge_val = self.shared_edge_dilation[0] if len(self.shared_edge_dilation) == 1 else self.shared_edge_dilation

                        if _HAS_DILATION and edge_dilation_is_enabled(edge_val):
                            dilate_h, dilate_w = compute_dilation_target_dims(frame_h, frame_w)
                            with torch.inference_mode():
                                depth_for_dilate_src = upsample_depth(depth_norm_raw, dilate_h, dilate_w)
                                depth_for_dilate = 1.0 - depth_for_dilate_src
                                try:
                                    depth_for_dilate = dilate_edge(depth_for_dilate, edge_val)
                                    depth_dilated = 1.0 - depth_for_dilate
                                    depth_norm = (
                                        upsample_depth(depth_dilated, frame_h, frame_w)
                                        if (dilate_h, dilate_w) != (frame_h, frame_w)
                                        else depth_dilated
                                    )
                                except RuntimeError as _dilate_err:
                                    print(f"[Dilation] skipped due to runtime error: {_dilate_err}", flush=True)
                                    depth_norm = upsample_depth(depth_norm_raw, frame_h, frame_w)
                        else:
                            depth_norm = upsample_depth(depth_norm_raw, frame_h, frame_w)

                        left_t, right_t = make_stereo(
                            frame, depth_norm, divergence, convergence, self.device,
                            already_normalized=True,
                            preserve_screen_border=bool(self.shared_preserve_border[0])
                        )


                        if auto_crop_on and self._content_process_h is not None and self._content_process_h < self.process_h:
                            left_t = pad_to_target_size(left_t, self.process_h, self.process_w)
                            right_t = pad_to_target_size(right_t, self.process_h, self.process_w)
                        mode = str(getattr(self.args, "mode", "openxr")).lower()
                        if mode in ("pc", "dual_monitor"):
                            fmt = str(getattr(self.args, "format", "hsbs"))
                            packed = pack_frame_gpu(left_t, right_t, fmt)
                            out = packed[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).contiguous()
                            try:
                                self.result_q.put_nowait(out)
                            except queue.Full:
                                try:
                                    self.result_q.get_nowait()
                                except queue.Empty:
                                    pass
                                try:
                                    self.result_q.put_nowait(out)
                                except queue.Full:
                                    pass
                        else:
                            left_out = left_t[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).contiguous()
                            right_out = right_t[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).contiguous()
                            try:
                                self.result_q.put_nowait(("openxr", left_out, right_out))
                            except queue.Full:
                                try:
                                    self.result_q.get_nowait()
                                except queue.Empty:
                                    pass
                                try:
                                    self.result_q.put_nowait(("openxr", left_out, right_out))
                                except queue.Full:
                                    pass


                    self._frame_count += 1


                    if self._frame_count % 30 == 0 and torch.cuda.is_available():
                        try:
                            total_mem = torch.cuda.get_device_properties(0).total_memory
                            reserved_mem = torch.cuda.memory_reserved(0)
                            if (reserved_mem / total_mem) > 0.85:
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                except Exception as _iw_e:
                    consecutive_errors += 1
                    print(f"[Inference] Frame processing error ({consecutive_errors}): {_iw_e}", flush=True)
                    if consecutive_errors >= 10:
                        print("[Inference] Too many consecutive errors, stopping.", flush=True)
                        self.running.clear()
                    continue
                else:
                    consecutive_errors = 0
        finally:
            print("[Inference] Stopped")


TEXTS = {
    "en": {
        "title": "Depth Live 3D  |  © 2026 Wake-82",
        "save": "Save", "load": "Load", "delete": "Delete", "start": "Start 3D Conversion", "stop": "Stop",
        "mode": "Mode", "format": "3D Format", "process_size": "Process Size", "target_fps": "Target FPS", "dynamic_res": "Dynamic Resolution Auto Adjust",
        "low_vram": "Low VRAM Mode (latest frame only, lowest latency)",
        "restart_notice_low_vram": "To apply the Low VRAM setting, please restart the conversion.",
        "cpu_perf": "CPU Performance",
        "cpu_perf_low": "Low", "cpu_perf_balanced": "Balanced", "cpu_perf_high": "High",
        "restart_notice_cpu_perf": "To apply the CPU Performance setting, please restart the conversion.",
        "divergence": "3D Strength", "auto_mode": "Auto Mode", "convergence": "Convergence", "edge": "Edge Fix",
        "ema": "Flicker Reduction", "preserve": "Preserve Screen Border",
        "auto_crop": "Auto Letterbox Crop (Remove Black Bars)",
        "input_size": "Depth Resolution", "hotkeys": "Keyboard Hotkeys",
        "hk_exit": "Exit Key", "hk_div_inc": "3D Strength +", "hk_div_dec": "3D Strength -",
        "hk_conv_inc": "Convergence +", "hk_conv_dec": "Convergence -", "hk_edge_inc": "Edge Fix +", "hk_edge_dec": "Edge Fix -",
        "hk_ema_inc": "Flicker Reduction +", "hk_ema_dec": "Flicker Reduction -", "hk_fps": "FPS Display", "exit_hold": "Exit Hold (sec)",
        "hk_vr_dist_inc": "VR Screen Dist +", "hk_vr_dist_dec": "VR Screen Dist -",
        "hk_vr_size_inc": "VR Screen Size +", "hk_vr_size_dec": "VR Screen Size -",
        "hk_vr_height_inc": "VR Screen Height +", "hk_vr_height_dec": "VR Screen Height -",
        "hk_vr_curve_inc": "VR Screen Curve +", "hk_vr_curve_dec": "VR Screen Curve -",
        "hk_vr_recenter": "VR Screen Recenter",
        "hk_vr_reset": "VR Screen Reset",
        "hk_fsbs_size_inc": "Full SBS Screen Size +",
        "hk_fsbs_size_dec": "Full SBS Screen Size -",
        "input_monitor": "Input Monitor",
        "output_monitor": "Output Monitor",
        "second_monitor_required": "Please assign a second monitor.",
        "auto_crop_on": "Letterbox Remove: ON",
        "auto_crop_off": "Letterbox Remove: OFF",
        "hk_auto_crop": "Letterbox Remove Toggle",
        "hk_auto_crop_state": "State",
        "auto_crop_hotkey_notice": "Please assign a hotkey to use the letterbox removal feature.",
        "fps_display": "FPS Display",
        "vr_dist": "VR Screen Dist", "vr_size": "VR Screen Size", "vr_curve": "VR Screen Curve", "vr_height": "VR Screen Height",
        "vr_recenter": "VR Screen Recenter", "vr_reset": "VR Screen Reset",
        "fsbs_screen_size": "Full SBS Screen Size",
        "mouse": "Mouse Cursor", "cursor_scale": "Cursor Scale", "cursor_color": "Cursor Color (R,G,B)",
        "cursor_outline": "Outline Color (R,G,B)", "cursor_autohide": "Auto Hide (while watching video)",
        "reset": "Reset Settings", "apply": "Apply", "log": "Log",
        "confirm_reset": "Reset all settings to default?", "yes": "Yes", "no": "No",
        "error_range": "Error: Unsupported value.\n{msg}", "script_not_found": "Cannot find engine.",
        "already_running": "Already running.", "started": "Conversion started.", "stopped": "Conversion stopped.",
        "invalid_hotkey": "Invalid hotkey format.", "applied": "Settings saved successfully.",
        "restart_notice": "To apply hotkeys/mouse settings, please restart the program.",
        "hk_placeholder": "Click & press key / Backspace to clear",
    },
}

class HotkeyLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setClearButtonEnabled(False)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            event.accept()
            return
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.clear()
            event.accept()
            return

        mods = []
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier: mods.append("ctrl")
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier: mods.append("shift")
        if event.modifiers() & Qt.KeyboardModifier.AltModifier: mods.append("alt")

        key_text = ""
        if Qt.Key.Key_F1 <= key <= Qt.Key.Key_F12: key_text = f"f{key - Qt.Key.Key_F1 + 1}"
        elif key == Qt.Key.Key_Escape: key_text = "esc"
        elif key == Qt.Key.Key_Space: key_text = "space"
        elif key == Qt.Key.Key_Tab: key_text = "tab"
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter): key_text = "enter"
        elif key == Qt.Key.Key_Up: key_text = "up"
        elif key == Qt.Key.Key_Down: key_text = "down"
        elif key == Qt.Key.Key_Left: key_text = "left"
        elif key == Qt.Key.Key_Right: key_text = "right"
        elif key == Qt.Key.Key_BracketLeft: key_text = "["
        elif key == Qt.Key.Key_BracketRight: key_text = "]"
        elif 0x20 <= key <= 0x7E: key_text = chr(key).lower()
        else:
            event.ignore()
            return

        if key_text:
            parts = mods + [key_text]
            self.setText("+".join(parts))
        event.accept()


class EmaComboBox(QComboBox):

    valueChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(False)
        for label, _ in EMA_LEVELS:
            self.addItem(label)
        self.currentIndexChanged.connect(self._emit_value_changed)

    def _emit_value_changed(self, _index: int):
        self.valueChanged.emit(self.value())

    def value(self) -> float:
        return ema_label_to_value(self.currentText())

    def setValue(self, value: float):
        label = ema_value_to_label(value)
        idx = self.findText(label)
        if idx >= 0:
            self.setCurrentIndex(idx)


class Live3DGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.lang = "en"
        self.process: QProcess | None = None
        self._build_ui()
        self._load_config()
        self._apply_language()
        self.setWindowTitle(self.t("title"))
        self.resize(880, 600)


        atexit.register(_force_letterbox_off_in_config)
        self._register_force_exit_handlers()

    def _register_force_exit_handlers(self):
        def _handler(signum, frame):
            _force_letterbox_off_in_config()

            sys.exit(0)

        for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    signal.signal(sig, _handler)
                except (ValueError, OSError, RuntimeError):

                    pass


        prev_excepthook = sys.excepthook

        def _excepthook(exc_type, exc_value, exc_tb):
            _force_letterbox_off_in_config()
            prev_excepthook(exc_type, exc_value, exc_tb)

        sys.excepthook = _excepthook

    def t(self, key: str) -> str:
        lang = getattr(self, "lang", "en")
        return TEXTS.get(lang, TEXTS["en"]).get(key, key)

    def _is_running(self) -> bool:
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 6)
        root.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setSpacing(10)
        scroll.setWidget(content)
        root.addWidget(scroll, stretch=1)


        self.grp_size = QGroupBox()
        s_lay = QGridLayout(self.grp_size)

        _combo_white_style = (
            "QComboBox { background-color: white; color: black; }"
            "QComboBox:disabled { background-color: white; color: #a0a0a0; }"
            "QComboBox QAbstractItemView { background-color: white; color: black; }"
        )
        self.cmb_mode = QComboBox()
        self.cmb_mode.setEditable(False)
        self.cmb_mode.setStyleSheet(_combo_white_style)
        self.cmb_mode.addItem("PC", "pc")
        self.cmb_mode.addItem("Dual-Monitor 3D Output", "dual_monitor")
        self.cmb_mode.addItem("OpenXR", "openxr")
        self.cmb_mode.setCurrentIndex(2)  # default OpenXR
        self.lbl_mode = QLabel()
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)

        self.cmb_process = QComboBox()
        self.cmb_process.setEditable(False)
        self.cmb_process.setStyleSheet(_combo_white_style)
        self.cmb_process.addItems(["3840x2160", "2560x1440", "1920x1080", "1280x720"])
        self.cmb_process.setCurrentText("1280x720")
        self.cmb_fps = QComboBox()
        self.cmb_fps.setEditable(False)
        self.cmb_fps.setStyleSheet(_combo_white_style)
        self.cmb_fps.addItems(["30", "60"])
        self.cmb_fps.setCurrentText("30")

        self.lbl_process = QLabel()
        self.lbl_fps = QLabel()
        self.cmb_input_mon = QComboBox()
        self.cmb_input_mon.setEditable(False)
        self.cmb_input_mon.setStyleSheet(_combo_white_style)
        self._populate_input_monitor_combo()
        self.lbl_input_mon = QLabel()
        self.btn_refresh_input_mon = QPushButton("\u21bb")
        self.btn_refresh_input_mon.setFixedWidth(28)
        self.btn_refresh_input_mon.setToolTip("Refresh monitor list")
        self.btn_refresh_input_mon.clicked.connect(lambda: self._populate_input_monitor_combo(keep_selection=True))

        self.lbl_output_mon = QLabel()
        self.cmb_output_mon = QComboBox()
        self.cmb_output_mon.setEditable(False)
        self.cmb_output_mon.setStyleSheet(_combo_white_style)
        self._populate_output_monitor_combo()
        self.btn_refresh_output_mon = QPushButton("\u21bb")
        self.btn_refresh_output_mon.setFixedWidth(28)
        self.btn_refresh_output_mon.setToolTip("Refresh monitor list")
        self.btn_refresh_output_mon.clicked.connect(lambda: self._populate_output_monitor_combo(keep_selection=True))
        self.cmb_input_mon.currentIndexChanged.connect(self._on_input_monitor_changed)
        self.cmb_output_mon.currentIndexChanged.connect(self._on_output_monitor_changed)
        # Output monitor row is only relevant to Dual-Monitor 3D Output mode;
        # hidden by default (only the label + row container are toggled, so
        # the combo/button keep their own default-visible state and appear
        # correctly the moment the row is shown).
        self.lbl_output_mon.setVisible(False)

        self.chk_dynamic_res = QCheckBox()
        self.chk_dynamic_res.setChecked(True)


        self.cmb_cpu_perf = QComboBox()
        self.cmb_cpu_perf.setStyleSheet(_combo_white_style)
        self.cmb_cpu_perf.setEditable(False)
        self._cpu_perf_ratios = {"low": 0.3, "balanced": 0.5, "high": 0.7}
        self.cmb_cpu_perf.addItem("Low", "low")
        self.cmb_cpu_perf.addItem("Balanced", "balanced")
        self.cmb_cpu_perf.addItem("High", "high")
        self.cmb_cpu_perf.setCurrentIndex(1)
        self.lbl_cpu_perf = QLabel()
        self.cmb_cpu_perf.currentIndexChanged.connect(self._on_cpu_perf_changed)

        self.chk_low_vram = QCheckBox()
        self.chk_low_vram.setChecked(False)
        self.chk_low_vram.toggled.connect(self._on_low_vram_toggled)

        s_lay.addWidget(self.lbl_mode, 0, 0)
        s_lay.addWidget(self.cmb_mode, 0, 1)
        s_lay.addWidget(self.lbl_process, 1, 0)
        s_lay.addWidget(self.cmb_process, 1, 1)
        s_lay.addWidget(self.lbl_fps, 2, 0)
        s_lay.addWidget(self.cmb_fps, 2, 1)
        s_lay.addWidget(self.lbl_input_mon, 3, 0)
        _input_mon_row = QWidget()
        _input_mon_row_lay = QHBoxLayout(_input_mon_row)
        _input_mon_row_lay.setContentsMargins(0, 0, 0, 0)
        _input_mon_row_lay.setSpacing(4)
        _input_mon_row_lay.addWidget(self.cmb_input_mon, stretch=1)
        _input_mon_row_lay.addWidget(self.btn_refresh_input_mon)
        s_lay.addWidget(_input_mon_row, 3, 1)

        s_lay.addWidget(self.lbl_output_mon, 4, 0)
        _output_mon_row = QWidget()
        _output_mon_row_lay = QHBoxLayout(_output_mon_row)
        _output_mon_row_lay.setContentsMargins(0, 0, 0, 0)
        _output_mon_row_lay.setSpacing(4)
        _output_mon_row_lay.addWidget(self.cmb_output_mon, stretch=1)
        _output_mon_row_lay.addWidget(self.btn_refresh_output_mon)
        s_lay.addWidget(_output_mon_row, 4, 1)
        self._output_mon_row = _output_mon_row
        self._output_mon_row.setVisible(False)

        s_lay.addWidget(self.chk_dynamic_res, 5, 0, 1, 2)
        s_lay.addWidget(self.lbl_cpu_perf, 6, 0)
        s_lay.addWidget(self.cmb_cpu_perf, 6, 1)
        s_lay.addWidget(self.chk_low_vram, 7, 0, 1, 2)
        self.content_layout.addWidget(self.grp_size)

        self.grp_stereo = QGroupBox()
        st_lay = QGridLayout(self.grp_stereo)


        self.cmb_format = QComboBox()
        self.cmb_format.setEditable(False)
        self.cmb_format.setStyleSheet(_combo_white_style)
        # Labels remain "hsbs"/"tb" (half-SBS / half-TB look), but packing is full-res SBS/TB
        # stretched to fill the 16:9 monitor with no letterbox.
        # "fsbs" packs the same as "hsbs" but is shown at its true aspect ratio
        # (no stretch) — a normal, correctly-proportioned Full SBS output.
        self.cmb_format.addItems(["hsbs", "fsbs", "tb", "anaglyph", "half-anaglyph"])
        self.cmb_format.setCurrentText("hsbs")
        self.lbl_format = QLabel()

        self.spin_div = QDoubleSpinBox()
        self.spin_div.setRange(0.5, 2.5)
        self.spin_div.setSingleStep(0.5)
        self.spin_div.setDecimals(1)
        self.spin_div.setValue(1.0)

        self.spin_conv = QDoubleSpinBox()
        self.spin_conv.setRange(0.0, 1.0)
        self.spin_conv.setSingleStep(0.1)
        self.spin_conv.setDecimals(1)
        self.spin_conv.setValue(0.5)

        self.spin_edge = QSpinBox()
        self.spin_edge.setRange(0, 5)
        self.spin_edge.setValue(2)

        self.spin_ema = EmaComboBox()
        self.spin_ema.setStyleSheet(_combo_white_style)
        self.spin_ema.setValue(0.0)


        self.chk_auto_mode = QCheckBox()
        self.chk_auto_mode.setChecked(True)

        self.chk_preserve = QCheckBox()
        self.chk_preserve.setChecked(True)


        self.chk_auto_crop = QCheckBox()
        self.chk_auto_crop.setChecked(False)
        self.chk_auto_crop.setVisible(False)

        self.cmb_input_size = QComboBox()
        self.cmb_input_size.setEditable(False)
        self.cmb_input_size.setStyleSheet(_combo_white_style)
        self.cmb_input_size.addItems(["256", "384", "512"])
        self.cmb_input_size.setCurrentText("256")


        self.chk_show_fps = QCheckBox()
        self.chk_show_fps.setChecked(False)


        self.spin_fsbs_size = QDoubleSpinBox()
        self.spin_fsbs_size.setRange(0.0, 50.0)
        self.spin_fsbs_size.setSingleStep(0.5)
        self.spin_fsbs_size.setDecimals(2)
        self.spin_fsbs_size.setValue(0.0)

        self.spin_vr_dist = QDoubleSpinBox()
        self.spin_vr_dist.setRange(0.3, 20.0)
        self.spin_vr_dist.setSingleStep(0.1)
        self.spin_vr_dist.setDecimals(2)
        self.spin_vr_dist.setValue(2.0)

        self.spin_vr_size = QDoubleSpinBox()
        self.spin_vr_size.setRange(0.2, 10.0)
        self.spin_vr_size.setSingleStep(0.05)
        self.spin_vr_size.setDecimals(2)
        self.spin_vr_size.setValue(1.5)

        self.spin_vr_curve = QDoubleSpinBox()
        self.spin_vr_curve.setRange(0.0, 100.0)
        self.spin_vr_curve.setSingleStep(1.0)
        self.spin_vr_curve.setDecimals(0)
        self.spin_vr_curve.setValue(0.0)

        self._vr_height_offset = 0.0

        self.btn_vr_recenter = QPushButton()
        self.btn_vr_reset = QPushButton()
        self.btn_vr_height_inc = QPushButton()
        self.btn_vr_height_dec = QPushButton()

        self.lbl_div = QLabel()
        self.lbl_conv = QLabel()
        self.lbl_edge = QLabel()
        self.lbl_ema = QLabel()
        self.lbl_input_size = QLabel()
        self.lbl_vr_dist = QLabel()
        self.lbl_vr_size = QLabel()
        self.lbl_vr_curve = QLabel()
        self.lbl_fsbs_size = QLabel()

        st_lay.addWidget(self.chk_auto_mode, 0, 0, 1, 2)
        st_lay.addWidget(self.lbl_format, 1, 0)
        st_lay.addWidget(self.cmb_format, 1, 1)
        st_lay.addWidget(self.lbl_div, 2, 0)
        st_lay.addWidget(self.spin_div, 2, 1)
        st_lay.addWidget(self.lbl_conv, 3, 0)
        st_lay.addWidget(self.spin_conv, 3, 1)
        st_lay.addWidget(self.lbl_edge, 4, 0)
        st_lay.addWidget(self.spin_edge, 4, 1)
        st_lay.addWidget(self.lbl_ema, 5, 0)
        st_lay.addWidget(self.spin_ema, 5, 1)
        st_lay.addWidget(self.chk_preserve, 6, 0, 1, 2)

        st_lay.addWidget(self.lbl_input_size, 7, 0)
        st_lay.addWidget(self.cmb_input_size, 7, 1)

        self.content_layout.addWidget(self.grp_stereo)


        self.grp_vr_screen = QGroupBox()
        vr_lay = QGridLayout(self.grp_vr_screen)
        vr_lay.addWidget(self.chk_show_fps, 0, 0, 1, 2)
        vr_lay.addWidget(self.lbl_fsbs_size, 1, 0)
        vr_lay.addWidget(self.spin_fsbs_size, 1, 1)
        vr_lay.addWidget(self.lbl_vr_dist, 2, 0)
        vr_lay.addWidget(self.spin_vr_dist, 2, 1)
        vr_lay.addWidget(self.lbl_vr_size, 3, 0)
        vr_lay.addWidget(self.spin_vr_size, 3, 1)
        vr_lay.addWidget(self.lbl_vr_curve, 4, 0)
        vr_lay.addWidget(self.spin_vr_curve, 4, 1)


        btn_vr_row = QHBoxLayout()
        btn_vr_row.addWidget(self.btn_vr_recenter)
        btn_vr_row.addWidget(self.btn_vr_height_dec)
        btn_vr_row.addWidget(self.btn_vr_height_inc)
        btn_vr_row.addWidget(self.btn_vr_reset)
        btn_vr_row.addStretch(1)
        vr_lay.addLayout(btn_vr_row, 5, 0, 1, 2)

        self.content_layout.addWidget(self.grp_vr_screen)

        self.spin_div.valueChanged.connect(self._on_div_changed)
        self.spin_conv.valueChanged.connect(self._send_params_to_process)
        self.spin_edge.valueChanged.connect(self._send_params_to_process)
        self.spin_ema.valueChanged.connect(self._send_params_to_process)
        self.chk_auto_mode.toggled.connect(self._on_auto_mode_toggled)
        self.chk_show_fps.toggled.connect(self._on_show_fps_toggled)
        self.chk_auto_crop.toggled.connect(self._on_auto_crop_toggled)
        self.chk_preserve.toggled.connect(self._on_preserve_toggled)
        self.cmb_format.currentTextChanged.connect(self._on_mode_changed)
        self.spin_fsbs_size.valueChanged.connect(self._send_fsbs_size)
        self.spin_vr_dist.valueChanged.connect(self._send_vr_panel)
        self.spin_vr_size.valueChanged.connect(self._send_vr_panel)
        self.spin_vr_curve.valueChanged.connect(self._send_vr_panel)
        self.btn_vr_recenter.clicked.connect(self._cmd_vr_recenter)
        self.btn_vr_reset.clicked.connect(self._cmd_vr_reset)
        self.btn_vr_height_inc.clicked.connect(self._cmd_vr_height_inc)
        self.btn_vr_height_dec.clicked.connect(self._cmd_vr_height_dec)

        self.grp_hotkey = QGroupBox()
        hk_lay = QGridLayout(self.grp_hotkey)

        self.edt_hk_exit = HotkeyLineEdit()
        self.edt_hk_exit.setText("esc")
        self.spin_exit_hold = QDoubleSpinBox()
        self.spin_exit_hold.setRange(0.5, 10.0)
        self.spin_exit_hold.setSingleStep(0.5)
        self.spin_exit_hold.setValue(2.0)
        self.edt_hk_div_inc = HotkeyLineEdit()
        self.edt_hk_div_inc.setText("]")
        self.edt_hk_div_dec = HotkeyLineEdit()
        self.edt_hk_div_dec.setText("[")
        self.edt_hk_conv_inc = HotkeyLineEdit()
        self.edt_hk_conv_dec = HotkeyLineEdit()
        self.edt_hk_edge_inc = HotkeyLineEdit()
        self.edt_hk_edge_dec = HotkeyLineEdit()
        self.edt_hk_ema_inc = HotkeyLineEdit()
        self.edt_hk_ema_dec = HotkeyLineEdit()
        self.edt_hk_auto_crop = HotkeyLineEdit()
        self.edt_hk_fps = HotkeyLineEdit()
        self.edt_hk_vr_dist_inc = HotkeyLineEdit()
        self.edt_hk_vr_dist_dec = HotkeyLineEdit()
        self.edt_hk_vr_size_inc = HotkeyLineEdit()
        self.edt_hk_vr_size_dec = HotkeyLineEdit()
        self.edt_hk_vr_height_inc = HotkeyLineEdit()
        self.edt_hk_vr_height_dec = HotkeyLineEdit()
        self.edt_hk_vr_curve_inc = HotkeyLineEdit()
        self.edt_hk_vr_curve_dec = HotkeyLineEdit()
        self.edt_hk_vr_recenter = HotkeyLineEdit()
        self.edt_hk_vr_reset = HotkeyLineEdit()
        self.edt_hk_fsbs_size_inc = HotkeyLineEdit()
        self.edt_hk_fsbs_size_dec = HotkeyLineEdit()

        self.lbl_hk_exit = QLabel()
        self.lbl_exit_hold = QLabel()
        self.lbl_hk_div_inc = QLabel()
        self.lbl_hk_div_dec = QLabel()
        self.lbl_hk_conv_inc = QLabel()
        self.lbl_hk_conv_dec = QLabel()
        self.lbl_hk_edge_inc = QLabel()
        self.lbl_hk_edge_dec = QLabel()
        self.lbl_hk_ema_inc = QLabel()
        self.lbl_hk_ema_dec = QLabel()
        self.lbl_auto_crop_notice = QLabel()
        self.lbl_auto_crop_notice.setWordWrap(True)
        self.lbl_hk_auto_crop = QLabel()
        self.lbl_hk_fps = QLabel()
        self.lbl_hk_vr_dist_inc = QLabel()
        self.lbl_hk_vr_dist_dec = QLabel()
        self.lbl_hk_vr_size_inc = QLabel()
        self.lbl_hk_vr_size_dec = QLabel()
        self.lbl_hk_vr_height_inc = QLabel()
        self.lbl_hk_vr_height_dec = QLabel()
        self.lbl_hk_vr_curve_inc = QLabel()
        self.lbl_hk_vr_curve_dec = QLabel()
        self.lbl_hk_vr_recenter = QLabel()
        self.lbl_hk_vr_reset = QLabel()
        self.lbl_hk_fsbs_size_inc = QLabel()
        self.lbl_hk_fsbs_size_dec = QLabel()

        row = 0
        for lbl, w in [
            (self.lbl_hk_exit, self.edt_hk_exit), (self.lbl_exit_hold, self.spin_exit_hold),
            (self.lbl_hk_div_inc, self.edt_hk_div_inc), (self.lbl_hk_div_dec, self.edt_hk_div_dec),
            (self.lbl_hk_conv_inc, self.edt_hk_conv_inc), (self.lbl_hk_conv_dec, self.edt_hk_conv_dec),
            (self.lbl_hk_edge_inc, self.edt_hk_edge_inc), (self.lbl_hk_edge_dec, self.edt_hk_edge_dec),
            (self.lbl_hk_ema_inc, self.edt_hk_ema_inc), (self.lbl_hk_ema_dec, self.edt_hk_ema_dec),
        ]:
            hk_lay.addWidget(lbl, row, 0)
            hk_lay.addWidget(w, row, 1)
            row += 1


        for lbl, w in [
            (self.lbl_hk_fps, self.edt_hk_fps),
            (self.lbl_hk_fsbs_size_inc, self.edt_hk_fsbs_size_inc), (self.lbl_hk_fsbs_size_dec, self.edt_hk_fsbs_size_dec),
            (self.lbl_hk_vr_dist_inc, self.edt_hk_vr_dist_inc), (self.lbl_hk_vr_dist_dec, self.edt_hk_vr_dist_dec),
            (self.lbl_hk_vr_size_inc, self.edt_hk_vr_size_inc), (self.lbl_hk_vr_size_dec, self.edt_hk_vr_size_dec),
            (self.lbl_hk_vr_height_inc, self.edt_hk_vr_height_inc), (self.lbl_hk_vr_height_dec, self.edt_hk_vr_height_dec),
            (self.lbl_hk_vr_curve_inc, self.edt_hk_vr_curve_inc), (self.lbl_hk_vr_curve_dec, self.edt_hk_vr_curve_dec),
            (self.lbl_hk_vr_recenter, self.edt_hk_vr_recenter),
            (self.lbl_hk_vr_reset, self.edt_hk_vr_reset),
        ]:
            hk_lay.addWidget(lbl, row, 0)
            hk_lay.addWidget(w, row, 1)
            row += 1


        hk_lay.addWidget(self.lbl_auto_crop_notice, row, 0, 1, 2)
        row += 1
        hk_lay.addWidget(self.lbl_hk_auto_crop, row, 0)
        hk_lay.addWidget(self.edt_hk_auto_crop, row, 1)
        row += 1

        self.btn_apply_hotkey = QPushButton()
        self.btn_apply_hotkey.clicked.connect(self._apply_hotkeys)
        hk_lay.addWidget(self.btn_apply_hotkey, row, 0, 1, 2)
        self.content_layout.addWidget(self.grp_hotkey)

        self.grp_mouse = QGroupBox()
        mo_lay = QGridLayout(self.grp_mouse)

        self.spin_cursor_scale = QDoubleSpinBox()
        self.spin_cursor_scale.setRange(0.5, 5.0)
        self.spin_cursor_scale.setSingleStep(0.1)
        self.spin_cursor_scale.setValue(2.0)
        self.edt_cursor_color = QLineEdit("255,255,255")
        self.edt_cursor_outline = QLineEdit("30,30,30")
        self.chk_autohide = QCheckBox()
        self.chk_autohide.setChecked(True)
        self.lbl_cursor_scale = QLabel()
        self.lbl_cursor_color = QLabel()
        self.lbl_cursor_outline = QLabel()

        mo_lay.addWidget(self.lbl_cursor_scale, 0, 0)
        mo_lay.addWidget(self.spin_cursor_scale, 0, 1)
        mo_lay.addWidget(self.lbl_cursor_color, 1, 0)
        mo_lay.addWidget(self.edt_cursor_color, 1, 1)
        mo_lay.addWidget(self.lbl_cursor_outline, 2, 0)
        mo_lay.addWidget(self.edt_cursor_outline, 2, 1)
        mo_lay.addWidget(self.chk_autohide, 3, 0, 1, 2)

        self.btn_apply_mouse = QPushButton()
        self.btn_apply_mouse.clicked.connect(self._apply_mouse)
        mo_lay.addWidget(self.btn_apply_mouse, 4, 0, 1, 2)
        self.content_layout.addWidget(self.grp_mouse)

        self.btn_reset = QPushButton()
        self.btn_reset.clicked.connect(self._reset_settings)
        self.content_layout.addWidget(self.btn_reset)
        self.content_layout.addStretch()


        btn_row = QHBoxLayout()
        self.btn_start = QPushButton()
        self.btn_stop = QPushButton()
        self.btn_start.setMinimumHeight(36)
        self.btn_stop.setMinimumHeight(36)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_start, stretch=1)
        btn_row.addWidget(self.btn_stop, stretch=1)
        root.addLayout(btn_row)

        self.lbl_log = QLabel()
        root.addWidget(self.lbl_log)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(56)
        self.log.setFont(QFont("Consolas", 9))
        root.addWidget(self.log)

        self._apply_openxr_state()
        self._apply_mode_state()
        self._update_fps_options()
        self._update_auto_mode_enabled_state()

    def _update_auto_mode_enabled_state(self):
        enabled = not self.chk_auto_mode.isChecked()
        for w in (
            self.spin_edge, self.lbl_edge, self.spin_ema, self.lbl_ema,
            self.edt_hk_edge_inc, self.edt_hk_edge_dec,
            self.lbl_hk_edge_inc, self.lbl_hk_edge_dec,
            self.edt_hk_ema_inc, self.edt_hk_ema_dec,
            self.lbl_hk_ema_inc, self.lbl_hk_ema_dec,
        ):
            w.setEnabled(enabled)

    def _apply_auto_values(self):
        edge, ema = compute_auto_edge_ema(self.spin_div.value())
        self.spin_edge.blockSignals(True)
        self.spin_ema.blockSignals(True)
        self.spin_edge.setValue(edge)
        self.spin_ema.setValue(ema)
        self.spin_edge.blockSignals(False)
        self.spin_ema.blockSignals(False)

    def _on_div_changed(self, _value):
        if self.chk_auto_mode.isChecked():
            self._apply_auto_values()
        self._send_params_to_process()

    def _on_auto_mode_toggled(self, checked: bool):
        self._update_auto_mode_enabled_state()
        if checked:
            self._apply_auto_values()
        self._send_auto_mode_to_process()
        self._send_params_to_process()

    def _send_auto_mode_to_process(self):
        if self._is_running():
            msg = f"SET_AUTO_MODE:{1 if self.chk_auto_mode.isChecked() else 0}\n"
            self.process.write(msg.encode("utf-8"))

    def _update_auto_crop_label(self):
        on = self.chk_auto_crop.isChecked()
        self.chk_auto_crop.setText(self.t("auto_crop_on") if on else self.t("auto_crop_off"))

        state_str = "ON" if on else "OFF"
        self.lbl_hk_auto_crop.setText(
            f"{self.t('hk_auto_crop')} ({self.t('hk_auto_crop_state')}: {state_str})"
        )

    def _populate_input_monitor_combo(self, keep_selection: bool = False):
        """Fill the input-monitor combo with '<index> - <name>' entries."""
        prev_data = self.cmb_input_mon.currentData() if keep_selection else None
        self.cmb_input_mon.blockSignals(True)
        self.cmb_input_mon.clear()
        try:
            monitors = get_monitor_list()
        except Exception as e:
            print(f"[Warning] Could not enumerate monitors for UI: {e}")
            monitors = []
        if not monitors:
            monitors = [{"index": 0, "name": "Unknown Display"}]
        for mon in monitors:
            idx = mon.get("index", 0)
            name = mon.get("name") or "Unknown Display"
            w = mon.get("width")
            h = mon.get("height")
            if w and h:
                label = f"{idx} - {name} ({w}x{h})"
            else:
                label = f"{idx} - {name}"
            self.cmb_input_mon.addItem(label, idx)
        if prev_data is not None:
            found = self.cmb_input_mon.findData(prev_data)
            if found >= 0:
                self.cmb_input_mon.setCurrentIndex(found)
        self.cmb_input_mon.blockSignals(False)

    def _input_monitor_value(self) -> int:
        data = self.cmb_input_mon.currentData()
        return int(data) if data is not None else 0

    def _set_input_monitor_value(self, value: int):
        found = self.cmb_input_mon.findData(int(value))
        if found >= 0:
            self.cmb_input_mon.setCurrentIndex(found)
        elif self.cmb_input_mon.count() > 0:
            self.cmb_input_mon.setCurrentIndex(0)

    def _populate_output_monitor_combo(self, keep_selection: bool = False):
        """Fill the output-monitor combo the same way as the input-monitor
        combo (always populated, independent of which mode is currently
        selected). If no second monitor is currently detected, show a single
        explicit 'None' entry instead of just re-listing monitor 0."""
        prev_data = self.cmb_output_mon.currentData() if keep_selection else None
        self.cmb_output_mon.blockSignals(True)
        self.cmb_output_mon.clear()
        try:
            monitors = get_monitor_list()
        except Exception as e:
            print(f"[Warning] Could not enumerate monitors for UI: {e}")
            monitors = []
        if len(monitors) < 2:
            self.cmb_output_mon.addItem("None (no second monitor detected)", None)
        else:
            for mon in monitors:
                idx = mon.get("index", 0)
                name = mon.get("name") or "Unknown Display"
                w = mon.get("width")
                h = mon.get("height")
                if w and h:
                    label = f"{idx} - {name} ({w}x{h})"
                else:
                    label = f"{idx} - {name}"
                self.cmb_output_mon.addItem(label, idx)
            if prev_data is not None:
                found = self.cmb_output_mon.findData(prev_data)
                if found >= 0:
                    self.cmb_output_mon.setCurrentIndex(found)
            else:
                # Default the output monitor to the second detected monitor
                # (index 1), so it doesn't default to the same monitor as
                # the input by default.
                found = self.cmb_output_mon.findData(1)
                if found >= 0:
                    self.cmb_output_mon.setCurrentIndex(found)
        self.cmb_output_mon.blockSignals(False)

    def _output_monitor_value(self) -> int:
        data = self.cmb_output_mon.currentData()
        return int(data) if data is not None else -1

    def _set_output_monitor_value(self, value):
        if value is None:
            return
        found = self.cmb_output_mon.findData(int(value))
        if found >= 0:
            self.cmb_output_mon.setCurrentIndex(found)

    def _on_input_monitor_changed(self, _idx: int = -1):
        """If the user picks an input monitor that matches the current
        output monitor (only meaningful in Dual-Monitor 3D Output mode),
        automatically move the output monitor to a different one."""
        if self._current_mode() != "dual_monitor":
            return
        in_val = self._input_monitor_value()
        out_val = self._output_monitor_value()
        if out_val is None or out_val < 0 or out_val != in_val:
            return
        alt = self._find_alternate_monitor_value(exclude=in_val)
        if alt is not None:
            self._set_output_monitor_value(alt)

    def _on_output_monitor_changed(self, _idx: int = -1):
        """If the user picks an output monitor that matches the current
        input monitor (only meaningful in Dual-Monitor 3D Output mode),
        automatically move the input monitor to a different one."""
        if self._current_mode() != "dual_monitor":
            return
        out_val = self._output_monitor_value()
        if out_val is None or out_val < 0:
            return
        in_val = self._input_monitor_value()
        if out_val != in_val:
            return
        alt = self._find_alternate_monitor_value(exclude=out_val)
        if alt is not None:
            self._set_input_monitor_value(alt)

    def _find_alternate_monitor_value(self, exclude: int):
        """Return a monitor index (from the input-monitor combo's available
        entries) that is not `exclude`, preferring the one right after it,
        e.g. 0 -> 1, 1 -> 0. Returns None if no alternate is available."""
        values = []
        for i in range(self.cmb_input_mon.count()):
            data = self.cmb_input_mon.itemData(i)
            if data is not None:
                values.append(int(data))
        candidates = [v for v in values if v != exclude]
        if not candidates:
            return None
        # Prefer exclude+1 if it exists among the candidates (mirrors the
        # common 0/1 dual-monitor case), otherwise just take the first
        # available alternate.
        if (exclude + 1) in candidates:
            return exclude + 1
        return candidates[0]

    def _on_low_vram_toggled(self, _checked: bool):


        if self._is_running():
            QMessageBox.information(self, "DepthLive3D", self.t("restart_notice_low_vram"))

    def _on_cpu_perf_changed(self, _index: int):


        if self._is_running():
            QMessageBox.information(self, "DepthLive3D", self.t("restart_notice_cpu_perf"))

    def _apply_language(self):
        t = self.t
        self.setWindowTitle(t("title"))
        self.btn_start.setText(t("start"))
        self.btn_stop.setText(t("stop"))
        self.lbl_input_mon.setText(t("input_monitor"))
        self.lbl_output_mon.setText(t("output_monitor"))
        self.grp_size.setTitle(t("process_size") + " / " + t("target_fps"))
        self.lbl_mode.setText(t("mode"))
        self.lbl_process.setText(t("process_size"))
        self.lbl_fps.setText(t("target_fps"))
        self.chk_dynamic_res.setText(t("dynamic_res"))
        self.chk_low_vram.setText(t("low_vram"))
        self.lbl_cpu_perf.setText(t("cpu_perf"))
        self.cmb_cpu_perf.setItemText(0, t("cpu_perf_low"))
        self.cmb_cpu_perf.setItemText(1, t("cpu_perf_balanced"))
        self.cmb_cpu_perf.setItemText(2, t("cpu_perf_high"))
        self.grp_stereo.setTitle("")
        self.lbl_format.setText(t("format"))
        self.lbl_div.setText(t("divergence"))
        self.chk_auto_mode.setText(t("auto_mode"))
        self.lbl_conv.setText(t("convergence"))
        self.lbl_edge.setText(t("edge"))
        self.lbl_ema.setText(t("ema"))
        self.chk_preserve.setText(t("preserve"))
        self._update_auto_crop_label()
        self.lbl_input_size.setText(t("input_size"))
        self.chk_show_fps.setText(t("fps_display"))
        self.lbl_fsbs_size.setText(t("fsbs_screen_size"))
        self.lbl_vr_dist.setText(t("vr_dist"))
        self.lbl_vr_size.setText(t("vr_size"))
        self.lbl_vr_curve.setText(t("vr_curve"))
        self.btn_vr_recenter.setText(t("vr_recenter"))
        self.btn_vr_reset.setText(t("vr_reset"))
        self.btn_vr_height_inc.setText(t("hk_vr_height_inc"))
        self.btn_vr_height_dec.setText(t("hk_vr_height_dec"))
        self.grp_hotkey.setTitle(t("hotkeys"))
        self.lbl_hk_exit.setText(t("hk_exit"))
        self.lbl_exit_hold.setText(t("exit_hold"))
        self.lbl_hk_div_inc.setText(t("hk_div_inc"))
        self.lbl_hk_div_dec.setText(t("hk_div_dec"))
        self.lbl_hk_conv_inc.setText(t("hk_conv_inc"))
        self.lbl_hk_conv_dec.setText(t("hk_conv_dec"))
        self.lbl_hk_edge_inc.setText(t("hk_edge_inc"))
        self.lbl_hk_edge_dec.setText(t("hk_edge_dec"))
        self.lbl_hk_ema_inc.setText(t("hk_ema_inc"))
        self.lbl_hk_ema_dec.setText(t("hk_ema_dec"))
        self.lbl_auto_crop_notice.setText(t("auto_crop_hotkey_notice"))
        self._update_auto_crop_label()
        self.lbl_hk_fps.setText(t("hk_fps"))
        self.lbl_hk_fsbs_size_inc.setText(t("hk_fsbs_size_inc"))
        self.lbl_hk_fsbs_size_dec.setText(t("hk_fsbs_size_dec"))
        self.lbl_hk_vr_dist_inc.setText(t("hk_vr_dist_inc"))
        self.lbl_hk_vr_dist_dec.setText(t("hk_vr_dist_dec"))
        self.lbl_hk_vr_size_inc.setText(t("hk_vr_size_inc"))
        self.lbl_hk_vr_size_dec.setText(t("hk_vr_size_dec"))
        self.lbl_hk_vr_curve_inc.setText(t("hk_vr_curve_inc"))
        self.lbl_hk_vr_curve_dec.setText(t("hk_vr_curve_dec"))
        self.lbl_hk_vr_height_inc.setText(t("hk_vr_height_inc"))
        self.lbl_hk_vr_height_dec.setText(t("hk_vr_height_dec"))
        self.lbl_hk_vr_recenter.setText(t("hk_vr_recenter"))
        self.lbl_hk_vr_reset.setText(t("hk_vr_reset"))
        self.btn_apply_hotkey.setText(t("apply"))
        self.grp_mouse.setTitle(t("mouse"))
        self.lbl_cursor_scale.setText(t("cursor_scale"))
        self.lbl_cursor_color.setText(t("cursor_color"))
        self.lbl_cursor_outline.setText(t("cursor_outline"))
        self.chk_autohide.setText(t("cursor_autohide"))
        self.btn_apply_mouse.setText(t("apply"))
        self.btn_reset.setText(t("reset"))
        self.lbl_log.setText(t("log"))
        for hk_edt in [self.edt_hk_exit, self.edt_hk_div_inc, self.edt_hk_div_dec,
                       self.edt_hk_conv_inc, self.edt_hk_conv_dec,
                       self.edt_hk_edge_inc, self.edt_hk_edge_dec,
                       self.edt_hk_ema_inc, self.edt_hk_ema_dec,
                       self.edt_hk_auto_crop,
                       self.edt_hk_fps,
                       self.edt_hk_fsbs_size_inc, self.edt_hk_fsbs_size_dec,
                       self.edt_hk_vr_dist_inc, self.edt_hk_vr_dist_dec,
                       self.edt_hk_vr_size_inc, self.edt_hk_vr_size_dec,
                       self.edt_hk_vr_height_inc, self.edt_hk_vr_height_dec,
                       self.edt_hk_vr_curve_inc, self.edt_hk_vr_curve_dec,
                       self.edt_hk_vr_recenter,
                       self.edt_hk_vr_reset]:
            hk_edt.setPlaceholderText(t("hk_placeholder"))

    
    def _current_mode(self) -> str:
        if not hasattr(self, "cmb_mode") or self.cmb_mode is None:
            return "openxr"
        try:
            data = self.cmb_mode.currentData()
            if data:
                return str(data)
            txt = (self.cmb_mode.currentText() or "").strip().lower()
            if txt in ("pc", "openxr", "dual_monitor"):
                return txt
            return "openxr"
        except Exception:
            return "openxr"

    def _on_mode_changed(self, *_):
        if hasattr(self, "cmb_mode") and self.cmb_mode is not None:
            self._apply_mode_state()
        self._update_fps_options()

    def _pc_fast_capture_selected(self) -> bool:
        """True when the fast dxcam capture path applies: OpenXR mode always,
        Dual-Monitor 3D Output mode always (no 3D-format restriction), or PC
        mode with anaglyph / half-anaglyph format. False for PC mode with
        hsbs / fsbs / tb, which stays on the ~30 FPS-capped Magnification path."""
        _mode = self._current_mode()
        if _mode != "pc":
            return True
        if not hasattr(self, "cmb_format") or self.cmb_format is None:
            return False
        fmt = self.cmb_format.currentText().strip().lower()
        return fmt in ("anaglyph", "half-anaglyph")

    def _update_fps_options(self):
        if not hasattr(self, "cmb_fps") or self.cmb_fps is None:
            return
        fast = self._pc_fast_capture_selected()
        current = self.cmb_fps.currentText().strip()
        desired_items = ["30", "60"] if fast else ["30"]
        existing_items = [self.cmb_fps.itemText(i) for i in range(self.cmb_fps.count())]
        if existing_items == desired_items:
            return
        self.cmb_fps.blockSignals(True)
        try:
            self.cmb_fps.clear()
            self.cmb_fps.addItems(desired_items)
            self.cmb_fps.setCurrentText(current if current in desired_items else "30")
        finally:
            self.cmb_fps.blockSignals(False)

    def _apply_mode_state(self):
        """Enable PC-only or OpenXR-only controls. Safe if widgets not yet built."""
        if not hasattr(self, "cmb_mode") or self.cmb_mode is None:
            return
        _mode = self._current_mode()
        is_pc = _mode == "pc"
        is_dual = _mode == "dual_monitor"
        is_oxr = not (is_pc or is_dual)
        if hasattr(self, "cmb_format") and self.cmb_format is not None:
            self.cmb_format.setEnabled(is_pc or is_dual)
        if hasattr(self, "lbl_format") and self.lbl_format is not None:
            self.lbl_format.setEnabled(is_pc or is_dual)

        # Output-monitor row only applies to Dual-Monitor 3D Output mode;
        # show/hide immediately when the mode changes, hidden for every
        # other mode.
        for w in (
            getattr(self, "lbl_output_mon", None),
            getattr(self, "_output_mon_row", None),
        ):
            if w is not None:
                try:
                    w.setVisible(is_dual)
                except Exception:
                    pass
        # FPS display, VR Screen (dist/size/recenter/reset), and Full SBS
        # Screen Size are always enabled regardless of mode — no longer
        # gated by is_oxr/is_pc/is_fsbs.
        for w in (
            getattr(self, "chk_show_fps", None),
            getattr(self, "spin_vr_dist", None),
            getattr(self, "spin_vr_size", None),
            getattr(self, "spin_vr_curve", None),
            getattr(self, "lbl_vr_dist", None),
            getattr(self, "lbl_vr_size", None),
            getattr(self, "lbl_vr_curve", None),
            getattr(self, "btn_vr_recenter", None),
            getattr(self, "btn_vr_reset", None),
            getattr(self, "btn_vr_height_inc", None),
            getattr(self, "btn_vr_height_dec", None),
            getattr(self, "grp_vr_screen", None),
            getattr(self, "grp_vr", None),
            getattr(self, "spin_fsbs_size", None),
            getattr(self, "lbl_fsbs_size", None),
        ):
            if w is not None:
                try:
                    w.setEnabled(True)
                except Exception:
                    pass
        for name in ("grp_hotkeys_vr", "grp_vr_hotkeys", "grp_openxr"):
            w = getattr(self, name, None)
            if w is not None:
                try:
                    w.setEnabled(True)
                except Exception:
                    pass

    def _apply_openxr_state(self):

        self.cmb_input_mon.setEnabled(True)
        self.lbl_input_mon.setEnabled(True)

    def _round_input_size(self, val: int) -> int:

        return max(64, int(round(val / 32.0) * 32))

    def _validate(self) -> str | None:
        if not (0.1 <= self.spin_div.value() <= 5.0): return "divergence"
        if not (0.0 <= self.spin_conv.value() <= 1.0): return "convergence"
        if not (0 <= self.spin_edge.value() <= 10): return "edge-dilation"
        try:
            w, h = self.cmb_process.currentText().lower().split("x")
            int(w), int(h)
        except Exception: return "process-size"
        _fps_txt = self.cmb_fps.currentText().strip().lower()
        try: float(_fps_txt)
        except Exception: return "target-fps"
        return None

    def _build_cmd(self) -> list[str]:

        if getattr(sys, "frozen", False):

            cmd = [sys.executable, "--run-engine"]
        else:

            cmd = [sys.executable, os.path.abspath(__file__), "--run-engine"]


        mode = self._current_mode()
        cmd += ["--mode", mode]
        if mode in ("pc", "dual_monitor"):
            cmd += ["--format", self.cmb_format.currentText().strip()]
        if mode == "dual_monitor":
            cmd += ["--output-monitor", str(self._output_monitor_value())]

        cmd += ["--input-monitor", str(self._input_monitor_value())]

        cmd += ["--process-size", self.cmb_process.currentText().strip()]
        _fps_val = self.cmb_fps.currentText().strip()
        cmd += ["--target-fps", _fps_val]
        if self.chk_dynamic_res.isChecked():
            cmd += ["--dynamic-resolution"]

        _cpu_key = self.cmb_cpu_perf.currentData()
        _cpu_ratio = self._cpu_perf_ratios.get(_cpu_key, 0.5)
        cmd += ["--cpu-headroom-ratio", f"{_cpu_ratio:.2f}"]

        if self.chk_low_vram.isChecked():
            cmd += ["--low-vram"]
        else:
            cmd += ["--no-low-vram"]
        cmd += ["--divergence", f"{self.spin_div.value():.1f}"]
        cmd += ["--convergence", f"{self.spin_conv.value():.1f}"]
        cmd += ["--edge-dilation", str(self.spin_edge.value())]
        cmd += ["--ema-decay", f"{self.spin_ema.value():.2f}"]
        if self.chk_auto_mode.isChecked():
            cmd += ["--auto-mode"]

        if self.chk_preserve.isChecked(): cmd += ["--preserve-screen-border"]
        else: cmd += ["--no-preserve-screen-border"]

        if self.chk_auto_crop.isChecked():
            cmd += ["--auto-crop"]

        try:
            isz = self._round_input_size(int(self.cmb_input_size.currentText().strip()))
        except ValueError: isz = 384
        self.cmb_input_size.setCurrentText(str(isz))
        cmd += ["--input-size", str(isz)]

        def add_hk(flag, edt):
            v = edt.text().strip()
            if v: cmd.extend([flag, v])

        add_hk("--hk-exit", self.edt_hk_exit)
        cmd += ["--exit-hold", f"{self.spin_exit_hold.value():.1f}"]
        add_hk("--hk-div-inc", self.edt_hk_div_inc)
        add_hk("--hk-div-dec", self.edt_hk_div_dec)
        add_hk("--hk-conv-inc", self.edt_hk_conv_inc)
        add_hk("--hk-conv-dec", self.edt_hk_conv_dec)
        add_hk("--hk-edge-inc", self.edt_hk_edge_inc)
        add_hk("--hk-edge-dec", self.edt_hk_edge_dec)
        add_hk("--hk-ema-inc", self.edt_hk_ema_inc)
        add_hk("--hk-ema-dec", self.edt_hk_ema_dec)
        add_hk("--hk-auto-crop", self.edt_hk_auto_crop)
        add_hk("--hk-fps-toggle", self.edt_hk_fps)
        add_hk("--hk-fsbs-size-inc", self.edt_hk_fsbs_size_inc)
        add_hk("--hk-fsbs-size-dec", self.edt_hk_fsbs_size_dec)
        add_hk("--hk-vr-dist-inc", self.edt_hk_vr_dist_inc)
        add_hk("--hk-vr-dist-dec", self.edt_hk_vr_dist_dec)
        add_hk("--hk-vr-size-inc", self.edt_hk_vr_size_inc)
        add_hk("--hk-vr-size-dec", self.edt_hk_vr_size_dec)
        add_hk("--hk-vr-height-inc", self.edt_hk_vr_height_inc)
        add_hk("--hk-vr-height-dec", self.edt_hk_vr_height_dec)
        add_hk("--hk-vr-curve-inc", self.edt_hk_vr_curve_inc)
        add_hk("--hk-vr-curve-dec", self.edt_hk_vr_curve_dec)
        add_hk("--hk-vr-recenter", self.edt_hk_vr_recenter)
        add_hk("--hk-vr-reset", self.edt_hk_vr_reset)

        cmd += ["--cursor-scale", f"{self.spin_cursor_scale.value():.1f}"]
        cmd += ["--cursor-color", self.edt_cursor_color.text().strip()]
        cmd += ["--cursor-outline", self.edt_cursor_outline.text().strip()]
        if self.chk_autohide.isChecked(): cmd += ["--cursor-autohide"]
        if self.chk_show_fps.isChecked(): cmd += ["--show-fps"]
        cmd += ["--fsbs-screen-size", f"{self.spin_fsbs_size.value():.2f}"]

        return cmd

    def _apply_hotkeys(self):
        self._save_config()
        if self._is_running(): QMessageBox.information(self, "DepthLive3D", self.t("restart_notice"))
        else: self.log.append(self.t("applied"))

    def _apply_mouse(self):
        self._save_config()
        if self._is_running(): QMessageBox.information(self, "DepthLive3D", self.t("restart_notice"))
        else: self.log.append(self.t("applied"))

    def _start(self):
        if self._is_running():
            QMessageBox.warning(self, "DepthLive3D", self.t("already_running"))
            return

        if self._current_mode() == "dual_monitor":
            if self._output_monitor_value() < 0:
                QMessageBox.warning(self, "DepthLive3D", self.t("second_monitor_required"))
                return

        err = self._validate()
        if err:
            QMessageBox.critical(self, "DepthLive3D", self.t("error_range").format(msg=err))
            return

        cmd = self._build_cmd()
        self.log.append(">>> " + " ".join(cmd))
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")


        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)

        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.finished.connect(self._on_finished)
        self.process.start(cmd[0], cmd[1:])
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log.append(self.t("started"))

    def _stop(self):
        if self._is_running():

            try:
                self.process.write(b"CMD_EXIT\n")
                self.process.waitForBytesWritten(500)
            except Exception:
                pass


            if not self.process.waitForFinished(3000):
                self.process.terminate()
                if not self.process.waitForFinished(2000):
                    self.process.kill()
            self.log.append(self.t("stopped"))
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.chk_auto_crop.isChecked():
            self.chk_auto_crop.blockSignals(True)
            self.chk_auto_crop.setChecked(False)
            self.chk_auto_crop.blockSignals(False)
            self._update_auto_crop_label()

    def _send_params_to_process(self):
        if self._is_running():
            msg = f"SET_PARAMS:{self.spin_div.value():.1f}:{self.spin_conv.value():.1f}:{self.spin_edge.value()}:{self.spin_ema.value():.2f}\n"
            self.process.write(msg.encode("utf-8"))

    def _on_show_fps_toggled(self, checked: bool):
        if self._is_running():
            msg = f"SET_SHOW_FPS:{1 if checked else 0}\n"
            self.process.write(msg.encode("utf-8"))

    def _on_auto_crop_toggled(self, checked: bool):
        self._update_auto_crop_label()
        if self._is_running():
            msg = f"SET_AUTO_CROP:{1 if checked else 0}\n"
            self.process.write(msg.encode("utf-8"))

    def _on_preserve_toggled(self, checked: bool):
        if self._is_running():
            msg = f"SET_PRESERVE_BORDER:{1 if checked else 0}\n"
            self.process.write(msg.encode("utf-8"))

    def _send_fsbs_size(self):
        if self._is_running():
            msg = f"SET_FSBS_SIZE:{self.spin_fsbs_size.value():.2f}\n"
            self.process.write(msg.encode("utf-8"))

    def _send_vr_panel(self):
        if self._is_running():
            msg = f"SET_VR_PANEL:{self.spin_vr_dist.value():.2f}:{self.spin_vr_size.value():.2f}:{self._vr_height_offset:.2f}:{self.spin_vr_curve.value():.1f}\n"
            self.process.write(msg.encode("utf-8"))

    def _cmd_vr_recenter(self):
        if self._is_running():
            self.process.write(b"CMD_VR_RECENTER\n")

    def _cmd_vr_reset(self):
        if self._is_running():
            self.process.write(b"CMD_VR_RESET\n")
        else:
            self.spin_vr_dist.blockSignals(True)
            self.spin_vr_size.blockSignals(True)
            self.spin_vr_curve.blockSignals(True)
            self.spin_vr_dist.setValue(2.0)
            self.spin_vr_size.setValue(1.5)
            self.spin_vr_curve.setValue(0.0)
            self.spin_vr_dist.blockSignals(False)
            self.spin_vr_size.blockSignals(False)
            self.spin_vr_curve.blockSignals(False)
            self._vr_height_offset = 0.0

    def _cmd_vr_height_inc(self):
        if self._is_running():
            self.process.write(b"CMD_VR_HEIGHT_INC\n")

    def _cmd_vr_height_dec(self):
        if self._is_running():
            self.process.write(b"CMD_VR_HEIGHT_DEC\n")

    def _on_stdout(self):
        if self.process:
            data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
            for line in data.splitlines():
                if line.startswith("SYNC_PARAMS:"):
                    try:
                        parts = line.split(":")
                        div = parts[1]; conv = parts[2]; edge = parts[3]
                        self.spin_div.blockSignals(True)
                        self.spin_conv.blockSignals(True)
                        self.spin_edge.blockSignals(True)
                        self.spin_ema.blockSignals(True)
                        self.spin_div.setValue(float(div))
                        self.spin_conv.setValue(float(conv))
                        self.spin_edge.setValue(int(edge))
                        if len(parts) >= 5: self.spin_ema.setValue(float(parts[4]))
                        self.spin_div.blockSignals(False)
                        self.spin_conv.blockSignals(False)
                        self.spin_edge.blockSignals(False)
                        self.spin_ema.blockSignals(False)
                    except (ValueError, IndexError): pass
                elif line.startswith("SYNC_SHOW_FPS:"):
                    try:
                        val = line.split(":")[1].strip() == "1"
                        self.chk_show_fps.blockSignals(True)
                        self.chk_show_fps.setChecked(val)
                        self.chk_show_fps.blockSignals(False)
                    except Exception:
                        pass
                elif line.startswith("SYNC_AUTO_CROP:"):
                    try:
                        val = line.split(":")[1].strip() == "1"
                        self.chk_auto_crop.blockSignals(True)
                        self.chk_auto_crop.setChecked(val)
                        self.chk_auto_crop.blockSignals(False)
                        self._update_auto_crop_label()
                    except Exception:
                        pass
                elif line.startswith("SYNC_PRESERVE_BORDER:"):
                    try:
                        val = line.split(":")[1].strip() == "1"
                        self.chk_preserve.blockSignals(True)
                        self.chk_preserve.setChecked(val)
                        self.chk_preserve.blockSignals(False)
                    except Exception:
                        pass
                elif line.startswith("SYNC_FSBS_SIZE:"):
                    try:
                        v = float(line.split(":")[1])
                        self.spin_fsbs_size.blockSignals(True)
                        self.spin_fsbs_size.setValue(v)
                        self.spin_fsbs_size.blockSignals(False)
                    except Exception:
                        pass
                elif line.startswith("SYNC_VR_PANEL:"):
                    try:
                        parts = line.split(":")
                        d = float(parts[1]); h = float(parts[2]); y = float(parts[3])
                        c = float(parts[4]) if len(parts) >= 5 else self.spin_vr_curve.value()
                        self.spin_vr_dist.blockSignals(True)
                        self.spin_vr_size.blockSignals(True)
                        self.spin_vr_curve.blockSignals(True)
                        self.spin_vr_dist.setValue(d)
                        self.spin_vr_size.setValue(h)
                        self.spin_vr_curve.setValue(c)
                        self.spin_vr_dist.blockSignals(False)
                        self.spin_vr_size.blockSignals(False)
                        self.spin_vr_curve.blockSignals(False)
                        self._vr_height_offset = y
                    except Exception:
                        pass
                elif line.startswith("[OpenXR] drawing frame") or line.startswith("[EMA]"):

                    pass
                else:
                    self.log.moveCursor(self.log.textCursor().MoveOperation.End)
                    self.log.insertPlainText(line + "\n")
                    self.log.moveCursor(self.log.textCursor().MoveOperation.End)

    def _on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log.append("--- process finished ---")

    def _reset_settings(self):
        reply = QMessageBox.question(self, "DepthLive3D", self.t("confirm_reset"), QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes: return

        self._set_input_monitor_value(0)
        self._set_output_monitor_value(1)
        self.cmb_process.setCurrentText("1280x720")
        self.cmb_fps.setCurrentText("30")
        self.chk_dynamic_res.setChecked(True)
        self.cmb_cpu_perf.setCurrentIndex(1)
        self.chk_low_vram.setChecked(False)
        self.chk_auto_mode.setChecked(True)
        self.spin_div.setValue(1.0)
        self.spin_conv.setValue(0.5)
        self.spin_edge.setValue(2)
        self.spin_ema.setValue(0.00)
        self.chk_preserve.setChecked(True)
        self.chk_auto_crop.setChecked(False)
        self.cmb_input_size.setCurrentText("256")
        self.chk_show_fps.setChecked(False)
        self.spin_fsbs_size.setValue(0.0)
        self._send_fsbs_size()
        self.spin_vr_dist.setValue(2.0)
        self.spin_vr_size.setValue(1.5)
        self.spin_vr_curve.setValue(0.0)
        self._vr_height_offset = 0.0
        self._send_vr_panel()
        self.edt_hk_exit.setText("esc")
        self.spin_exit_hold.setValue(2.0)
        self.edt_hk_div_inc.setText("]")
        self.edt_hk_div_dec.setText("[")
        self.edt_hk_conv_inc.clear()
        self.edt_hk_conv_dec.clear()
        self.edt_hk_edge_inc.clear()
        self.edt_hk_edge_dec.clear()
        self.edt_hk_ema_inc.clear()
        self.edt_hk_ema_dec.clear()
        self.edt_hk_auto_crop.clear()
        self.edt_hk_fps.clear()
        self._update_auto_crop_label()
        self.edt_hk_vr_dist_inc.clear()
        self.edt_hk_vr_dist_dec.clear()
        self.edt_hk_vr_size_inc.clear()
        self.edt_hk_vr_size_dec.clear()
        self.edt_hk_vr_height_inc.clear()
        self.edt_hk_vr_height_dec.clear()
        self.edt_hk_vr_curve_inc.clear()
        self.edt_hk_vr_curve_dec.clear()
        self.edt_hk_vr_recenter.clear()
        self.edt_hk_vr_reset.clear()
        self.edt_hk_fsbs_size_inc.clear()
        self.edt_hk_fsbs_size_dec.clear()
        self.spin_cursor_scale.setValue(2.0)
        self.edt_cursor_color.setText("255,255,255")
        self.edt_cursor_outline.setText("30,30,30")
        self.chk_autohide.setChecked(True)
        self.log.append(self.t("applied"))

    def _collect_config(self) -> dict:
        return {
            "input_monitor": self._input_monitor_value(),
            "output_monitor": self._output_monitor_value(),
            "mode": self._current_mode(), "format": self.cmb_format.currentText(),
            "process_size": self.cmb_process.currentText(),
            "target_fps": self.cmb_fps.currentText(),
            "dynamic_res": self.chk_dynamic_res.isChecked(),
            "cpu_perf": self.cmb_cpu_perf.currentData(),
            "low_vram": self.chk_low_vram.isChecked(),
            "divergence": self.spin_div.value(),
            "convergence": self.spin_conv.value(),
            "edge": self.spin_edge.value(),
            "ema_decay": self.spin_ema.value(),
            "auto_mode": self.chk_auto_mode.isChecked(),
            "preserve": self.chk_preserve.isChecked(),
            "auto_crop": self.chk_auto_crop.isChecked(),
            "input_size": self.cmb_input_size.currentText(),
            "show_fps": self.chk_show_fps.isChecked(),
            "fsbs_screen_size": self.spin_fsbs_size.value(),
            "hk_fsbs_size_inc": self.edt_hk_fsbs_size_inc.text(),
            "hk_fsbs_size_dec": self.edt_hk_fsbs_size_dec.text(),
            "openxr_panel_distance": self.spin_vr_dist.value(),
            "openxr_panel_height": self.spin_vr_size.value(),
            "openxr_panel_y_offset": self._vr_height_offset,
            "openxr_panel_curve": self.spin_vr_curve.value(),
            "hk_exit": self.edt_hk_exit.text(),
            "exit_hold": self.spin_exit_hold.value(),
            "hk_div_inc": self.edt_hk_div_inc.text(),
            "hk_div_dec": self.edt_hk_div_dec.text(),
            "hk_conv_inc": self.edt_hk_conv_inc.text(),
            "hk_conv_dec": self.edt_hk_conv_dec.text(),
            "hk_edge_inc": self.edt_hk_edge_inc.text(),
            "hk_edge_dec": self.edt_hk_edge_dec.text(),
            "hk_ema_inc": self.edt_hk_ema_inc.text(),
            "hk_ema_dec": self.edt_hk_ema_dec.text(),
            "hk_auto_crop": self.edt_hk_auto_crop.text(),
            "hk_fps": self.edt_hk_fps.text(),
            "hk_vr_dist_inc": self.edt_hk_vr_dist_inc.text(),
            "hk_vr_dist_dec": self.edt_hk_vr_dist_dec.text(),
            "hk_vr_size_inc": self.edt_hk_vr_size_inc.text(),
            "hk_vr_size_dec": self.edt_hk_vr_size_dec.text(),
            "hk_vr_height_inc": self.edt_hk_vr_height_inc.text(),
            "hk_vr_height_dec": self.edt_hk_vr_height_dec.text(),
            "hk_vr_curve_inc": self.edt_hk_vr_curve_inc.text(),
            "hk_vr_curve_dec": self.edt_hk_vr_curve_dec.text(),
            "hk_vr_recenter": self.edt_hk_vr_recenter.text(),
            "hk_vr_reset": self.edt_hk_vr_reset.text(),
            "cursor_scale": self.spin_cursor_scale.value(),
            "cursor_color": self.edt_cursor_color.text(),
            "cursor_outline": self.edt_cursor_outline.text(),
            "cursor_autohide": self.chk_autohide.isChecked(),
        }

    def _apply_config(self, cfg: dict):
        self.lang = "en"

        self._set_input_monitor_value(cfg.get("input_monitor", cfg.get("in_mon", 0)))
        self._set_output_monitor_value(cfg.get("output_monitor", 1))
        self.cmb_process.setCurrentText(cfg.get("process_size", cfg.get("process", "1280x720")))
        _mode = cfg.get("mode", "openxr")
        for i in range(self.cmb_mode.count()):
            if self.cmb_mode.itemData(i) == _mode or self.cmb_mode.itemText(i).lower() == str(_mode).lower():
                self.cmb_mode.setCurrentIndex(i)
                break
        self.cmb_format.setCurrentText(cfg.get("format", "hsbs"))
        self.cmb_fps.setCurrentText(str(cfg.get("target_fps", cfg.get("fps", "30"))))
        self.chk_dynamic_res.setChecked(cfg.get("dynamic_res", True))
        _cpu_perf_idx = {"low": 0, "balanced": 1, "high": 2}.get(cfg.get("cpu_perf", "balanced"), 1)
        self.cmb_cpu_perf.setCurrentIndex(_cpu_perf_idx)
        self.chk_low_vram.setChecked(cfg.get("low_vram", False))
        self.chk_auto_mode.setChecked(cfg.get("auto_mode", True))
        self.spin_div.setValue(cfg.get("divergence", 1.0))
        self.spin_conv.setValue(cfg.get("convergence", 0.5))
        self.spin_edge.setValue(cfg.get("edge", 2))
        self.spin_ema.setValue(cfg.get("ema_decay", 0.00))
        self.chk_preserve.setChecked(cfg.get("preserve", True))
        self.chk_auto_crop.setChecked(cfg.get("auto_crop", False))
        self.cmb_input_size.setCurrentText(str(cfg.get("input_size", "256")))
        self.chk_show_fps.setChecked(cfg.get("show_fps", False))
        self.spin_fsbs_size.setValue(cfg.get("fsbs_screen_size", 0.0))
        self.edt_hk_fsbs_size_inc.setText(cfg.get("hk_fsbs_size_inc", ""))
        self.edt_hk_fsbs_size_dec.setText(cfg.get("hk_fsbs_size_dec", ""))
        self.spin_vr_dist.setValue(cfg.get("openxr_panel_distance", cfg.get("vr_dist", 2.0)))
        self.spin_vr_size.setValue(cfg.get("openxr_panel_height", cfg.get("vr_size", 1.5)))
        self.spin_vr_curve.setValue(cfg.get("openxr_panel_curve", cfg.get("vr_curve", 0.0)))
        self._vr_height_offset = cfg.get("openxr_panel_y_offset", cfg.get("vr_height", 0.0))
        self.edt_hk_exit.setText(cfg.get("hk_exit", "esc"))
        self.spin_exit_hold.setValue(cfg.get("exit_hold", 2.0))
        self.edt_hk_div_inc.setText(cfg.get("hk_div_inc", "]"))
        self.edt_hk_div_dec.setText(cfg.get("hk_div_dec", "["))
        self.edt_hk_conv_inc.setText(cfg.get("hk_conv_inc", ""))
        self.edt_hk_conv_dec.setText(cfg.get("hk_conv_dec", ""))
        self.edt_hk_edge_inc.setText(cfg.get("hk_edge_inc", ""))
        self.edt_hk_edge_dec.setText(cfg.get("hk_edge_dec", ""))
        self.edt_hk_ema_inc.setText(cfg.get("hk_ema_inc", ""))
        self.edt_hk_ema_dec.setText(cfg.get("hk_ema_dec", ""))
        self.edt_hk_auto_crop.setText(cfg.get("hk_auto_crop", ""))
        self.edt_hk_fps.setText(cfg.get("hk_fps", ""))
        self._update_auto_crop_label()
        self.edt_hk_vr_dist_inc.setText(cfg.get("hk_vr_dist_inc", ""))
        self.edt_hk_vr_dist_dec.setText(cfg.get("hk_vr_dist_dec", ""))
        self.edt_hk_vr_size_inc.setText(cfg.get("hk_vr_size_inc", ""))
        self.edt_hk_vr_size_dec.setText(cfg.get("hk_vr_size_dec", ""))
        self.edt_hk_vr_height_inc.setText(cfg.get("hk_vr_height_inc", ""))
        self.edt_hk_vr_height_dec.setText(cfg.get("hk_vr_height_dec", ""))
        self.edt_hk_vr_curve_inc.setText(cfg.get("hk_vr_curve_inc", ""))
        self.edt_hk_vr_curve_dec.setText(cfg.get("hk_vr_curve_dec", ""))
        self.edt_hk_vr_recenter.setText(cfg.get("hk_vr_recenter", ""))
        self.edt_hk_vr_reset.setText(cfg.get("hk_vr_reset", ""))
        self.spin_cursor_scale.setValue(cfg.get("cursor_scale", cfg.get("c_scale", 2.0)))
        self.edt_cursor_color.setText(cfg.get("cursor_color", cfg.get("c_color", "255,255,255")))
        self.edt_cursor_outline.setText(cfg.get("cursor_outline", cfg.get("c_out", "30,30,30")))
        self.chk_autohide.setChecked(cfg.get("cursor_autohide", cfg.get("c_auto", True)))
        self._apply_openxr_state()
        self._apply_mode_state()
        self._update_fps_options()

    def _save_config(self):
        try:
            CONFIG_FILE.write_text(
                json.dumps(self._collect_config(), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass

    def _load_config(self):
        if not CONFIG_FILE.exists():
            return
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            self._apply_config(cfg)
        except Exception:
            pass

    def closeEvent(self, event):
        self._stop()


        self.chk_auto_crop.setChecked(False)
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
        painter.drawText(pix.rect(), Qt.AlignCenter, "Live3D\nLoading...")
    finally:
        painter.end()
    return pix


def launch_gui():
    app = QApplication(sys.argv)

    splash = QSplashScreen(_build_splash_pixmap())
    splash.show()
    app.processEvents()

    state = {"gui": None}

    def _init_and_show():
        gui = Live3DGui()
        state["gui"] = gui
        gui.show()
        splash.finish(gui)

    # Defer the heavy GUI construction until after the event loop starts,
    # so the OS sees the app pumping messages immediately instead of
    # showing a "busy" (hourglass) cursor while widgets are being built.
    QTimer.singleShot(0, _init_and_show)

    sys.exit(app.exec())


def main():
    ap = argparse.ArgumentParser(description="Live3D realtime (Win32GL + CUDA interop) + GUI Launcher")

    ap.add_argument("--run-engine", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--mode", choices=["pc", "openxr", "dual_monitor"], default="openxr",
                    help="pc: single-monitor overlay (WDA_NONE + Mag capture) | openxr: headset direct submit | "
                         "dual_monitor: dxcam-captures the input monitor and outputs the converted 3D frame "
                         "full-screen (WDA_NONE, Virtual Desktop compatible) onto a separate output monitor")
    ap.add_argument("--format", choices=["hsbs", "fsbs", "tb", "anaglyph", "half-anaglyph"], default="hsbs",
                    help="Stereo pack format for PC mode overlay. "
                         "hsbs/tb pack at full eye resolution (full-SBS / full-TB) then stretch to fill "
                         "the 16:9 monitor (no letterbox) so the visual matches classic half-SBS / half-TB. "
                         "fsbs packs identically to hsbs but is displayed at its true, correct aspect ratio "
                         "(letterboxed/pillarboxed, no stretch) — a normal full-SBS output.")
    ap.add_argument("--no-click-through", action="store_true", default=False,
                    help="PC overlay receives mouse clicks (default: click-through)")


    ap.add_argument("--input-monitor", type=int, default=0)
    ap.add_argument("--output-monitor", type=int, default=1,
                     help="Dual-Monitor 3D Output mode only: monitor index the converted 3D frame is displayed on")
    ap.add_argument("--process-size", default="1280x720")
    ap.add_argument("--target-fps", type=float, default=60)
    ap.add_argument("--dynamic-resolution", action="store_true", help="Automatically change monitor resolution to process-size and restore on exit")
    ap.add_argument("--cpu-headroom-ratio", type=float, default=0.5,
                     help="CPU Performance: ratio of remaining cores used for computation (Low=0.3 / Balanced=0.5 / High=0.7). "
                          "Already applied before torch/cv2 import; used here only for record-keeping/logging.")
    ap.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=False,
                     help="OFF (default): queue size 2/4, rarely drops intermediate frames, smoother playback on 12GB-class VRAM. "
                          "ON: queue size 1/1, uses only the latest frame to minimize latency (recommended for low-VRAM environments).")
    ap.add_argument("--divergence", "-d", type=float, default=1.0)
    ap.add_argument("--convergence", "-c", type=float, default=0.5)
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--invert-depth", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--cursor-ox", type=int, default=0)
    ap.add_argument("--cursor-oy", type=int, default=0)
    ap.add_argument("--cursor-scale", type=float, default=1.0)
    ap.add_argument("--cursor-color", type=parse_bgr, default="255,255,255")
    ap.add_argument("--cursor-outline", type=parse_bgr, default="30,30,30")
    ap.add_argument("--cursor-outline-width", type=int, default=1)
    ap.add_argument("--edge-dilation", type=parse_edge_dilation, default="2")
    ap.add_argument("--preserve-screen-border", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cursor-autohide", action="store_true")
    ap.add_argument("--show-fps", action="store_true")
    ap.add_argument("--auto-crop", action="store_true", help="Automatically remove letterbox in real time (top/bottom black bars)")
    ap.add_argument("--ema-decay", type=float, default=0.0)
    ap.add_argument("--auto-mode", action="store_true", default=False)
    ap.add_argument("--div-step", type=float, default=0.5)
    ap.add_argument("--conv-step", type=float, default=0.1)
    ap.add_argument("--edge-step", type=int, default=1)
    ap.add_argument("--hk-exit", type=str, default="esc")
    ap.add_argument("--exit-hold", type=float, default=2.0)
    ap.add_argument("--hk-div-inc", type=str, default="]")
    ap.add_argument("--hk-div-dec", type=str, default="[")
    ap.add_argument("--hk-conv-inc", type=str, default="")
    ap.add_argument("--hk-conv-dec", type=str, default="")
    ap.add_argument("--hk-edge-inc", type=str, default="")
    ap.add_argument("--hk-edge-dec", type=str, default="")
    ap.add_argument("--hk-ema-inc", type=str, default="")
    ap.add_argument("--hk-ema-dec", type=str, default="")
    ap.add_argument("--hk-auto-crop", type=str, default="", help="Hotkey to toggle letterbox removal")
    ap.add_argument("--hk-fps-toggle", type=str, default="", help="Hotkey to toggle FPS display")
    ap.add_argument("--hk-vr-dist-inc", type=str, default="")
    ap.add_argument("--hk-vr-dist-dec", type=str, default="")
    ap.add_argument("--hk-vr-size-inc", type=str, default="")
    ap.add_argument("--hk-vr-size-dec", type=str, default="")
    ap.add_argument("--hk-vr-height-inc", type=str, default="")
    ap.add_argument("--hk-vr-height-dec", type=str, default="")
    ap.add_argument("--hk-vr-curve-inc", type=str, default="")
    ap.add_argument("--hk-vr-curve-dec", type=str, default="")
    ap.add_argument("--hk-vr-recenter", type=str, default="")
    ap.add_argument("--hk-vr-reset", type=str, default="")
    ap.add_argument("--hk-fsbs-size-inc", type=str, default="")
    ap.add_argument("--hk-fsbs-size-dec", type=str, default="")
    ap.add_argument("--fsbs-screen-size", type=float, default=0.0)
    ap.add_argument("--fsbs-size-step", type=float, default=0.5)
    ap.add_argument("--ema-step", type=float, default=0.05)
    ap.add_argument("--vr-dist-step", type=float, default=0.1)
    ap.add_argument("--vr-size-step", type=float, default=0.05)
    ap.add_argument("--vr-height-step", type=float, default=0.05)
    ap.add_argument("--vr-curve-step", type=float, default=5.0)

    args = ap.parse_args()

    if not args.run_engine:
        launch_gui()
        return


    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    print(f"[CPU Performance] ratio={_CPU_HEADROOM_RATIO:.2f} (arg={getattr(args, 'cpu_headroom_ratio', None)}) "
          f"-> threads={_CPU_THREADS} (cpu_count={_CPU_COUNT})", flush=True)

    args.ema_decay = clamp_ema_value(args.ema_decay)

    try:
        hk_exit = parse_hotkey(args.hk_exit)
        hk_div_inc = parse_hotkey(args.hk_div_inc)
        hk_div_dec = parse_hotkey(args.hk_div_dec)
        hk_conv_inc = parse_hotkey(args.hk_conv_inc)
        hk_conv_dec = parse_hotkey(args.hk_conv_dec)
        hk_edge_inc = parse_hotkey(args.hk_edge_inc)
        hk_edge_dec = parse_hotkey(args.hk_edge_dec)
        hk_ema_inc = parse_hotkey(args.hk_ema_inc)
        hk_ema_dec = parse_hotkey(args.hk_ema_dec)
        hk_fps_toggle = parse_hotkey(args.hk_fps_toggle)
        hk_auto_crop = parse_hotkey(getattr(args, "hk_auto_crop", ""))
        hk_vr_dist_inc = parse_hotkey(args.hk_vr_dist_inc)
        hk_vr_dist_dec = parse_hotkey(args.hk_vr_dist_dec)
        hk_vr_size_inc = parse_hotkey(args.hk_vr_size_inc)
        hk_vr_size_dec = parse_hotkey(args.hk_vr_size_dec)
        hk_vr_height_inc = parse_hotkey(args.hk_vr_height_inc)
        hk_vr_height_dec = parse_hotkey(args.hk_vr_height_dec)
        hk_vr_curve_inc = parse_hotkey(getattr(args, "hk_vr_curve_inc", ""))
        hk_vr_curve_dec = parse_hotkey(getattr(args, "hk_vr_curve_dec", ""))
        hk_vr_recenter = parse_hotkey(args.hk_vr_recenter)
        hk_vr_reset = parse_hotkey(args.hk_vr_reset)
        hk_fsbs_size_inc = parse_hotkey(getattr(args, "hk_fsbs_size_inc", ""))
        hk_fsbs_size_dec = parse_hotkey(getattr(args, "hk_fsbs_size_dec", ""))
    except argparse.ArgumentTypeError as e:
        print(f"[Error] Hotkey parsing failed: {e}")
        sys.exit(1)

    winmm = ctypes.windll.winmm
    winmm.timeBeginPeriod(1)
    try: ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try: user32.SetProcessDPIAware()
        except Exception: pass

    monitors = get_monitor_list()
    print(f"Detected monitors: {len(monitors)}")


    capture_idx = args.input_monitor
    out_mon = monitors[0] if monitors else {"left": 0, "top": 0, "width": 1920, "height": 1080}
    vr_mode = False
    _mode_label = str(getattr(args, "mode", "openxr")).lower()
    print(f"[{_mode_label.upper()} Mode] Capture: monitor {capture_idx}")


    try:
        _cap_mon = monitors[capture_idx]
        capture_monitor_left = int(_cap_mon["left"])
        capture_monitor_top = int(_cap_mon["top"])
    except Exception:
        capture_monitor_left, capture_monitor_top = 0, 0


    original_display_mode = None
    if getattr(args, "dynamic_resolution", False):
        try:
            import win32api

            devices = []
            i = 0
            while True:
                try:
                    d = win32api.EnumDisplayDevices(None, i)
                    if d.StateFlags & 1:
                        devices.append(d)
                    i += 1
                except Exception:
                    break
            if capture_idx < len(devices):
                device_name = devices[capture_idx].DeviceName
            else:
                device_name = None

            original_display_mode = get_current_display_mode(device_name)
            process_w, process_h = parse_size(args.process_size)
            print(f"[Dynamic Resolution] Changing {original_display_mode['width']}x{original_display_mode['height']} → {process_w}x{process_h}")
            ok = change_display_resolution(process_w, process_h, device_name=device_name)
            if ok:
                print("[Dynamic Resolution] Resolution changed successfully.")
                time.sleep(1.0)


                monitors = get_monitor_list()
                print(f"[Dynamic Resolution] Monitors re-enumerated: {len(monitors)}")


                import atexit
                def _atexit_restore():
                    if original_display_mode is not None:
                        try:
                            print("[Dynamic Resolution] atexit: restoring original resolution...", flush=True)
                            restore_display_mode(original_display_mode)
                        except Exception as e:
                            print(f"[Warning] atexit restore failed: {e}", flush=True)
                atexit.register(_atexit_restore)
            else:
                print("[Warning] Failed to change display resolution. Continuing with original resolution.")
                original_display_mode = None
        except Exception as e:
            print(f"[Warning] Dynamic resolution change failed: {e}")
            original_display_mode = None

    process_w, process_h = parse_size(args.process_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not _HAS_OPENGL:
        print("[Error] PyOpenGL is missing. (pip install PyOpenGL PyOpenGL_accelerate)")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("[Error] CUDA is not available.")
        sys.exit(1)

    print("Loading ZipDepth...")
    predictor = load_zipdepth(input_size=args.input_size, fp16=not args.no_fp16)

    overlay = None
    openxr_viewer = None

    def _cleanup_leftover_overlays():
        def enum_handler(hwnd, _):
            try:
                title = win32gui.GetWindowText(hwnd)
                cls = win32gui.GetClassName(hwnd)
                if "Live3D" in title or "Live3D_Win32GL_Class" in cls:
                    try:
                        user32.SetWindowDisplayAffinity(int(hwnd), WDA_NONE)
                    except Exception:
                        pass
                    win32gui.DestroyWindow(hwnd)
            except Exception:
                pass
            return True
        try:
            win32gui.EnumWindows(enum_handler, None)
        except Exception:
            pass

    mode = str(getattr(args, "mode", "openxr")).lower()
    is_dual_monitor = (mode == "dual_monitor")
    if mode == "openxr":
        if not _HAS_OPENXR:
            print("[Error] pyopenxr is required. (pip install pyopenxr)")
            sys.exit(1)
        openxr_viewer = OpenXRStereoViewer()
        print("[Mode] OpenXR Direct Submit")
    elif is_dual_monitor:
        # Dual-Monitor 3D Output mode: dxcam captures the INPUT monitor
        # (capture_idx) at full FPS, and the converted stereo frame is
        # displayed full-screen on a *separate* OUTPUT monitor
        # (args.output_monitor). The output window always uses WDA_NONE
        # (capture_visible=True) so it is recognized by virtual-display /
        # remote-streaming software (e.g. Virtual Desktop), and always
        # stretches to fill the output monitor regardless of 3D format
        # (preserve_aspect=False), same fill behavior as PC mode.
        output_idx = int(getattr(args, "output_monitor", 1))
        if not (0 <= output_idx < len(monitors)):
            print(f"[Warning] Output monitor index {output_idx} out of range; falling back to monitor 0.")
            output_idx = 0
        _pc_fast_capture = True  # always dxcam, no 3D-format FPS restriction
        for attempt in range(20):
            _cleanup_leftover_overlays()
            time.sleep(0.15)
            try:
                out_mon = monitors[output_idx]
                overlay = Win32GLOverlay(
                    int(out_mon["width"]), int(out_mon["height"]),
                    left=int(out_mon["left"]), top=int(out_mon["top"]),
                    click_through=not bool(getattr(args, "no_click_through", False)),
                    vr_mode=False,
                    capture_visible=True,  # always WDA_NONE: visible to virtual displays / remote streaming
                    preserve_aspect=False,  # always fill the output monitor fully
                )
                try:
                    overlay._apply_exclude_capture()
                except Exception:
                    pass
                print(
                    f"[Mode] Dual-Monitor 3D Output (dxcam capture=monitor {capture_idx}, "
                    f"output=monitor {output_idx}, WDA_NONE, Virtual Desktop compatible) "
                    f"format={getattr(args, 'format', 'hsbs')} hwnd={overlay.hwnd}",
                    flush=True,
                )
                break
            except Exception as e:
                if attempt < 19:
                    time.sleep(1.0 + attempt * 0.3)
                else:
                    print(f"[Error] Failed to create Win32GLOverlay: {e}")
                    sys.exit(1)
    else:
        # PC mode: Win32GL overlay on capture monitor.
        # anaglyph / half-anaglyph formats use the fast dxcam capture path (like OpenXR):
        #   overlay gets WDA_EXCLUDEFROMCAPTURE (self-excluded from local capture, full FPS,
        #   not visible to other capture/streaming tools).
        # hsbs / fsbs / tb formats keep the original Magnification-API path:
        #   overlay stays WDA_NONE (Virtual-Desktop/OBS visible) and capture is limited to
        #   ~30 FPS by the Magnification API/DWM.
        _pc_fast_capture = str(getattr(args, "format", "hsbs")).lower() in ("anaglyph", "half-anaglyph")
        for attempt in range(20):
            _cleanup_leftover_overlays()
            time.sleep(0.15)
            try:
                out_mon = monitors[capture_idx] if 0 <= capture_idx < len(monitors) else monitors[0]
                overlay = Win32GLOverlay(
                    int(out_mon["width"]), int(out_mon["height"]),
                    left=int(out_mon["left"]), top=int(out_mon["top"]),
                    click_through=not bool(getattr(args, "no_click_through", False)),
                    vr_mode=False,
                    capture_visible=not _pc_fast_capture,  # anaglyph/half-anaglyph -> WDA_EXCLUDEFROMCAPTURE; else WDA_NONE + Mag exclude
                    preserve_aspect=(str(getattr(args, "format", "hsbs")) == "fsbs"),
                )
                # Force affinity again after full window/GL init (style changes can reset it)
                try:
                    overlay._apply_exclude_capture()
                except Exception:
                    pass
                print(
                    (
                        f"[Mode] PC Overlay (dxcam fast path, WDA_EXCLUDEFROMCAPTURE) "
                        if _pc_fast_capture else
                        f"[Mode] PC Overlay (Virtual Desktop compat=ON, WDA_NONE + MagCapture) "
                    ) + f"format={getattr(args, 'format', 'hsbs')} hwnd={overlay.hwnd}",
                    flush=True,
                )
                break
            except Exception as e:
                if attempt < 19:
                    time.sleep(1.0 + attempt * 0.3)
                else:
                    print(f"[Error] Failed to create Win32GLOverlay: {e}")
                    sys.exit(1)
        # System cursor on the PC monitor:
        # - hsbs / fsbs / tb: Do NOT hide here (SetSystemCursor affects the whole
        #   OS, and the user wants the real Windows/system cursor left alone).
        #   The cursor baked into the *captured/converted* output is handled
        #   separately via MagShowSystemCursor in the Magnification capture
        #   worker, which does not touch the real system cursor.
        #   Visibility is controlled in the main loop by --cursor-autohide:
        #     OFF → always show Windows cursor on the monitor
        #     ON  → hide only while video is playing and mouse is idle
        # - anaglyph / half-anaglyph: the overlay is shown directly on the PC
        #   monitor (glasses mode) with the real system cursor on the very
        #   same screen, and the app also draws its own converted cursor into
        #   the frame — so the real Windows cursor must be hidden for the
        #   whole time conversion is running, or the user sees it doubled.
        #   Hide it immediately here and keep it hidden (see main loop);
        #   the finally: block always restores it on stop/crash.
        try:
            if _pc_fast_capture:
                hide_windows_system_cursor()
                # Safety net: guarantees the real cursor comes back even if an
                # exception happens somewhere between here and the main loop's
                # try/finally (e.g. during model/setup code), not just on a
                # clean stop or an error caught inside the main loop.
                atexit.register(restore_windows_system_cursor)
            else:
                restore_windows_system_cursor()
        except Exception as e:
            print(f"[Cursor] cursor state on start warning: {e}", flush=True)


    _low_vram = bool(getattr(args, "low_vram", False))
    if _low_vram:
        _frame_q_size, _result_q_size = 1, 1
    else:
        _frame_q_size, _result_q_size = 2, 4
    frame_q = queue.Queue(maxsize=_frame_q_size)
    result_q = queue.Queue(maxsize=_result_q_size)
    print(f"[Queue] low_vram={_low_vram} frame_q={_frame_q_size} result_q={_result_q_size}", flush=True)
    running = threading.Event()
    running.set()

    shared_divergence = [float(args.divergence)]
    shared_convergence = [float(args.convergence)]
    shared_edge_dilation = list(args.edge_dilation) if isinstance(args.edge_dilation, (list, tuple)) else [args.edge_dilation]
    shared_fps = [0.0]
    shared_show_fps = [bool(getattr(args, "show_fps", False))]
    shared_auto_crop = [bool(getattr(args, "auto_crop", False))]
    shared_auto_mode = [bool(getattr(args, "auto_mode", False))]
    _openxr_letterbox_seq_seen = [0]
    shared_preserve_border = [bool(getattr(args, "preserve_screen_border", True))]
    # Full SBS screen size: 0 = current full-screen size, higher values shrink the
    # displayed Full SBS image (adds letterbox/pillarbox border) in real time.
    shared_fsbs_scale = [max(0.0, float(getattr(args, "fsbs_screen_size", 0.0)))]
    if overlay is not None:
        overlay.fsbs_scale_ref = shared_fsbs_scale


    shared_cursor = [0.5, 0.5, False, 0.0, 0]
    # Track whether we currently have the system cursor hidden (PC monitor).
    # anaglyph/half-anaglyph already hid it right after overlay creation above,
    # so start this flag in sync with that (avoids an immediate spurious
    # restore on the first main-loop iteration).
    _sys_cursor_hidden_state = [bool(mode == "pc" and str(getattr(args, "format", "hsbs")).lower() in ("anaglyph", "half-anaglyph"))]
    _last_cursor_reassert = [time.perf_counter()]
    if openxr_viewer is not None:
        openxr_viewer.shared_auto_crop = shared_auto_crop
        openxr_viewer.shared_cursor = shared_cursor
        openxr_viewer.shared_divergence = shared_divergence
        openxr_viewer.shared_auto_mode = shared_auto_mode
        openxr_viewer.shared_edge_dilation = shared_edge_dilation
        openxr_viewer.args = args
    prev_state = [shared_divergence[0], shared_convergence[0], shared_edge_dilation[0], args.ema_decay]

    display_frame_count = 0
    display_fps_timer = time.perf_counter()
    display_fps_ema = 0.0

    def read_gui_input():
        for line in sys.stdin:
            line = line.strip()
            if line.startswith("SET_PARAMS:"):
                try:
                    parts = line.split(":")
                    shared_divergence[0] = float(parts[1])
                    shared_convergence[0] = float(parts[2])
                    shared_edge_dilation[0] = int(parts[3])
                    if len(parts) >= 5:
                        args.ema_decay = clamp_ema_value(float(parts[4]))
                except Exception:
                    pass
            elif line.startswith("SET_SHOW_FPS:"):
                try:
                    shared_show_fps[0] = (line.split(":")[1] == "1")
                except Exception:
                    pass
            elif line.startswith("SET_AUTO_CROP:"):
                try:
                    shared_auto_crop[0] = (line.split(":")[1] == "1")
                except Exception:
                    pass
            elif line.startswith("SET_AUTO_MODE:"):
                try:
                    shared_auto_mode[0] = (line.split(":")[1] == "1")
                except Exception:
                    pass
            elif line.startswith("SET_PRESERVE_BORDER:"):
                try:
                    shared_preserve_border[0] = (line.split(":")[1] == "1")
                except Exception:
                    pass
            elif line.startswith("SET_VR_PANEL:"):
                try:
                    parts = line.split(":")
                    d = float(parts[1])
                    h = float(parts[2])
                    y = float(parts[3])
                    c = float(parts[4]) if len(parts) >= 5 else None
                    if openxr_viewer is not None:
                        with openxr_viewer._panel_lock:
                            openxr_viewer.panel_distance = max(0.3, min(20.0, d))
                            openxr_viewer.panel_height = max(0.2, min(10.0, h))
                            openxr_viewer.panel_y_offset = max(-5.0, min(5.0, y))
                            if c is not None:
                                lo, hi = openxr_viewer._panel_curve_range
                                openxr_viewer.panel_curve = max(lo, min(hi, c))
                        openxr_viewer._save_panel_config()
                except Exception:
                    pass
            elif line.startswith("SET_FSBS_SIZE:"):
                try:
                    v = float(line.split(":")[1])
                    shared_fsbs_scale[0] = max(0.0, min(50.0, v))
                except Exception:
                    pass
            elif line == "CMD_VR_RECENTER":
                if openxr_viewer is not None:
                    openxr_viewer.recenter_panel()

                    with openxr_viewer._panel_lock:
                        d = openxr_viewer.panel_distance
                        h = openxr_viewer.panel_height
                        y = openxr_viewer.panel_y_offset
                        cv = openxr_viewer.panel_curve
                    print(f"SYNC_VR_PANEL:{d:.2f}:{h:.2f}:{y:.2f}:{cv:.1f}", flush=True)
            elif line == "CMD_VR_RESET":
                if openxr_viewer is not None:
                    openxr_viewer.reset_panel()
                    with openxr_viewer._panel_lock:
                        d = openxr_viewer.panel_distance
                        h = openxr_viewer.panel_height
                        y = openxr_viewer.panel_y_offset
                        cv = openxr_viewer.panel_curve
                    print(f"SYNC_VR_PANEL:{d:.2f}:{h:.2f}:{y:.2f}:{cv:.1f}", flush=True)
            elif line == "CMD_VR_HEIGHT_INC":
                if openxr_viewer is not None:
                    openxr_viewer.adjust_panel_height_offset(args.vr_height_step)
                    with openxr_viewer._panel_lock:
                        d = openxr_viewer.panel_distance
                        h = openxr_viewer.panel_height
                        y = openxr_viewer.panel_y_offset
                        cv = openxr_viewer.panel_curve
                    print(f"SYNC_VR_PANEL:{d:.2f}:{h:.2f}:{y:.2f}:{cv:.1f}", flush=True)
            elif line == "CMD_VR_HEIGHT_DEC":
                if openxr_viewer is not None:
                    openxr_viewer.adjust_panel_height_offset(-args.vr_height_step)
                    with openxr_viewer._panel_lock:
                        d = openxr_viewer.panel_distance
                        h = openxr_viewer.panel_height
                        y = openxr_viewer.panel_y_offset
                        cv = openxr_viewer.panel_curve
                    print(f"SYNC_VR_PANEL:{d:.2f}:{h:.2f}:{y:.2f}:{cv:.1f}", flush=True)
            elif line == "CMD_VR_CURVE_INC":
                if openxr_viewer is not None:
                    openxr_viewer.adjust_panel_curve(args.vr_curve_step)
                    with openxr_viewer._panel_lock:
                        d = openxr_viewer.panel_distance
                        h = openxr_viewer.panel_height
                        y = openxr_viewer.panel_y_offset
                        cv = openxr_viewer.panel_curve
                    print(f"SYNC_VR_PANEL:{d:.2f}:{h:.2f}:{y:.2f}:{cv:.1f}", flush=True)
            elif line == "CMD_VR_CURVE_DEC":
                if openxr_viewer is not None:
                    openxr_viewer.adjust_panel_curve(-args.vr_curve_step)
                    with openxr_viewer._panel_lock:
                        d = openxr_viewer.panel_distance
                        h = openxr_viewer.panel_height
                        y = openxr_viewer.panel_y_offset
                        cv = openxr_viewer.panel_curve
                    print(f"SYNC_VR_PANEL:{d:.2f}:{h:.2f}:{y:.2f}:{cv:.1f}", flush=True)
            elif line == "CMD_EXIT":

                print("[Engine] CMD_EXIT received, shutting down...", flush=True)
                running.clear()
    threading.Thread(target=read_gui_input, daemon=True).start()

    _pc_mode = (str(getattr(args, "mode", "openxr")).lower() == "pc")
    _pc_fast_capture = _pc_mode and str(getattr(args, "format", "hsbs")).lower() in ("anaglyph", "half-anaglyph")
    use_mag = (_pc_mode and not _pc_fast_capture and _HAS_MAGNIFICATION)
    exclude_hwnds = []
    if overlay is not None and getattr(overlay, "hwnd", None):
        exclude_hwnds = [int(overlay.hwnd)]
    if use_mag:
        print("[Capture] PC mode (hsbs/fsbs/tb) → Magnification API (overlay HWND excluded, capped ~30 FPS)", flush=True)
        capture_worker = MagCaptureWorker(
            capture_idx, args.target_fps, frame_q, running, exclude_hwnds=exclude_hwnds
        )
    else:
        if _pc_mode and not _pc_fast_capture and not _HAS_MAGNIFICATION:
            print("[Warning] Magnification.dll missing — dxcam fallback (mirror risk)", flush=True)
        elif _pc_fast_capture:
            print("[Capture] PC mode (anaglyph/half-anaglyph) → dxcam (Desktop Duplication, overlay excluded via WDA_EXCLUDEFROMCAPTURE, full FPS)", flush=True)
        capture_worker = CaptureWorker(capture_idx, args.target_fps, frame_q, running)

    inference_worker = InferenceWorker(args, predictor, device, frame_q, result_q, running, shared_divergence, shared_convergence, shared_edge_dilation, use_gl=True, shared_fps=shared_fps,
                                                                        shared_auto_crop=shared_auto_crop, shared_preserve_border=shared_preserve_border,
                                         shared_cursor=shared_cursor,
                                         monitor_left=capture_monitor_left, monitor_top=capture_monitor_top)
    capture_worker.start()
    if isinstance(capture_worker, MagCaptureWorker):
        if not capture_worker.wait_ready(5.0):
            print("[Warning] MagCapture did not become ready in time", flush=True)
        if overlay is not None and getattr(overlay, "hwnd", None):
            capture_worker.set_exclude_hwnds([int(overlay.hwnd)])
    inference_worker.start()

    target = args.target_fps if args.target_fps > 0 else 0
    frame_interval = 1.0 / target if target > 0 else 0.0
    next_frame_time = time.perf_counter()
    exit_held_since = None
    hud_text = ""; hud_expire = 0.0
    next_topmost_check = 0.0
    last_frame = None

    try:
        while running.is_set():
            if overlay is not None:
                overlay.pump()

            if is_hotkey_down(hk_exit, edge=False):
                if exit_held_since is None: exit_held_since = time.perf_counter()
                elif time.perf_counter() - exit_held_since >= args.exit_hold: break
            else: exit_held_since = None

            # --- Input monitor system cursor (Windows cursor) ---
            # PC mode, anaglyph / half-anaglyph: the real Windows cursor lives on
            # the same screen as the overlay's own drawn cursor, so it stays
            # force-hidden for the entire time conversion is running (independent
            # of --cursor-autohide / video-playing / idle detection) to avoid the
            # doubled-cursor look. It is restored on stop/crash by the
            # finally: block below.
            # PC mode, hsbs / fsbs / tb: unchanged — respect --cursor-autohide:
            #   OFF → always show system cursor on the PC monitor
            #   ON  → hide only when video is playing AND mouse has been idle
            # Dual-Monitor 3D Output mode: the input monitor is the user's normal
            # working screen (dxcam captures its real system cursor directly into
            # the raw frame, which our own drawn/composited cursor can't erase),
            # so it is never force-hidden — only --cursor-autohide's "video
            # playing AND idle" condition ever hides it, matching hsbs/fsbs/tb.
            _cur_mode = str(getattr(args, "mode", "")).lower()
            if _cur_mode in ("pc", "dual_monitor"):
                try:
                    # Keep activity timestamp fresh for idle detection
                    if detect_real_mouse_activity():
                        shared_cursor[3] = time.perf_counter()
                        shared_cursor[4] = 1

                    if _cur_mode == "pc" and str(getattr(args, "format", "hsbs")).lower() in ("anaglyph", "half-anaglyph"):
                        want_hide_sys = True
                    else:
                        want_hide_sys = False
                        if bool(getattr(args, "cursor_autohide", False)):
                            idle = (
                                len(shared_cursor) >= 4
                                and shared_cursor[3] > 0.0
                                and (time.perf_counter() - shared_cursor[3]) >= _CURSOR_HIDE_SECONDS
                            )
                            want_hide_sys = bool(is_video_playing() and idle)

                    # Debounce: only call hide/restore when state changes.
                    # For anaglyph/half-anaglyph, also periodically re-assert
                    # the hide even when state hasn't "changed" — clicking into
                    # another window/dialog can occasionally make Windows
                    # reload its default cursors out from under us.
                    if want_hide_sys != _sys_cursor_hidden_state[0]:
                        if want_hide_sys:
                            hide_windows_system_cursor()
                        else:
                            restore_windows_system_cursor()
                        _sys_cursor_hidden_state[0] = want_hide_sys
                    elif want_hide_sys and (time.perf_counter() - _last_cursor_reassert[0]) >= 2.0:
                        hide_windows_system_cursor()
                        _last_cursor_reassert[0] = time.perf_counter()
                except Exception:
                    pass

            if is_hotkey_down(hk_div_inc, edge=True):
                shared_divergence[0] = min(2.5, shared_divergence[0] + args.div_step)
                if shared_auto_mode[0]:
                    _auto_edge, _auto_ema = compute_auto_edge_ema(shared_divergence[0])
                    shared_edge_dilation[0] = _auto_edge
                    args.ema_decay = _auto_ema
            if is_hotkey_down(hk_div_dec, edge=True):
                shared_divergence[0] = max(0.5, shared_divergence[0] - args.div_step)
                if shared_auto_mode[0]:
                    _auto_edge, _auto_ema = compute_auto_edge_ema(shared_divergence[0])
                    shared_edge_dilation[0] = _auto_edge
                    args.ema_decay = _auto_ema
            if is_hotkey_down(hk_conv_inc, edge=True): shared_convergence[0] = min(1.0, shared_convergence[0] + args.conv_step)
            if is_hotkey_down(hk_conv_dec, edge=True): shared_convergence[0] = max(0.0, shared_convergence[0] - args.conv_step)

            if not shared_auto_mode[0]:
                if is_hotkey_down(hk_edge_inc, edge=True): shared_edge_dilation[0] = min(5, int(shared_edge_dilation[0]) + args.edge_step)
                if is_hotkey_down(hk_edge_dec, edge=True): shared_edge_dilation[0] = max(0, int(shared_edge_dilation[0]) - args.edge_step)
                if is_hotkey_down(hk_ema_inc, edge=True): args.ema_decay = ema_step_value(args.ema_decay, increase=True)
                if is_hotkey_down(hk_ema_dec, edge=True): args.ema_decay = ema_step_value(args.ema_decay, increase=False)
            if is_hotkey_down(hk_fps_toggle, edge=True):
                shared_show_fps[0] = not shared_show_fps[0]
                print(f"SYNC_SHOW_FPS:{1 if shared_show_fps[0] else 0}", flush=True)

            if is_hotkey_down(hk_fsbs_size_inc, edge=True):
                shared_fsbs_scale[0] = max(0.0, min(50.0, shared_fsbs_scale[0] + args.fsbs_size_step))
                print(f"SYNC_FSBS_SIZE:{shared_fsbs_scale[0]:.2f}", flush=True)
                hud_text = f"Full SBS Screen Size {shared_fsbs_scale[0]:.2f}"
                hud_expire = time.perf_counter() + 2.0
            if is_hotkey_down(hk_fsbs_size_dec, edge=True):
                shared_fsbs_scale[0] = max(0.0, min(50.0, shared_fsbs_scale[0] - args.fsbs_size_step))
                print(f"SYNC_FSBS_SIZE:{shared_fsbs_scale[0]:.2f}", flush=True)
                hud_text = f"Full SBS Screen Size {shared_fsbs_scale[0]:.2f}"
                hud_expire = time.perf_counter() + 2.0

            if is_hotkey_down(hk_auto_crop, edge=True):
                shared_auto_crop[0] = not shared_auto_crop[0]
                state = "ON" if shared_auto_crop[0] else "OFF"
                print(f"SYNC_AUTO_CROP:{1 if shared_auto_crop[0] else 0}", flush=True)
                hud_text = f"Letterbox {state}"
                hud_expire = time.perf_counter() + 2.0


            if openxr_viewer is not None:
                vr_changed = False
                recenter_pressed = is_hotkey_down(hk_vr_recenter, edge=True)
                reset_pressed = is_hotkey_down(hk_vr_reset, edge=True)

                if is_hotkey_down(hk_vr_dist_inc, edge=True):
                    openxr_viewer.adjust_panel_distance(args.vr_dist_step)
                    vr_changed = True
                if is_hotkey_down(hk_vr_dist_dec, edge=True):
                    openxr_viewer.adjust_panel_distance(-args.vr_dist_step)
                    vr_changed = True
                if is_hotkey_down(hk_vr_size_inc, edge=True):
                    openxr_viewer.adjust_panel_size(args.vr_size_step)
                    vr_changed = True
                if is_hotkey_down(hk_vr_size_dec, edge=True):
                    openxr_viewer.adjust_panel_size(-args.vr_size_step)
                    vr_changed = True
                if is_hotkey_down(hk_vr_height_inc, edge=True):
                    openxr_viewer.adjust_panel_height_offset(args.vr_height_step)
                    vr_changed = True
                if is_hotkey_down(hk_vr_height_dec, edge=True):
                    openxr_viewer.adjust_panel_height_offset(-args.vr_height_step)
                    vr_changed = True
                if is_hotkey_down(hk_vr_curve_inc, edge=True):
                    openxr_viewer.adjust_panel_curve(args.vr_curve_step)
                    vr_changed = True
                if is_hotkey_down(hk_vr_curve_dec, edge=True):
                    openxr_viewer.adjust_panel_curve(-args.vr_curve_step)
                    vr_changed = True

                if recenter_pressed:
                    openxr_viewer.recenter_panel()
                    hud_text = "VR Screen Recentered"
                    hud_expire = time.perf_counter() + 2.0
                elif reset_pressed:
                    openxr_viewer.reset_panel()
                    hud_text = "VR Screen Reset"
                    hud_expire = time.perf_counter() + 2.0
                elif vr_changed:
                    with openxr_viewer._panel_lock:
                        d = openxr_viewer.panel_distance
                        h = openxr_viewer.panel_height
                        y = openxr_viewer.panel_y_offset
                        cv = openxr_viewer.panel_curve
                    hud_text = f"Dist {d:.2f} | Size {h:.2f} | Height {y:.2f} | Curve {cv:.0f}"
                    hud_expire = time.perf_counter() + 2.0

                if vr_changed or recenter_pressed or reset_pressed:
                    with openxr_viewer._panel_lock:
                        d = openxr_viewer.panel_distance
                        h = openxr_viewer.panel_height
                        y = openxr_viewer.panel_y_offset
                        cv = openxr_viewer.panel_curve
                    print(f"SYNC_VR_PANEL:{d:.2f}:{h:.2f}:{y:.2f}:{cv:.1f}", flush=True)

            curr_state = [shared_divergence[0], shared_convergence[0], shared_edge_dilation[0], args.ema_decay]
            if curr_state != prev_state:
                print(f"SYNC_PARAMS:{curr_state[0]:.1f}:{curr_state[1]:.1f}:{curr_state[2]}:{curr_state[3]:.2f}", flush=True)
                hud_text = f"Div {curr_state[0]:.1f} | Conv {curr_state[1]:.2f} | Edge {curr_state[2]} | EMA {ema_value_to_label(curr_state[3])}"
                hud_expire = time.perf_counter() + 2.0
                prev_state = curr_state


            now = time.perf_counter()
            if frame_interval > 0:
                remaining = next_frame_time - now
                if remaining > 0.002:


                    time.sleep(remaining - 0.0003)


                while time.perf_counter() < next_frame_time:
                    pass

                next_frame_time += frame_interval

                if time.perf_counter() - next_frame_time > frame_interval * 2:
                    next_frame_time = time.perf_counter() + frame_interval

            got_new = False
            try:
                if _low_vram:

                    while True:
                        last_frame = result_q.get_nowait()
                        got_new = True
                else:


                    if result_q.qsize() >= 4:
                        while result_q.qsize() > 1:
                            try:
                                result_q.get_nowait()
                            except queue.Empty:
                                break
                    last_frame = result_q.get_nowait()
                    got_new = True
            except queue.Empty:
                pass

            if last_frame is None:
                time.sleep(0.001)
                continue


            if isinstance(last_frame, tuple) and len(last_frame) == 3 and last_frame[0] == "openxr":
                _, left_frame, right_frame = last_frame


                if openxr_viewer is not None and openxr_viewer._letterbox_notify_seq != _openxr_letterbox_seq_seen[0]:
                    _openxr_letterbox_seq_seen[0] = openxr_viewer._letterbox_notify_seq
                    state = "ON" if shared_auto_crop[0] else "OFF"
                    print(f"SYNC_AUTO_CROP:{1 if shared_auto_crop[0] else 0}", flush=True)
                    hud_text = f"Letterbox {state}"
                    hud_expire = time.perf_counter() + 2.0


                need_draw = (hud_text and time.perf_counter() < hud_expire) or shared_show_fps[0]
                if need_draw:
                    active_hud_text = hud_text if (hud_text and time.perf_counter() < hud_expire) else ""
                    for gpu_frame in (left_frame, right_frame):
                        if isinstance(gpu_frame, torch.Tensor):
                            draw_hud_overlay_gpu(
                                gpu_frame, hud_text=active_hud_text,
                                show_fps=bool(shared_show_fps[0]), fps_value=float(shared_fps[0]),
                            )

                if openxr_viewer is not None:
                    openxr_viewer.submit(left_frame, right_frame)


                if got_new:
                    display_frame_count += 1
                    now_fps = time.perf_counter()
                    if now_fps - display_fps_timer >= 0.5:
                        inst_fps = display_frame_count / (now_fps - display_fps_timer)
                        display_fps_ema = inst_fps if display_fps_ema == 0.0 else display_fps_ema * 0.6 + inst_fps * 0.4
                        shared_fps[0] = display_fps_ema
                        display_frame_count = 0
                        display_fps_timer = now_fps
                continue

            if isinstance(last_frame, torch.Tensor):
                # PC overlay path: keep topmost + HUD + blit
                if overlay is not None:
                    now_topmost = time.perf_counter()
                    if now_topmost >= next_topmost_check:
                        overlay.keep_topmost()
                        next_topmost_check = now_topmost + 0.5

                need_draw = (hud_text and time.perf_counter() < hud_expire) or shared_show_fps[0]
                if need_draw:
                    active_hud_text = hud_text if (hud_text and time.perf_counter() < hud_expire) else ""
                    draw_hud_overlay_gpu(
                        last_frame, hud_text=active_hud_text,
                        show_fps=bool(shared_show_fps[0]), fps_value=float(shared_fps[0]),
                    )

                if overlay is not None:
                    overlay.blit_tensor(last_frame)

                if got_new:
                    display_frame_count += 1
                    now_fps = time.perf_counter()
                    if now_fps - display_fps_timer >= 0.5:
                        inst_fps = display_frame_count / (now_fps - display_fps_timer)
                        display_fps_ema = inst_fps if display_fps_ema == 0.0 else display_fps_ema * 0.6 + inst_fps * 0.4
                        shared_fps[0] = display_fps_ema
                        display_frame_count = 0
                        display_fps_timer = now_fps

    except KeyboardInterrupt: pass
    finally:
        running.clear()
        capture_worker.join(timeout=1.5)
        inference_worker.join(timeout=1.5)
        if openxr_viewer is not None:
            openxr_viewer.destroy()
        if overlay is not None:
            try:
                overlay.destroy()
            except Exception:
                pass
        try:
            restore_windows_system_cursor()
        except Exception:
            pass

        if original_display_mode is not None:
            try:
                print(f"[Dynamic Resolution] Restoring {original_display_mode['width']}x{original_display_mode['height']}...", flush=True)
                ok = restore_display_mode(original_display_mode)
                if ok:
                    print("[Dynamic Resolution] Original resolution restored.", flush=True)
                else:
                    print("[Warning] restore_display_mode returned False", flush=True)
            except Exception as e:
                print(f"[Warning] Failed to restore display resolution: {e}", flush=True)

        try:
            winmm.timeEndPeriod(1)
        except Exception:
            pass

if __name__ == "__main__":
    main()
