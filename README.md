# DepthLive3D

![Icon](https://raw.githubusercontent.com/wake-82/DepthLive3D/refs/heads/main/icon.ico)

## What is DepthLive3D?

DepthLive3D is a free program that uses AI-based depth mapping and OpenXR technology to convert PC screen into 3D screen in real time within a VR environment.

## Requirements

- **OS:** Windows 10/11 (64-bit)
- **GPU:** NVIDIA GPU with CUDA support
  - GTX 10xx series (Pascal) or older → CUDA 12.6
  - RTX 20xx / 30xx / 40xx / 50xx series → CUDA 12.8
- **NVIDIA Driver:** Latest driver recommended (required for CUDA 12.6 / 12.8 support)
- **Python:** 3.12
- **Git:** Latest version 
- **Disk space:** ~10 GB free (for PyTorch, models, and dependencies)
- **VR headset (optional):** OpenXR-compatible headset, if using VR output
  
---

## DepthLive3D Windows 10 & 11 Quick Installation Guide

1. Install the Microsoft Visual C++ Redistributable x64 package first (vc_redist.x64.exe):
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

2. Download the DepthLive 3D installation CMD script from the Releases page and extract it to C:\DepthLive3D.
https://github.com/wake-82/DepthLive3D/releases/tag/VR

3. Run old-gpu-install.bat if you are using an NVIDIA GTX 10-series GPU, or new-gpu-install.bat if you have an RTX 20-series GPU or higher.

4. Once the installation is complete, run the DepthLive3D-Run.bat file.

5. Launch an OpenXR-supported PC program (such as Virtual Desktop or SteamVR) on your VR headset and click 'Start 3D Conversion' to view your 3D-converted PC screen on a virtual VR display.

---

## Development Environment, Installation, and Usage

### 1. Install Python 3.12 and Git

### 2. Create a folder
```
mkdir c:\DepthLive3D
mkdir c:\DepthLive3D\DepthLive3D
```

### 3. Move into the folder
```
cd c:\DepthLive3D
```

*(Optional)* If you want to install inside a virtual environment, run the following commands in order to activate it:
```
python -m venv venv
venv\Scripts\activate
```

### 4. Install ZipDepth
```
cd c:\DepthLive3D\DepthLive3D
git clone https://github.com/fabiotosi92/ZipDepth.git
```

Move into the ZipDepth folder:
```
cd ZipDepth
```

Install the ZipDepth library:
```
pip install -r requirements.txt
pip install -e .
```

### 5. Install the DepthLive3D libraries
```
pip install PySide6 pywin32 dxcam PyOpenGL PyOpenGL_accelerate pyopenxr glfw psutil
```

### 6. Move back to the base folder
```
cd c:\DepthLive3D
```

### 7. Install DepthLive3D
```
git clone https://github.com/wake-82/DepthLive3D.git
```

### 8. Install PyTorch (CUDA)

- Older GPUs (e.g., GTX 1000 series, Pascal architecture): `cu126`
- Newer GPUs (e.g., RTX 20/30/40/50 series): `cu128`

Run only the one line that matches your graphics card:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Check whether the GPU version was installed correctly (if the CPU version was installed instead, reinstall):
```
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Example output:
```
2.11.0+cu128
True
```

### 9. Run the program
```
cd c:\DepthLive3D\DepthLive3D
python DepthLive3D.py
```

If you installed it inside a virtual environment, run the following instead:
```
cd c:\DepthLive3D\DepthLive3D
venv\Scripts\activate
python DepthLive3D.py
```

### 10. Start the conversion

Switch to OpenXR mode in Virtual Desktop, SteamVR, or a similar application, then click the "Start 3D Conversion" button in DepthLive3D. The program will capture your computer screen and output a real-time 3D-converted image to the virtual screen.

---

## Options

- **Process Size** — Sets the capture resolution. 720p or 1080p is recommended.
- **Target FPS** — Sets the capture frame rate. You can choose between 30 and 60. If the frame rate is unstable, try switching to 30.
- **Input Monitor** — Selects which monitor to capture. Set to `0` for the main monitor.
- **Dynamic Resolution Auto Adjust** — Automatically adjusts the resolution dynamically based on the Process Size setting. Enabling this skips resizing and improves frame rate.
- **CPU Performance** — Selects how many CPU cores to use. Setting this to "High" does not always improve performance — try each mode to find the best setting for your system.
- **Low VRAM Mode** — Enable this option if the VR screen freezes or the background flickers due to insufficient GPU VRAM. Note that enabling this will reduce the frame rate. 
<br>Additionally, since this mode uses only the most recent frame, it offers the fastest screen response time. Therefore, if game responsiveness is important to you, enabling this mode can be advantageous. 
- **Auto Mode** — When enabled, Edge Fix and Flicker Reduction are automatically set to optimal values based on the 3D Strength value.
- **3D Strength** — Adjusts the strength of the 3D effect. Higher strength increases artifacts and depth map flickering. You can adjust this in real time using the `[` and `]` keys.
- **Convergence** — Adjusts how much the screen appears to pop out or recede. At `0.0`, the background is flat and the foreground pops out. At `1.0`, the background appears deeper and the foreground recedes into the screen. `0.5` is a balanced midpoint between the two.
- **Edge Fix** — Expands the edges of foreground objects. As 3D Strength increases, foreground shapes may distort — Edge Fix helps correct this.
- **Flicker Reduction** — Smooths out depth map flickering by blending frames. Higher values reduce flickering but can introduce ghosting/afterimages, which may cause eye or brain fatigue.
- **Preserve Screen Border** — When enabled, protects the edges of the screen. Recommended when using a high 3D Strength value.
- **Depth Resolution** — Adjusts the resolution of the depth map. Higher resolution improves 3D quality but reduces frame rate.
- **FPS Display** — Displays the current average conversion FPS on screen.
- **VR Screen Options** — Configure screen size, height, distance, center position, and screen reset.
- **Keyboard Hotkeys** — Assign shortcut keys for each option.
- **Letterbox Remove Toggle** — Automatically detects and removes letterboxing. This removes artifacts that can appear above and below the letterbox bars in letterboxed videos. Quest users can toggle this via controller input; other headsets must assign a keyboard shortcut. Make sure to turn this OFF when you're done, to prevent malfunctions.
- **Mouse Cursor** — Adjust the size and color of the mouse cursor used for controller input.
  - **Auto Hide** — When enabled, the mouse cursor is automatically hidden during video playback.
- **Reset Settings** — Restores all settings to their default values.
- **Start 3D Conversion / Stop** — Starts or stops the program. You can also hold the `ESC` key for 2 seconds to exit the program.

> Any option that does not prompt a "restart required" notice can be adjusted in real time while the program is running.

---

## Controller Guide (Meta Quest series controllers)

| Input | Function |
|---|---|
| Analog stick (up/down) | Mouse scroll |
| Analog stick (left/right) | Adjust 3D Strength |
| Analog stick button | Recenter view |
| Trigger button | Left mouse click |
| Grip button | Hold the grip button and move the controller left/right to adjust screen size, or up/down to adjust screen distance. |
| A / X button | Right mouse click |
| B / Y button | Toggle Letterbox Remove ON/OFF |

---

## QnA

Q: The letterbox removal feature isn't working.<br>
A: If there is text inside the letterbox or the resolution is not the standard 16:9 aspect ratio, the letterbox may not be recognized.

Q: I only see a black screen on streaming sites like Netflix or Disney+.<br>
A: Screen capture is blocked due to DRM protection policies.<br> Try disabling hardware acceleration in your web browser settings.

Q: The game screen is cropped or displaying incorrectly.<br>
A: Because this app relies on screen capture, full-screen mode may not work properly.<br> Try changing the game's display settings to 'Windowed' or 'Borderless Windowed' mode.

Q: My mouse cursor looks weird, or it disappears when I hover over something clickable.<br>
A: Go to 'Mouse pointer style and color' in Windows Settings and reset it to the default cursor.<br> It should display normally now.

Q: The mouse cursor isn't moving.<br>
A: If you are running a program that requires administrator privileges, the controller's mouse cursor will not work.<br> Please use your computer mouse to close the program.

---

## Credits (Acknowledgements)

This project uses source code from [IW3](https://github.com/nagadomi/nunif/) (MIT License).
This project uses source code from [ZipDepth](https://github.com/fabiotosi92/ZipDepth) (MIT License).

See `THIRD_PARTY_LICENSES/` for the full license texts of the above.

Built with the help of:
- [PyTorch](https://pytorch.org/) — BSD-style license
- [PySide6](https://doc.qt.io/qtforpython/) — LGPLv3
- [dxcam](https://github.com/ra1nty/dxcam) — MIT License
- [OpenCV (opencv-python)](https://opencv.org/) — Apache 2.0 License
- [NumPy](https://numpy.org/) — BSD License
- [PyOpenGL](http://pyopengl.sourceforge.net/) — BSD License
- [pyopenxr](https://github.com/cmbruns/pyopenxr) — Apache 2.0 License, Copyright 2021 Christopher Bruns
- [glfw](https://www.glfw.org/) — zlib/libpng License
- [pywin32](https://github.com/mhammond/pywin32) — PSF License
- [psutil](https://github.com/giampaolo/psutil) — BSD-3-Clause License

Full license texts for each third-party library are included in the THIRD_PARTY_LICENSES/ folder.
