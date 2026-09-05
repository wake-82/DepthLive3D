# DepthLive3D

![Icon](https://raw.githubusercontent.com/wake-82/DepthLive3D/refs/heads/main/icon.ico)

## What is DepthLive3D?
![preview](./preview.png)
DepthLive3D is a free program that uses AI-based depth mapping and OpenXR technology to convert your live PC screen into 3D in real time within a VR environment, alongside offline 2D-to-3D video file conversion.

## Requirements

- **OS:** Windows 10/11 (64-bit)
- **GPU:** NVIDIA GPU with CUDA support
  - GTX 10xx series (Pascal) through RTX 40xx series → CUDA 12.6
  - RTX 50xx series (Blackwell) or newer → CUDA 12.8
  - The installer detects your GPU automatically, so you don't need to pick a CUDA version yourself.
- **NVIDIA Driver:** Latest driver recommended (required for CUDA 12.6 / 12.8 support)
- **Python:** 3.12
- **Git:** Latest version 
- **Disk space:** ~10 GB free (for PyTorch, models, and dependencies)
- **VR headset (optional):** OpenXR-compatible headset, if using VR output
  
---

## DepthLive3D Windows 10 & 11 Quick Installation Guide

1. Install the Microsoft Visual C++ Redistributable x64 package first (vc_redist.x64.exe):
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

2. Download the DepthLive3D installation files from the Releases page and extract them to C:\DepthLive3D. 
https://github.com/wake-82/DepthLive3D/releases/tag/VR

3. Run `install.bat`. 

4. Once the installation is complete, run the DepthLive3D-Run.bat file.

5. Select Converted 3D to convert video files, or Live 3D to convert your computer screen to 3D in real time.


---

## Development Environment, Installation, and Usage

### 1. Install Python 3.12 and Git

### 2. Create a folder
```
mkdir c:\DepthLive3D
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

### 5. Install the pinned dependency versions

Move back to the base folder:
```
cd c:\DepthLive3D
```
```
pip install -r requirements-lock.txt
```

### 6. Move back to the base folder
```
cd c:\DepthLive3D
```

### 7. Install DepthLive3D
```
git clone https://github.com/wake-82/DepthLive3D.git
move C:\DepthLive3D\ZipDepth C:\DepthLive3D\DepthLive3D\
```

### 8. Install Video-Depth-Anything

Move into the DepthLive3D folder:
```
cd c:\DepthLive3D\DepthLive3D
```

Clone the Video-Depth-Anything repository:
```
git clone https://github.com/DepthAnything/Video-Depth-Anything.git
```

Create the checkpoints folder and download the VDA-S checkpoint:
```
mkdir Video-Depth-Anything\checkpoints
curl -L -o Video-Depth-Anything\checkpoints\video_depth_anything_vits.pth "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true"
```

Also download the Metric VDA-S checkpoint:
```
curl -L -o Video-Depth-Anything\checkpoints\metric_video_depth_anything_vits.pth "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth?download=true"
```

Move back to the base folder:
```
cd c:\DepthLive3D
```

### 9. Install PyTorch (CUDA)

The exact pinned/tested combination is **torch 2.13.0 + torchvision 0.28.0**.

- GTX 10xx through RTX 40xx (Pascal through Ada): use `cu126` — this is the combination validated for correct performance.
- RTX 50xx (Blackwell) or newer: use `cu128`. As of this writing, `cu126` cannot run on Blackwell GPUs, but the `cu128` channel does not yet offer a torch 2.13.0 build — install the latest version available on that channel instead and expect it to differ slightly from the cu126 reference build.

CPU version uninstall:
```
pip uninstall torch torchvision torchaudio -y
```

Run only the one line that matches your graphics card:
```
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Check whether the GPU version was installed correctly (if the CPU version was installed instead, reinstall):
```
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Example output (cu126):
```
2.13.0+cu126
True
```

### 10. Run the program
```
cd c:\DepthLive3D\DepthLive3D
python DepthLive3D.py
```

If you installed it inside a virtual environment, run the following instead:
```
cd c:\DepthLive3D\
venv\Scripts\activate
cd c:\DepthLive3D\DepthLive3D
python DepthLive3D.py
```

### 11. Start the conversion
Select Converted 3D to convert video files, or Live 3D to convert your computer screen to 3D in real time.

---

## Live 3D Options

- **Mode** — Selects between OpenXR, PC, and Dual-Monitor 3D Output modes.<br> Choose OpenXR for VR devices, and PC or Dual-Monitor 3D Output for 3D TV, AR glasses, or standard monitors.<br> Note that Dual-Monitor 3D Output mode requires an external monitor or a virtual monitor.
- **Process Size** — Sets the capture resolution. 720p or 1080p is recommended.
- **Target FPS** — Sets the capture frame rate. You can choose between 30 and 60. If the frame rate is unstable, try switching to 30.
- **Input Monitor** — Selects which monitor to capture. Set to `0` for the main monitor.
- **Output Monitor** — Enabled in Dual-Monitor 3D Output mode.<br> Select the monitor to capture in Input Monitor, and select the monitor to display the 3D conversion output in Output Monitor.
- **Dynamic Resolution Auto Adjust** — Automatically adjusts the resolution dynamically based on the Process Size setting. Enabling this skips resizing and improves frame rate.
- **CPU Performance** — Selects how many CPU cores to use. Setting this to "High" does not always improve performance — try each mode to find the best setting for your system.
- **Low VRAM Mode** — Enable this option if the VR screen freezes or the background flickers due to insufficient GPU VRAM. Note that enabling this will reduce the frame rate. 
<br>Additionally, since this mode uses only the most recent frame, it offers the fastest screen response time. Therefore, if game responsiveness is important to you, enabling this mode can be advantageous. 
- **Auto Mode** — When enabled, Edge Fix and Flicker Reduction are automatically set to optimal values based on the 3D Strength value.
- **3D Format** — Enabled in PC mode. Select the 3D output format that matches your device.
- **3D Strength** — Adjusts the strength of the 3D effect. Higher strength increases artifacts and depth map flickering. You can adjust this in real time using the `[` and `]` keys.
- **Convergence** — Adjusts how much the screen appears to pop out or recede. At `0.0`, the background is flat and the foreground pops out. At `1.0`, the background appears deeper and the foreground recedes into the screen. `0.5` is a balanced midpoint between the two.
- **Edge Fix** — Expands the edges of foreground objects. As 3D Strength increases, foreground shapes may distort — Edge Fix helps correct this.
- **Flicker Reduction** — Smooths out depth map flickering by blending frames. Higher values reduce flickering but can introduce ghosting/afterimages, which may cause eye or brain fatigue.
- **Preserve Screen Border** — When enabled, protects the edges of the screen. Recommended when using a high 3D Strength value.
- **Depth Models** - You can choose between Zipdepth and VDA.<br> Zipdepth is lightweight and fast, while VDA is heavy but offers superior depth consistency.
- **Depth Resolution** — Adjusts the resolution of the depth map. Higher resolution improves 3D quality but reduces frame rate.
- **FPS Display** — Displays the current average conversion FPS on screen.
- **Full SBS Screen Size** — Designed for AR glass users. Adjusts the screen size when outputting in Full SBS (fsbs).
- **VR Screen Options** — Configure screen size, height, distance, center position, background and screen reset.
- **Keyboard Hotkeys** — Assign shortcut keys for each option.
- **Letterbox Remove Toggle** — Automatically detects and removes letterboxing. This removes artifacts that can appear above and below the letterbox bars in letterboxed videos. Quest users can toggle this via controller input; other headsets must assign a keyboard shortcut. Make sure to turn this OFF when you're done, to prevent malfunctions.
- **Mouse Cursor** — Adjust the size and color of the mouse cursor used for controller input.
  - **Auto Hide** — When enabled, the mouse cursor is automatically hidden during video playback.
- **Reset Settings** — Restores all settings to their default values.
- **Start 3D Conversion / Stop** — Starts or stops the program. You can also hold the `ESC` key for 2 seconds to exit the program.

> Any option that does not prompt a "restart required" notice can be adjusted in real time while the program is running.

---

## Conversion 3D Options

- **Input File** — Select a video file. Selecting a folder will sequentially convert all videos inside it.
- **Depthmap Input File** — Select both the original video and a depthmap video to skip depthmap generation and start conversion immediately.
- **Output Folder** — Select the destination folder where converted files will be saved.
- **Preset** — Save and load your custom 3D settings.
- **Output Format** — Select the 3D output format.
- **3D Options** — Please refer to the Live 3D options documentation.
- **Extract Raw Depthmap** — When checked, exports and saves the raw depthmap video alongside the output file.
- **Extract Corrected Depthmap** - You can extract depth map videos with edge fix and flicker reduction applied.<br> Use this for debugging to inspect the results.
- **Screen Border Protection** — Please refer to the Live 3D options documentation.
- **Use FP16** — When checked, processes using FP16; when unchecked, generates depthmaps using FP32.<br> FP16 offers faster processing speeds, whereas FP32 provides slightly higher precision.
- **Auto Mode** — When checked, automatically applies optimized parameters based on the 3D depth strength.
- **Video Codec** — Selects the video encoding codec.<br> libx uses CPU processing, while nvenc utilizes NVIDIA GPU acceleration for faster conversion speeds.
- **Resize Resolution** — Changes the resolution of the video prior to conversion.
- **MKV HDR Normalize** — When checked, re-encodes HDR video into a standard video format before starting the conversion process.
- **Auto Letterbox Crop** — Automatically detects and crops top and bottom letterboxes.<br> Note that detection may fail if text or subtitles overlap the letterbox area.
- **Pad to 16:9** — Pads non-16:9 videos with letterboxes to conform to a 16:9 aspect ratio.<br> Recommended for use in conjunction with Auto Letterbox Crop.
- **Start Time, End Time** — When enabled, allows specifying a custom time frame to convert a specific clip of the video.
- **Start / Stop Buttons** — Starts or halts the video conversion process.

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

**Q: The letterbox removal feature isn't working.**<br> 
A: If there is text inside the letterbox or the resolution is not the standard 16:9 aspect ratio, the letterbox may not be recognized.

**Q: I only see a black screen on streaming sites like Netflix or Disney+.**<br>
A: Screen capture is blocked due to DRM protection policies.<br> Try disabling hardware acceleration in your web browser settings.

**Q: The game screen is cropped or displaying incorrectly.**<br>
A: Because this app relies on screen capture, full-screen mode may not work properly.<br> Try changing the game's display settings to 'Windowed' or 'Borderless Windowed' mode.

**Q: My mouse cursor looks weird, or it disappears when I hover over something clickable.**<br>
A: Go to 'Mouse pointer style and color' in Windows Settings and reset it to the default cursor.<br> It should display normally now.

**Q: The mouse cursor isn't moving.**<br>
A: If you are running a program that requires administrator privileges, the controller's mouse cursor will not work.<br> Please use your computer mouse to close the program.

**Q: Target FPS is capped at 30.**<br> 
A: In PC mode, SBS and TB settings only support up to 30 FPS due to limitations in the capture method.

**Q: An error occurs as soon as conversion starts when the NVENC codec is selected.**<br>
A: Driver compatibility for FFmpeg's NVENC codec varies depending on the build version. Please update to the latest NVIDIA graphics driver.

**Q: Anaglyph output in PC mode does not work when using Virtual Desktop.**<br>
A: Anaglyph output in PC mode does not support 3D rendering in virtual display environments like Virtual Desktop. However, by switching to Output mode, you can display the anaglyph screen in Virtual Desktop.

**Q: How do I use the Dual-Monitor 3D Output mode?**<br>
A: This mode requires two monitors in total: one for capture and one for output.<br>
Setup Preparation: If you lack physical monitors, add a second display using a virtual display driver, an HDMI dummy plug, or Virtual Desktop's virtual display feature.<br>
Configuration: Set your Windows display settings to Extend, then select different monitors for input and output in the program. (e.g., Input: Monitor 0 / Output: Monitor 1)<br>
Important: Do not assign the same monitor to both input and output.

---

## Credits (Acknowledgements)

This project uses source code from [IW3](https://github.com/nagadomi/nunif/) (MIT License).<br>
This project uses source code from [ZipDepth](https://github.com/fabiotosi92/ZipDepth) (MIT License).<br>
This project uses source code from [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) (Apache 2.0 License, Copyright 2025 ByteDance).<br>

See `THIRD_PARTY_LICENSES/` for the full license texts of the above.

Built with the help of:
- [PyTorch](https://pytorch.org/) — BSD-style license
- [PySide6](https://doc.qt.io/qtforpython/) — LGPLv3
- [dxcam](https://github.com/ra1nty/dxcam) — MIT License
- [OpenCV (opencv-python)](https://opencv.org/) — Apache 2.0 License, Copyright 2026 OpenCV team
- [NumPy](https://numpy.org/) — BSD License
- [PyOpenGL](http://pyopengl.sourceforge.net/) — BSD License
- [pyopenxr](https://github.com/cmbruns/pyopenxr) — Apache 2.0 License, Copyright 2021 Christopher Bruns
- [glfw](https://www.glfw.org/) — zlib/libpng License
- [pywin32](https://github.com/mhammond/pywin32) — PSF License
- [psutil](https://github.com/giampaolo/psutil) — BSD-3-Clause License
- [FFmpeg](https://ffmpeg.org/) — invoked as an external executable;<br>
  builds fetched from https://github.com/BtbN/FFmpeg-Builds are licensed GPL v3

Full license texts for each third-party library are included in the THIRD_PARTY_LICENSES/ folder.
