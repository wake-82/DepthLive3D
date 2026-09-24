# DepthLive3D

![Icon](https://raw.githubusercontent.com/wake-82/DepthLive3D/refs/heads/main/icon.ico)

## DepthLive3D란?
![preview](./preview.png)
DepthLive3D는 AI 기반 깊이 매핑과 OpenXR 기술을 활용하여 PC 화면을 VR 환경 내에서 실시간으로 3D로 변환하고, 오프라인 2D-to-3D 비디오 파일 변환도 지원하는 무료 프로그램입니다.

## 3D 변환 샘플 영상
[Big Buck Bunny](https://youtu.be/GoZfcv_eMXA)
- 2D Source 720p, VDA-S Stream, Depth Resolution 630, Stereo Strength 2.0, Convergence 0.5, Auto Mode, Fast Inpaint]

## 업데이트 내역
- **v1.0:** 최초 배포
- **v2.0:**  VDA 모델 추가
- **v3.0:** 인페인트 모드 및 Depth AA 모드 추가

## 요구 사항

- **OS:** Windows 10/11 (64비트)
- **GPU:** CUDA를 지원하는 NVIDIA GPU
  - GTX 10xx 시리즈(Pascal) ~ RTX 40xx 시리즈 → CUDA 12.6
  - RTX 50xx 시리즈(Blackwell) 이상 → CUDA 12.8
  - 설치 프로그램이 GPU를 자동으로 감지하므로 CUDA 버전을 직접 선택할 필요가 없습니다.
- **NVIDIA 드라이버:** 최신 드라이버 권장 (CUDA 12.6 / 12.8 지원에 필요)
- **Python:** 3.12
- **Git:** 최신 버전 
- **디스크 공간:** 여유 공간 약 10 GB 이상 (PyTorch, 모델, 의존성 패키지 포함)
- **VR 헤드셋 (선택 사항):** VR 출력을 사용하는 경우 OpenXR 호환 헤드셋
  
---

## DepthLive3D Windows 10 & 11 빠른 설치 가이드

1. 먼저 Microsoft Visual C++ 재배포 가능 패키지 x64를 설치합니다 (vc_redist.x64.exe):
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

2. Releases 페이지에서 DepthLive3D 설치 파일을 다운로드한 후 C:\DepthLive3D에 압축을 해제합니다. 
https://github.com/wake-82/DepthLive3D/releases/tag/VR

3. `install.bat`을 실행합니다. 

4. 설치가 완료되면 DepthLive3D-Run.bat 파일을 실행합니다.

5. 비디오 파일을 변환하려면 Converted 3D를, 컴퓨터 화면을 실시간으로 3D로 변환하려면 Live 3D를 선택합니다.


---

## 개발 환경, 설치 및 사용법

### 1. Python 3.12 및 Git 설치

### 2. 폴더 생성
```
mkdir c:\DepthLive3D
```

### 3. 폴더로 이동
```
cd c:\DepthLive3D
```

*(선택 사항)* 가상 환경 내에 설치하려면 다음 명령어를 순서대로 실행하여 활성화합니다:
```
python -m venv venv
venv\Scripts\activate
```

### 4. ZipDepth 설치
```
git clone https://github.com/fabiotosi92/ZipDepth.git
```

ZipDepth 폴더로 이동:
```
cd ZipDepth
```

ZipDepth 라이브러리 설치:
```
pip install -r requirements.txt
pip install -e .
```

### 5. 기본 폴더로 이동
```
cd c:\DepthLive3D
```

### 6. DepthLive3D 설치
```
git clone https://github.com/wake-82/DepthLive3D.git
move C:\DepthLive3D\ZipDepth C:\DepthLive3D\DepthLive3D\
```
### 7. 고정된 의존성 버전 설치

```
pip install -r requirements-lock.txt
```

### 8. Video-Depth-Anything 설치

DepthLive3D 폴더로 이동:
```
cd c:\DepthLive3D\DepthLive3D
```

Video-Depth-Anything 저장소 클론:
```
git clone https://github.com/DepthAnything/Video-Depth-Anything.git
```

checkpoints 폴더를 생성하고 VDA-S 체크포인트를 다운로드:
```
mkdir Video-Depth-Anything\checkpoints
curl -L -o Video-Depth-Anything\checkpoints\video_depth_anything_vits.pth "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true"
```

Metric VDA-S 체크포인트도 다운로드:
```
curl -L -o Video-Depth-Anything\checkpoints\metric_video_depth_anything_vits.pth "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth?download=true"
```

기본 폴더로 이동:
```
cd c:\DepthLive3D
```

### 9. PyTorch (CUDA) 설치

검증된 조합은 **torch 2.13.0 + torchvision 0.28.0**입니다.

- GTX 10xx ~ RTX 40xx (Pascal ~ Ada): `cu126` 사용 — 올바른 성능이 검증된 조합입니다.
- RTX 50xx (Blackwell) 이상: `cu128` 사용. 현재 `cu126`은 Blackwell GPU에서 실행되지 않으며, `cu128` 채널에는 아직 torch 2.13.0 빌드가 제공되지 않으므로 해당 채널에서 이용 가능한 최신 버전을 설치하고 cu126 기준 빌드와 다소 차이가 있을 수 있음을 감안하세요.

CPU 버전 제거:
```
pip uninstall torch torchvision torchaudio -y
```

본인의 그래픽 카드에 맞는 한 줄만 실행하세요:
```
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

GPU 버전이 올바르게 설치되었는지 확인 (CPU 버전이 설치된 경우 재설치 필요):
```
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

출력 예시 (cu126):
```
2.13.0+cu126
True
```

### 10. 프로그램 실행
```
cd c:\DepthLive3D\DepthLive3D
python DepthLive3D.py
```

가상 환경에 설치한 경우 다음 명령어를 실행하세요:
```
cd c:\DepthLive3D\
venv\Scripts\activate
cd c:\DepthLive3D\DepthLive3D
python DepthLive3D.py
```

### 11. 변환 시작
비디오 파일을 변환하려면 Converted 3D를, 컴퓨터 화면을 실시간으로 3D로 변환하려면 Live 3D를 선택합니다.

---

## Live 3D 옵션

- **Mode** — OpenXR, PC, 듀얼 모니터 3D 출력 모드 중 선택합니다.<br> VR 기기는 OpenXR을, 3D TV, AR 글래스, 일반 모니터는 PC 또는 듀얼 모니터 3D 출력 모드를 선택합니다.<br> 듀얼 모니터 3D 출력 모드는 외부 모니터 또는 가상 모니터가 필요합니다.
- **Process Size** — 캡처 해상도를 설정합니다. 720p 또는 1080p를 권장합니다.
- **Target FPS** — 캡처 프레임 레이트를 설정합니다. 30 또는 60 중 선택할 수 있습니다. 프레임 레이트가 불안정한 경우 30으로 전환해 보세요.
- **Input Monitor** — 캡처할 모니터를 선택합니다. 주 모니터는 `0`으로 설정합니다.
- **Output Monitor** — 듀얼 모니터 3D 출력 모드에서 활성화됩니다.<br> Input Monitor에서 캡처할 모니터를, Output Monitor에서 3D 변환 결과를 출력할 모니터를 선택합니다.
- **Dynamic Resolution Auto Adjust** — Process Size 설정에 따라 해상도를 동적으로 자동 조정합니다. 활성화 시 리사이징을 건너뛰어 프레임 레이트가 향상됩니다.
- **CPU Performance** — 사용할 CPU 코어 수를 선택합니다. "High"로 설정한다고 해서 항상 성능이 향상되지는 않으므로 각 모드를 직접 시험해 최적의 설정을 찾으세요.
- **Low VRAM Mode** — GPU VRAM 부족으로 VR 화면이 멈추거나 배경이 깜빡이는 경우 이 옵션을 활성화하세요. 단, 활성화하면 프레임 레이트가 감소합니다. 
<br>또한 이 모드는 가장 최근 프레임만 사용하므로 화면 응답 속도가 가장 빠릅니다. 게임 반응 속도가 중요한 경우 이 모드를 활성화하면 유리할 수 있습니다. 
- **Auto Mode** — 활성화하면 3D Strength 값을 기준으로 Edge Fix와 Flicker Reduction이 자동으로 최적값으로 설정됩니다.
- **3D Format** — PC 모드에서 활성화됩니다. 기기에 맞는 3D 출력 형식을 선택합니다.
- **3D Strength** — 3D 효과의 강도를 조정합니다. 강도가 높을수록 아티팩트와 깊이 맵 깜빡임이 증가합니다. `[` 및 `]` 키로 실시간 조정이 가능합니다.
- **Convergence** — 화면이 튀어나오거나 들어가 보이는 정도를 조정합니다. `0.0`에서는 배경이 평평하고 전경이 튀어나옵니다. `1.0`에서는 배경이 더 깊어 보이고 전경이 화면 안으로 들어갑니다. `0.5`는 두 효과의 균형 중간점입니다.
- **Edge Fix** — 전경 오브젝트의 가장자리를 확장합니다. 3D Strength가 높아질수록 전경 형태가 왜곡될 수 있으며, Edge Fix가 이를 보정합니다.
- **Flicker Reduction** — 프레임을 블렌딩하여 깊이 맵 깜빡임을 완화합니다. 값이 높을수록 깜빡임은 줄어들지만 잔상(고스팅)이 발생할 수 있으며, 눈과 뇌 피로를 유발할 수 있습니다.
- **Preserve Screen Border** — 활성화하면 화면 가장자리를 보호합니다. 3D Strength 값이 높을 때 권장합니다.
- **Depth Models** - Zipdepth와 VDA 중 선택할 수 있습니다.<br> Zipdepth는 가볍고 빠르며, VDA는 무겁지만 깊이 일관성이 뛰어납니다.
- **Depth Resolution** — 깊이 맵의 해상도를 조정합니다. 해상도가 높을수록 3D 품질이 향상되지만 프레임 레이트가 감소합니다.
- **Method** — Normal과 FAST Inpaint 중 선택할 수 있습니다. Normal은 빠른 변환을 제공하며, FAST Inpaint는 느리지만 3D 비디오 품질을 향상시킵니다.
- **Depth AA** — 깊이 맵에 안티앨리어싱을 적용하여 가장자리를 부드럽게 합니다. VDA 모델 사용 시 권장합니다.
- **FPS Display** — 현재 평균 변환 FPS를 화면에 표시합니다.
- **Full SBS Screen Size** — AR 글래스 사용자를 위한 옵션입니다. Full SBS(fsbs) 출력 시 화면 크기를 조정합니다.
- **VR Screen Options** — 화면 크기, 높이, 거리, 중앙 위치, 배경 및 화면 초기화를 설정합니다.
- **Keyboard Hotkeys** — 각 옵션에 단축키를 할당합니다.
- **Letterbox Remove Toggle** — 레터박스를 자동으로 감지하고 제거합니다. 레터박스 비디오에서 검은 띠 위아래에 나타날 수 있는 아티팩트를 제거합니다. Quest 사용자는 컨트롤러 입력으로 토글할 수 있으며, 그 외 헤드셋은 키보드 단축키를 할당해야 합니다. 사용 후 반드시 OFF로 설정하여 오작동을 방지하세요.
- **Mouse Cursor** — 컨트롤러 입력에 사용되는 마우스 커서의 크기와 색상을 조정합니다.
  - **Auto Hide** — 활성화하면 동영상 재생 중 마우스 커서가 자동으로 숨겨집니다.
- **Reset Settings** — 모든 설정을 기본값으로 초기화합니다.
- **Start 3D Conversion / Stop** — 프로그램을 시작하거나 중지합니다. `ESC` 키를 2초간 길게 눌러 프로그램을 종료할 수도 있습니다.

> "재시작 필요" 안내가 표시되지 않는 옵션은 프로그램 실행 중 실시간으로 조정할 수 있습니다.

---

## Conversion 3D 옵션

- **Input File** — 비디오 파일을 선택합니다. 폴더를 선택하면 폴더 내 모든 비디오를 순서대로 변환합니다.
- **Depthmap Input File** — 원본 비디오와 뎁스맵 비디오를 함께 선택하면 뎁스맵 생성 과정을 건너뛰고 즉시 변환을 시작합니다.
- **Output Folder** — 변환된 파일을 저장할 대상 폴더를 선택합니다.
- **Preset** — 사용자 지정 3D 설정을 저장하거나 불러옵니다.
- **Output Format** — 3D 출력 형식을 선택합니다.
- **3D Options** — Live 3D 옵션 설명을 참조하세요.
- **Extract Raw Depthmap** — 체크 시 원본 뎁스맵 비디오를 출력 파일과 함께 내보내어 저장합니다.
- **Extract Corrected Depthmap** - Edge Fix와 Flicker Reduction이 적용된 깊이 맵 비디오를 추출할 수 있습니다.<br> 결과를 확인하는 디버깅용으로 활용하세요.
- **Screen Border Protection** — Live 3D 옵션 설명을 참조하세요.
- **Use FP16** — 체크 시 FP16으로 처리하고, 체크 해제 시 FP32로 뎁스맵을 생성합니다.<br> FP16은 처리 속도가 빠르며, FP32는 정밀도가 약간 더 높습니다.
- **Auto Mode** — 체크 시 3D 깊이 강도를 기준으로 최적화된 파라미터를 자동으로 적용합니다.
- **Video Codec** — 비디오 인코딩 코덱을 선택합니다.<br> libx는 CPU를 사용하고, nvenc는 NVIDIA GPU 가속을 활용하여 더 빠른 변환 속도를 제공합니다.
- **Resize Resolution** — 변환 전 비디오 해상도를 변경합니다.
- **MKV HDR Normalize** — 체크 시 변환 시작 전 HDR 비디오를 표준 비디오 형식으로 재인코딩합니다.
- **Auto Letterbox Crop** — 위아래 레터박스를 자동으로 감지하고 잘라냅니다.<br> 레터박스 영역에 자막이나 텍스트가 겹쳐 있으면 감지에 실패할 수 있습니다.
- **Pad to 16:9** — 16:9 비율이 아닌 비디오에 레터박스를 추가하여 16:9 비율로 맞춥니다.<br> Auto Letterbox Crop과 함께 사용하는 것을 권장합니다.
- **Sound Booster** — 활성화 시 오디오 음량이 400% 증폭되어 변환됩니다.
- **Start Time, End Time** — 활성화하면 비디오의 특정 구간을 지정하여 변환할 수 있습니다.
- **Start / Stop Buttons** — 비디오 변환을 시작하거나 중지합니다.

---

## 오디오 싱크

- 3D 비디오 변환 후 오디오 싱크 문제를 수정할 때 사용합니다. 딜레이 값을 조정하여 오디오를 앞당기거나 늦춰 싱크를 맞출 수 있습니다.

---

## 컨트롤러 가이드 (Meta Quest 시리즈 컨트롤러)

| 입력 | 기능 |
|---|---|
| 아날로그 스틱 (위/아래) | 마우스 스크롤 |
| 아날로그 스틱 (좌/우) | 3D Strength 조정 |
| 아날로그 스틱 버튼 | 화면 중심 재조정 |
| 트리거 버튼 | 마우스 왼쪽 클릭 |
| 그립 버튼 | 그립 버튼을 누른 상태에서 컨트롤러를 좌우로 움직이면 화면 크기, 위아래로 움직이면 화면 거리를 조정합니다. |
| A / X 버튼 | 마우스 오른쪽 클릭 |
| B / Y 버튼 | Letterbox Remove ON/OFF 토글 |

---

## Q&A

**Q: 레터박스 제거 기능이 작동하지 않습니다.**<br> 
A: 레터박스 안에 텍스트가 있거나 해상도가 표준 16:9 비율이 아닌 경우 레터박스를 인식하지 못할 수 있습니다.

**Q: Netflix나 Disney+ 같은 스트리밍 사이트에서 검은 화면만 보입니다.**<br>
A: DRM 보호 정책으로 인해 화면 캡처가 차단됩니다.<br> 웹 브라우저 설정에서 하드웨어 가속을 비활성화해 보세요.

**Q: 게임 화면이 잘리거나 제대로 표시되지 않습니다.**<br>
A: 이 앱은 화면 캡처 방식을 사용하기 때문에 전체 화면 모드에서 제대로 작동하지 않을 수 있습니다.<br> 게임의 디스플레이 설정을 '창 모드' 또는 '창 테두리 없음' 모드로 변경해 보세요.

**Q: 마우스 커서가 이상하게 보이거나, 클릭 가능한 요소에 마우스를 올리면 커서가 사라집니다.**<br>
A: Windows 설정의 '마우스 포인터 스타일 및 색상'으로 이동하여 기본 커서로 초기화하세요.<br> 이후 정상적으로 표시됩니다.

**Q: 마우스 커서가 움직이지 않습니다.**<br>
A: 관리자 권한이 필요한 프로그램을 실행 중인 경우 컨트롤러의 마우스 커서가 작동하지 않습니다.<br> 컴퓨터 마우스를 사용하여 해당 프로그램을 종료하세요.

**Q: Target FPS가 30으로 제한됩니다.**<br> 
A: PC 모드에서 SBS 및 TB 설정은 캡처 방식의 한계로 인해 최대 30 FPS만 지원합니다.

**Q: NVENC 코덱 선택 시 변환 시작과 동시에 오류가 발생합니다.**<br>
A: FFmpeg의 NVENC 코덱에 대한 드라이버 호환성은 빌드 버전에 따라 다를 수 있습니다. 최신 NVIDIA 그래픽 드라이버로 업데이트해 주세요.

**Q: Virtual Desktop 사용 시 PC 모드의 아나글리프 출력이 작동하지 않습니다.**<br>
A: PC 모드의 아나글리프 출력은 Virtual Desktop과 같은 가상 디스플레이 환경에서 3D 렌더링을 지원하지 않습니다. 다만 출력 모드로 전환하면 Virtual Desktop에서 아나글리프 화면을 표시할 수 있습니다.

**Q: 듀얼 모니터 3D 출력 모드는 어떻게 사용하나요?**<br>
A: 이 모드는 캡처용과 출력용 모니터 총 2대가 필요합니다.<br>
준비 사항: 물리적 모니터가 부족한 경우 가상 디스플레이 드라이버, HDMI 더미 플러그, 또는 Virtual Desktop의 가상 디스플레이 기능을 사용하여 두 번째 디스플레이를 추가하세요.<br>
설정 방법: Windows 디스플레이 설정을 확장 모드로 설정한 후, 프로그램에서 입력과 출력에 서로 다른 모니터를 선택합니다. (예: 입력: 모니터 0 / 출력: 모니터 1)<br>
주의: 입력과 출력에 동일한 모니터를 할당하지 마세요.

**Q: 인페인트 모드 사용 시 화면에 검은 선이 나타납니다.**<br>
A: 'Screen Border Protection' 옵션을 비활성화하시면 검은 선이 사라집니다.

---

## 크레딧 (감사의 말)

이 프로젝트는 [IW3](https://github.com/nagadomi/nunif/)의 소스 코드를 사용합니다 (MIT 라이선스).<br>
이 프로젝트는 [ZipDepth](https://github.com/fabiotosi92/ZipDepth)의 소스 코드를 사용합니다 (MIT 라이선스).<br>
이 프로젝트는 [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)의 소스 코드를 사용합니다 (Apache 2.0 라이선스, Copyright 2025 ByteDance).<br>

전체 라이선스 텍스트는 `THIRD_PARTY_LICENSES/`를 참조하세요.

다음 프로젝트의 도움을 받아 제작되었습니다:
- [PyTorch](https://pytorch.org/) — BSD 스타일 라이선스
- [PySide6](https://doc.qt.io/qtforpython/) — LGPLv3
- [dxcam](https://github.com/ra1nty/dxcam) — MIT 라이선스
- [OpenCV (opencv-python)](https://opencv.org/) — Apache 2.0 라이선스, Copyright 2026 OpenCV team
- [NumPy](https://numpy.org/) — BSD 라이선스
- [PyOpenGL](http://pyopengl.sourceforge.net/) — BSD 라이선스
- [pyopenxr](https://github.com/cmbruns/pyopenxr) — Apache 2.0 라이선스, Copyright 2021 Christopher Bruns
- [glfw](https://www.glfw.org/) — zlib/libpng 라이선스
- [pywin32](https://github.com/mhammond/pywin32) — PSF 라이선스
- [winsdk](https://github.com/pywinrt/python-winsdk) — MIT 라이선스
- [psutil](https://github.com/giampaolo/psutil) — BSD-3-Clause 라이선스
- [FFmpeg](https://ffmpeg.org/) — 외부 실행 파일로 호출됨;<br>
  https://github.com/BtbN/FFmpeg-Builds 에서 제공되는 빌드는 GPL v3 라이선스 적용

각 서드파티 라이브러리의 전체 라이선스 텍스트는 THIRD_PARTY_LICENSES/ 폴더에 포함되어 있습니다.

## 참고 문헌
이 프로젝트의 스테레오 생성 접근 방식은 다음 연구에서 영감을 받았습니다:
> Hachaj, T. (2023). Adaptable 2D to 3D Stereo Vision Image Conversion Based on a
> Deep Convolutional Neural Network and Fast Inpaint Algorithm.
> *Entropy*, 25(8), 1212. https://doi.org/10.3390/e25081212

## 3D 데모 영상 – 2D 소스: Big Buck Bunny
© 2008 Blender Foundation / https://peach.blender.org/

크리에이티브 커먼즈 저작자표시 3.0 (CC BY 3.0) 라이선스에 따라 이용할 수 있습니다.

Original: https://www.bigbuckbunny.org
License: https://creativecommons.org/licenses/by/3.0/

본 데모 영상은 DepthLive3D를 사용하여 2D에서 3D로 변환되었습니다.

## 다음 업데이트

