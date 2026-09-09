# 🎬 AutoClipper Studio

> **AI-Powered 9:16 Vertical Video Cutter, Face Tracker & Automated Project Converter**

AutoClipper Studio is a desktop suite built to convert horizontal videos (podcasts, gaming streams, interviews, YouTube videos) into optimized 9:16 vertical short-form content (TikTok, YouTube Shorts, Instagram Reels) with AI face tracking, auto-subtitles, hardware acceleration, and ultra-fast parallel rendering.

---

## ✨ Key Features

### ⚡ 1. Ultra Fast Performance Engine
- **⚡ Ultra Fast Mode Toggle**: Multi-threaded parallel clip rendering (`ThreadPoolExecutor` with 4+ workers) to render multiple output clips concurrently on your GPU/CPU encoder.
- **CUDA GPU AI Acceleration**: Automatic PyTorch CUDA GPU detection (`device='cuda'`) for YOLO subject tracking (boosting tracking speed from ~15 FPS to **150+ FPS**).
- **Zero-Cost Frame Grabbing**: Optimized frame skipping using native OpenCV `cap.grab()` jumps instead of slow per-frame decoding.
- **Audio-Only Silence Scan**: Fast FFmpeg `-vn` audio stream decoding downsampled to 16kHz mono (`-ar 16000 -ac 1`) for instant speech pause detection.
- **Hardware-Accelerated Scaling**: Ultra-fast bicubic hardware scaling (`-hwaccel cuda`) and low-latency encoder presets (`h264_nvenc p1`, `h264_qsv veryfast`, `libx264 ultrafast`).

### 🎯 2. AI Face Tracking & Smart Vertical Crop
- **YOLO & YuNet Subject Detection**: Automatically tracks faces and active speakers frame-by-frame (`yolo26n.pt`, `yolo12n.pt`, `yolov8n.pt`, `yunet.onnx`).
- **Smooth Camera Movement**: Exponential moving average smoothing so the camera glides smoothly across active speakers.
- **Multiple Layout Modes**:
  - **Smart Dynamic 9:16 Crop**: Tracks the active speaker and crops dynamically.
  - **Podcast Stack (Top/Bottom 2 Speakers)**: Detects 2 speakers and creates a 2-host vertical split frame.
  - **Gaming Split (Cam + Gameplay)**: Crops webcam box onto top/bottom and displays gameplay on the rest.
  - **Blurred Background Stack**: Fits full video centered with a blurred, expanded background.
- **Real-Time Live Preview Window**: Displays live frame-by-frame preview during AI tracking and clip rendering with Tkinter memory-protection.

### 📺 3. Integrated YouTube Downloader & Project Prep
- **YouTube / Web Downloader**: Built-in `yt-dlp` integration to download YouTube videos directly into your `Input/` folder.
- **Seamless Project Prep**:
  - Download video from URL.
  - Trim custom time range (`HH:MM:SS` or seconds).
  - Generate AI subtitles with OpenAI Whisper (`.srt`, `.vtt`, `.ass`).
  - Automatically organize into project folders for easy NLE editing (Premiere, CapCut, DaVinci).

### ✂️ 4. Quick Video Splitter & Merger
- **Split Video**:
  - **By Time Interval**: Split video into equal time chunks (e.g., 60-second parts).
  - **By Equal Parts**: Automatically calculates total duration and splits into $N$ equal parts.
- **Merge Videos**: Batch merge multiple video clips into one file using fast, lossless FFmpeg stream concatenation (`-c copy`).

### ⚙️ 5. One-Click Settings & Tools Hub
- Download and manage YOLO models (`yolo26n.pt`, `yolo12n.pt`, `yolov8n-face.pt`, `yunet.onnx`).
- Download required binaries (`yt-dlp.exe`, `ffmpeg.exe`) straight from the UI.
- Auto-detects NVIDIA NVENC (`h264_nvenc`), Intel QSV (`h264_qsv`), AMD AMF (`h264_amf`), and Windows Media Foundation (`h264_mf`).

---

## 🚀 Quick Start

### 1. Requirements
- **Windows 10 / 11**
- **Python 3.8+**
- **FFmpeg & yt-dlp** (Can be downloaded automatically within the app under `⚙️ Hub → 🔧 Tools`)

### 2. Installation
Clone the repository and install required packages:

```bash
git clone https://github.com/your-username/AutoClipper.git
cd AutoClipper

# Install python dependencies
pip install -r requirements.txt
```

### 3. Run the App

#### Option A: Launch GUI (Recommended)
Double-click `start_cliping.bat` or run:
```bash
python app.py
```

#### Option B: Headless CLI Batch Mode
```bash
python app.py --cli --input Input/ --output Output/ --duration 30 --mode crop --preset ultra_fast --subtitles
```

---

## 🖥️ Command Line Usage (CLI)

```bash
python app.py --cli [OPTIONS]

Options:
  --input PATH       Input file or directory (default: Input)
  --output PATH      Output directory (default: Output)
  --duration SECS    Clip length in seconds (0 = full video)
  --mode MODE        crop | podcast | gaming | blurred
  --cam-pos POS      top | bottom (for gaming mode)
  --smart-cut        Cut clips on audio silence pauses
  --subtitles        Generate AI Whisper subtitles
  --top-crop PCT     Top crop % (0.0 - 0.3)
  --bottom-crop PCT  Bottom crop % (0.0 - 0.3)
  --res WxH          Output resolution (default: 1080x1920)
  --preset PRESET    ultra_fast | fast | quality (default: ultra_fast)
  --encoder ENC      auto | h264_nvenc | h264_qsv | h264_amf | libx264
  --detector DET     auto | yolo | yunet
  --model NAME       Specific model file name (e.g. yolo26n.pt)
```

---

## 📁 Project Structure

```
AutoClipper/
├── app.py                 # Application entry point (GUI, Web UI & CLI launcher)
├── gui.py                 # CustomTkinter Desktop GUI Studio
├── web_app.py              # Gradio Web & Cloud Studio Interface
├── clipper_engine.py      # Core AI VideoProcessor, YOLO/YuNet face tracker & FFmpeg engine
├── AutoClipper_Colab.ipynb # 1-Click Google Colab Notebook
├── start_cliping.bat      # Windows batch launcher (Desktop App)
├── start_cloud_app.bat    # Windows batch launcher (Web UI)
├── start_cloud_app.sh     # Linux / Colab launcher script (Web UI)
├── requirements.txt       # Python dependency list
├── models/                # AI YOLO & YuNet models storage folder
├── tools/                 # Portable binaries folder (ffmpeg.exe, yt-dlp.exe)
├── Input/                 # Default input folder for videos
└── Output/                # Default output directory for vertical clips & project files
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
