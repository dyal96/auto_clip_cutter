import os
import sys
import re
import math
import subprocess
import json
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import urllib.request

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
os.makedirs(MODELS_DIR, exist_ok=True)
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "yunet.onnx")

AVAILABLE_MODELS_CATALOG = {
    "yolo26n.pt": {
        "name": "YOLO26 Nano (PyTorch)",
        "description": "Ultra Fast subject & face tracking (Recommended)",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26n.pt",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo26n.pt",
        "size_mb": 5.5,
        "type": "yolo"
    },
    "yolo26n-pose.pt": {
        "name": "YOLO26 Nano Pose",
        "description": "Fast subject detection with human pose keypoints",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26n-pose.pt",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo26n-pose.pt",
        "size_mb": 7.8,
        "type": "yolo"
    },
    "yolo26s.pt": {
        "name": "YOLO26 Small",
        "description": "Higher accuracy subject detection",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26s.pt",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo26s.pt",
        "size_mb": 19.5,
        "type": "yolo"
    },
    "yolo26m.pt": {
        "name": "YOLO26 Medium",
        "description": "High precision subject detection",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26m.pt",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo26m.pt",
        "size_mb": 40.5,
        "type": "yolo"
    },
    "yolo12n.pt": {
        "name": "YOLO12 Nano",
        "description": "Fast subject detector",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo12n.pt",
        "size_mb": 5.6,
        "type": "yolo"
    },
    "yolo12m.pt": {
        "name": "YOLO12 Medium",
        "description": "Medium YOLO12 model",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo12m.pt",
        "size_mb": 40.9,
        "type": "yolo"
    },
    "yolo11n.pt": {
        "name": "YOLO11 Nano",
        "description": "Compact YOLO11 detection model",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt",
        "size_mb": 5.4,
        "type": "yolo"
    },
    "yolov8n.pt": {
        "name": "YOLOv8 Nano",
        "description": "Standard legacy YOLOv8 detection model",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "size_mb": 6.2,
        "type": "yolo"
    },
    "FastSAM-s.pt": {
        "name": "FastSAM Small",
        "description": "Segment Anything model for subject segmentation",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/FastSAM-s.pt",
        "size_mb": 23.8,
        "type": "yolo"
    },
    "yunet.onnx": {
        "name": "YuNet ONNX (Face)",
        "description": "Ultra lightweight face detector (ONNX)",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "fallback_url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "size_mb": 0.2,
        "type": "yunet"
    }
}

def get_models_dir():
    return MODELS_DIR

def list_installed_models():
    """Returns list of model filenames present in the models/ directory."""
    if not os.path.exists(MODELS_DIR):
        return []
    valid_exts = ('.pt', '.onnx', '.xml', '.engine')
    return [f for f in os.listdir(MODELS_DIR) if f.lower().endswith(valid_exts)]

def resolve_model_path(model_identifier):
    """
    Resolves full absolute path of a model file.
    Checks inside models/ directory first, then root directory, or returns path in models/.
    """
    if not model_identifier:
        return os.path.join(MODELS_DIR, "yolo26n.pt")
    
    if os.path.isabs(model_identifier) and os.path.exists(model_identifier):
        return model_identifier

    filename = os.path.basename(model_identifier)
    in_models = os.path.join(MODELS_DIR, filename)
    if os.path.exists(in_models):
        return in_models
    
    in_root = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(in_root):
        return in_root
    
    return in_models

def download_model_file(model_filename, progress_callback=None, status_callback=None):
    """
    Downloads a model file directly from GitHub releases or OpenCV zoo into models/ folder.
    Supports progress callback (fraction 0.0 to 1.0) and status updates.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    target_path = os.path.join(MODELS_DIR, model_filename)
    tmp_path = target_path + ".tmp"

    meta = AVAILABLE_MODELS_CATALOG.get(model_filename, {})
    urls_to_try = []
    if "url" in meta:
        urls_to_try.append(meta["url"])
    if "fallback_url" in meta:
        urls_to_try.append(meta["fallback_url"])
    
    urls_to_try.append(f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_filename}")
    urls_to_try.append(f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_filename}")

    last_error = None
    for url in urls_to_try:
        try:
            if status_callback:
                status_callback(f"Connecting to download {model_filename}...")
            
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(tmp_path, 'wb') as out_file:
                total_length = response.getheader('content-length')
                total_bytes = int(total_length) if total_length and total_length.isdigit() else 0
                
                downloaded = 0
                chunk_size = 64 * 1024
                while True:
                    buffer = response.read(chunk_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_bytes > 0 and progress_callback:
                        progress_callback(downloaded / total_bytes)
            
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000:
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.rename(tmp_path, target_path)
                if status_callback:
                    status_callback(f"Downloaded {model_filename} successfully!")
                return target_path
        except Exception as e:
            last_error = e
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    raise RuntimeError(f"Failed to download {model_filename}: {last_error}")

_DETECTED_ENCODERS = None

def get_available_gpu_encoders():
    """
    Probes system FFmpeg installation to auto-detect working GPU hardware encoders.
    Returns list of tuples: [('encoder_id', 'Human Label'), ...]
    """
    global _DETECTED_ENCODERS
    if _DETECTED_ENCODERS is not None:
        return _DETECTED_ENCODERS

    supported = []
    test_encoders = [
        ("h264_nvenc", "⚡ NVIDIA NVENC (GPU)"),
        ("h264_qsv", "⚡ Intel QuickSync (GPU)"),
        ("h264_amf", "⚡ AMD AMF (GPU)"),
        ("h264_mf", "⚡ Media Foundation (GPU)"),
        ("libx264", "CPU (libx264 Software)")
    ]

    for enc_id, label in test_encoders:
        try:
            cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=1", "-c:v", enc_id, "-frames:v", "1", "-f", "null", "-"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if res.returncode == 0:
                supported.append((enc_id, label))
        except Exception:
            pass

    if not supported:
        supported = [("libx264", "CPU (libx264 Software)")]

    _DETECTED_ENCODERS = supported
    return _DETECTED_ENCODERS


class SubjectTracker:
    """
    Detects subjects (persons & faces) using YOLO26, YOLO12, YOLO11, YOLOv8, YuNet ONNX,
    or Haar Cascade fallback, maintaining smoothed tracking coordinates across frames.
    Optimized with automatic frame downscaling for 60+ FPS processing.
    """
    def __init__(self, model_path=None, score_threshold=0.5, nms_threshold=0.3, detector_type="auto"):
        self.score_thresh = score_threshold
        self.nms_thresh = nms_threshold
        self.detector_type = detector_type
        self.yunet_detector = None
        self.haar_cascade = None
        self.yolo_model = None
        self.active_engine = "Haar Cascade"

        # Determine target model file
        target_model = model_path
        if not target_model or target_model == "auto":
            if detector_type == "yunet":
                target_model = "yunet.onnx"
            elif detector_type == "yolo":
                target_model = "yolo26n.pt"
            else:
                installed = list_installed_models()
                pref_order = ["yolo26n.pt", "yolo26n.onnx", "yolo26n-pose.pt", "yolo26s.pt", "yolo12n.pt", "yolo11n.pt", "yolov8n.pt", "yunet.onnx"]
                found = None
                for p in pref_order:
                    if p in installed:
                        found = p
                        break
                target_model = found or "yolo26n.pt"

        resolved_path = resolve_model_path(target_model)
        is_yunet = "yunet" in target_model.lower()

        # 1. Try loading with Ultralytics YOLO if not purely YuNet
        if not is_yunet or detector_type in ("auto", "yolo"):
            try:
                from ultralytics import YOLO
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                if os.path.exists(resolved_path):
                    self.yolo_model = YOLO(resolved_path)
                else:
                    # Allow YOLO to resolve or auto-download by name
                    self.yolo_model = YOLO(target_model)
                
                if self.device == "cuda":
                    try:
                        self.yolo_model.to("cuda")
                    except Exception:
                        pass

                model_base = os.path.basename(resolved_path)
                dev_label = "GPU CUDA" if self.device == "cuda" else "CPU"
                self.active_engine = f"YOLO ({model_base}) [{dev_label}]"
                print(f"[Tracker] Initialized YOLO detector: {model_base} on {dev_label}")
            except Exception as e:
                print(f"[Warning] YOLO detector ({target_model}) failed to initialize: {e}")
                self.yolo_model = None

        # 2. Try YuNet ONNX if YOLO not loaded
        yunet_path = resolve_model_path("yunet.onnx")
        if self.yolo_model is None and os.path.exists(yunet_path):
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    yunet_path, "", (300, 300), score_threshold, nms_threshold
                )
                self.active_engine = "YuNet ONNX (Face)"
                print(f"[Tracker] Initialized YuNet detector: {yunet_path}")
            except Exception as e:
                print(f"[Warning] Failed to initialize YuNet: {e}. Falling back to Haar Cascades.")
                self.yunet_detector = None

        # 3. Fallback to Haar Cascade
        if self.yolo_model is None and self.yunet_detector is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self.haar_cascade = cv2.CascadeClassifier(cascade_path)
                self.active_engine = "Haar Cascade"

    def detect_faces(self, frame, max_dim=640):
        """
        Detect subjects/faces in a BGR image frame.
        Downscales input image to max_dim for speedup while maintaining scale.
        Returns list of bounding boxes [(x, y, w, h, score), ...] in original frame coordinates.
        """
        h, w = frame.shape[:2]
        faces_out = []

        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            proc_w = max(32, int(w * scale))
            proc_h = max(32, int(h * scale))
            proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            proc_frame = frame
            proc_w, proc_h = w, h

        # A. YOLO detection (Person class = 0)
        if self.yolo_model is not None:
            try:
                device_arg = getattr(self, "device", "cpu")
                try:
                    results = self.yolo_model(proc_frame, verbose=False, conf=self.score_thresh, classes=[0], device=device_arg)
                except Exception:
                    results = self.yolo_model(proc_frame, verbose=False, conf=self.score_thresh, device=device_arg)

                for r in results:
                    boxes = getattr(r, 'boxes', None)
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            cls = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) > 0 else 0
                            if cls == 0 or len(boxes) == 1:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                conf = float(box.conf[0])
                                pw = x2 - x1
                                ph = y2 - y1
                                fx = int(x1)
                                fy = int(y1)
                                fw = int(pw)
                                fh = int(max(30, ph * 0.45))
                                if scale != 1.0:
                                    fx = int(fx / scale)
                                    fy = int(fy / scale)
                                    fw = int(fw / scale)
                                    fh = int(fh / scale)
                                faces_out.append((fx, fy, fw, fh, conf))
            except Exception as e:
                pass

        # B. YuNet detection
        if not faces_out and self.yunet_detector is not None:
            self.yunet_detector.setInputSize((proc_w, proc_h))
            _, faces = self.yunet_detector.detect(proc_frame)
            if faces is not None:
                for face in faces:
                    fx, fy, fw, fh = map(int, face[:4])
                    score = float(face[-1])
                    if scale != 1.0:
                        fx = int(fx / scale)
                        fy = int(fy / scale)
                        fw = int(fw / scale)
                        fh = int(fh / scale)
                    faces_out.append((fx, fy, fw, fh, score))

        # C. Haar Cascade detection
        if not faces_out and self.haar_cascade is not None:
            gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            detected = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (fx, fy, fw, fh) in detected:
                if scale != 1.0:
                    fx = int(fx / scale)
                    fy = int(fy / scale)
                    fw = int(fw / scale)
                    fh = int(fh / scale)
                faces_out.append((int(fx), int(fy), int(fw), int(fh), 0.8))

        return faces_out

    def get_primary_face_center(self, frame, max_dim=640):
        """
        Finds primary face/subject (largest & closest to center) and returns (center_x, center_y, width, height).
        """
        faces = self.detect_faces(frame, max_dim=max_dim)
        if not faces:
            return None

        h, w = frame.shape[:2]
        frame_cx = w / 2.0

        best_face = None
        best_score = -1.0

        for (fx, fy, fw, fh, score) in faces:
            cx = fx + fw / 2.0
            dist_from_center = abs(cx - frame_cx) / w
            area = fw * fh
            metric = (area / (w * h)) * (1.0 - 0.3 * dist_from_center) + score * 0.1
            if metric > best_score:
                best_score = metric
                best_face = (cx, fy + fh / 2.0, fw, fh)

        return best_face

    def get_dual_speakers(self, frame, max_dim=640):
        """
        For Podcast Mode: Finds two distinct faces/subjects (Left speaker and Right speaker).
        Returns tuple of (speaker_left, speaker_right) or None.
        """
        faces = self.detect_faces(frame, max_dim=max_dim)
        if len(faces) < 2:
            return None

        h, w = frame.shape[:2]
        sorted_faces = sorted(faces, key=lambda f: f[0])
        face1 = sorted_faces[0]
        face2 = sorted_faces[-1]

        if (face2[0] - face1[0]) > (w * 0.15):
            left_center = (face1[0] + face1[2] / 2.0, face1[1] + face1[3] / 2.0, face1[2], face1[3])
            right_center = (face2[0] + face2[2] / 2.0, face2[1] + face2[3] / 2.0, face2[2], face2[3])
            return left_center, right_center

        return None

# Backwards compatibility alias
FaceTracker = SubjectTracker


def calculate_crop_window(video_width, video_height, face_center_x, face_center_y, 
                          top_crop_pct=0.0, bottom_crop_pct=0.0, target_aspect_ratio=9/16):
    """
    Computes (crop_x, crop_y, crop_w, crop_h) that:
    1. Fits target aspect ratio inside original frame.
    2. Excludes top_crop_pct and bottom_crop_pct (watermark avoidance).
    3. Centers around (face_center_x, face_center_y).
    """
    y_min = int(video_height * top_crop_pct)
    y_max = int(video_height * (1.0 - bottom_crop_pct))
    usable_h = max(1, y_max - y_min)

    crop_h = usable_h
    crop_w = int(crop_h * target_aspect_ratio)

    if crop_w > video_width:
        crop_w = video_width
        crop_h = int(crop_w / target_aspect_ratio)

    crop_w = (crop_w // 2) * 2
    crop_h = (crop_h // 2) * 2

    if face_center_x is None:
        face_center_x = video_width / 2.0
    crop_x = int(face_center_x - crop_w / 2.0)
    crop_x = max(0, min(video_width - crop_w, crop_x))

    if face_center_y is None:
        face_center_y = y_min + usable_h / 2.0

    crop_y = int(face_center_y - crop_h / 2.0)
    crop_y = max(y_min, min(y_max - crop_h, crop_y))

    return crop_x, crop_y, crop_w, crop_h


def build_dynamic_crop_expressions(clip_trajectory, start_time, default_crop_box):
    """
    Builds time-dependent FFmpeg expressions x='...' and y='...' for the crop filter.
    Interpolates smoothly between trajectory keyframe points over time t (seconds).
    """
    if not clip_trajectory:
        cx, cy, cw, ch = default_crop_box
        return str(cx), str(cy), cw, ch

    pts = []
    for pt in clip_trajectory:
        t_rel = max(0.0, pt[0] - start_time)
        cx, cy, cw, ch = pt[1], pt[2], pt[3], pt[4]
        pts.append((t_rel, cx, cy, cw, ch))

    pts.sort(key=lambda item: item[0])
    cw, ch = pts[0][3], pts[0][4]

    filtered = [pts[0]]
    for p in pts[1:]:
        prev = filtered[-1]
        dt = p[0] - prev[0]
        dx = abs(p[1] - prev[1])
        dy = abs(p[2] - prev[2])
        if dt >= 0.5 or dx >= 4 or dy >= 4:
            filtered.append(p)

    if len(filtered) == 1:
        return str(filtered[0][1]), str(filtered[0][2]), cw, ch

    last_pt = filtered[-1]
    x_expr = str(int(last_pt[1]))
    y_expr = str(int(last_pt[2]))

    for i in range(len(filtered) - 2, -1, -1):
        curr_t, curr_x, curr_y = filtered[i][0], filtered[i][1], filtered[i][2]
        next_t, next_x, next_y = filtered[i+1][0], filtered[i+1][1], filtered[i+1][2]
        dt = next_t - curr_t

        if dt > 0.001:
            x_seg = f"{curr_x:.1f}+({next_x - curr_x:.1f})*(t-{curr_t:.2f})/{dt:.2f}"
            y_seg = f"{curr_y:.1f}+({next_y - curr_y:.1f})*(t-{curr_t:.2f})/{dt:.2f}"
            x_expr = f"if(lte(t,{next_t:.2f}),{x_seg},{x_expr})"
            y_expr = f"if(lte(t,{next_t:.2f}),{y_seg},{y_expr})"

    x_expr = x_expr.replace(",", "\\,")
    y_expr = y_expr.replace(",", "\\,")
    return x_expr, y_expr, cw, ch


class SubtitleGenerator:
    """
    Transcribes video audio using Whisper AI model and exports styled ASS subtitles.
    """
    @staticmethod
    def generate_ass_subtitles(video_path, output_ass_path, model_name="base", target_resolution=(1080, 1920)):
        try:
            import whisper
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            print("[Warning] Whisper not installed. Skipping auto subtitles.")
            return None

        print(f"[Whisper] Loading Whisper '{model_name}' model on device '{device}'...")
        model = whisper.load_model(model_name, device=device)
        result = model.transcribe(video_path, verbose=False, fp16=(device == "cuda"))

        segments = result.get("segments", [])
        if not segments:
            return None

        res_x, res_y = target_resolution
        margin_v = int(res_y * 0.115)

        ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,52,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,30,30,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        def format_time(seconds):
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            cs = int((seconds - int(seconds)) * 100)
            return f"{hrs:01d}:{mins:02d}:{secs:02d}.{cs:02d}"

        lines = [ass_header]
        for seg in segments:
            start_str = format_time(seg["start"])
            end_str = format_time(seg["end"])
            text = seg["text"].strip().replace("\n", " ")
            if text:
                lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}\n")

        with open(output_ass_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return output_ass_path


class AudioSilenceDetector:
    """
    Uses FFmpeg silencedetect filter to find speech pauses / discussion end boundaries.
    """
    @staticmethod
    def detect_silences(video_path, noise_threshold_db=-35, min_silence_sec=0.4):
        cmd = [
            "ffmpeg", "-vn", "-i", video_path,
            "-ar", "16000", "-ac", "1",
            "-af", f"silencedetect=noise={noise_threshold_db}dB:d={min_silence_sec}",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stderr = result.stderr

        silence_starts = [float(m) for m in re.findall(r"silence_start:\s*([\d\.]+)", stderr)]
        silence_ends = [float(m) for m in re.findall(r"silence_end:\s*([\d\.]+)", stderr)]

        silences = []
        for i in range(min(len(silence_starts), len(silence_ends))):
            silences.append((silence_starts[i], silence_ends[i]))

        return silences


class VideoProcessor:
    """
    Processes video files with face tracking, continuous dynamic panning, smart silence cutting,
    Podcast dual-speaker stack, Gaming split screen, auto subtitles, and high-quality Lanczos rendering.
    """
    def __init__(self, tracker=None, detector_type="auto"):
        self.tracker = tracker or SubjectTracker(detector_type=detector_type)

    @staticmethod
    def probe_video(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration
        }

    def analyze_face_trajectory(self, video_path, sample_fps=2.5, smoothing_alpha=0.15,
                                  top_crop_pct=0.0, bottom_crop_pct=0.0, target_aspect=9/16,
                                  progress_callback=None, frame_callback=None, max_dim=640):
        meta = self.probe_video(video_path)
        width, height = meta["width"], meta["height"]
        video_fps = meta["fps"]
        total_frames = meta["total_frames"]

        cap = cv2.VideoCapture(video_path)
        frame_interval = max(1, int(video_fps / sample_fps))

        smoothed_cx = width / 2.0
        smoothed_cy = height / 2.0

        trajectory = []
        dual_speaker_points = []
        frame_idx = 0

        while cap.isOpened() and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp = frame_idx / video_fps
            
            dual = self.tracker.get_dual_speakers(frame, max_dim=max_dim)
            if dual:
                dual_speaker_points.append((timestamp, dual[0], dual[1]))

            face = self.tracker.get_primary_face_center(frame, max_dim=max_dim)
            if face is not None:
                raw_cx, raw_cy, _, _ = face
                smoothed_cx = (1 - smoothing_alpha) * smoothed_cx + smoothing_alpha * raw_cx
                smoothed_cy = (1 - smoothing_alpha) * smoothed_cy + smoothing_alpha * raw_cy

            cx, cy, cw, ch = calculate_crop_window(
                width, height, smoothed_cx, smoothed_cy,
                top_crop_pct, bottom_crop_pct, target_aspect
            )
            trajectory.append((timestamp, cx, cy, cw, ch))

            if frame_callback:
                try:
                    frame_callback(frame, frame_idx + 1, total_frames)
                except Exception:
                    pass

            if progress_callback and meta["total_frames"] > 0:
                progress_callback(frame_idx / meta["total_frames"])

            frame_idx += frame_interval
            for _ in range(frame_interval - 1):
                if not cap.grab():
                    break

        cap.release()
        return trajectory, dual_speaker_points, meta

    def calculate_smart_clip_intervals(self, video_path, target_duration=30.0, smart_cut=True):
        meta = self.probe_video(video_path)
        total_duration = meta["duration"]

        if not smart_cut or target_duration is None or target_duration <= 0 or target_duration >= total_duration:
            if target_duration is None or target_duration <= 0 or target_duration >= total_duration:
                return [(0.0, total_duration)]
            
            intervals = []
            cur = 0.0
            while cur < total_duration:
                nxt = min(cur + target_duration, total_duration)
                if nxt - cur >= 5.0:
                    intervals.append((cur, nxt))
                cur = nxt
            return intervals

        silences = AudioSilenceDetector.detect_silences(video_path)
        if not silences:
            return self.calculate_smart_clip_intervals(video_path, target_duration, smart_cut=False)

        silence_midpoints = [(st + et) / 2.0 for (st, et) in silences]

        intervals = []
        cur_start = 0.0

        while cur_start < total_duration:
            ideal_end = cur_start + target_duration
            if ideal_end >= total_duration - 5.0:
                intervals.append((cur_start, total_duration))
                break

            candidates = [t for t in silence_midpoints if abs(t - ideal_end) <= 8.0 and t > cur_start + 10.0]
            if candidates:
                best_end = min(candidates, key=lambda t: abs(t - ideal_end))
            else:
                best_end = ideal_end

            intervals.append((cur_start, best_end))
            cur_start = best_end

        return intervals

    def render_clip(self, video_path, start_time, end_time, output_path, crop_box,
                    mode="crop", target_resolution=(1080, 1920), top_crop_pct=0.0, bottom_crop_pct=0.0,
                    dual_speakers=None, cam_pos="top", ass_subtitle_path=None, encoder="auto",
                    clip_trajectory=None, speed_preset="fast"):
        """
        Renders clip with dynamic camera trajectory, optimized hardware acceleration,
        fast scaling filters, and configurable speed/quality presets.
        """
        crop_x, crop_y, crop_w, crop_h = crop_box
        out_w, out_h = target_resolution
        half_h = out_h // 2

        meta = self.probe_video(video_path)
        vid_w, vid_h = meta["width"], meta["height"]

        # Build continuous dynamic crop expression for smooth tracking
        x_expr, y_expr, crop_w, crop_h = build_dynamic_crop_expressions(
            clip_trajectory, start_time, crop_box
        )

        scale_flags = "flags=bicubic" if speed_preset == "fast" else "flags=lanczos+accurate_rnd"
        unsharp_filter = "" if speed_preset == "fast" else ",unsharp=5:5:0.8:5:5:0.4"

        if mode == "crop":
            # Single Speaker Dynamic Face/Body Crop
            filter_graph = f"[0:v]crop={crop_w}:{crop_h}:{x_expr}:{y_expr},scale={out_w}:{out_h}:{scale_flags}{unsharp_filter}[vout]"

        elif mode == "podcast":
            # Podcast Mode: Dual Speaker Stack
            if dual_speakers:
                spk_l, spk_r = dual_speakers
                lx, ly, lw, lh = spk_l
                rx, ry, rw, rh = spk_r

                cx_l, cy_l, cw_l, ch_l = calculate_crop_window(vid_w, vid_h, lx, ly, top_crop_pct, bottom_crop_pct, target_aspect_ratio=1080/960)
                cx_r, cy_r, cw_r, ch_r = calculate_crop_window(vid_w, vid_h, rx, ry, top_crop_pct, bottom_crop_pct, target_aspect_ratio=1080/960)
            else:
                cw_l, ch_l = int(vid_w * 0.45), int(vid_h * 0.8)
                cx_l, cy_l = 0, int(vid_h * top_crop_pct)
                cw_r, ch_r = cw_l, ch_l
                cx_r, cy_r = int(vid_w * 0.55), cy_l

            filter_graph = (
                f"[0:v]crop={cw_l}:{ch_l}:{cx_l}:{cy_l},scale={out_w}:{half_h}:{scale_flags}{unsharp_filter}[top];"
                f"[0:v]crop={cw_r}:{ch_r}:{cx_r}:{cy_r},scale={out_w}:{half_h}:{scale_flags}{unsharp_filter}[bot];"
                f"[top][bot]vstack=inputs=2[vout]"
            )

        elif mode == "gaming":
            # Gaming Mode: Dynamic Cam Face + Gameplay
            cam_cw, cam_ch = crop_w, crop_h

            game_cw = vid_w
            game_ch = int(vid_w * (9/16))
            game_cx = 0
            game_cy = max(0, int((vid_h - game_ch) / 2))

            if cam_pos == "top":
                filter_graph = (
                    f"[0:v]crop={cam_cw}:{cam_ch}:{x_expr}:{y_expr},scale={out_w}:{half_h}:{scale_flags}{unsharp_filter}[cam];"
                    f"[0:v]crop={game_cw}:{game_ch}:{game_cx}:{game_cy},scale={out_w}:{half_h}:{scale_flags}{unsharp_filter}[game];"
                    f"[cam][game]vstack=inputs=2[vout]"
                )
            else:
                filter_graph = (
                    f"[0:v]crop={game_cw}:{game_ch}:{game_cx}:{game_cy},scale={out_w}:{half_h}:{scale_flags}{unsharp_filter}[game];"
                    f"[0:v]crop={cam_cw}:{cam_ch}:{x_expr}:{y_expr},scale={out_w}:{half_h}:{scale_flags}{unsharp_filter}[cam];"
                    f"[game][cam]vstack=inputs=2[vout]"
                )

        elif mode == "fit":
            # Fit Mode: Letterbox with black background padding (No crop)
            usable_y = int(vid_h * top_crop_pct)
            usable_h = max(1, int(vid_h * (1.0 - top_crop_pct - bottom_crop_pct)))
            filter_graph = (
                f"[0:v]crop=iw:{usable_h}:0:{usable_y},"
                f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:{scale_flags}{unsharp_filter},"
                f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black[vout]"
            )

        elif mode in ("fit_blur", "blurred"):
            # Fit Mode: Blurred background padding (No crop)
            usable_y = int(vid_h * top_crop_pct)
            usable_h = max(1, int(vid_h * (1.0 - top_crop_pct - bottom_crop_pct)))
            filter_graph = (
                f"[0:v]crop=iw:{usable_h}:0:{usable_y},"
                f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase:flags=bicubic,crop={out_w}:{out_h},boxblur=10:3[bg];"
                f"[0:v]crop=iw:{usable_h}:0:{usable_y},"
                f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:{scale_flags}{unsharp_filter}[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2[vout]"
            )

        else:
            # Default fallback: Smart Crop
            filter_graph = f"[0:v]crop={crop_w}:{crop_h}:{x_expr}:{y_expr},scale={out_w}:{out_h}:{scale_flags}{unsharp_filter}[vout]"

        if ass_subtitle_path and os.path.exists(ass_subtitle_path):
            clean_ass_path = ass_subtitle_path.replace("\\", "/").replace(":", "\\:")
            filter_graph += f";[vout]subtitles='{clean_ass_path}'[vfinal]"
            out_label = "[vfinal]"
        else:
            out_label = "[vout]"

        enc_id = encoder
        if enc_id == "auto":
            gpu_encoders = get_available_gpu_encoders()
            enc_id = gpu_encoders[0][0]

        video_codec_args = []
        if enc_id == "h264_nvenc":
            nv_preset = "p1" if speed_preset in ("fast", "ultra_fast") else "p4"
            video_codec_args = ["-c:v", "h264_nvenc", "-preset", nv_preset, "-cq", "20", "-spatial-aq", "1"]
        elif enc_id == "h264_qsv":
            qsv_preset = "veryfast" if speed_preset in ("fast", "ultra_fast") else "medium"
            video_codec_args = ["-c:v", "h264_qsv", "-preset", qsv_preset, "-global_quality", "20"]
        elif enc_id == "h264_amf":
            amf_qual = "speed" if speed_preset in ("fast", "ultra_fast") else "quality"
            video_codec_args = ["-c:v", "h264_amf", "-quality", amf_qual, "-rc", "cqp", "-qp_p", "20", "-qp_i", "20"]
        elif enc_id == "h264_mf":
            video_codec_args = ["-c:v", "h264_mf", "-b:v", "10M"]
        else:
            x264_preset = "ultrafast" if speed_preset in ("fast", "ultra_fast") else "medium"
            video_codec_args = ["-c:v", "libx264", "-preset", x264_preset, "-crf", "20", "-threads", "0"]

        hwaccel_args = []
        if enc_id == "h264_nvenc":
            hwaccel_args = ["-hwaccel", "cuda"]
        elif enc_id in ("h264_qsv", "h264_amf"):
            hwaccel_args = ["-hwaccel", "auto"]

        duration = end_time - start_time
        cmd = [
            "ffmpeg", "-y"
        ] + hwaccel_args + [
            "-ss", f"{start_time:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-avoid_negative_ts", "make_zero",
            "-filter_complex", filter_graph,
            "-map", out_label,
            "-map", "0:a?"
        ] + video_codec_args + [
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-max_muxing_queue_size", "1024",
            output_path
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Fallback 1: If hardware input decoding failed (e.g. AV1 or missing sequence header), retry with CPU decoding + GPU encoding
        if result.returncode != 0 and hwaccel_args:
            cmd_no_hwaccel = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.3f}",
                "-i", video_path,
                "-t", f"{duration:.3f}",
                "-avoid_negative_ts", "make_zero",
                "-filter_complex", filter_graph,
                "-map", out_label,
                "-map", "0:a?"
            ] + video_codec_args + [
                "-c:a", "aac",
                "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-max_muxing_queue_size", "1024",
                output_path
            ]
            result = subprocess.run(cmd_no_hwaccel, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Fallback 2: If GPU encoder failed, retry with CPU libx264 software encoder
        if result.returncode != 0 and enc_id != "libx264":
            x264_preset = "ultrafast" if speed_preset in ("fast", "ultra_fast") else "medium"
            sw_codec_args = ["-c:v", "libx264", "-preset", x264_preset, "-crf", "20", "-threads", "0"]
            cmd_sw_all = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.3f}",
                "-i", video_path,
                "-t", f"{duration:.3f}",
                "-avoid_negative_ts", "make_zero",
                "-filter_complex", filter_graph,
                "-map", out_label,
                "-map", "0:a?"
            ] + sw_codec_args + [
                "-c:a", "aac",
                "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-max_muxing_queue_size", "1024",
                output_path
            ]
            result = subprocess.run(cmd_sw_all, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg rendering failed after fallback attempts:\n{result.stderr}")

        return output_path

    def process_video_to_clips(self, video_path, output_dir, clip_duration=30.0,
                               top_crop_pct=0.0, bottom_crop_pct=0.0, mode="crop",
                               cam_pos="top", smart_cut=True, auto_subtitles=False,
                               smoothing_alpha=0.15, target_resolution=(1080, 1920),
                               encoder="auto", speed_preset="fast", sample_fps=3.0,
                               progress_callback=None, status_callback=None,
                               frame_callback=None):
        if status_callback:
            gpu_encoders = get_available_gpu_encoders()
            active_label = dict(gpu_encoders).get(encoder, gpu_encoders[0][1])
            engine_name = getattr(self.tracker, "active_engine", "Subject Tracker")
            mode_label = "⚡ ULTRA FAST" if speed_preset == "ultra_fast" else speed_preset.upper()
            status_callback(f"Tracking subjects ({engine_name}) • Preset: {mode_label} • Engine: {active_label}...")

        actual_sample_fps = 2.0 if speed_preset == "ultra_fast" else sample_fps
        max_dim = 320 if speed_preset == "ultra_fast" else 640

        trajectory, dual_speaker_points, meta = self.analyze_face_trajectory(
            video_path,
            sample_fps=actual_sample_fps,
            smoothing_alpha=smoothing_alpha,
            top_crop_pct=top_crop_pct,
            bottom_crop_pct=bottom_crop_pct,
            target_aspect=target_resolution[0] / target_resolution[1],
            progress_callback=lambda p: progress_callback(p * 0.3) if progress_callback else None,
            frame_callback=frame_callback,
            max_dim=max_dim
        )

        if status_callback and smart_cut:
            status_callback("Detecting audio silences & speech cut points...")

        clip_intervals = self.calculate_smart_clip_intervals(
            video_path, target_duration=clip_duration, smart_cut=smart_cut
        )

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        output_files = []

        ass_path = None
        if auto_subtitles:
            if status_callback:
                status_callback("Generating AI auto-subtitles with Whisper...")
            temp_ass = os.path.join(output_dir, f"{base_name}_temp_captions.ass")
            ass_path = SubtitleGenerator.generate_ass_subtitles(video_path, temp_ass, target_resolution=target_resolution)

        num_clips = len(clip_intervals)

        def _render_single(item):
            idx, (st, et) = item
            if status_callback and speed_preset != "ultra_fast":
                status_callback(f"Rendering clip {idx+1}/{num_clips} ({st:.1f}s - {et:.1f}s)...")

            mid_t = (st + et) / 2.0
            if trajectory:
                closest_point = min(trajectory, key=lambda pt: abs(pt[0] - mid_t))
                crop_box = closest_point[1:]
                clip_traj = [pt for pt in trajectory if st <= pt[0] <= et]
            else:
                default_cx, default_cy, default_cw, default_ch = calculate_crop_window(
                    meta["width"], meta["height"], meta["width"] / 2.0, meta["height"] / 2.0,
                    top_crop_pct, bottom_crop_pct, target_aspect_ratio=target_resolution[0] / target_resolution[1]
                )
                crop_box = (default_cx, default_cy, default_cw, default_ch)
                clip_traj = None

            dual_spk = None
            if dual_speaker_points:
                closest_dual = min(dual_speaker_points, key=lambda pt: abs(pt[0] - mid_t))
                dual_spk = (closest_dual[1], closest_dual[2])

            out_filename = f"{base_name}_{mode}_clip_{idx+1:03d}.mp4"
            out_filepath = os.path.join(output_dir, out_filename)

            self.render_clip(
                video_path=video_path,
                start_time=st,
                end_time=et,
                output_path=out_filepath,
                crop_box=crop_box,
                mode=mode,
                target_resolution=target_resolution,
                top_crop_pct=top_crop_pct,
                bottom_crop_pct=bottom_crop_pct,
                dual_speakers=dual_spk,
                cam_pos=cam_pos,
                ass_subtitle_path=ass_path,
                encoder=encoder,
                clip_trajectory=clip_traj,
                speed_preset=speed_preset
            )

            # Emit a live preview frame at the midpoint of the clip
            if frame_callback:
                try:
                    _cap = cv2.VideoCapture(video_path)
                    _cap.set(cv2.CAP_PROP_POS_MSEC, mid_t * 1000)
                    _ret, _frame = _cap.read()
                    _cap.release()
                    if _ret and _frame is not None:
                        frame_callback(_frame, idx + 1, num_clips)
                except Exception:
                    pass

            return out_filepath

        if speed_preset == "ultra_fast" and num_clips > 1:
            import concurrent.futures
            workers = min(4, max(1, os.cpu_count() or 2))
            if status_callback:
                status_callback(f"Rendering {num_clips} clips in parallel ({workers} workers)...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_render_single, item): item[0] for item in enumerate(clip_intervals)}
                done_count = 0
                for fut in concurrent.futures.as_completed(futures):
                    output_files.append(fut.result())
                    done_count += 1
                    if progress_callback:
                        progress_callback(0.3 + 0.7 * (done_count / num_clips))
            output_files.sort()
        else:
            for idx, (st, et) in enumerate(clip_intervals):
                res_path = _render_single((idx, (st, et)))
                output_files.append(res_path)
                if progress_callback:
                    progress_callback(0.3 + 0.7 * ((idx + 1) / num_clips))

        if ass_path and os.path.exists(ass_path):
            try:
                os.remove(ass_path)
            except Exception:
                pass

        if status_callback:
            status_callback(f"Done! Exported {len(output_files)} clip(s) to '{output_dir}'")

        return output_files
