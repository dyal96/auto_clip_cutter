#!/usr/bin/env python3
"""
AutoClipper Studio - Cloud & Google Colab Web Interface
Powered by Gradio, PyTorch, OpenCV & FFmpeg

Supports:
- Outsourced Cloud GPU Compute (Google Colab T4/A100/V100)
- Local Area Network (LAN) Server Hosting (0.0.0.0:7860)
- Drag-and-drop video upload & YouTube URL downloader
- AI Face & Subject Tracking (YOLO / YuNet)
- Previewing clips in browser & individual MP4 downloads
- 1-Click ZIP Download for all processed clips
- Zero ngrok required (uses Gradio built-in --share tunnel)
"""

import os
import sys
import glob
import shutil
import zipfile
import subprocess
import argparse
import torch

import gradio as gr
from clipper_engine import VideoProcessor, SubjectTracker, list_installed_models, download_model_file, AVAILABLE_MODELS_CATALOG

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "Input"))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "Output"))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "models"))

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def get_gpu_status():
    """Returns human-readable string about GPU hardware status."""
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"🟢 GPU CUDA Active: {device_name} ({vram_gb:.1f} GB VRAM)"
    else:
        return "🟠 CPU Mode (No CUDA GPU detected)"


def download_video_from_url(url, cookies_file_path=None, extra_args=None):
    """Downloads video from YouTube or Web URL using yt-dlp with cookie support and format optimization."""
    if not url or not url.strip():
        return None, "Error: Please enter a valid URL."
    
    url = url.strip()
    target_path = os.path.join(INPUT_DIR, "web_download.mp4")
    if os.path.exists(target_path):
        try:
            os.remove(target_path)
        except Exception:
            pass

    # Prefer AVC1 (H.264) or VP9 to avoid AV1 hardware decoding issues on OpenCV/FFmpeg
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[vcodec^=vp9]+bestaudio/best[ext=mp4]/best",
        "-o", target_path,
        "--no-playlist",
        "--no-warnings"
    ]

    # Resolve cookies file
    ck_path = None
    if cookies_file_path and os.path.exists(cookies_file_path):
        ck_path = cookies_file_path
    else:
        for candidate in ["cookies.txt", "youtube_cookies.txt", os.path.join(INPUT_DIR, "cookies.txt")]:
            if os.path.exists(candidate):
                ck_path = candidate
                break

    if ck_path:
        cmd.extend(["--cookies", ck_path])

    if extra_args and extra_args.strip():
        import shlex
        try:
            cmd.extend(shlex.split(extra_args.strip()))
        except Exception:
            pass

    cmd.append(url)

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if res.returncode == 0 and os.path.exists(target_path):
            return target_path, f"Successfully downloaded URL video to {os.path.basename(target_path)}"
        else:
            # Try direct yt-dlp executable fallback
            fallback_cmd = ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best", "-o", target_path, "--no-playlist"]
            if ck_path:
                fallback_cmd.extend(["--cookies", ck_path])
            fallback_cmd.append(url)

            res_fb = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=300)
            if res_fb.returncode == 0 and os.path.exists(target_path):
                return target_path, f"Successfully downloaded URL video to {os.path.basename(target_path)}"
            err_output = (res.stderr or res_fb.stderr or "Unknown download error").strip()
            return None, f"Download failed: {err_output}"
    except Exception as e:
        return None, f"Download error: {str(e)}"


def process_video_ui(
    input_file,
    youtube_url,
    cookies_file,
    extra_ytdlp_args,
    clip_duration,
    mode,
    cam_pos,
    resolution_str,
    encoder,
    speed_preset,
    detector_type,
    model_name,
    top_crop_pct,
    bottom_crop_pct,
    smart_cut,
    auto_subtitles,
    smoothing_alpha
):
    """
    Main handler function for processing video to vertical 9:16 clips.
    """
    logs = []
    def log(msg):
        logs.append(msg)
        print(f"[AutoClipper] {msg}")

    log("=" * 50)
    log("Starting AutoClipper AI Processing Pipeline...")
    log(f"System Hardware: {get_gpu_status()}")

    # Determine input video path
    video_path = None
    if youtube_url and youtube_url.strip():
        log(f"Fetching input from URL: {youtube_url.strip()}")
        ck_path = None
        if cookies_file is not None:
            if hasattr(cookies_file, 'name'):
                ck_path = cookies_file.name
            elif isinstance(cookies_file, str):
                ck_path = cookies_file

        dl_path, msg = download_video_from_url(youtube_url.strip(), cookies_file_path=ck_path, extra_args=extra_ytdlp_args)
        log(msg)
        if dl_path and os.path.exists(dl_path):
            video_path = dl_path

    if not video_path and input_file is not None:
        if isinstance(input_file, str):
            video_path = input_file
        elif hasattr(input_file, 'name'):
            video_path = input_file.name

    if not video_path or not os.path.exists(video_path):
        log("❌ Error: No valid input video file or URL provided!")
        return "\n".join(logs), [], None

    log(f"Input Video File: {os.path.basename(video_path)}")

    # Parse resolution WxH
    try:
        parts = resolution_str.split('x')
        target_res = (int(parts[0]), int(parts[1]))
    except Exception:
        target_res = (1080, 1920)

    # Clean existing clips in Output folder before new run
    clear_output_dir(OUTPUT_DIR)

    log(f"Editing Mode: {mode.upper()}")
    log(f"Target Clip Duration: {clip_duration if clip_duration > 0 else 'Full Video'}s")
    log(f"Output Resolution: {target_res[0]}x{target_res[1]}")
    log(f"AI Detector: {detector_type} (Model: {model_name})")
    log(f"Encoder Engine: {encoder} | Speed Preset: {speed_preset}")

    try:
        tracker = SubjectTracker(model_path=model_name, detector_type=detector_type)
        processor = VideoProcessor(tracker=tracker)

        processor.process_video_to_clips(
            video_path=video_path,
            output_dir=OUTPUT_DIR,
            clip_duration=float(clip_duration) if float(clip_duration) > 0 else None,
            top_crop_pct=float(top_crop_pct) / 100.0,
            bottom_crop_pct=float(bottom_crop_pct) / 100.0,
            mode=mode,
            cam_pos=cam_pos,
            smart_cut=smart_cut,
            auto_subtitles=auto_subtitles,
            smoothing_alpha=float(smoothing_alpha),
            target_resolution=target_res,
            encoder=encoder,
            speed_preset=speed_preset,
            status_callback=lambda m: log(f"  ➜ {m}")
        )

        log("✅ Processing completed successfully!")

        # Gather output clips
        clips = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")))
        log(f"Generated {len(clips)} vertical video clip(s).")

        # Create ZIP archive
        zip_path = create_zip_archive(OUTPUT_DIR)
        log(f"📦 ZIP Archive generated: {os.path.basename(zip_path)}")

        return "\n".join(logs), clips, zip_path

    except Exception as e:
        import traceback
        err_msg = f"❌ Processing failed with error: {e}\n{traceback.format_exc()}"
        log(err_msg)
        return "\n".join(logs), [], None


def clear_output_dir(dir_path):
    """Helper to remove existing mp4/zip files in output folder."""
    if os.path.exists(dir_path):
        for f in os.listdir(dir_path):
            file_p = os.path.join(dir_path, f)
            if os.path.isfile(file_p) and (f.endswith('.mp4') or f.endswith('.zip')):
                try:
                    os.remove(file_p)
                except Exception:
                    pass


def create_zip_archive(dir_path):
    """Zips all .mp4 files in dir_path into AutoClipper_Clips.zip."""
    zip_target = os.path.join(dir_path, "AutoClipper_Clips.zip")
    if os.path.exists(zip_target):
        try:
            os.remove(zip_target)
        except Exception:
            pass

    mp4_files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.mp4')])
    
    with zipfile.ZipFile(zip_target, 'w', zipfile.ZIP_DEFLATED) as zf:
        for mp4 in mp4_files:
            zf.write(mp4, os.path.basename(mp4))

    return zip_target


def trigger_zip_creation():
    """Generates and returns ZIP file of all current clips in Output directory."""
    zip_p = create_zip_archive(OUTPUT_DIR)
    if os.path.exists(zip_p):
        return zip_p, "✅ Created ZIP package with all processed clips!"
    return None, "No output clips found in Output folder to zip."


def download_ai_model_ui(model_filename):
    """UI handler to download selected YOLO / YuNet model file."""
    try:
        path = download_model_file(model_filename)
        return f"Successfully downloaded model: {os.path.basename(path)}"
    except Exception as e:
        return f"Failed to download model: {str(e)}"


# Build Gradio Interface
def create_ui():
    custom_css = """
    .container { max-width: 1400px; margin: auto; padding: 0.5rem; }
    .status-card { background: #0f172a; padding: 0.6rem 1rem; border-radius: 8px; font-weight: bold; border: 1px solid #1e293b; }
    .block { padding: 0.5rem !important; }
    """

    installed_models = list_installed_models()
    model_choices = installed_models if installed_models else ["yolo26n.pt", "yunet.onnx"]

    with gr.Blocks(title="AutoClipper Studio - Cloud Web AI", css=custom_css) as demo:
        gr.Markdown(
            """
            # 🎬 AutoClipper Studio (Cloud & Web Edition)
            ### AI-Powered 9:16 Vertical Video Cutter, Face Tracker & Subtitle Generator
            """
        )

        with gr.Row():
            gpu_info_box = gr.Markdown(value=get_gpu_status(), elem_classes=["status-card"])

        with gr.Tabs():
            # TAB 1: CLIPPER ENGINE
            with gr.TabItem("⚡ Video Clipper Studio"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("### 1. Select Input Source")
                        with gr.Row():
                            input_video = gr.Video(label="Upload Video File", sources=["upload"], height=160)
                            youtube_url = gr.Textbox(
                                label="OR Paste YouTube / Web Video URL",
                                placeholder="https://www.youtube.com/watch?v=...",
                                lines=3
                            )

                        with gr.Accordion("🔑 YouTube Download & Cookies Settings (Fix Bot / Sign-in Errors)", open=False):
                            with gr.Row():
                                cookies_file = gr.File(
                                    label="Upload cookies.txt",
                                    file_count="single"
                                )
                                extra_ytdlp_args = gr.Textbox(
                                    label="Extra yt-dlp Arguments (Optional)",
                                    placeholder="e.g. --js-runtimes node or --username ..."
                                )

                        gr.Markdown("### 2. Clip Settings & AI Controls")
                        with gr.Accordion("⚙️ Main Video Settings", open=True):
                            with gr.Row():
                                resolution_str = gr.Dropdown(
                                    choices=["1080x1080", "1080x1920", "1920x1080", "720x1280"],
                                    value="1080x1920",
                                    label="Output Resolution (WxH)"
                                )
                                mode = gr.Dropdown(
                                    choices=["crop", "fit", "fit_blur", "podcast", "gaming", "blurred"],
                                    value="crop",
                                    label="Editing Layout Mode",
                                    info="crop: Smart Face Track | fit: Black Letterbox | fit_blur: Blurred BG"
                                )
                            with gr.Row():
                                clip_duration = gr.Slider(
                                    minimum=0, maximum=120, step=5, value=30,
                                    label="Target Duration (Sec, 0 = Full Video)"
                                )
                                cam_pos = gr.Radio(
                                    choices=["top", "bottom"], value="top", label="Gaming Cam Position"
                                )

                        with gr.Accordion("🤖 AI Face Tracker & Encoder", open=False):
                            with gr.Row():
                                detector_type = gr.Dropdown(
                                    choices=["auto", "yolo", "yunet"], value="auto", label="Subject Detector Engine"
                                )
                                model_name = gr.Dropdown(
                                    choices=model_choices, value=model_choices[0], label="Model Weights File"
                                )
                            with gr.Row():
                                encoder = gr.Dropdown(
                                    choices=["auto", "h264_nvenc", "h264_qsv", "h264_amf", "libx264"],
                                    value="auto",
                                    label="Hardware Video Encoder Engine"
                                )
                                speed_preset = gr.Dropdown(
                                    choices=["ultra_fast", "fast", "quality"],
                                    value="ultra_fast",
                                    label="Rendering Speed Preset"
                                )
                                smoothing_alpha = gr.Slider(
                                    minimum=0.05, maximum=0.5, step=0.05, value=0.15,
                                    label="Motion Smoothness"
                                )

                        with gr.Accordion("✂️ Crop & Subtitle Enhancements", open=False):
                            with gr.Row():
                                top_crop_pct = gr.Slider(
                                    minimum=0, maximum=30, step=1, value=5, label="Top Watermark Crop (%)"
                                )
                                bottom_crop_pct = gr.Slider(
                                    minimum=0, maximum=30, step=1, value=10, label="Bottom Watermark Crop (%)"
                                )
                            with gr.Row():
                                smart_cut = gr.Checkbox(label="Smart Silence Cut (Speech pauses)", value=False)
                                auto_subtitles = gr.Checkbox(label="Auto Subtitles (Whisper AI)", value=False)

                        process_btn = gr.Button("⚡ Start AI Processing & Cut Clips", variant="primary", size="lg")

                    with gr.Column(scale=6):
                        gr.Markdown("### 3. Output Clips & Processing Logs")
                        status_logs = gr.Textbox(label="Execution Logs & Progress", lines=10, max_lines=16, interactive=False)
                        
                        output_gallery = gr.Gallery(
                            label="Generated Video Clips",
                            columns=2, height="auto", object_fit="contain"
                        )
                        
                        gr.Markdown("### 📦 Bulk Download Options")
                        with gr.Row():
                            create_zip_btn = gr.Button("📦 Download All Clips as ZIP Archive", variant="secondary")
                        
                        zip_download_file = gr.File(label="Download AutoClipper_Clips.zip Package", interactive=False)
                        zip_status = gr.Markdown("")

                # Event Bindings
                process_btn.click(
                    fn=process_video_ui,
                    inputs=[
                        input_video, youtube_url, cookies_file, extra_ytdlp_args,
                        clip_duration, mode, cam_pos, resolution_str, encoder, speed_preset,
                        detector_type, model_name, top_crop_pct, bottom_crop_pct,
                        smart_cut, auto_subtitles, smoothing_alpha
                    ],
                    outputs=[status_logs, output_gallery, zip_download_file]
                )

                create_zip_btn.click(
                    fn=trigger_zip_creation,
                    inputs=[],
                    outputs=[zip_download_file, zip_status]
                )

            # TAB 2: MODEL DOWNLOADER HUB
            with gr.TabItem("⚙️ AI Models & System Hub"):
                gr.Markdown("### 🧠 Download & Manage Tracking Models")
                model_download_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS_CATALOG.keys()),
                    value="yolo26n.pt",
                    label="Select Model to Download"
                )
                download_model_btn = gr.Button("📥 Download Model File to Cloud Storage")
                model_dl_status = gr.Textbox(label="Model Download Status", interactive=False)

                download_model_btn.click(
                    fn=download_ai_model_ui,
                    inputs=[model_download_dropdown],
                    outputs=[model_dl_status]
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="AutoClipper Cloud Web Interface")
    parser.add_argument("--share", action="store_true", help="Create free Gradio public tunnel URL (zero ngrok required)")
    parser.add_argument("--server-name", default="0.0.0.0", help="Server host IP (default: 0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on (default: 7860)")
    parser.add_argument("--inbrowser", action="store_true", help="Automatically open browser tab on start")
    args = parser.parse_args()

    demo = create_ui()
    print(f"Launching AutoClipper Web UI on http://{args.server_name}:{args.port}")
    if args.share:
        print("Creating Gradio public tunnel link...")

    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser
    )


if __name__ == "__main__":
    main()
