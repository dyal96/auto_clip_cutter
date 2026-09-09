import os
import sys
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from clipper_engine import VideoProcessor, FaceTracker

def run_cli(args):
    print("=" * 60)
    print("AutoClipper Studio - Automated AI Vertical Video Cutter")
    print("=" * 60)

    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_path):
        video_files = [input_path]
    elif os.path.isdir(input_path):
        video_files = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))
        ]
    else:
        print(f"Error: Input path '{input_path}' does not exist.")
        sys.exit(1)

    if not video_files:
        print(f"No valid video files found in '{input_path}'.")
        sys.exit(0)

    res_parts = args.res.split('x')
    target_res = (int(res_parts[0]), int(res_parts[1]))

    print(f"Found {len(video_files)} video file(s) to process.")
    print(f"Clip Duration: {args.duration if args.duration else 'Full Video'}")
    print(f"Editing Mode: {args.mode}")
    print(f"Gaming Cam Pos: {args.cam_pos}")
    print(f"Smart Speech Cut: {args.smart_cut}")
    print(f"Auto Subtitles: {args.subtitles}")
    print(f"Watermark Top Crop: {int(args.top_crop * 100)}%")
    print(f"Watermark Bottom Crop: {int(args.bottom_crop * 100)}%")
    print(f"Output Resolution: {target_res[0]}x{target_res[1]}")
    print(f"Hardware Encoder: {args.encoder}")
    print(f"Performance Preset: {args.preset.upper()}")
    print(f"Subject Detector: {args.detector} (Model: {args.model if args.model else 'Auto'})")
    print("-" * 60)

    tracker = FaceTracker(model_path=args.model, detector_type=args.detector)
    processor = VideoProcessor(tracker=tracker)

    for idx, vid_path in enumerate(video_files):
        print(f"\nProcessing [{idx+1}/{len(video_files)}]: {os.path.basename(vid_path)}")
        try:
            processor.process_video_to_clips(
                video_path=vid_path,
                output_dir=output_dir,
                clip_duration=args.duration if args.duration > 0 else None,
                top_crop_pct=args.top_crop,
                bottom_crop_pct=args.bottom_crop,
                mode=args.mode,
                cam_pos=args.cam_pos,
                smart_cut=args.smart_cut,
                auto_subtitles=args.subtitles,
                smoothing_alpha=args.smooth,
                target_resolution=target_res,
                encoder=args.encoder,
                speed_preset=args.preset,
                status_callback=lambda msg: print(f"  [Status] {msg}")
            )
        except Exception as e:
            print(f"  [Error] Failed to process {vid_path}: {e}")

    print("\n" + "=" * 60)
    print(f"Batch processing completed! All clips saved in: {output_dir}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="AutoClipper AI Studio - Video Face Tracking & Vertical Cutter")
    parser.add_argument("--cli", action="store_true", help="Run in CLI headless batch mode")
    parser.add_argument("--web", action="store_true", help="Launch Gradio Web & Cloud UI")
    parser.add_argument("--input", default="Input", help="Input video file or folder (default: Input)")
    parser.add_argument("--output", default="Output", help="Output directory (default: Output)")
    parser.add_argument("--duration", type=float, default=30.0, help="Target clip duration in seconds (default: 30)")
    parser.add_argument("--mode", choices=["crop", "fit", "fit_blur", "podcast", "gaming", "blurred"], default="crop", help="Editing mode: crop, fit (letterbox), fit_blur (blurred BG), podcast, gaming (default: crop)")
    parser.add_argument("--cam-pos", choices=["top", "bottom"], default="top", help="Gaming mode webcam position (default: top)")
    parser.add_argument("--smart-cut", action="store_true", help="Cut clips at audio silence / speech pauses")
    parser.add_argument("--subtitles", action="store_true", help="Generate AI auto-subtitles with Whisper")
    parser.add_argument("--top-crop", type=float, default=0.05, help="Top watermark crop fraction 0.0-0.3 (default: 0.05)")
    parser.add_argument("--bottom-crop", type=float, default=0.10, help="Bottom watermark crop fraction 0.0-0.3 (default: 0.10)")
    parser.add_argument("--res", default="1080x1920", help="Target resolution WxH, e.g., 1080x1080, 1080x1920, 1920x1080 (default: 1080x1920)")
    parser.add_argument("--smooth", type=float, default=0.15, help="Tracking smoothing alpha 0.05-0.5 (default: 0.15)")
    parser.add_argument("--encoder", choices=["auto", "h264_nvenc", "h264_qsv", "h264_amf", "h264_mf", "libx264"], default="auto", help="Video encoder engine (default: auto GPU detection)")
    parser.add_argument("--preset", choices=["fast", "quality", "ultra_fast"], default="ultra_fast", help="Performance speed preset (default: ultra_fast)")
    parser.add_argument("--detector", choices=["auto", "yolo", "yunet"], default="auto", help="Subject detector type (default: auto YOLO/YuNet)")
    parser.add_argument("--model", default=None, help="Specific model file name or path (e.g. yolo26n.pt, yolo26n-pose.pt, yolo12n.pt, yunet.onnx)")

    args = parser.parse_args()

    if args.web:
        from web_app import main as launch_web
        launch_web()
    elif args.cli:
        run_cli(args)
    else:
        from gui import AutoClipperGUI
        app = AutoClipperGUI(input_dir=args.input, output_dir=args.output)
        app.mainloop()

if __name__ == "__main__":
    main()
