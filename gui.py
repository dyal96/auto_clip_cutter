import os
import sys
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import customtkinter as ctk
from tkinter import filedialog, messagebox

from clipper_engine import VideoProcessor, FaceTracker, calculate_crop_window, get_available_gpu_encoders

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SettingsHubWindow(ctk.CTkToplevel):
    """
    Unified settings hub with three tabs:
      1. AI Models  – Download / manage YOLO26 / YOLO12 / YuNet models
      2. Tools      – Download yt-dlp & ffmpeg executables
      3. Download Video – Download YouTube / web videos to Input/ folder via yt-dlp
    """
    TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")

    TOOLS_CATALOG = {
        "yt-dlp.exe": {
            "name": "yt-dlp",
            "description": "YouTube & 1000+ site video downloader (CLI binary)",
            "url": "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe",
            "size_mb": 12.0,
        },
        "ffmpeg.exe": {
            "name": "FFmpeg",
            "description": "Video/audio conversion engine used for all rendering",
            "url": "https://github.com/yt-dlp/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip",
            "size_mb": 85.0,
            "is_zip": True,
            "zip_extract_glob": "*/bin/ffmpeg.exe",
        },
        "ffprobe.exe": {
            "name": "FFprobe",
            "description": "Bundled with FFmpeg – video metadata inspection tool",
            "url": None,  # obtained from same zip as ffmpeg
            "zip_extract_glob": "*/bin/ffprobe.exe",
            "size_mb": 0,
        },
    }

    def __init__(self, parent, initial_tab="models"):
        super().__init__(parent)
        self.parent = parent
        self.title("AutoClipper Settings Hub – Models, Tools & Video Downloader")
        self.geometry("700x650")
        self.resizable(True, True)
        self.minsize(660, 540)
        self.attributes("-topmost", True)
        os.makedirs(self.TOOLS_DIR, exist_ok=True)

        self._active_downloads = 0
        self.setup_ui(initial_tab)

    # ------------------------------------------------------------------ #
    # UI                                                                   #
    # ------------------------------------------------------------------ #
    def setup_ui(self, initial_tab="models"):
        self.tab_view = ctk.CTkTabview(self, anchor="nw")
        self.tab_view.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_models  = self.tab_view.add("🤖 AI Models")
        self.tab_tools   = self.tab_view.add("🔧 Tools")
        self.tab_dl      = self.tab_view.add("📺 Download Video")
        self.tab_project = self.tab_view.add("🎬 Project Prep")

        self._build_models_tab()
        self._build_tools_tab()
        self._build_dl_tab()
        self._build_project_tab()

        tab_map = {"models": "🤖 AI Models", "tools": "🔧 Tools",
                   "video": "📺 Download Video", "project": "🎬 Project Prep"}
        self.tab_view.set(tab_map.get(initial_tab, "🎬 Project Prep"))

    # -------- AI Models tab -------------------------------------------- #
    def _build_models_tab(self):
        tab = self.tab_models
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        ctk.CTkLabel(tab, text="Download AI detection models directly from Ultralytics GitHub releases:",
                     font=ctk.CTkFont(size=11), text_color="gray70").grid(row=0, column=0, sticky="w", padx=5, pady=(5, 2))

        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.grid(row=1, column=0, sticky="nsew", padx=0, pady=4)
        scroll.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        from clipper_engine import AVAILABLE_MODELS_CATALOG, list_installed_models
        installed = list_installed_models()

        self.model_status_labels = {}
        self.model_btn_map = {}

        for row_idx, (filename, meta) in enumerate(AVAILABLE_MODELS_CATALOG.items()):
            self._make_item_row(scroll, row_idx, filename, meta, installed,
                                label_store=self.model_status_labels,
                                btn_store=self.model_btn_map,
                                on_download=self._start_model_download)

        # Footer bar
        footer = ctk.CTkFrame(tab, fg_color="transparent", height=52)
        footer.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        footer.grid_columnconfigure(0, weight=1)
        self.model_progress = ctk.CTkProgressBar(footer)
        self.model_progress.grid(row=0, column=0, sticky="ew", padx=5, pady=(4, 2))
        self.model_progress.set(0)
        self.model_status_lbl = ctk.CTkLabel(footer, text="Select a model to download → saved in models/ folder",
                                             font=ctk.CTkFont(size=11), text_color="gray70")
        self.model_status_lbl.grid(row=1, column=0, sticky="w", padx=6)

    # -------- Tools tab ------------------------------------------------- #
    def _build_tools_tab(self):
        tab = self.tab_tools
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(tab, text="Download essential tools required by AutoClipper:",
                     font=ctk.CTkFont(size=11), text_color="gray70").grid(row=0, column=0, sticky="w", padx=5, pady=(5, 8))

        self.tool_status_labels = {}
        self.tool_btn_map = {}

        installed_tools = self._get_installed_tools()

        tool_rows = [k for k, v in self.TOOLS_CATALOG.items() if v.get("url")]  # skip ffprobe (auto)
        for row_idx, filename in enumerate(tool_rows):
            meta = self.TOOLS_CATALOG[filename]
            is_inst, install_note = self._tool_installed(filename)
            item = ctk.CTkFrame(tab, corner_radius=8)
            item.grid(row=row_idx + 1, column=0, padx=5, pady=5, sticky="ew")
            item.grid_columnconfigure(0, weight=1)

            info = ctk.CTkFrame(item, fg_color="transparent")
            info.grid(row=0, column=0, padx=10, pady=8, sticky="w")
            ctk.CTkLabel(info, text=f"{meta['name']}  ({filename})", font=ctk.CTkFont(weight="bold", size=13)).pack(anchor="w")
            ctk.CTkLabel(info, text=f"{meta['description']} • ~{meta['size_mb']} MB", font=ctk.CTkFont(size=10), text_color="gray70").pack(anchor="w")
            if install_note:
                ctk.CTkLabel(info, text=f"📍 {install_note}", font=ctk.CTkFont(size=10), text_color="#4ea8de").pack(anchor="w")

            btn_box = ctk.CTkFrame(item, fg_color="transparent")
            btn_box.grid(row=0, column=1, padx=10, pady=8, sticky="e")

            st_text = "✓ Installed" if is_inst else "Not Found"
            st_color = "#2fa572" if is_inst else "gray60"
            lbl = ctk.CTkLabel(btn_box, text=st_text, font=ctk.CTkFont(size=11, weight="bold"), text_color=st_color)
            lbl.pack(side="left", padx=8)
            self.tool_status_labels[filename] = lbl

            btn = ctk.CTkButton(
                btn_box,
                text="Re-download" if is_inst else "Download",
                width=95, height=26,
                fg_color="gray30" if is_inst else "#1f538d",
                hover_color="gray40" if is_inst else "#14375e",
                command=lambda fn=filename: self._start_tool_download(fn)
            )
            btn.pack(side="left")
            self.tool_btn_map[filename] = btn

        # Note about tools path
        note_row = len(tool_rows) + 1
        ctk.CTkLabel(
            tab,
            text=f"Tools are saved to: {self.TOOLS_DIR}\nAutoClipper will use them automatically when found in 'tools/' or system PATH.",
            font=ctk.CTkFont(size=10), text_color="gray60", justify="left"
        ).grid(row=note_row, column=0, sticky="w", padx=10, pady=(10, 4))

        # Tool footer
        tools_footer = ctk.CTkFrame(tab, fg_color="transparent", height=52)
        tools_footer.grid(row=note_row + 1, column=0, sticky="ew", pady=(4, 0))
        tools_footer.grid_columnconfigure(0, weight=1)
        self.tool_progress = ctk.CTkProgressBar(tools_footer)
        self.tool_progress.grid(row=0, column=0, sticky="ew", padx=5, pady=(4, 2))
        self.tool_progress.set(0)
        self.tool_status_lbl = ctk.CTkLabel(tools_footer, text="Select a tool to download",
                                            font=ctk.CTkFont(size=11), text_color="gray70")
        self.tool_status_lbl.grid(row=1, column=0, sticky="w", padx=6)

    # -------- Download Video tab ---------------------------------------- #
    def _build_dl_tab(self):
        tab = self.tab_dl
        tab.grid_columnconfigure(0, weight=1)

        # Heading
        ctk.CTkLabel(tab, text="📺 Download any YouTube or web video into your Input/ folder",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        ctk.CTkLabel(tab, text="Requires yt-dlp  (download it from the 🔧 Tools tab if missing)",
                     font=ctk.CTkFont(size=11), text_color="gray60").grid(row=1, column=0, sticky="w", padx=8, pady=(0, 10))

        # URL input row
        url_frame = ctk.CTkFrame(tab, fg_color="transparent")
        url_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        url_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(url_frame, text="Video URL:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.url_entry = ctk.CTkEntry(url_frame, placeholder_text="https://www.youtube.com/watch?v=...", height=34)
        self.url_entry.grid(row=1, column=0, sticky="ew")

        # Quality options
        quality_frame = ctk.CTkFrame(tab, fg_color="transparent")
        quality_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=(8, 4))
        quality_frame.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(quality_frame, text="Quality:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", columnspan=3, pady=(0, 4))
        self.quality_var = ctk.CTkSegmentedButton(
            quality_frame,
            values=["Best", "1080p", "720p", "480p", "Audio Only (MP3)"],
        )
        self.quality_var.set("Best")
        self.quality_var.grid(row=1, column=0, columnspan=3, sticky="ew")

        # Output folder indicator
        output_info = ctk.CTkFrame(tab, fg_color="transparent")
        output_info.grid(row=4, column=0, sticky="ew", padx=8, pady=(8, 4))
        output_info.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(output_info, text="Output Folder:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w")
        self.dl_output_label = ctk.CTkLabel(output_info, text="...", font=ctk.CTkFont(size=11), text_color="#4ea8de")
        self.dl_output_label.grid(row=1, column=0, sticky="w")
        # Will be set when parent is available

        # Cookie / custom args
        adv_frame = ctk.CTkFrame(tab, corner_radius=8)
        adv_frame.grid(row=5, column=0, sticky="ew", padx=8, pady=(4, 8))
        adv_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(adv_frame, text="Extra yt-dlp args (optional):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))
        self.extra_args_entry = ctk.CTkEntry(adv_frame, placeholder_text='e.g. --cookies-from-browser chrome  or  --playlist-start 1', height=30)
        self.extra_args_entry.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

        # Download button
        self.dl_btn = ctk.CTkButton(
            tab, text="⬇️  Download Video to Input/", height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1f538d", hover_color="#14375e",
            command=self._start_video_download
        )
        self.dl_btn.grid(row=6, column=0, sticky="ew", padx=8, pady=4)

        # Progress
        dl_footer = ctk.CTkFrame(tab, fg_color="transparent")
        dl_footer.grid(row=7, column=0, sticky="ew", padx=8, pady=(4, 0))
        dl_footer.grid_columnconfigure(0, weight=1)
        self.dl_progress = ctk.CTkProgressBar(dl_footer)
        self.dl_progress.grid(row=0, column=0, sticky="ew", pady=(4, 2))
        self.dl_progress.set(0)
        self.dl_status_lbl = ctk.CTkLabel(dl_footer, text="Paste a URL above and press Download",
                                          font=ctk.CTkFont(size=11), text_color="gray70")
        self.dl_status_lbl.grid(row=1, column=0, sticky="w")

        # Log box
        ctk.CTkLabel(tab, text="Download Log:", font=ctk.CTkFont(weight="bold", size=11)).grid(row=8, column=0, sticky="w", padx=8, pady=(8, 2))
        self.dl_log = ctk.CTkTextbox(tab, height=110, font=ctk.CTkFont(family="Consolas", size=10),
                                     text_color="#00FF66", fg_color="black")
        self.dl_log.grid(row=9, column=0, sticky="ew", padx=8, pady=(0, 8))

        # Set output label
        try:
            input_dir = os.path.abspath(self.parent.input_dir)
            self.dl_output_label.configure(text=input_dir)
        except Exception:
            self.dl_output_label.configure(text="Input/ folder")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _make_item_row(self, parent, row_idx, filename, meta, installed_list,
                       label_store, btn_store, on_download):
        item = ctk.CTkFrame(parent, corner_radius=8)
        item.grid(row=row_idx, column=0, padx=5, pady=4, sticky="ew")
        item.grid_columnconfigure(0, weight=1)

        info = ctk.CTkFrame(item, fg_color="transparent")
        info.grid(row=0, column=0, padx=10, pady=8, sticky="w")
        ctk.CTkLabel(info, text=f"{meta['name']}  ({filename})", font=ctk.CTkFont(weight="bold", size=12)).pack(anchor="w")
        ctk.CTkLabel(info, text=f"{meta['description']} • ~{meta['size_mb']} MB", font=ctk.CTkFont(size=10), text_color="gray70").pack(anchor="w")

        btn_box = ctk.CTkFrame(item, fg_color="transparent")
        btn_box.grid(row=0, column=1, padx=10, pady=8, sticky="e")

        is_inst = filename in installed_list
        st_text = "✓ Installed" if is_inst else "Not Installed"
        st_color = "#2fa572" if is_inst else "gray60"
        lbl = ctk.CTkLabel(btn_box, text=st_text, font=ctk.CTkFont(size=11, weight="bold"), text_color=st_color)
        lbl.pack(side="left", padx=10)
        label_store[filename] = lbl

        btn_text = "Re-download" if is_inst else "Download"
        btn = ctk.CTkButton(
            btn_box, text=btn_text, width=95, height=26,
            fg_color="gray30" if is_inst else "#1f538d",
            hover_color="gray40" if is_inst else "#14375e",
            command=lambda fn=filename: on_download(fn)
        )
        btn.pack(side="left")
        btn_store[filename] = btn

    def _get_installed_tools(self):
        found = {}
        for fn in self.TOOLS_CATALOG:
            local = os.path.join(self.TOOLS_DIR, fn)
            if os.path.exists(local):
                found[fn] = local
        return found

    def _tool_installed(self, filename):
        """Returns (is_installed, location_note)"""
        local = os.path.join(self.TOOLS_DIR, filename)
        if os.path.exists(local):
            return True, f"tools/{filename}"
        # Check PATH
        import shutil
        found = shutil.which(filename.replace(".exe", ""))
        if found:
            return True, found
        return False, None

    def _find_tool(self, filename):
        """Return executable path for a tool, checking tools/ then PATH."""
        local = os.path.join(self.TOOLS_DIR, filename)
        if os.path.exists(local):
            return local
        import shutil
        found = shutil.which(filename.replace(".exe", ""))
        return found

    # ------------------------------------------------------------------ #
    # Model downloads                                                      #
    # ------------------------------------------------------------------ #
    def _start_model_download(self, filename):
        for b in self.model_btn_map.values():
            b.configure(state="disabled")
        self.model_status_lbl.configure(text=f"Downloading {filename}...")
        self.model_progress.set(0)
        threading.Thread(target=self._model_dl_worker, args=(filename,), daemon=True).start()

    def _model_dl_worker(self, filename):
        from clipper_engine import download_model_file
        try:
            download_model_file(
                filename,
                progress_callback=lambda p: self.after(0, lambda p=p: self.model_progress.set(p)),
                status_callback=lambda msg: self.after(0, lambda msg=msg: self.model_status_lbl.configure(text=msg))
            )
            self.after(0, self._on_model_success, filename)
        except Exception as e:
            self.after(0, self._on_model_error, filename, str(e))

    def _on_model_success(self, filename):
        for b in self.model_btn_map.values():
            b.configure(state="normal")
        self.model_progress.set(1.0)
        self.model_status_lbl.configure(text=f"✅ {filename} downloaded successfully!")
        if filename in self.model_status_labels:
            self.model_status_labels[filename].configure(text="✓ Installed", text_color="#2fa572")
        if filename in self.model_btn_map:
            self.model_btn_map[filename].configure(text="Re-download", fg_color="gray30", hover_color="gray40")
        self.parent.log(f"Model downloaded: {filename} → models/")
        self.parent.refresh_model_dropdown(select_model=filename)

    def _on_model_error(self, filename, err):
        for b in self.model_btn_map.values():
            b.configure(state="normal")
        self.model_status_lbl.configure(text=f"❌ Error downloading {filename}")
        messagebox.showerror("Download Error", f"Failed to download {filename}:\n{err}", parent=self)

    # ------------------------------------------------------------------ #
    # Tool downloads (yt-dlp, ffmpeg)                                     #
    # ------------------------------------------------------------------ #
    def _start_tool_download(self, filename):
        for b in self.tool_btn_map.values():
            b.configure(state="disabled")
        self.tool_status_lbl.configure(text=f"Downloading {filename}...")
        self.tool_progress.set(0)
        threading.Thread(target=self._tool_dl_worker, args=(filename,), daemon=True).start()

    def _tool_dl_worker(self, filename):
        import urllib.request, zipfile, glob, shutil
        meta = self.TOOLS_CATALOG.get(filename, {})
        url = meta.get("url")
        is_zip = meta.get("is_zip", False)
        zip_glob = meta.get("zip_extract_glob", "")

        os.makedirs(self.TOOLS_DIR, exist_ok=True)
        target = os.path.join(self.TOOLS_DIR, filename)
        tmp = target + ".tmp"

        def set_status(msg):
            self.after(0, lambda msg=msg: self.tool_status_lbl.configure(text=msg))
        def set_prog(p):
            self.after(0, lambda p=p: self.tool_progress.set(p))

        try:
            set_status(f"Connecting to download {filename}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
                total = int(resp.getheader("content-length") or 0)
                done = 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    done += len(chunk)
                    f.write(chunk)
                    if total > 0:
                        set_prog(done / total)
                    set_status(f"Downloading {filename}... {done // 1024 // 1024} MB")

            if is_zip and zip_glob:
                set_status("Extracting from zip archive...")
                with zipfile.ZipFile(tmp, "r") as zf:
                    for member in zf.namelist():
                        import fnmatch
                        if fnmatch.fnmatch(member, zip_glob):
                            # Extract matching file flat into TOOLS_DIR
                            extracted_name = os.path.basename(member)
                            out_path = os.path.join(self.TOOLS_DIR, extracted_name)
                            with zf.open(member) as src, open(out_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            set_status(f"Extracted {extracted_name}")
                        # Also extract ffprobe if found
                        ffprobe_glob = self.TOOLS_CATALOG.get("ffprobe.exe", {}).get("zip_extract_glob", "")
                        if ffprobe_glob and fnmatch.fnmatch(member, ffprobe_glob):
                            out_path = os.path.join(self.TOOLS_DIR, "ffprobe.exe")
                            with zf.open(member) as src, open(out_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                os.remove(tmp)
            else:
                if os.path.exists(target):
                    os.remove(target)
                os.rename(tmp, target)

            self.after(0, self._on_tool_success, filename)
        except Exception as e:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            self.after(0, self._on_tool_error, filename, str(e))

    def _on_tool_success(self, filename):
        for b in self.tool_btn_map.values():
            b.configure(state="normal")
        self.tool_progress.set(1.0)

        # Update status labels
        for fn in list(self.tool_status_labels.keys()):
            is_inst, note = self._tool_installed(fn)
            if is_inst:
                self.tool_status_labels[fn].configure(text="✓ Installed", text_color="#2fa572")
                self.tool_btn_map[fn].configure(text="Re-download", fg_color="gray30", hover_color="gray40")

        self.tool_status_lbl.configure(text=f"✅ {filename} installed in tools/ folder!")
        self.parent.log(f"Tool installed: {filename} → tools/")

        if filename == "ffmpeg.exe":
            messagebox.showinfo("FFmpeg Installed",
                                f"FFmpeg has been installed to:\n{self.TOOLS_DIR}\n\nRestart AutoClipper to use it automatically.",
                                parent=self)

    def _on_tool_error(self, filename, err):
        for b in self.tool_btn_map.values():
            b.configure(state="normal")
        self.tool_status_lbl.configure(text=f"❌ Error: {filename}")
        messagebox.showerror("Tool Download Error", f"Failed to download {filename}:\n{err}", parent=self)

    # ------------------------------------------------------------------ #
    # Video download (yt-dlp → Input/)                                    #
    # ------------------------------------------------------------------ #
    def _start_video_download(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("No URL", "Please enter a video URL first.", parent=self)
            return

        ytdlp_path = self._find_tool("yt-dlp.exe")
        if not ytdlp_path:
            if messagebox.askyesno("yt-dlp Not Found",
                                   "yt-dlp is not installed.\nDownload it now from the 🔧 Tools tab?", parent=self):
                self.tab_view.set("🔧 Tools")
            return

        self.dl_btn.configure(state="disabled", text="⏳ Downloading...")
        self.dl_progress.set(0)
        self.dl_log.delete("1.0", "end")
        self.dl_status_lbl.configure(text="Starting download...")

        quality = self.quality_var.get()
        extra = self.extra_args_entry.get().strip()
        try:
            input_dir = os.path.abspath(self.parent.input_dir)
        except Exception:
            input_dir = os.path.abspath("Input")

        threading.Thread(
            target=self._video_dl_worker,
            args=(url, ytdlp_path, quality, extra, input_dir),
            daemon=True
        ).start()

    def _video_dl_worker(self, url, ytdlp_path, quality, extra_args, input_dir):
        import subprocess, re

        quality_map = {
            "Best":           ["-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"],
            "1080p":          ["-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]"],
            "720p":           ["-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]"],
            "480p":           ["-f", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]"],
            "Audio Only (MP3)": ["-f", "bestaudio", "--extract-audio", "--audio-format", "mp3"],
        }
        fmt_args = quality_map.get(quality, [])

        ffmpeg_path = self._find_tool("ffmpeg.exe")
        ffmpeg_args = ["--ffmpeg-location", ffmpeg_path] if ffmpeg_path else []

        cmd = [ytdlp_path] + fmt_args + ffmpeg_args + [
            "--merge-output-format", "mp4",
            "--output", os.path.join(input_dir, "%(title).80s.%(ext)s"),
            "--newline",
            "--progress",
        ]

        if extra_args:
            import shlex
            try:
                cmd += shlex.split(extra_args)
            except ValueError:
                cmd += extra_args.split()

        cmd.append(url)

        def log(msg):
            self.after(0, lambda msg=msg: (
                self.dl_log.insert("end", msg + "\n"),
                self.dl_log.see("end")
            ))
        def set_status(msg):
            self.after(0, lambda msg=msg: self.dl_status_lbl.configure(text=msg))
        def set_prog(p):
            self.after(0, lambda p=p: self.dl_progress.set(p))

        try:
            log(f"▶ Running: {' '.join(cmd)}\n")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace",
                                    bufsize=1)

            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    log(line)
                    # Parse progress percentage
                    m = re.search(r"(\d+\.\d+)%", line)
                    if m:
                        set_prog(float(m.group(1)) / 100.0)
                        set_status(f"Downloading… {m.group(1)}%")
                    elif "Destination" in line or "[download]" in line:
                        set_status(line[:80])
                    elif "Merging" in line:
                        set_status("Merging video+audio…")

            proc.wait()
            if proc.returncode == 0:
                self.after(0, self._on_video_dl_success, input_dir)
            else:
                self.after(0, self._on_video_dl_error, "yt-dlp exited with error. Check the log above.")
        except Exception as e:
            self.after(0, self._on_video_dl_error, str(e))

    def _on_video_dl_success(self, input_dir):
        self.dl_btn.configure(state="normal", text="⬇️  Download Video to Input/")
        self.dl_progress.set(1.0)
        self.dl_status_lbl.configure(text="✅ Download complete! Video saved to Input/ folder.")
        self.parent.log(f"Video downloaded → {input_dir}")
        self.parent.refresh_input_files()

    def _on_video_dl_error(self, err):
        self.dl_btn.configure(state="normal", text="⬇️  Download Video to Input/")
        self.dl_status_lbl.configure(text=f"❌ Download failed")
        self.dl_log.insert("end", f"\n❌ ERROR: {err}\n")
        self.dl_log.see("end")

    # ------------------------------------------------------------------ #
    # 🎬 Project Prep tab                                                  #
    # ------------------------------------------------------------------ #
    def _build_project_tab(self):
        tab = self.tab_project
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            tab,
            text="🎬 One-Click Project Prep  —  Download → Trim → Subtitles → Export",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(10, 2))
        ctk.CTkLabel(
            tab,
            text="Download a video, optionally trim a range, generate an SRT subtitle file, and pack everything into an organised project folder.",
            font=ctk.CTkFont(size=11), text_color="gray60", wraplength=640, justify="left"
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))

        # ── Step 1 : URL ─────────────────────────────────────────────────
        s1 = ctk.CTkFrame(tab, corner_radius=8)
        s1.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        s1.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(s1, text="① Source", font=ctk.CTkFont(weight="bold", size=12)).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))

        ctk.CTkLabel(s1, text="YouTube URL:", font=ctk.CTkFont(size=11)).grid(row=1, column=0, sticky="w", padx=12, pady=2)
        self.proj_url_entry = ctk.CTkEntry(s1, placeholder_text="https://www.youtube.com/watch?v=...", height=30)
        self.proj_url_entry.grid(row=1, column=1, sticky="ew", padx=(4, 12), pady=2)

        ctk.CTkLabel(s1, text="Quality:", font=ctk.CTkFont(size=11)).grid(row=2, column=0, sticky="w", padx=12, pady=(2, 8))
        self.proj_quality = ctk.CTkSegmentedButton(s1, values=["Best", "1080p", "720p", "480p"])
        self.proj_quality.set("1080p")
        self.proj_quality.grid(row=2, column=1, sticky="w", padx=(4, 12), pady=(2, 8))

        # ── Step 2 : Trim ─────────────────────────────────────────────────
        s2 = ctk.CTkFrame(tab, corner_radius=8)
        s2.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        s2.grid_columnconfigure((1, 3), weight=1)
        ctk.CTkLabel(s2, text="② Trim Range  (leave blank = keep full video)",
                     font=ctk.CTkFont(weight="bold", size=12)).grid(row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(8, 4))

        ctk.CTkLabel(s2, text="Start:", font=ctk.CTkFont(size=11)).grid(row=1, column=0, sticky="w", padx=12, pady=(2, 8))
        self.proj_trim_start = ctk.CTkEntry(s2, placeholder_text="00:00:00  or  30", width=140, height=28)
        self.proj_trim_start.grid(row=1, column=1, sticky="w", padx=(4, 16))

        ctk.CTkLabel(s2, text="End:", font=ctk.CTkFont(size=11)).grid(row=1, column=2, sticky="w", padx=(0, 4))
        self.proj_trim_end = ctk.CTkEntry(s2, placeholder_text="00:05:00  or  300", width=140, height=28)
        self.proj_trim_end.grid(row=1, column=3, sticky="w", padx=(4, 12), pady=(2, 8))

        # ── Step 3 : Subtitles ───────────────────────────────────────────
        s3 = ctk.CTkFrame(tab, corner_radius=8)
        s3.grid(row=4, column=0, sticky="ew", padx=8, pady=4)
        s3.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(s3, text="③ Subtitle Export",
                     font=ctk.CTkFont(weight="bold", size=12)).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 4))

        self.proj_sub_switch = ctk.CTkSwitch(s3, text="Generate SRT subtitle file (Whisper AI)",
                                             font=ctk.CTkFont(size=12))
        self.proj_sub_switch.select()
        self.proj_sub_switch.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 4))

        sub_fmt_box = ctk.CTkFrame(s3, fg_color="transparent")
        sub_fmt_box.grid(row=2, column=0, sticky="w", padx=12, pady=(0, 8))
        ctk.CTkLabel(sub_fmt_box, text="Format:", font=ctk.CTkFont(size=11)).pack(side="left")
        self.proj_sub_fmt = ctk.CTkSegmentedButton(sub_fmt_box, values=["SRT", "VTT", "ASS", "All"])
        self.proj_sub_fmt.set("SRT")
        self.proj_sub_fmt.pack(side="left", padx=8)

        # ── Step 4 : Project name / folder ───────────────────────────────
        s4 = ctk.CTkFrame(tab, corner_radius=8)
        s4.grid(row=5, column=0, sticky="ew", padx=8, pady=4)
        s4.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(s4, text="④ Project Name",
                     font=ctk.CTkFont(weight="bold", size=12)).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))
        ctk.CTkLabel(s4, text="Folder name:", font=ctk.CTkFont(size=11)).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 8))
        self.proj_name_entry = ctk.CTkEntry(s4, placeholder_text="my_project  (auto-filled from video title)", height=28)
        self.proj_name_entry.grid(row=1, column=1, sticky="ew", padx=(4, 12), pady=(0, 8))

        # ── Action button ─────────────────────────────────────────────────
        self.proj_btn = ctk.CTkButton(
            tab, text="🚀  Prepare Project", height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#2d6a4f", hover_color="#1b4332",
            command=self._start_project_prep
        )
        self.proj_btn.grid(row=6, column=0, sticky="ew", padx=8, pady=(6, 4))

        # ── Progress + log ────────────────────────────────────────────────
        proj_footer = ctk.CTkFrame(tab, fg_color="transparent")
        proj_footer.grid(row=7, column=0, sticky="ew", padx=8)
        proj_footer.grid_columnconfigure(0, weight=1)
        self.proj_progress = ctk.CTkProgressBar(proj_footer)
        self.proj_progress.grid(row=0, column=0, sticky="ew", pady=(4, 2))
        self.proj_progress.set(0)
        self.proj_status_lbl = ctk.CTkLabel(proj_footer, text="Fill in the fields above and press Prepare Project",
                                            font=ctk.CTkFont(size=11), text_color="gray70")
        self.proj_status_lbl.grid(row=1, column=0, sticky="w")

        ctk.CTkLabel(tab, text="Log:", font=ctk.CTkFont(weight="bold", size=11)).grid(row=8, column=0, sticky="w", padx=8, pady=(6, 2))
        self.proj_log = ctk.CTkTextbox(tab, height=95, font=ctk.CTkFont(family="Consolas", size=10),
                                       text_color="#00FF66", fg_color="black")
        self.proj_log.grid(row=9, column=0, sticky="ew", padx=8, pady=(0, 8))

    # ── Project Prep helpers ──────────────────────────────────────────────
    @staticmethod
    def _parse_timestamp(s):
        """Parse 'HH:MM:SS', 'MM:SS', or plain seconds string → float seconds."""
        s = s.strip()
        if not s:
            return None
        if ":" in s:
            parts = s.split(":")
            secs = 0.0
            for p in parts:
                secs = secs * 60 + float(p)
            return secs
        return float(s)

    def _proj_log(self, msg):
        self.after(0, lambda msg=msg: (
            self.proj_log.insert("end", msg + "\n"),
            self.proj_log.see("end")
        ))

    def _proj_status(self, msg, prog=None):
        self.after(0, lambda msg=msg: self.proj_status_lbl.configure(text=msg))
        if prog is not None:
            self.after(0, lambda p=prog: self.proj_progress.set(p))

    def _start_project_prep(self):
        url = self.proj_url_entry.get().strip()
        trim_start_str = self.proj_trim_start.get().strip()
        trim_end_str   = self.proj_trim_end.get().strip()
        gen_subs = self.proj_sub_switch.get() == 1
        sub_fmt  = self.proj_sub_fmt.get()
        proj_name = self.proj_name_entry.get().strip()
        quality  = self.proj_quality.get()

        if not url:
            messagebox.showwarning("No URL", "Enter a YouTube / web URL first.", parent=self)
            return

        ytdlp_path = self._find_tool("yt-dlp.exe")
        if not ytdlp_path:
            if messagebox.askyesno("yt-dlp Not Found",
                                   "yt-dlp is required.\nSwitch to 🔧 Tools tab to download it?", parent=self):
                self.tab_view.set("🔧 Tools")
            return

        trim_start = self._parse_timestamp(trim_start_str)
        trim_end   = self._parse_timestamp(trim_end_str)

        self.proj_btn.configure(state="disabled", text="⏳ Working...")
        self.proj_log.delete("1.0", "end")
        self.proj_progress.set(0)

        try:
            input_dir = os.path.abspath(self.parent.input_dir)
            output_dir = os.path.abspath(self.parent.output_dir)
        except Exception:
            input_dir  = os.path.abspath("Input")
            output_dir = os.path.abspath("Output")

        threading.Thread(
            target=self._project_prep_worker,
            args=(url, ytdlp_path, quality, trim_start, trim_end,
                  gen_subs, sub_fmt, proj_name, input_dir, output_dir),
            daemon=True
        ).start()

    def _project_prep_worker(self, url, ytdlp_path, quality, trim_start, trim_end,
                             gen_subs, sub_fmt, proj_name, input_dir, output_dir):
        import subprocess, re, shutil

        def log(m):  self._proj_log(m)
        def status(m, p=None): self._proj_status(m, p)

        # ── 1. Download ───────────────────────────────────────────────────
        status("Step 1/4 — Downloading video...", 0.05)
        log("\n▶ Step 1: Downloading from URL...")

        quality_map = {
            "Best":  "-f bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "1080p": "-f bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]",
            "720p":  "-f bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
            "480p":  "-f bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]",
        }
        fmt_str = quality_map.get(quality, quality_map["1080p"])

        ffmpeg_path = self._find_tool("ffmpeg.exe")
        ffmpeg_args = ["--ffmpeg-location", ffmpeg_path] if ffmpeg_path else []

        # Use a temp download name pattern; capture actual filename
        dl_template = os.path.join(input_dir, "%(title).100s.%(ext)s")
        cmd_dl = (
            [ytdlp_path] + fmt_str.split() + ffmpeg_args +
            ["--merge-output-format", "mp4",
             "--output", dl_template,
             "--print", "after_move:filepath",
             "--newline", url]
        )

        downloaded_path = None
        title_guess = proj_name or "project"
        try:
            proc = subprocess.Popen(
                cmd_dl, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1
            )
            for line in proc.stdout:
                line = line.rstrip()
                log(line)
                if line and os.path.exists(line) and line.endswith(".mp4"):
                    downloaded_path = line
                m = re.search(r"(\d+\.\d+)%", line)
                if m:
                    status(f"Downloading… {m.group(1)}%", float(m.group(1)) / 100 * 0.30)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError("yt-dlp exited with error – check log.")
        except Exception as e:
            self.after(0, self._proj_done_error, str(e))
            return

        # Fallback: find newest mp4 in input_dir
        if not downloaded_path or not os.path.exists(downloaded_path):
            mp4s = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".mp4")]
            if mp4s:
                downloaded_path = max(mp4s, key=os.path.getmtime)

        if not downloaded_path or not os.path.exists(downloaded_path):
            self.after(0, self._proj_done_error, "Could not find downloaded video file.")
            return

        log(f"\n✅ Downloaded: {downloaded_path}")
        status("Step 2/4 — Setting up project folder...", 0.32)

        # ── 2. Create project folder ──────────────────────────────────────
        base_title = proj_name or re.sub(r'[<>:"/\\|?*]', '_', os.path.splitext(os.path.basename(downloaded_path))[0])[:60]
        proj_dir   = os.path.join(output_dir, base_title)
        os.makedirs(proj_dir, exist_ok=True)
        log(f"\n📁 Project folder: {proj_dir}")

        # Copy original into project folder
        orig_dest = os.path.join(proj_dir, "original_" + os.path.basename(downloaded_path))
        shutil.copy2(downloaded_path, orig_dest)
        log(f"   Copied original → {os.path.basename(orig_dest)}")

        # ── 3. Trim (if requested) ────────────────────────────────────────
        trimmed_path = None
        if trim_start is not None or trim_end is not None:
            status("Step 3/4 — Trimming video...", 0.50)
            log("\n✂️  Step 3: Trimming...")
            trim_out = os.path.join(proj_dir, "trimmed_" + os.path.basename(downloaded_path))

            trim_cmd = [ffmpeg_path or "ffmpeg", "-y"]
            if trim_start is not None:
                trim_cmd += ["-ss", str(trim_start)]
            trim_cmd += ["-i", downloaded_path]
            if trim_end is not None:
                duration = trim_end - (trim_start or 0)
                trim_cmd += ["-t", str(duration)]
            trim_cmd += ["-c", "copy", trim_out]

            try:
                r = subprocess.run(trim_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
                log(r.stdout)
                if r.returncode == 0:
                    trimmed_path = trim_out
                    log(f"   ✅ Trimmed → {os.path.basename(trim_out)}")
                else:
                    log(f"   ⚠️  Trim failed, using original.")
            except Exception as e:
                log(f"   ⚠️  Trim error: {e}")
        else:
            status("Step 3/4 — Skipping trim (no range set)", 0.55)
            log("\n⏩ Step 3: No trim range set – skipping.")

        # ── 4. Generate subtitles ─────────────────────────────────────────
        subtitle_src = trimmed_path or orig_dest
        sub_files = []
        if gen_subs:
            status("Step 4/4 — Generating subtitles (Whisper AI)...", 0.65)
            log("\n💬 Step 4: Generating subtitles with Whisper AI...")
            try:
                import whisper, torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                wmodel = whisper.load_model("base", device=device)
                result = wmodel.transcribe(subtitle_src, verbose=False, fp16=(device=="cuda"))
                segments = result.get("segments", [])

                def fmt_t(secs, vtt=False):
                    h = int(secs // 3600)
                    m = int((secs % 3600) // 60)
                    s = int(secs % 60)
                    ms = int((secs - int(secs)) * 1000)
                    sep = "." if vtt else ","
                    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"

                fmts = ["SRT", "VTT", "ASS"] if sub_fmt == "All" else [sub_fmt]

                for fmt in fmts:
                    sub_out = os.path.join(proj_dir, f"{base_title}.{fmt.lower()}")
                    with open(sub_out, "w", encoding="utf-8") as f:
                        if fmt == "SRT":
                            for i, seg in enumerate(segments, 1):
                                text = seg["text"].strip()
                                f.write(f"{i}\n{fmt_t(seg['start'])} --> {fmt_t(seg['end'])}\n{text}\n\n")
                        elif fmt == "VTT":
                            f.write("WEBVTT\n\n")
                            for i, seg in enumerate(segments, 1):
                                text = seg["text"].strip()
                                f.write(f"{i}\n{fmt_t(seg['start'], True)} --> {fmt_t(seg['end'], True)}\n{text}\n\n")
                        elif fmt == "ASS":
                            f.write("[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n\n[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,Bold,Alignment\nStyle: Default,Arial,36,&H00FFFFFF,-1,2\n\n[Events]\nFormat: Layer,Start,End,Style,Name,Text\n")
                            for seg in segments:
                                text = seg["text"].strip().replace("\n"," ")
                                f.write(f"Dialogue: 0,{fmt_t(seg['start'])},{fmt_t(seg['end'])},Default,,{text}\n")
                    sub_files.append(sub_out)
                    log(f"   ✅ {fmt} subtitles → {os.path.basename(sub_out)}")
            except ImportError:
                log("   ⚠️  Whisper not installed. Skipping subtitle generation.")
            except Exception as e:
                log(f"   ⚠️  Subtitle error: {e}")
        else:
            status("Step 4/4 — Skipping subtitles", 0.80)
            log("\n⏩ Step 4: Subtitle generation disabled.")

        # ── 5. Write project notes README ────────────────────────────────
        readme = os.path.join(proj_dir, "PROJECT_NOTES.txt")
        with open(readme, "w", encoding="utf-8") as f:
            f.write(f"AutoClipper Project: {base_title}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Source URL    : {url}\n")
            f.write(f"Downloaded    : {os.path.basename(orig_dest)}\n")
            if trimmed_path:
                f.write(f"Trimmed clip  : {os.path.basename(trimmed_path)}")
                if trim_start is not None: f.write(f"  (from {trim_start:.1f}s")
                if trim_end   is not None: f.write(f" to {trim_end:.1f}s")
                f.write(")\n")
            for sf in sub_files:
                f.write(f"Subtitles     : {os.path.basename(sf)}\n")
            f.write("\nFiles ready for editing in Premiere Pro / DaVinci Resolve / CapCut\n")
        log(f"\n📄 Project notes → {os.path.basename(readme)}")

        self.after(0, self._proj_done_success, proj_dir)

    def _proj_done_success(self, proj_dir):
        self.proj_btn.configure(state="normal", text="🚀  Prepare Project")
        self.proj_progress.set(1.0)
        self.proj_status_lbl.configure(text=f"✅ Project ready! → {proj_dir}")
        try:
            self.parent.log(f"Project prepared → {proj_dir}")
            self.parent.refresh_input_files()
        except Exception:
            pass
        messagebox.showinfo(
            "Project Ready!",
            f"Your project folder is ready:\n{proj_dir}\n\nContents:\n• Original video\n• Trimmed clip (if set)\n• Subtitle file(s) (if enabled)\n• PROJECT_NOTES.txt",
            parent=self
        )
        import subprocess as sp
        try:
            sp.Popen(["explorer", proj_dir])
        except Exception:
            pass

    def _proj_done_error(self, err):
        self.proj_btn.configure(state="normal", text="🚀  Prepare Project")
        self.proj_status_lbl.configure(text=f"❌ Failed")
        self._proj_log(f"\n❌ ERROR: {err}")
        messagebox.showerror("Project Prep Error", err, parent=self)


# Backwards-compat alias so existing open_model_downloader() call works
ModelDownloaderWindow = SettingsHubWindow


class AutoClipperGUI(ctk.CTk):
    def __init__(self, input_dir="Input", output_dir="Output"):
        super().__init__()

        self.title("AutoClipper - AI Studio & Face Tracking Vertical Cutter")
        self.geometry("1240x780")
        self.minsize(1050, 650)

        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Inject tools/ directory into PATH so ffmpeg/yt-dlp are auto-found
        tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
        os.makedirs(tools_dir, exist_ok=True)
        if tools_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = tools_dir + os.pathsep + os.environ.get("PATH", "")

        self.detected_encoders = get_available_gpu_encoders()

        self.tracker = FaceTracker()
        self.processor = VideoProcessor(tracker=self.tracker)

        self.selected_file = None
        self.sample_frame = None
        self.sample_faces = []
        self.is_processing = False
        self.hub_win = None
        self._log_buffer = []  # Messages queued before log_textbox is built

        self.setup_ui()
        self.refresh_input_files()

        # Log GPU Status on Startup
        top_enc_id, top_enc_label = self.detected_encoders[0]
        if "GPU" in top_enc_label:
            self.log(f"GPU Hardware Acceleration Active: {top_enc_label}")
        else:
            self.log(f"Rendering Engine: {top_enc_label}")

    def setup_ui(self):
        # ── 2-column root layout ───────────────────────────────────────────
        self.grid_columnconfigure(0, weight=1, minsize=470)
        self.grid_columnconfigure(1, weight=2, minsize=650)
        self.grid_rowconfigure(0, weight=1)

        # ══════════════════════════════════════════════════════════════════
        # LEFT PANEL
        # ══════════════════════════════════════════════════════════════════
        self.left_frame = ctk.CTkFrame(self, corner_radius=12)
        self.left_frame.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(1, weight=1)

        # ── Compact header ─────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        hdr.grid(row=0, column=0, padx=12, pady=(10, 2), sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="AutoClipper AI Studio",
                     font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(hdr, text="⚙️ Hub", width=70, height=26,
                      fg_color="gray25", hover_color="gray35",
                      font=ctk.CTkFont(size=11),
                      command=lambda: self.open_hub("models")).grid(row=0, column=1, sticky="e")

        # ── Main tab view ──────────────────────────────────────────────────
        self.left_tab = ctk.CTkTabview(self.left_frame, anchor="nw")
        self.left_tab.grid(row=1, column=0, padx=6, pady=(0, 6), sticky="nsew")

        tab_proc = self.left_tab.add("⚡ Process")
        tab_dl   = self.left_tab.add("📺 Download & Prep")
        tab_sm   = self.left_tab.add("✂️ Split & Merge")

        tab_proc.grid_columnconfigure(0, weight=1)
        tab_proc.grid_rowconfigure(1, weight=1)
        tab_dl.grid_columnconfigure(0, weight=1)
        tab_sm.grid_columnconfigure(0, weight=1)

        # ── ⚡ PROCESS TAB ─────────────────────────────────────────────────
        # File selector
        file_section = ctk.CTkFrame(tab_proc, fg_color="transparent")
        file_section.grid(row=0, column=0, padx=4, pady=(4, 2), sticky="ew")
        file_section.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(file_section, text="Input Video File:",
                     font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, sticky="w")
        self.file_dropdown = ctk.CTkOptionMenu(
            file_section, values=["No videos found"], height=28, command=self.on_file_selected
        )
        self.file_dropdown.grid(row=1, column=0, pady=(2, 0), sticky="ew")

        fb = ctk.CTkFrame(file_section, fg_color="transparent")
        fb.grid(row=2, column=0, sticky="ew")
        fb.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(fb, text="Browse File", height=26, command=self.browse_file).grid(
            row=0, column=0, padx=(0, 4), pady=4, sticky="ew")
        ctk.CTkButton(fb, text="🔄 Refresh", height=26, fg_color="gray30", hover_color="gray40",
                      command=self.refresh_input_files).grid(
            row=0, column=1, padx=(4, 0), pady=4, sticky="ew")

        # Settings compact frame (Fits 100% in Viewport without vertical scrolling)
        self.settings_frame = ctk.CTkFrame(tab_proc, fg_color="transparent")
        self.settings_frame.grid(row=1, column=0, padx=2, pady=2, sticky="nsew")
        self.settings_frame.grid_columnconfigure((0, 1), weight=1)

        sf = self.settings_frame  # shorthand

        # Row 0: Resolution & Editing Mode (2 columns)
        r0_l = ctk.CTkFrame(sf, fg_color="transparent")
        r0_l.grid(row=0, column=0, padx=(0, 4), pady=(2, 4), sticky="ew")
        r0_l.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(r0_l, text="Output Resolution / Aspect:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, sticky="w")
        self.res_menu = ctk.CTkOptionMenu(
            r0_l, values=["1080x1080 (1:1 Rectangular / Square)",
                        "1080x1920 (9:16 Vertical Short / Reel)",
                        "1920x1080 (16:9 Landscape / Standard)",
                        "720x1280 (9:16 Fast HD)"],
            height=28, font=ctk.CTkFont(size=11),
            command=self.on_setting_changed
        )
        self.res_menu.set("1080x1920 (9:16 Vertical Short / Reel)")
        self.res_menu.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        r0_r = ctk.CTkFrame(sf, fg_color="transparent")
        r0_r.grid(row=0, column=1, padx=(4, 0), pady=(2, 4), sticky="ew")
        r0_r.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(r0_r, text="Video Editing Mode:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, sticky="w")
        self.mode_menu = ctk.CTkOptionMenu(
            r0_r, values=["Smart Dynamic AI Crop (Subject Track)",
                        "Fit Video - Black Letterbox (No Crop)",
                        "Fit Video - Blurred Background (No Crop)",
                        "Podcast Stack (2 Speakers Top/Bottom)",
                        "Gaming Split (Cam + Gameplay)"],
            height=28, font=ctk.CTkFont(size=11),
            command=self.on_mode_changed
        )
        self.mode_menu.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        # Hidden gaming cam layout
        self.cam_pos_label = ctk.CTkLabel(sf, text="Gaming Cam Layout:", font=ctk.CTkFont(size=11, weight="bold"))
        self.cam_pos_menu = ctk.CTkSegmentedButton(
            sf, values=["Cam Top / Game Bot", "Game Top / Cam Bot"],
            command=self.on_setting_changed)
        self.cam_pos_menu.set("Cam Top / Game Bot")

        # Row 1: Duration & Cut Strategy
        r1 = ctk.CTkFrame(sf, fg_color="transparent")
        r1.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 4))
        r1.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(r1, text="Clip Duration & Cut Strategy:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, sticky="w")

        self.duration_menu = ctk.CTkSegmentedButton(
            r1, values=["15s", "30s", "60s", "Custom", "Full Video"],
            height=26, font=ctk.CTkFont(size=11),
            command=self.on_duration_changed
        )
        self.duration_menu.set("30s")
        self.duration_menu.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        self.custom_dur_frame = ctk.CTkFrame(r1, fg_color="transparent")
        self.custom_dur_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.custom_dur_frame, text="Custom length (sec):", font=ctk.CTkFont(size=10)).pack(side="left", padx=(0, 4))
        self.custom_dur_entry = ctk.CTkEntry(self.custom_dur_frame, width=70, height=24, placeholder_text="e.g. 45")
        self.custom_dur_entry.pack(side="left")

        # Row 2: AI Detector & Hardware Encoder (2 columns)
        r2_l = ctk.CTkFrame(sf, fg_color="transparent")
        r2_l.grid(row=4, column=0, padx=(0, 4), pady=(2, 4), sticky="ew")
        r2_l.grid_columnconfigure(0, weight=1)
        det_hdr = ctk.CTkFrame(r2_l, fg_color="transparent")
        det_hdr.grid(row=0, column=0, sticky="ew")
        det_hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(det_hdr, text="AI Detector:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, sticky="w")
        self.dl_models_btn = ctk.CTkButton(
            det_hdr, text="📥 Models", width=65, height=20,
            font=ctk.CTkFont(size=9, weight="bold"),
            fg_color="#1f538d", hover_color="#14375e",
            command=self.open_model_downloader)
        self.dl_models_btn.grid(row=0, column=1, sticky="e")
        self.detector_menu = ctk.CTkOptionMenu(r2_l, values=["Loading..."], height=28, font=ctk.CTkFont(size=11), command=self.on_detector_changed)
        self.detector_menu.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.refresh_model_dropdown()

        r2_r = ctk.CTkFrame(sf, fg_color="transparent")
        r2_r.grid(row=4, column=1, padx=(4, 0), pady=(2, 4), sticky="ew")
        r2_r.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(r2_r, text="Hardware GPU Encoder:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, sticky="w")
        encoder_labels = [label for enc_id, label in self.detected_encoders]
        self.encoder_menu = ctk.CTkOptionMenu(r2_r, values=encoder_labels, height=28, font=ctk.CTkFont(size=11))
        self.encoder_menu.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        # Row 3: Watermark Cuts (2 columns)
        r3_l = ctk.CTkFrame(sf, fg_color="transparent")
        r3_l.grid(row=5, column=0, padx=(0, 4), pady=(2, 4), sticky="ew")
        r3_l.grid_columnconfigure(0, weight=1)
        self.top_crop_label = ctk.CTkLabel(r3_l, text="Top Cut: 5%", font=ctk.CTkFont(size=11, weight="bold"))
        self.top_crop_label.grid(row=0, column=0, sticky="w")
        self.top_crop_slider = ctk.CTkSlider(r3_l, from_=0, to=0.30, number_of_steps=30, command=self.on_top_crop_slide, height=14)
        self.top_crop_slider.set(0.05)
        self.top_crop_slider.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        r3_r = ctk.CTkFrame(sf, fg_color="transparent")
        r3_r.grid(row=5, column=1, padx=(4, 0), pady=(2, 4), sticky="ew")
        r3_r.grid_columnconfigure(0, weight=1)
        self.bottom_crop_label = ctk.CTkLabel(r3_r, text="Bottom Cut: 10%", font=ctk.CTkFont(size=11, weight="bold"))
        self.bottom_crop_label.grid(row=0, column=0, sticky="w")
        self.bottom_crop_slider = ctk.CTkSlider(r3_r, from_=0, to=0.30, number_of_steps=30, command=self.on_bottom_crop_slide, height=14)
        self.bottom_crop_slider.set(0.10)
        self.bottom_crop_slider.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        # Row 4: Smoothness & Switches
        r4 = ctk.CTkFrame(sf, fg_color="transparent")
        r4.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(2, 4))
        r4.grid_columnconfigure((0, 1), weight=1)

        r4_l = ctk.CTkFrame(r4, fg_color="transparent")
        r4_l.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        r4_l.grid_columnconfigure(0, weight=1)
        self.smooth_label = ctk.CTkLabel(r4_l, text="Smoothness: 0.15", font=ctk.CTkFont(size=11, weight="bold"))
        self.smooth_label.grid(row=0, column=0, sticky="w")
        self.smooth_slider = ctk.CTkSlider(r4_l, from_=0.05, to=0.50, number_of_steps=45, command=self.on_smooth_slide, height=14)
        self.smooth_slider.set(0.15)
        self.smooth_slider.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        r4_r = ctk.CTkFrame(r4, fg_color="transparent")
        r4_r.grid(row=0, column=1, padx=(4, 0), sticky="ew")
        self.ultra_fast_switch = ctk.CTkSwitch(r4_r, text="⚡ Ultra Fast Mode", font=ctk.CTkFont(size=11, weight="bold"), progress_color="#00FF66", command=self.on_ultra_fast_toggled)
        self.ultra_fast_switch.select()
        self.ultra_fast_switch.grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.smart_cut_switch = ctk.CTkSwitch(r4_r, text="Smart Silence Cut", font=ctk.CTkFont(size=11))
        self.smart_cut_switch.select()
        self.smart_cut_switch.grid(row=1, column=0, sticky="w", pady=(2, 2))
        self.subtitles_switch = ctk.CTkSwitch(r4_r, text="Auto Subtitles (Whisper)", font=ctk.CTkFont(size=11))
        self.subtitles_switch.grid(row=2, column=0, sticky="w", pady=(2, 0))

        # ── Start button (bottom of Process tab) ──────────────────────────
        proc_action = ctk.CTkFrame(tab_proc, fg_color="transparent")
        proc_action.grid(row=2, column=0, padx=4, pady=(4, 6), sticky="ew")
        proc_action.grid_columnconfigure(0, weight=1)

        self.start_btn = ctk.CTkButton(
            proc_action, text="🚀 START AUTOCLIPPING",
            font=ctk.CTkFont(size=14, weight="bold"), height=38,
            fg_color="#1f538d", hover_color="#14375e",
            command=self.start_processing)
        self.start_btn.grid(row=0, column=0, sticky="ew")

        # ── 📺 DOWNLOAD & PREP TAB ─────────────────────────────────────────
        dl_scroll = ctk.CTkScrollableFrame(tab_dl, fg_color="transparent")
        dl_scroll.grid(row=0, column=0, sticky="nsew", padx=0)
        dl_scroll.grid_columnconfigure(0, weight=1)
        tab_dl.grid_rowconfigure(0, weight=1)

        ds = dl_scroll  # shorthand

        ctk.CTkLabel(ds, text="🎬 Download → Trim → Subtitle → Export",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=4, pady=(8, 2))
        ctk.CTkLabel(ds, text="All steps run in one click. Files saved to Output/<project>/",
                     font=ctk.CTkFont(size=10), text_color="gray60").grid(
            row=1, column=0, sticky="w", padx=4, pady=(0, 6))

        # ① Source
        s1 = ctk.CTkFrame(ds, corner_radius=8)
        s1.grid(row=2, column=0, sticky="ew", padx=4, pady=3)
        s1.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(s1, text="① Source", font=ctk.CTkFont(weight="bold", size=11)).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))
        ctk.CTkLabel(s1, text="URL:", font=ctk.CTkFont(size=11)).grid(
            row=1, column=0, sticky="w", padx=10)
        self.proj_url_entry = ctk.CTkEntry(
            s1, placeholder_text="https://www.youtube.com/watch?v=...", height=30)
        self.proj_url_entry.grid(row=1, column=1, sticky="ew", padx=(4, 10), pady=(0, 4))
        ctk.CTkLabel(s1, text="Quality:", font=ctk.CTkFont(size=11)).grid(
            row=2, column=0, sticky="w", padx=10, pady=(0, 8))
        self.proj_quality = ctk.CTkSegmentedButton(s1, values=["Best", "1080p", "720p", "480p"])
        self.proj_quality.set("1080p")
        self.proj_quality.grid(row=2, column=1, sticky="w", padx=(4, 10), pady=(0, 8))

        # ② Trim
        s2 = ctk.CTkFrame(ds, corner_radius=8)
        s2.grid(row=3, column=0, sticky="ew", padx=4, pady=3)
        s2.grid_columnconfigure((1, 3), weight=1)
        ctk.CTkLabel(s2, text="② Trim  (blank = full video)",
                     font=ctk.CTkFont(weight="bold", size=11)).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(8, 4))
        ctk.CTkLabel(s2, text="Start:", font=ctk.CTkFont(size=11)).grid(row=1, column=0, padx=10)
        self.proj_trim_start = ctk.CTkEntry(s2, placeholder_text="00:01:30 or 90", width=130, height=26)
        self.proj_trim_start.grid(row=1, column=1, sticky="w", padx=(4, 10))
        ctk.CTkLabel(s2, text="End:", font=ctk.CTkFont(size=11)).grid(row=1, column=2)
        self.proj_trim_end = ctk.CTkEntry(s2, placeholder_text="00:05:00 or 300", width=130, height=26)
        self.proj_trim_end.grid(row=1, column=3, sticky="w", padx=(4, 10), pady=(0, 8))

        # ③ Subtitles
        s3 = ctk.CTkFrame(ds, corner_radius=8)
        s3.grid(row=4, column=0, sticky="ew", padx=4, pady=3)
        s3.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(s3, text="③ Subtitle Export",
                     font=ctk.CTkFont(weight="bold", size=11)).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 4))
        self.proj_sub_switch = ctk.CTkSwitch(
            s3, text="Generate SRT subtitle file (Whisper AI)", font=ctk.CTkFont(size=11))
        self.proj_sub_switch.select()
        self.proj_sub_switch.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 4))
        sub_fmt_row = ctk.CTkFrame(s3, fg_color="transparent")
        sub_fmt_row.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 8))
        ctk.CTkLabel(sub_fmt_row, text="Format:", font=ctk.CTkFont(size=11)).pack(side="left")
        self.proj_sub_fmt = ctk.CTkSegmentedButton(sub_fmt_row, values=["SRT", "VTT", "ASS", "All"])
        self.proj_sub_fmt.set("SRT")
        self.proj_sub_fmt.pack(side="left", padx=8)

        # ④ Project Name
        s4 = ctk.CTkFrame(ds, corner_radius=8)
        s4.grid(row=5, column=0, sticky="ew", padx=4, pady=3)
        s4.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(s4, text="④ Project Name",
                     font=ctk.CTkFont(weight="bold", size=11)).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))
        ctk.CTkLabel(s4, text="Name:", font=ctk.CTkFont(size=11)).grid(
            row=1, column=0, sticky="w", padx=10)
        self.proj_name_entry = ctk.CTkEntry(
            s4, placeholder_text="auto-filled from video title", height=28)
        self.proj_name_entry.grid(row=1, column=1, sticky="ew", padx=(4, 10), pady=(0, 8))

        # ⑤ Extra args
        s5 = ctk.CTkFrame(ds, corner_radius=8)
        s5.grid(row=6, column=0, sticky="ew", padx=4, pady=3)
        s5.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(s5, text="Extra yt-dlp args (optional):",
                     font=ctk.CTkFont(size=10)).grid(
            row=0, column=0, sticky="w", padx=10, pady=(6, 2))
        self.extra_args_entry = ctk.CTkEntry(
            s5, placeholder_text="--cookies-from-browser chrome",
            height=26, font=ctk.CTkFont(size=10))
        self.extra_args_entry.grid(row=0, column=1, sticky="ew", padx=(4, 10), pady=(6, 6))

        # Prepare button
        self.proj_btn = ctk.CTkButton(
            ds, text="🚀  Prepare Project",
            height=40, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2d6a4f", hover_color="#1b4332",
            command=self._start_project_prep_inline)
        self.proj_btn.grid(row=7, column=0, sticky="ew", padx=4, pady=(6, 4))

        # Progress
        dl_prog_frame = ctk.CTkFrame(ds, fg_color="transparent")
        dl_prog_frame.grid(row=8, column=0, sticky="ew", padx=4)
        dl_prog_frame.grid_columnconfigure(0, weight=1)
        self.proj_progress = ctk.CTkProgressBar(dl_prog_frame)
        self.proj_progress.grid(row=0, column=0, sticky="ew", pady=(4, 2))
        self.proj_progress.set(0)
        self.proj_status_lbl = ctk.CTkLabel(
            dl_prog_frame, text="Paste URL above → press Prepare Project",
            font=ctk.CTkFont(size=10), text_color="gray60")
        self.proj_status_lbl.grid(row=1, column=0, sticky="w")

        ctk.CTkLabel(ds, text="Log:", font=ctk.CTkFont(weight="bold", size=10)).grid(
            row=9, column=0, sticky="w", padx=4, pady=(6, 2))
        self.proj_log = ctk.CTkTextbox(
            ds, height=90, font=ctk.CTkFont(family="Consolas", size=10),
            text_color="#00FF66", fg_color="black")
        self.proj_log.grid(row=10, column=0, sticky="ew", padx=4, pady=(0, 8))

        # ── ✂️ SPLIT & MERGE TAB ───────────────────────────────────────────
        sm_scroll = ctk.CTkScrollableFrame(tab_sm, fg_color="transparent")
        sm_scroll.grid(row=0, column=0, sticky="nsew", padx=0, pady=4)
        sm_scroll.grid_columnconfigure(0, weight=1)
        sms = sm_scroll  # shorthand

        # --- Section 1: Split Video ---
        split_card = ctk.CTkFrame(sms, corner_radius=8)
        split_card.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 8))
        split_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(split_card, text="✂️ Split Video",
                     font=ctk.CTkFont(weight="bold", size=13)).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))

        ctk.CTkLabel(split_card, text="Input File:", font=ctk.CTkFont(size=11)).grid(
            row=1, column=0, sticky="w", padx=10, pady=(2, 4))
        self.split_file_dropdown = ctk.CTkOptionMenu(split_card, values=["No videos found"], height=28)
        self.split_file_dropdown.grid(row=1, column=1, sticky="ew", padx=(4, 10), pady=(2, 4))

        ctk.CTkLabel(split_card, text="Split Mode:", font=ctk.CTkFont(size=11)).grid(
            row=2, column=0, sticky="w", padx=10, pady=(2, 4))
        self.split_mode_btn = ctk.CTkSegmentedButton(
            split_card, values=["By Time Interval", "By Equal Parts"],
            command=self._on_split_mode_changed)
        self.split_mode_btn.set("By Time Interval")
        self.split_mode_btn.grid(row=2, column=1, sticky="w", padx=(4, 10), pady=(2, 4))

        self.split_val_lbl = ctk.CTkLabel(split_card, text="Interval (seconds):", font=ctk.CTkFont(size=11))
        self.split_val_lbl.grid(row=3, column=0, sticky="w", padx=10, pady=(2, 8))
        self.split_val_entry = ctk.CTkEntry(split_card, placeholder_text="60", width=120, height=28)
        self.split_val_entry.insert(0, "60")
        self.split_val_entry.grid(row=3, column=1, sticky="w", padx=(4, 10), pady=(2, 8))

        self.split_action_btn = ctk.CTkButton(
            split_card, text="🔪 Split Video Now", height=34,
            font=ctk.CTkFont(weight="bold", size=12),
            fg_color="#2d6a4f", hover_color="#1b4332",
            command=self._start_split_video)
        self.split_action_btn.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))

        # --- Section 2: Merge Videos ---
        merge_card = ctk.CTkFrame(sms, corner_radius=8)
        merge_card.grid(row=1, column=0, sticky="ew", padx=4, pady=(4, 8))
        merge_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(merge_card, text="🔗 Merge Videos",
                     font=ctk.CTkFont(weight="bold", size=13)).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))

        self.merge_files_box = ctk.CTkTextbox(merge_card, height=90, font=ctk.CTkFont(size=10))
        self.merge_files_box.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(2, 4))
        self.merge_files_list = []

        m_btn_bar = ctk.CTkFrame(merge_card, fg_color="transparent")
        m_btn_bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 6))
        ctk.CTkButton(m_btn_bar, text="➕ Add Videos", width=100, height=26,
                      command=self._merge_add_files).pack(side="left", padx=(0, 4))
        ctk.CTkButton(m_btn_bar, text="➕ Add All Input/", width=110, height=26, fg_color="gray30", hover_color="gray40",
                      command=self._merge_add_all_input).pack(side="left", padx=4)
        ctk.CTkButton(m_btn_bar, text="🗑️ Clear", width=65, height=26, fg_color="#991b1b", hover_color="#7f1d1d",
                      command=self._merge_clear_files).pack(side="right", padx=(4, 0))

        out_name_row = ctk.CTkFrame(merge_card, fg_color="transparent")
        out_name_row.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))
        out_name_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(out_name_row, text="Output Name:", font=ctk.CTkFont(size=11)).grid(row=0, column=0, sticky="w")
        self.merge_output_entry = ctk.CTkEntry(out_name_row, placeholder_text="merged_video.mp4", height=28)
        self.merge_output_entry.insert(0, "merged_video.mp4")
        self.merge_output_entry.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.merge_action_btn = ctk.CTkButton(
            merge_card, text="🔗 Merge Videos Now", height=34,
            font=ctk.CTkFont(weight="bold", size=12),
            fg_color="#1d4ed8", hover_color="#1e40af",
            command=self._start_merge_videos)
        self.merge_action_btn.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))

        # --- Footer: Progress & Log ---
        sm_footer = ctk.CTkFrame(sms, fg_color="transparent")
        sm_footer.grid(row=2, column=0, sticky="ew", padx=4)
        sm_footer.grid_columnconfigure(0, weight=1)

        self.sm_progress = ctk.CTkProgressBar(sm_footer)
        self.sm_progress.grid(row=0, column=0, sticky="ew", pady=(4, 2))
        self.sm_progress.set(0)
        self.sm_status_lbl = ctk.CTkLabel(
            sm_footer, text="Select operation above", font=ctk.CTkFont(size=10), text_color="gray60")
        self.sm_status_lbl.grid(row=1, column=0, sticky="w")

        ctk.CTkLabel(sms, text="Split & Merge Log:", font=ctk.CTkFont(weight="bold", size=10)).grid(
            row=3, column=0, sticky="w", padx=4, pady=(6, 2))
        self.sm_log = ctk.CTkTextbox(
            sms, height=90, font=ctk.CTkFont(family="Consolas", size=10),
            text_color="#00FF66", fg_color="black")
        self.sm_log.grid(row=4, column=0, sticky="ew", padx=4, pady=(0, 8))

        # ══════════════════════════════════════════════════════════════════
        # RIGHT PANEL: Preview + Progress + Log
        # ══════════════════════════════════════════════════════════════════
        self.right_frame = ctk.CTkFrame(self, corner_radius=12)
        self.right_frame.grid(row=0, column=1, padx=12, pady=12, sticky="nsew")
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=3)
        self.right_frame.grid_rowconfigure(3, weight=1)

        # Preview header row
        prev_hdr = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        prev_hdr.grid(row=0, column=0, padx=12, pady=(10, 4), sticky="ew")
        prev_hdr.grid_columnconfigure(0, weight=1)

        self.preview_title_lbl = ctk.CTkLabel(
            prev_hdr, text="Interactive Visual Crop & Layout Preview",
            font=ctk.CTkFont(size=15, weight="bold"))
        self.preview_title_lbl.grid(row=0, column=0, sticky="w")

        self.live_badge = ctk.CTkLabel(
            prev_hdr, text="🔴 LIVE PROCESSING",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#ff4444")
        # Will be shown only while processing

        self.preview_label = ctk.CTkLabel(
            self.right_frame,
            text="Select a video file to view the interactive crop preview",
            fg_color="#0b0f19", corner_radius=10)
        self.preview_label.grid(row=1, column=0, padx=12, pady=6, sticky="nsew")

        progress_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        progress_frame.grid(row=2, column=0, padx=12, pady=4, sticky="ew")
        progress_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            progress_frame, text="Ready", font=ctk.CTkFont(size=11), text_color="gray80")
        self.status_label.grid(row=0, column=0, sticky="w", pady=(0, 2))

        self.progress_bar = ctk.CTkProgressBar(progress_frame, height=12)
        self.progress_bar.grid(row=1, column=0, sticky="ew")
        self.progress_bar.set(0.0)

        log_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        log_frame.grid(row=3, column=0, padx=12, pady=(4, 12), sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(log_frame, text="Activity Log Console:",
                     font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, sticky="w")

        self.log_textbox = ctk.CTkTextbox(
            log_frame, font=ctk.CTkFont(family="Consolas", size=10),
            text_color="#38bdf8", fg_color="#090d16", corner_radius=8)
        self.log_textbox.grid(row=1, column=0, sticky="nsew", pady=(2, 0))

        # Flush any messages buffered before the textbox was created
        for buffered_msg in self._log_buffer:
            self.log_textbox.insert("end", f"> {buffered_msg}\n")
        self._log_buffer.clear()
        if self.log_textbox.get("1.0", "end").strip():
            self.log_textbox.see("end")

        self.log("AutoClipper Studio initialized with GPU Acceleration & AI Subtitles.")

    # ---------------- EVENT HANDLERS & PREVIEW ----------------

    def log(self, message):
        if hasattr(self, 'log_textbox'):
            self.log_textbox.insert("end", f"> {message}\n")
            self.log_textbox.see("end")
        else:
            # log_textbox not yet built — buffer for later
            self._log_buffer.append(message)

    def refresh_input_files(self):
        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))]
        if not files:
            self.file_dropdown.configure(values=["No videos in Input/ folder"])
            self.file_dropdown.set("No videos in Input/ folder")
            if hasattr(self, "split_file_dropdown"):
                self.split_file_dropdown.configure(values=["No videos in Input/ folder"])
                self.split_file_dropdown.set("No videos in Input/ folder")
            self.selected_file = None
            self.log("No video files found in 'Input/' directory.")
        else:
            self.file_dropdown.configure(values=files)
            self.file_dropdown.set(files[0])
            if hasattr(self, "split_file_dropdown"):
                self.split_file_dropdown.configure(values=files)
                self.split_file_dropdown.set(files[0])
            self.on_file_selected(files[0])
            self.log(f"Found {len(files)} video file(s).")

    # ── ✂️ Split & Merge Handlers ──────────────────────────────────────────
    def _on_split_mode_changed(self, mode):
        if mode == "By Time Interval":
            self.split_val_lbl.configure(text="Interval (seconds):")
            self.split_val_entry.delete(0, "end")
            self.split_val_entry.insert(0, "60")
        else:
            self.split_val_lbl.configure(text="Number of Parts:")
            self.split_val_entry.delete(0, "end")
            self.split_val_entry.insert(0, "4")

    def _sm_log(self, msg):
        self.after(0, lambda msg=msg: (
            self.sm_log.insert("end", msg + "\n"),
            self.sm_log.see("end")
        ))

    def _sm_status(self, msg, prog=None):
        self.after(0, lambda msg=msg: self.sm_status_lbl.configure(text=msg))
        if prog is not None:
            self.after(0, lambda p=prog: self.sm_progress.set(p))

    def _merge_add_files(self):
        files = filedialog.askopenfilenames(
            title="Select Video Files to Merge",
            filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv *.webm")]
        )
        if files:
            for f in files:
                if f not in self.merge_files_list:
                    self.merge_files_list.append(f)
            self._update_merge_textbox()

    def _merge_add_all_input(self):
        files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir)
                 if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))]
        if not files:
            messagebox.showinfo("No Files", "No video files found in Input/ folder.", parent=self)
            return
        for f in files:
            if f not in self.merge_files_list:
                self.merge_files_list.append(f)
        self._update_merge_textbox()

    def _merge_clear_files(self):
        self.merge_files_list.clear()
        self._update_merge_textbox()

    def _update_merge_textbox(self):
        self.merge_files_box.delete("1.0", "end")
        if not self.merge_files_list:
            self.merge_files_box.insert("1.0", "(No videos selected. Click Add Videos above)")
        else:
            for idx, f in enumerate(self.merge_files_list, 1):
                self.merge_files_box.insert("end", f"{idx}. {os.path.basename(f)}\n")

    def _find_ffmpeg(self):
        import shutil
        local_ffmpeg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "ffmpeg.exe")
        if os.path.exists(local_ffmpeg):
            return local_ffmpeg
        sys_ffmpeg = shutil.which("ffmpeg")
        if sys_ffmpeg:
            return sys_ffmpeg
        return None

    def _start_split_video(self):
        chosen_file = self.split_file_dropdown.get()
        if not chosen_file or chosen_file.startswith("No videos"):
            messagebox.showwarning("No Input Video", "Select a video file to split.", parent=self)
            return

        filepath = os.path.join(self.input_dir, chosen_file)
        if not os.path.exists(filepath):
            messagebox.showerror("File Not Found", f"Video file not found: {filepath}", parent=self)
            return

        ffmpeg_path = self._find_ffmpeg()
        if not ffmpeg_path:
            messagebox.showerror("FFmpeg Missing", "FFmpeg is required. Please download it from ⚙️ Hub → 🔧 Tools.", parent=self)
            return

        mode = self.split_mode_btn.get()
        val_str = self.split_val_entry.get().strip()

        try:
            val = float(val_str)
            if val <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid positive number.", parent=self)
            return

        self.split_action_btn.configure(state="disabled", text="⏳ Splitting...")
        self.sm_log.delete("1.0", "end")
        self.sm_progress.set(0)

        threading.Thread(
            target=self._split_video_worker,
            args=(filepath, mode, val, ffmpeg_path),
            daemon=True
        ).start()

    def _split_video_worker(self, filepath, mode, val, ffmpeg_path):
        import subprocess, re, shutil

        def log(m): self._sm_log(m)
        def status(m, p=None): self._sm_status(m, p)

        filename = os.path.basename(filepath)
        name_no_ext = os.path.splitext(filename)[0]
        split_out_dir = os.path.join(self.output_dir, f"{name_no_ext}_splits")
        os.makedirs(split_out_dir, exist_ok=True)

        log(f"▶ Starting split for: {filename}")
        log(f"  Output folder: {split_out_dir}")

        ffprobe_path = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe.exe")
        if not os.path.exists(ffprobe_path):
            ffprobe_path = shutil.which("ffprobe") or "ffprobe"

        try:
            if mode == "By Time Interval":
                interval = float(val)
                status("Splitting by time interval...", 0.3)
                out_pattern = os.path.join(split_out_dir, f"{name_no_ext}_part%03d.mp4")
                cmd = [
                    ffmpeg_path, "-y", "-i", filepath,
                    "-c", "copy",
                    "-segment_time", str(interval),
                    "-f", "segment",
                    "-reset_timestamps", "1",
                    out_pattern
                ]
                log(f"▶ Running: {' '.join(cmd)}")
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
                if proc.returncode != 0:
                    raise Exception(f"FFmpeg split failed:\n{proc.stdout[-500:]}")
            else:
                num_parts = int(val)
                status("Getting video duration...", 0.1)
                cmd_dur = [
                    ffprobe_path, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    filepath
                ]
                res = subprocess.run(cmd_dur, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                total_duration = float(res.stdout.strip())
                log(f"Video total duration: {total_duration:.2f} seconds")

                part_duration = total_duration / num_parts
                log(f"Splitting into {num_parts} equal parts (~{part_duration:.2f}s each)")

                for i in range(num_parts):
                    start_time = i * part_duration
                    status(f"Splitting part {i+1}/{num_parts}...", (i + 1) / num_parts)
                    out_part = os.path.join(split_out_dir, f"{name_no_ext}_part{i+1:03d}.mp4")
                    cmd_part = [
                        ffmpeg_path, "-y",
                        "-ss", f"{start_time:.3f}",
                        "-i", filepath,
                        "-t", f"{part_duration:.3f}",
                        "-c", "copy",
                        out_part
                    ]
                    proc = subprocess.run(cmd_part, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
                    if proc.returncode != 0:
                        raise Exception(f"Failed to extract part {i+1}:\n{proc.stdout[-300:]}")
                    log(f"  ✓ Part {i+1}/{num_parts} done -> {os.path.basename(out_part)}")

            self.after(0, self._on_split_success, split_out_dir)
        except Exception as e:
            self.after(0, self._on_split_error, str(e))

    def _on_split_success(self, out_dir):
        self.split_action_btn.configure(state="normal", text="🔪 Split Video Now")
        self.sm_progress.set(1.0)
        self.sm_status_lbl.configure(text="✅ Video split completed!")
        self._sm_log(f"\n✅ SUCCESS: Split clips saved to:\n{out_dir}")
        messagebox.showinfo("Split Complete", f"Successfully split video!\nOutput folder:\n{out_dir}", parent=self)
        try:
            import subprocess
            subprocess.Popen(["explorer", os.path.abspath(out_dir)])
        except Exception:
            pass

    def _on_split_error(self, err):
        self.split_action_btn.configure(state="normal", text="🔪 Split Video Now")
        self.sm_status_lbl.configure(text="❌ Split failed")
        self._sm_log(f"\n❌ ERROR: {err}")
        messagebox.showerror("Split Error", err, parent=self)

    def _start_merge_videos(self):
        if not self.merge_files_list:
            messagebox.showwarning("No Videos Selected", "Add videos to merge first.", parent=self)
            return

        ffmpeg_path = self._find_ffmpeg()
        if not ffmpeg_path:
            messagebox.showerror("FFmpeg Missing", "FFmpeg is required. Download it from ⚙️ Hub → 🔧 Tools.", parent=self)
            return

        out_name = self.merge_output_entry.get().strip()
        if not out_name:
            out_name = "merged_video.mp4"
        if not out_name.lower().endswith(".mp4"):
            out_name += ".mp4"

        self.merge_action_btn.configure(state="disabled", text="⏳ Merging...")
        self.sm_log.delete("1.0", "end")
        self.sm_progress.set(0)

        threading.Thread(
            target=self._merge_videos_worker,
            args=(list(self.merge_files_list), out_name, ffmpeg_path),
            daemon=True
        ).start()

    def _merge_videos_worker(self, file_list, out_name, ffmpeg_path):
        import subprocess, tempfile

        def log(m): self._sm_log(m)
        def status(m, p=None): self._sm_status(m, p)

        out_path = os.path.join(self.output_dir, out_name)
        log(f"▶ Merging {len(file_list)} videos into: {out_name}")

        status("Creating file list...", 0.2)
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
                list_file_path = tf.name
                for fpath in file_list:
                    abs_f = os.path.abspath(fpath).replace("\\", "/")
                    tf.write(f"file '{abs_f}'\n")

            status("Merging videos with FFmpeg...", 0.5)
            cmd = [
                ffmpeg_path, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file_path,
                "-c", "copy",
                out_path
            ]
            log(f"▶ Running concat: {' '.join(cmd)}")
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")

            try:
                os.remove(list_file_path)
            except Exception:
                pass

            if proc.returncode != 0:
                raise Exception(f"FFmpeg concat failed:\n{proc.stdout[-500:]}")

            self.after(0, self._on_merge_success, out_path)
        except Exception as e:
            self.after(0, self._on_merge_error, str(e))

    def _on_merge_success(self, out_path):
        self.merge_action_btn.configure(state="normal", text="🔗 Merge Videos Now")
        self.sm_progress.set(1.0)
        self.sm_status_lbl.configure(text="✅ Videos merged successfully!")
        self._sm_log(f"\n✅ SUCCESS: Merged file saved to:\n{out_path}")
        messagebox.showinfo("Merge Complete", f"Successfully merged videos!\nSaved to:\n{out_path}", parent=self)
        try:
            import subprocess
            subprocess.Popen(["explorer", self.output_dir])
        except Exception:
            pass

    def _on_merge_error(self, err):
        self.merge_action_btn.configure(state="normal", text="🔗 Merge Videos Now")
        self.sm_status_lbl.configure(text="❌ Merge failed")
        self._sm_log(f"\n❌ ERROR: {err}")
        messagebox.showerror("Merge Error", err, parent=self)

    def browse_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv *.webm")]
        )
        if filepath:
            self.selected_file = filepath
            filename = os.path.basename(filepath)
            self.file_dropdown.configure(values=[filename])
            self.file_dropdown.set(filename)
            self.log(f"Selected file: {filepath}")
            self.load_sample_frame()

    def on_file_selected(self, choice):
        if choice.startswith("No videos"):
            self.selected_file = None
            return
        candidate = os.path.join(self.input_dir, choice)
        if os.path.exists(candidate):
            self.selected_file = candidate
            self.log(f"Selected: {choice}")
            self.load_sample_frame()

    def on_mode_changed(self, choice):
        if "Gaming" in choice:
            self.cam_pos_label.grid(row=2, column=0, sticky="w", pady=(2, 2))
            self.cam_pos_menu.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        else:
            self.cam_pos_label.grid_forget()
            self.cam_pos_menu.grid_forget()

        self.update_preview()

    def on_duration_changed(self, choice):
        """Show/hide the Custom seconds entry based on segmented button selection."""
        if choice == "Custom":
            self.custom_dur_frame.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        else:
            self.custom_dur_frame.grid_forget()
        self.update_preview()
    # ── Inline Project Prep (in Download & Prep tab) ─────────────────────
    def _start_project_prep_inline(self):
        """Delegates to the SettingsHub project-prep logic without opening a separate window."""
        url = self.proj_url_entry.get().strip()
        if not url:
            messagebox.showwarning("No URL", "Enter a YouTube / web URL first.")
            return

        # Resolve tools
        tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
        import shutil
        ytdlp_path = (os.path.join(tools_dir, "yt-dlp.exe")
                      if os.path.exists(os.path.join(tools_dir, "yt-dlp.exe"))
                      else shutil.which("yt-dlp"))
        if not ytdlp_path:
            if messagebox.askyesno("yt-dlp Not Found",
                                   "yt-dlp is not installed.\nOpen the ⚙️ Hub → 🔧 Tools tab to download it?"):
                self.open_hub("tools")
            return

        try:
            trim_start = SettingsHubWindow._parse_timestamp(self.proj_trim_start.get())
            trim_end   = SettingsHubWindow._parse_timestamp(self.proj_trim_end.get())
        except ValueError:
            messagebox.showerror("Invalid Time", "Trim timestamps must be HH:MM:SS or seconds.")
            return

        self.proj_btn.configure(state="disabled", text="⏳ Working...")
        self.proj_log.delete("1.0", "end")
        self.proj_progress.set(0)

        quality   = self.proj_quality.get()
        gen_subs  = self.proj_sub_switch.get() == 1
        sub_fmt   = self.proj_sub_fmt.get()
        proj_name = self.proj_name_entry.get().strip()

        # Build a minimal proxy SettingsHubWindow to run the worker
        class _Proxy:
            input_dir  = self.input_dir
            output_dir = self.output_dir
            def log(_, m): self.log(m)   # noqa: E306
            def refresh_input_files(_): self.refresh_input_files()  # noqa: E306

        proxy = _Proxy()
        hub = SettingsHubWindow.__new__(SettingsHubWindow)
        hub.parent = proxy
        hub.TOOLS_DIR = tools_dir
        hub.proj_log = self.proj_log
        hub.proj_progress = self.proj_progress
        hub.proj_status_lbl = self.proj_status_lbl
        hub.proj_btn = self.proj_btn
        hub.after = self.after

        # Patch done callbacks to reset the inline button text
        def _done_ok(proj_dir):
            self.proj_btn.configure(state="normal", text="🚀  Prepare Project")
            self.proj_progress.set(1.0)
            self.proj_status_lbl.configure(text=f"✅ Ready → {os.path.basename(proj_dir)}")
            self.log(f"Project prepared → {proj_dir}")
            self.refresh_input_files()
            messagebox.showinfo("Project Ready!",
                f"Your project folder is ready:\n{proj_dir}\n\nOpening folder...")
            try:
                import subprocess as sp
                sp.Popen(["explorer", proj_dir])
            except Exception:
                pass

        def _done_err(err):
            self.proj_btn.configure(state="normal", text="🚀  Prepare Project")
            self.proj_status_lbl.configure(text="❌ Failed")
            self.proj_log.insert("end", f"\n❌ ERROR: {err}\n")
            messagebox.showerror("Project Prep Error", err)

        hub._proj_done_success = _done_ok
        hub._proj_done_error   = _done_err
        hub._find_tool  = lambda fn: (os.path.join(tools_dir, fn)
                                      if os.path.exists(os.path.join(tools_dir, fn))
                                      else shutil.which(fn.replace(".exe", "")))

        threading.Thread(
            target=hub._project_prep_worker,
            args=(url, ytdlp_path, quality, trim_start, trim_end,
                  gen_subs, sub_fmt, proj_name, self.input_dir, self.output_dir),
            daemon=True
        ).start()


    def open_hub(self, tab="models"):
        """Open (or focus) the Settings Hub window, optionally on a specific tab."""
        if self.hub_win is not None and self.hub_win.winfo_exists():
            self.hub_win.focus()
            tab_map = {"models": "🤖 AI Models", "tools": "🔧 Tools",
                       "video": "📺 Download Video", "project": "🎬 Project Prep"}
            if tab in tab_map:
                self.hub_win.tab_view.set(tab_map[tab])
        else:
            self.hub_win = SettingsHubWindow(self, initial_tab=tab)

    def open_model_downloader(self):
        self.open_hub("models")

    def refresh_model_dropdown(self, select_model=None):
        from clipper_engine import list_installed_models, AVAILABLE_MODELS_CATALOG
        installed = list_installed_models()

        options = []
        model_mapping = {}

        # 1. Add installed models first
        if installed:
            for fn in installed:
                meta = AVAILABLE_MODELS_CATALOG.get(fn, {})
                display_name = meta.get("name", fn)
                label = f"⚡ {display_name} ({fn})"
                options.append(label)
                model_mapping[label] = fn

        # 2. Add uninstalled catalog models
        for fn, meta in AVAILABLE_MODELS_CATALOG.items():
            if fn not in installed:
                label = f"📥 [Download Required] {meta['name']} ({fn})"
                options.append(label)
                model_mapping[label] = fn

        if not options:
            options = ["⚡ YOLO26 Nano (yolo26n.pt)", "⚡ YuNet ONNX (yunet.onnx)"]

        self.model_mapping = model_mapping
        self.detector_menu.configure(values=options)

        if select_model:
            for lbl, fn in model_mapping.items():
                if fn == select_model:
                    self.detector_menu.set(lbl)
                    self.on_detector_changed(lbl)
                    return

        # Default selection priority: yolo26n.pt -> first installed -> first option
        default_label = None
        for lbl, fn in model_mapping.items():
            if fn == "yolo26n.pt" and fn in installed:
                default_label = lbl
                break

        if not default_label and installed:
            default_label = options[0]

        if not default_label and options:
            default_label = options[0]

        if default_label:
            self.detector_menu.set(default_label)
            self.on_detector_changed(default_label)

    def on_detector_changed(self, choice):
        fn = getattr(self, "model_mapping", {}).get(choice, choice)

        if choice.startswith("📥 [Download Required]"):
            if messagebox.askyesno("Model Download Required", f"The model '{fn}' is not downloaded yet.\nWould you like to open the AI Model Downloader now?"):
                self.open_model_downloader()
            return

        from clipper_engine import SubjectTracker, VideoProcessor
        self.tracker = SubjectTracker(model_path=fn)
        self.processor = VideoProcessor(tracker=self.tracker)
        engine_name = getattr(self.tracker, "active_engine", "Subject Tracker")
        self.log(f"Switched AI Subject Tracker Model to: {engine_name}")
        if self.sample_frame is not None:
            self.load_sample_frame()

    def on_top_crop_slide(self, val):
        pct = int(val * 100)
        self.top_crop_label.configure(text=f"Watermark Top Cut: {pct}%")
        self.update_preview()

    def on_bottom_crop_slide(self, val):
        pct = int(val * 100)
        self.bottom_crop_label.configure(text=f"Watermark Bottom Cut: {pct}%")
        self.update_preview()

    def on_smooth_slide(self, val):
        self.smooth_label.configure(text=f"Motion Smoothness: {val:.2f}")

    def on_ultra_fast_toggled(self):
        is_on = getattr(self, "ultra_fast_switch", None) and self.ultra_fast_switch.get() == 1
        preset_label = "⚡ ULTRA FAST (Parallel GPU Rendering + 320p AI)" if is_on else "Standard Fast"
        self.log(f"Switched Performance Preset to: {preset_label}")

    def on_setting_changed(self, val=None):
        self.update_preview()

    def load_sample_frame(self):
        if not self.selected_file or not os.path.exists(self.selected_file):
            return

        cap = cv2.VideoCapture(self.selected_file)
        if not cap.isOpened():
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total_frames * 0.25)))
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        cap.release()

        if ret and frame is not None:
            self.sample_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.sample_faces = self.tracker.detect_faces(frame)
            self.update_preview()

    def update_preview(self):
        if self.sample_frame is None:
            return

        h, w = self.sample_frame.shape[:2]
        img = Image.fromarray(self.sample_frame.copy())
        draw = ImageDraw.Draw(img, "RGBA")

        top_crop = self.top_crop_slider.get()
        bottom_crop = self.bottom_crop_slider.get()
        mode_str = self.mode_menu.get()

        # 1. Red Watermark Exclusion Zones
        if top_crop > 0:
            top_y = int(h * top_crop)
            draw.rectangle([0, 0, w, top_y], fill=(255, 0, 0, 90), outline=(255, 0, 0, 200), width=2)
            draw.text((15, 10), f"WATERMARK TOP CUT ({int(top_crop*100)}%)", fill=(255, 255, 255, 220))

        if bottom_crop > 0:
            bot_y = int(h * (1.0 - bottom_crop))
            draw.rectangle([0, bot_y, w, h], fill=(255, 0, 0, 90), outline=(255, 0, 0, 200), width=2)
            draw.text((15, bot_y + 10), f"WATERMARK BOTTOM CUT ({int(bottom_crop*100)}%)", fill=(255, 255, 255, 220))

        # 2. Draw Detected Face Box(es)
        primary_face = self.tracker.get_primary_face_center(self.sample_frame)
        dual = self.tracker.get_dual_speakers(self.sample_frame)

        if self.sample_faces:
            for (fx, fy, fw, fh, score) in self.sample_faces:
                draw.rectangle([fx, fy, fx + fw, fy + fh], outline=(0, 255, 0, 220), width=3)

        # 3. Draw Layout Boxes based on Editing Mode
        if "Podcast" in mode_str:
            # Podcast Dual Speaker Stack Preview
            if dual:
                spk_l, spk_r = dual
                cx_l, cy_l, cw_l, ch_l = calculate_crop_window(w, h, spk_l[0], spk_l[1], top_crop, bottom_crop, 1080/960)
                cx_r, cy_r, cw_r, ch_r = calculate_crop_window(w, h, spk_r[0], spk_r[1], top_crop, bottom_crop, 1080/960)
            else:
                cw_l, ch_l = int(w * 0.45), int(h * 0.8)
                cx_l, cy_l = 0, int(h * top_crop)
                cw_r, ch_r = cw_l, ch_l
                cx_r, cy_r = int(w * 0.55), cy_l

            draw.rectangle([cx_l, cy_l, cx_l + cw_l, cy_l + ch_l], outline=(0, 220, 255, 255), width=3)
            draw.text((cx_l + 10, cy_l + 10), "PODCAST TOP SPEAKER A", fill=(0, 220, 255, 255))

            draw.rectangle([cx_r, cy_r, cx_r + cw_r, cy_r + ch_r], outline=(255, 220, 0, 255), width=3)
            draw.text((cx_r + 10, cy_r + 10), "PODCAST BOTTOM SPEAKER B", fill=(255, 220, 0, 255))

        elif "Gaming" in mode_str:
            # Gaming Cam Split Preview
            if primary_face:
                fcx, fcy, _, _ = primary_face
            else:
                fcx, fcy = w / 2.0, h / 2.0

            cx, cy, cw, ch = calculate_crop_window(w, h, fcx, fcy, top_crop, bottom_crop, 9/16)
            draw.rectangle([cx, cy, cx + cw, cy + ch], outline=(0, 220, 255, 255), width=3)
            draw.text((cx + 10, cy + 10), "GAMING WEBCAM FACE CROP", fill=(0, 220, 255, 255))

            # Gameplay screen box
            game_h = int(w * (9/16))
            game_y = int((h - game_h) / 2)
            draw.rectangle([0, game_y, w, game_y + game_h], outline=(255, 0, 220, 255), width=3)
            draw.text((15, game_y + 10), "GAMEPLAY SCREEN CROP", fill=(255, 0, 220, 255))

        else:
            # Standard Single Face Dynamic Crop
            if primary_face:
                fcx, fcy, _, _ = primary_face
            else:
                fcx, fcy = w / 2.0, h / 2.0

            cx, cy, cw, ch = calculate_crop_window(w, h, fcx, fcy, top_crop, bottom_crop, 9/16)
            draw.rectangle([cx, cy, cx + cw, cy + ch], outline=(0, 220, 255, 255), width=4)
            draw.text((cx + 10, cy + 10), "DYNAMIC 9:16 FACE CROP", fill=(0, 220, 255, 255))

        # Render onto CTk Label
        max_pw, max_ph = 600, 360
        img_aspect = w / h
        if img_aspect > (max_pw / max_ph):
            nw = max_pw
            nh = int(max_pw / img_aspect)
        else:
            nh = max_ph
            nw = int(max_ph * img_aspect)

        img_resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        self._preview_ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(nw, nh))

        self.preview_label.configure(image=self._preview_ctk_img, text="")

    # ---------------- BACKGROUND WORKER ----------------

    def start_processing(self):
        if self.is_processing:
            messagebox.showwarning("In Progress", "Processing is already running!")
            return

        if not self.selected_file or not os.path.exists(self.selected_file):
            messagebox.showerror("Error", "Please select a valid input video file.")
            return

        dur_choice = self.duration_menu.get()
        if dur_choice == "15s":
            clip_dur = 15.0
        elif dur_choice == "30s":
            clip_dur = 30.0
        elif dur_choice == "60s":
            clip_dur = 60.0
        elif dur_choice == "Custom":
            try:
                clip_dur = float(self.custom_dur_entry.get().strip())
                if clip_dur <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Duration",
                    "Please enter a valid positive number of seconds for the custom clip duration.")
                return
        else:
            clip_dur = None  # Full Video

        res_str = self.res_menu.get() if hasattr(self, "res_menu") else "1080x1920"
        if "1080x1080" in res_str:
            target_res = (1080, 1080)
        elif "1920x1080" in res_str:
            target_res = (1920, 1080)
        elif "720x1280" in res_str:
            target_res = (720, 1280)
        else:
            target_res = (1080, 1920)

        mode_str = self.mode_menu.get()
        if "Podcast" in mode_str:
            mode_choice = "podcast"
        elif "Gaming" in mode_str:
            mode_choice = "gaming"
        elif "Black Letterbox" in mode_str:
            mode_choice = "fit"
        elif "Blurred" in mode_str:
            mode_choice = "fit_blur"
        else:
            mode_choice = "crop"

        cam_pos = "top" if "Top" in self.cam_pos_menu.get() else "bottom"
        smart_cut = self.smart_cut_switch.get() == 1
        auto_subs = self.subtitles_switch.get() == 1

        top_crop = self.top_crop_slider.get()
        bottom_crop = self.bottom_crop_slider.get()
        smoothing = self.smooth_slider.get()

        selected_label = self.encoder_menu.get()
        encoder_id = "auto"
        for eid, lbl in self.detected_encoders:
            if lbl == selected_label:
                encoder_id = eid
                break

        self.is_processing = True
        self.start_btn.configure(state="disabled", text="⚡ AUTOCLIPPING VIDEO...")
        self.progress_bar.set(0.0)
        self.log(f"Starting AutoClipper job (Res: {target_res[0]}x{target_res[1]}, Mode: {mode_choice}, Encoder: {selected_label})...")
        # Show LIVE badge
        self.live_badge.grid(row=0, column=1, sticky="e")

        thread = threading.Thread(
            target=self._worker_thread,
            args=(self.selected_file, self.output_dir, clip_dur, top_crop, bottom_crop,
                  mode_choice, cam_pos, smart_cut, auto_subs, smoothing, encoder_id, target_res),
            daemon=True
        )
        thread.start()

    def _worker_thread(self, video_path, output_dir, clip_dur, top_crop, bottom_crop,
                        mode, cam_pos, smart_cut, auto_subs, smoothing, encoder_id, target_res=(1080, 1920)):
        try:
            preset_choice = "ultra_fast" if (hasattr(self, "ultra_fast_switch") and self.ultra_fast_switch.get() == 1) else "fast"
            out_files = self.processor.process_video_to_clips(
                video_path=video_path,
                output_dir=output_dir,
                clip_duration=clip_dur,
                top_crop_pct=top_crop,
                bottom_crop_pct=bottom_crop,
                mode=mode,
                cam_pos=cam_pos,
                smart_cut=smart_cut,
                auto_subtitles=auto_subs,
                smoothing_alpha=smoothing,
                target_resolution=target_res,
                encoder=encoder_id,
                speed_preset=preset_choice,
                progress_callback=self._update_progress_from_thread,
                status_callback=self._update_status_from_thread,
                frame_callback=self._update_live_preview
            )
            self.after(0, self._on_processing_complete, out_files)
        except Exception as e:
            self.after(0, self._on_processing_error, str(e))

    def _update_live_preview(self, frame_bgr, clip_idx, total_clips):
        """Receive a BGR numpy frame from the engine and display it as the live preview."""
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            # Fit within preview area
            max_pw, max_ph = 600, 340
            iw, ih = img.size
            scale = min(max_pw / iw, max_ph / ih, 1.0)
            nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
            img_resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(nw, nh))

            def _update(ci=ctk_img, idx=clip_idx, tot=total_clips):
                self._preview_ctk_img = ci
                self.preview_label.configure(image=self._preview_ctk_img, text="")
                title_txt = f"⚡ Live Processing — Frame {idx}/{tot}" if tot > 50 else f"⚡ Live Processing — Clip {idx}/{tot}"
                self.preview_title_lbl.configure(text=title_txt)
            self.after(0, _update)
        except Exception:
            pass

    def _update_progress_from_thread(self, val):
        self.after(0, lambda: self.progress_bar.set(val))

    def _update_status_from_thread(self, msg):
        self.after(0, lambda: (self.status_label.configure(text=msg), self.log(msg)))

    def _on_processing_complete(self, out_files):
        self.is_processing = False
        self.start_btn.configure(state="normal", text="🚀 START AUTOCLIPPING")
        self.progress_bar.set(1.0)
        self.status_label.configure(text="Finished successfully!")
        # Hide live badge, restore preview title
        self.live_badge.grid_forget()
        self.preview_title_lbl.configure(text="Interactive Visual Crop & Layout Preview")
        self.log(f"SUCCESS: Exported {len(out_files)} video clips to '{self.output_dir}'")

        messagebox.showinfo(
            "Completed!",
            f"AutoClipper successfully generated {len(out_files)} vertical clip(s)!\nSaved in: {self.output_dir}"
        )

    def _on_processing_error(self, err_msg):
        self.is_processing = False
        self.start_btn.configure(state="normal", text="🚀 START AUTOCLIPPING")
        self.status_label.configure(text="Error occurred!")
        self.live_badge.grid_forget()
        self.preview_title_lbl.configure(text="Interactive Visual Crop & Layout Preview")
        self.log(f"ERROR: {err_msg}")
        messagebox.showerror("Processing Error", f"An error occurred during video clipping:\n{err_msg}")


if __name__ == "__main__":
    app = AutoClipperGUI()
    app.mainloop()
