#!/usr/bin/env python3
"""
Refined Fireshare processing script — fixed to ensure ffmpeg never receives files with a ".processing" suffix.

Behavioral summary:
- Watch for incoming files that end with ".mp4.processing" or ".mp4.processed" in the watch directory
- Wait until file size stabilizes (upload finished)
- Move the file to a temp directory (atomic move) and rename it to remove the ".processing" suffix (so ffmpeg sees a normal .mp4)
- Compress + strip metadata in temp
- Move final uniquely-named MP4 into the Fireshare watched output directory
- Trigger `fireshare scan-video --path` synchronously and log output

Usage:
  ./process.py --watch-dir ~/fireshare/clips/uploads --output-dir ~/fireshare/clips/videos --temp-dir ~/fireshare/clips/temp

Dependencies:
- python3
- watchdog (pip install watchdog)
- ffmpeg & ffprobe in PATH
- fireshare CLI for scan triggering (optional — script will still produce file)

"""

from __future__ import annotations
import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception:
    print("Missing dependency: watchdog. Install with `pip install watchdog`.")
    raise

# ---------- Logging ----------
LOG_FILE = "fireshare_processor.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("fireshare_processor")

# ---------- Constants ----------
VIDEO_SUFFIXES = (".mp4.processing", ".mp4.processed")
DEFAULT_CHECK_INTERVAL = 2.0
DEFAULT_STABLE_CHECKS = 3
DEFAULT_TIMEOUT = 300

FFMPEG_DEFAULT = {
    "vcodec": "libx264",
    "crf": "23",
    "preset": "medium",
    "audio_bitrate": "192k",
}

# ---------- Helpers ----------

def is_processing_file(p: Path) -> bool:
    return any(p.name.lower().endswith(suffix) for suffix in VIDEO_SUFFIXES)


def stable_file_wait(path: Path, timeout: int = DEFAULT_TIMEOUT, interval: float = DEFAULT_CHECK_INTERVAL, stable_checks: int = DEFAULT_STABLE_CHECKS) -> bool:
    start = time.time()
    last_size = -1
    stable = 0
    while time.time() - start < timeout:
        if not path.exists():
            logger.warning("File disappeared while waiting for stability: %s", path)
            return False
        try:
            size = path.stat().st_size
        except Exception as exc:
            logger.debug("stat failed for %s: %s", path, exc)
            size = -1
        if size > 0 and size == last_size:
            stable += 1
            if stable >= stable_checks:
                logger.info("File stabilized: %s (%d bytes)", path.name, size)
                return True
        else:
            stable = 0
        last_size = size
        time.sleep(interval)
    logger.error("Timeout waiting for file stability: %s", path)
    return False


def unique_name(dst_dir: Path, base_name: str) -> Path:
    suffix = Path(base_name).suffix
    stem = Path(base_name).stem
    unique = f"{int(time.time())}_{uuid.uuid4().hex}_{stem}{suffix}"
    return dst_dir / unique

# ---------- Video processor ----------
class VideoProcessor:
    def __init__(self, temp_dir: Path, ffmpeg_opts: dict | None = None):
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_opts = ffmpeg_opts or FFMPEG_DEFAULT

    def ffmpeg_available(self) -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception:
            return False

    def compress_and_strip(self, src: Path, dst: Path, width: Optional[int] = None, height: Optional[int] = None) -> bool:
        if not src.exists():
            logger.error("Source missing for compression: %s", src)
            return False

        vf = None
        if width and height:
            vf = f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i", str(src),
            "-map_metadata", "-1",
            "-map_chapters", "-1",
        ]
        if vf:
            cmd += ["-vf", vf]
        cmd += [
            "-c:v", self.ffmpeg_opts.get("vcodec", "libx264"),
            "-preset", self.ffmpeg_opts.get("preset", "medium"),
            "-crf", str(self.ffmpeg_opts.get("crf", "23")),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", self.ffmpeg_opts.get("audio_bitrate", "192k"),
            "-movflags", "+faststart",
            str(dst),
        ]

        logger.info("ffmpeg command: %s %s -> %s", cmd[0], src.name, dst.name)
        try:
            completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if completed.returncode != 0:
                logger.error("ffmpeg failed (%d): %s", completed.returncode, completed.stderr[-1000:])
                return False
            if not dst.exists() or dst.stat().st_size == 0:
                logger.error("ffmpeg produced no output: %s", dst)
                return False
            orig = src.stat().st_size
            new = dst.stat().st_size
            if orig > 0:
                reduction = (1.0 - (new / orig)) * 100.0
                logger.info("Compression finished: %s (reduction %.1f%%)", dst.name, reduction)
            else:
                logger.info("Compression finished: %s", dst.name)
            return True
        except Exception as exc:
            logger.exception("Exception running ffmpeg: %s", exc)
            return False

# ---------- FS Event Handler ----------
class ProcessingFileHandler(FileSystemEventHandler):
    def __init__(self, processor: VideoProcessor, watch_dir: Path, output_dir: Path, temp_dir: Path):
        self.processor = processor
        self.watch_dir = watch_dir
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self._processing = set()

    def on_created(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if is_processing_file(p):
            logger.info("Detected new processing file: %s", p)
            self._handle(p)

    def on_moved(self, event):
        if event.is_directory:
            return
        p = Path(event.dest_path)
        if is_processing_file(p):
            logger.info("Detected moved/renamed processing file: %s", p)
            self._handle(p)

    def _handle(self, src: Path):
        key = str(src.resolve())
        if key in self._processing:
            logger.info("Already processing: %s", src)
            return
        self._processing.add(key)
        try:
            if not stable_file_wait(src):
                logger.error("Source didn't stabilize: %s", src)
                return

            # Move the file to temp and rename it to remove the .processing suffix so ffmpeg can open it
            temp_move = self.temp_dir / src.name
            try:
                logger.info("Moving to temp: %s -> %s", src, temp_move)
                shutil.move(str(src), str(temp_move))
            except Exception as exc:
                logger.exception("Move to temp failed: %s", exc)
                return

            # derive mp4 name and rename in temp
            name = temp_move.name
            mp4_name = name
            for s in VIDEO_SUFFIXES:
                if name.lower().endswith(s):
                    mp4_name = name[:-len(s)]
                    break
            temp_mp4 = self.temp_dir / mp4_name
            try:
                logger.info("Renaming in temp: %s -> %s", temp_move.name, temp_mp4.name)
                temp_move.rename(temp_mp4)
            except Exception:
                logger.warning("Rename failed, will continue with moved file as-is")
                temp_mp4 = temp_move

            # Now run compression on the temp_mp4 (without .processing suffix)
            compressed_temp = self.temp_dir / f"compressed_{int(time.time())}_{uuid.uuid4().hex}_{temp_mp4.name}"
            ok = self.processor.compress_and_strip(temp_mp4, compressed_temp)
            if not ok:
                logger.error("Compression failed for %s", temp_mp4)
                return

            # Move compressed file into Fireshare watched output with a unique filename
            final_target = unique_name(self.output_dir, temp_mp4.name)
            try:
                logger.info("Moving compressed to output: %s -> %s", compressed_temp, final_target)
                shutil.move(str(compressed_temp), str(final_target))
            except Exception as exc:
                logger.exception("Failed to move compressed to output: %s", exc)
                return

            # Trigger fireshare scan synchronously and capture output
            try:
                scan_cmd = ["fireshare", "scan-video", "--path", str(final_target)]
                logger.info("Running scan: %s", " ".join(scan_cmd))
                res = subprocess.run(scan_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                logger.info("fireshare scan exit=%s stdout=%s stderr=%s", res.returncode, (res.stdout or "").strip(), (res.stderr or "").strip())
                if res.returncode != 0:
                    logger.warning("fireshare scan may have failed for %s", final_target)
            except Exception as exc:
                logger.warning("Failed to run fireshare scan: %s", exc)

            logger.info("Finished processing %s", final_target)

        finally:
            try:
                if 'temp_mp4' in locals() and temp_mp4.exists():
                    temp_mp4.unlink()
            except Exception:
                pass
            self._processing.discard(key)

# ---------- Main / CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Fireshare video processing watcher")
    p.add_argument("--watch-dir", default="~/fireshare/clips/uploads", help="Directory to watch for .mp4.processing files")
    p.add_argument("--output-dir", default="~/fireshare/clips/videos", help="Directory Fireshare watches (final MP4s go here)")
    p.add_argument("--temp-dir", default="~/fireshare/clips/temp", help="Temp working directory")
    p.add_argument("--crf", type=int, default=23, help="CRF for compression (lower = better quality)")
    p.add_argument("--preset", default="medium", help="ffmpeg preset")
    p.add_argument("--audio-bitrate", default="192k")
    p.add_argument("--once", action="store_true", help="Process matching files once and exit (no watcher)")
    return p.parse_args()


def main():
    args = parse_args()
    watch_dir = Path(os.path.expanduser(args.watch_dir)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    temp_dir = Path(os.path.expanduser(args.temp_dir)).resolve()

    watch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_opts = {"vcodec": "libx264", "crf": args.crf, "preset": args.preset, "audio_bitrate": args.audio_bitrate}
    processor = VideoProcessor(temp_dir=temp_dir, ffmpeg_opts=ffmpeg_opts)

    if not processor.ffmpeg_available():
        logger.error("ffmpeg/ffprobe not available in PATH. Aborting.")
        sys.exit(1)

    handler = ProcessingFileHandler(processor=processor, watch_dir=watch_dir, output_dir=output_dir, temp_dir=temp_dir)

    if args.once:
        for p in watch_dir.iterdir():
            if p.is_file() and is_processing_file(p):
                handler._handle(p)
        logger.info("One-shot processing complete.")
        return

    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()
    logger.info("Watching %s -> output %s (temp: %s)", watch_dir, output_dir, temp_dir)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
