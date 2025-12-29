#!/usr/bin/env python3
"""
Refined Fireshare processing script — guarantees ffmpeg never receives
files with a ".processing" suffix or missing .mp4 extension.
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
    return any(p.name.lower().endswith(s) for s in VIDEO_SUFFIXES)


def stable_file_wait(
    path: Path,
    timeout: int = DEFAULT_TIMEOUT,
    interval: float = DEFAULT_CHECK_INTERVAL,
    stable_checks: int = DEFAULT_STABLE_CHECKS,
) -> bool:
    start = time.time()
    last_size = -1
    stable = 0

    while time.time() - start < timeout:
        if not path.exists():
            logger.warning("File disappeared while waiting: %s", path)
            return False

        try:
            size = path.stat().st_size
        except Exception:
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


def strip_processing_suffix(name: str) -> str:
    lower = name.lower()
    for suffix in VIDEO_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return name


def unique_output_name(dst_dir: Path, base_mp4: str) -> Path:
    p = Path(base_mp4)
    if p.suffix.lower() != ".mp4":
        raise RuntimeError(f"Base filename is not .mp4: {base_mp4}")
    unique = f"{int(time.time())}_{uuid.uuid4().hex}_{p.name}"
    return dst_dir / unique

# ---------- Video Processor ----------

class VideoProcessor:
    def __init__(self, temp_dir: Path, ffmpeg_opts: dict):
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_opts = ffmpeg_opts

    def ffmpeg_available(self) -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception:
            return False

    def compress_and_strip(self, src: Path, dst: Path) -> bool:
        if src.suffix.lower() != ".mp4":
            logger.error("ffmpeg input is not .mp4: %s", src)
            return False
        if dst.suffix.lower() != ".mp4":
            logger.error("ffmpeg output is not .mp4: %s", dst)
            return False

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i", str(src),
            "-map_metadata", "-1",
            "-map_chapters", "-1",
            "-c:v", self.ffmpeg_opts["vcodec"],
            "-preset", self.ffmpeg_opts["preset"],
            "-crf", str(self.ffmpeg_opts["crf"]),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", self.ffmpeg_opts["audio_bitrate"],
            "-movflags", "+faststart",
            str(dst),
        ]

        logger.info("Running ffmpeg: %s -> %s", src.name, dst.name)

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            logger.error("ffmpeg failed: %s", proc.stderr[-1000:])
            return False

        if not dst.exists() or dst.stat().st_size == 0:
            logger.error("ffmpeg produced invalid output: %s", dst)
            return False

        return True

# ---------- Watchdog Handler ----------

class ProcessingFileHandler(FileSystemEventHandler):
    def __init__(self, processor: VideoProcessor, watch_dir: Path, output_dir: Path, temp_dir: Path):
        self.processor = processor
        self.watch_dir = watch_dir
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self._processing = set()

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_handle(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            self._maybe_handle(Path(event.dest_path))

    def _maybe_handle(self, path: Path):
        if is_processing_file(path):
            self._handle(path)

    def _handle(self, src: Path):
        key = str(src.resolve())
        if key in self._processing:
            return
        self._processing.add(key)

        try:
            if not stable_file_wait(src):
                return

            temp_raw = self.temp_dir / src.name
            shutil.move(str(src), str(temp_raw))

            clean_name = strip_processing_suffix(temp_raw.name)
            if not clean_name.lower().endswith(".mp4"):
                raise RuntimeError(f"Resulting filename is not mp4: {clean_name}")

            temp_mp4 = self.temp_dir / clean_name
            temp_raw.rename(temp_mp4)

            compressed = self.temp_dir / f"compressed_{uuid.uuid4().hex}.mp4"
            if not self.processor.compress_and_strip(temp_mp4, compressed):
                return

            final_path = unique_output_name(self.output_dir, temp_mp4.name)
            shutil.move(str(compressed), str(final_path))

            try:
                scan = subprocess.run(
                    ["fireshare", "scan-video", "--path", str(final_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                logger.info("fireshare scan exit=%s", scan.returncode)
            except Exception as exc:
                logger.warning("fireshare scan failed: %s", exc)

            logger.info("Finished: %s", final_path.name)

        finally:
            for p in (locals().get("temp_mp4"), locals().get("compressed")):
                try:
                    if p and p.exists():
                        p.unlink()
                except Exception:
                    pass
            self._processing.discard(key)

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-dir", default="~/fireshare/clips/uploads")
    parser.add_argument("--output-dir", default="~/fireshare/clips/videos")
    parser.add_argument("--temp-dir", default="~/fireshare/clips/temp")
    args = parser.parse_args()

    watch_dir = Path(os.path.expanduser(args.watch_dir)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    temp_dir = Path(os.path.expanduser(args.temp_dir)).resolve()

    for d in (watch_dir, output_dir, temp_dir):
        d.mkdir(parents=True, exist_ok=True)

    processor = VideoProcessor(
        temp_dir=temp_dir,
        ffmpeg_opts=FFMPEG_DEFAULT,
    )

    if not processor.ffmpeg_available():
        logger.error("ffmpeg not available")
        sys.exit(1)

    handler = ProcessingFileHandler(processor, watch_dir, output_dir, temp_dir)

    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()

    logger.info("Watching %s", watch_dir)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

if __name__ == "__main__":
    main()
