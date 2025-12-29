#!/usr/bin/env python3
"""
Fireshare Video Processor
- Watches a directory for files that end with .processing or .processed (commonly written
  as original.mp4.processing)
- Waits until the source file stops changing (fully written)
- Moves the file to a temp directory
- Renames it to remove the .processing/.processed suffix (becomes .mp4)
- Strips metadata and compresses with ffmpeg (single pass)
- Moves the compressed file to the output directory
- Triggers a fireshare scan on the moved file
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fireshare_processor.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("fireshare_processor")

# ---------- Quality presets ----------
QUALITY_PRESETS = {
    "low": {
        "1080p": {"resolution": (1920, 1080), "crf": "28", "preset": "fast", "audio_bitrate": "128k"},
        "720p": {"resolution": (1280, 720), "crf": "28", "preset": "fast", "audio_bitrate": "128k"},
    },
    "medium": {
        "1080p": {"resolution": (1920, 1080), "crf": "23", "preset": "medium", "audio_bitrate": "192k"},
        "720p": {"resolution": (1280, 720), "crf": "23", "preset": "medium", "audio_bitrate": "192k"},
    },
    "high": {
        "1080p": {"resolution": (1920, 1080), "crf": "20", "preset": "slow", "audio_bitrate": "256k"},
        "720p": {"resolution": (1280, 720), "crf": "20", "preset": "slow", "audio_bitrate": "256k"},
    },
}


# ---------- Video processing ----------
class VideoProcessor:
    def __init__(self, quality="medium", resolution="1080p", temp_dir=None):
        if quality not in QUALITY_PRESETS:
            raise ValueError(f"Invalid quality: {quality}. Choices: {list(QUALITY_PRESETS.keys())}")
        if resolution not in QUALITY_PRESETS[quality]:
            raise ValueError(f"Invalid resolution: {resolution}. Choices: {list(QUALITY_PRESETS[quality].keys())}")

        self.quality = quality
        self.resolution = resolution
        self.preset = QUALITY_PRESETS[quality][resolution]
        self.temp_dir = Path(temp_dir) if temp_dir else Path(os.path.expanduser("~/fireshare/temp"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"VideoProcessor initialized: quality={quality}, resolution={resolution}, temp={self.temp_dir}")

    def compress_and_strip(self, input_path: Path, output_path: Path) -> bool:
        """
        Run ffmpeg to strip metadata and compress into output_path.
        Returns True on success.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            logger.error("Input for compression does not exist: %s", input_path)
            return False

        width, height = self.preset["resolution"]
        crf = self.preset["crf"]
        preset = self.preset["preset"]
        audio_bitrate = self.preset["audio_bitrate"]

        # scale & pad to the requested resolution while preserving aspect ratio
        vf = f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i", str(input_path),
            "-map_metadata", "-1",        # strip metadata
            "-map_chapters", "-1",       # strip chapters
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", crf,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-movflags", "+faststart",
            str(output_path),
        ]

        logger.info("Running ffmpeg: %s -> %s", input_path.name, output_path.name)
        logger.debug("ffmpeg cmd: %s", " ".join(cmd))

        try:
            # Capture stderr to show ffmpeg progress logs
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, universal_newlines=True)
            stderr_lines = []
            # read stderr continuously
            while True:
                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    stderr_lines.append(line)
                    # log periodic progress lines at debug level
                    if "frame=" in line or "time=" in line or "speed=" in line:
                        logger.debug(line.strip())
            process.wait()
            if process.returncode != 0:
                logger.error("ffmpeg returned non-zero (%s). Last lines:\n%s", process.returncode, "".join(stderr_lines[-40:]))
                return False

            # Validate output
            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error("ffmpeg produced empty or missing output: %s", output_path)
                return False

            orig_size = input_path.stat().st_size
            new_size = output_path.stat().st_size
            reduction = (1.0 - (new_size / orig_size)) * 100 if orig_size > 0 else 0.0
            logger.info("Compression finished: %s (%.1f%% reduction)", output_path.name, reduction)
            return True

        except Exception as e:
            logger.exception("Compression failed: %s", e)
            return False


# ---------- File system event handler ----------
class ProcessingFileHandler(FileSystemEventHandler):
    """
    Reacts to new .processing/.processed files.
    Workflow:
      - Wait for source size to stabilize
      - Move to temp folder
      - Rename to remove .processing/.processed -> becomes original .mp4
      - Compress/strip into a temp output file
      - Move compressed file to output directory (unique name)
      - Trigger fireshare scan
    """

    def __init__(self, processor: VideoProcessor, watch_dir: Path, output_dir: Path, temp_dir: Path):
        self.processor = processor
        self.watch_dir = Path(watch_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.processing_files = set()

    @staticmethod
    def _is_processing_file(path: Path):
        # Accept suffix .processing or .processed (common variants). The file should include ".mp4" before suffix.
        name = path.name
        lower = name.lower()
        return (lower.endswith(".mp4.processing") or lower.endswith(".mp4.processed"))

    def on_created(self, event):
        if event.is_directory:
            return
        src = Path(event.src_path)
        if not self._is_processing_file(src):
            return
        logger.info("Created: %s", src)
        self._handle(src)

    def on_moved(self, event):
        # handle moved/renamed into the watched folder
        if event.is_directory:
            return
        dest = Path(event.dest_path)
        if not self._is_processing_file(dest):
            return
        logger.info("Moved/renamed: %s", dest)
        self._handle(dest)

    def _wait_for_stable(self, path: Path, timeout=300, interval=2, stable_checks=3) -> bool:
        """
        Wait until size is stable for `stable_checks` consecutive checks.
        """
        logger.info("Waiting for completion (stable size) of %s", path)
        start = time.time()
        last_size = -1
        stable = 0
        while time.time() - start < timeout:
            if not path.exists():
                logger.warning("File disappeared while waiting: %s", path)
                return False
            try:
                size = path.stat().st_size
            except Exception as e:
                logger.debug("Stat failed: %s", e)
                size = -1

            if size == last_size and size > 0:
                stable += 1
                if stable >= stable_checks:
                    logger.info("File appears complete: %s (%d bytes)", path.name, size)
                    return True
            else:
                stable = 0
            last_size = size
            time.sleep(interval)

        logger.error("Timeout waiting for file to stabilize: %s", path)
        return False

    def _unique_target(self, target: Path) -> Path:
        """
        If target exists, append _1, _2, ... before suffix to make unique.
        """
        if not target.exists():
            return target
        base = target.stem
        suffix = target.suffix  # e.g., '.mp4'
        parent = target.parent
        counter = 1
        while True:
            candidate = parent / f"{base}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def _handle(self, src_path: Path):
        str_src = str(src_path.resolve())
        if str_src in self.processing_files:
            logger.info("Already processing: %s", src_path)
            return
        self.processing_files.add(str_src)

        try:
            # Ensure file finished writing in the watch location before move
            if not self._wait_for_stable(src_path):
                logger.error("Source file not stable or disappeared: %s", src_path)
                return

            # Move the file to temp directory (preserve name initially)
            temp_move_path = self.temp_dir / src_path.name
            try:
                logger.info("Moving %s -> %s", src_path, temp_move_path)
                shutil.move(str(src_path), str(temp_move_path))
            except Exception as e:
                logger.exception("Failed to move to temp: %s", e)
                return

            # Determine mp4 name by removing processing suffix
            name = temp_move_path.name
            if name.lower().endswith(".mp4.processing"):
                mp4_name = name[:-len(".processing")]
            elif name.lower().endswith(".mp4.processed"):
                mp4_name = name[:-len(".processed")]
            else:
                # Unexpected, but keep original
                mp4_name = name

            temp_mp4_path = self.temp_dir / mp4_name
            # Rename to remove the suffix
            try:
                logger.info("Renaming in temp: %s -> %s", temp_move_path.name, temp_mp4_path.name)
                temp_move_path.rename(temp_mp4_path)
            except Exception as e:
                logger.exception("Rename in temp failed: %s", e)
                # try to continue by using the moved file as-is
                temp_mp4_path = temp_move_path

            # Prepare compressed output name in temp
            compressed_temp = self.temp_dir / f"compressed_{int(time.time())}_{temp_mp4_path.name}"
            # Run compression + metadata strip
            ok = self.processor.compress_and_strip(temp_mp4_path, compressed_temp)
            if not ok:
                logger.error("Processing failed for %s", temp_mp4_path)
                # leave original or try cleanup
                return

            # Move compressed file back to the output directory
            final_target = self.output_dir / compressed_temp.name
            final_target = self._unique_target(final_target)
            try:
                logger.info("Moving compressed -> output: %s -> %s", compressed_temp, final_target)
                shutil.move(str(compressed_temp), str(final_target))
            except Exception as e:
                logger.exception("Failed to move compressed to output: %s", e)
                return

            # Trigger fireshare scan
            try:
                logger.info("Triggering fireshare scan for: %s", final_target)
                subprocess.Popen(["fireshare", "scan-video", f"--path={final_target}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                logger.warning("Could not trigger fireshare scan: %s", e)

            logger.info("Processing completed: %s", final_target)

        finally:
            # cleanup temp source if still present
            try:
                # remove temp original mp4 if exists
                if 'temp_mp4_path' in locals() and temp_mp4_path.exists():
                    temp_mp4_path.unlink()
            except Exception:
                pass
            self.processing_files.discard(str_src)


# ---------- CLI / main ----------
def main():
    parser = argparse.ArgumentParser(description="Fireshare video processor (monitor + compress)")
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--resolution", choices=["720p", "1080p"], default="1080p")
    parser.add_argument("--watch-dir", default="~/fireshare/clips/uploads")
    parser.add_argument("--output-dir", default="~/fireshare/clips/uploads")
    parser.add_argument("--temp-dir", default="~/fireshare/clips/temp")
    args = parser.parse_args()

    watch_dir = Path(os.path.expanduser(args.watch_dir)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    temp_dir = Path(os.path.expanduser(args.temp_dir)).resolve()

    watch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Fireshare processor")
    logger.info("Watch dir: %s", watch_dir)
    logger.info("Output dir: %s", output_dir)
    logger.info("Temp dir: %s", temp_dir)
    logger.info("Quality: %s, Resolution: %s", args.quality, args.resolution)

    # check ffmpeg availability
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        logger.error("ffmpeg/ffprobe not found or not in PATH. Install ffmpeg.")
        sys.exit(1)

    # warn if fireshare command missing, but continue
    try:
        subprocess.run(["fireshare", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        logger.warning("fireshare command not found in PATH; scan triggers will likely fail.")

    processor = VideoProcessor(quality=args.quality, resolution=args.resolution, temp_dir=temp_dir)
    handler = ProcessingFileHandler(processor=processor, watch_dir=watch_dir, output_dir=output_dir, temp_dir=temp_dir)
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()
    try:
        logger.info("Monitoring started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping monitor...")
        observer.stop()
    observer.join()
    logger.info("Monitor stopped.")


if __name__ == "__main__":
    main()
