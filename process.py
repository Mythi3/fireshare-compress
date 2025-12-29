#!/usr/bin/env python3
"""
Production-grade Fireshare processor with quality presets, robust Docker scan,
and automatic poster (thumbnail) generation for each video.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import shlex
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception:
    print("Missing dependency: watchdog. Install with `pip install watchdog`.")
    raise

# ---------- Defaults & presets ----------
DEFAULT_CHECK_INTERVAL = 2.0
DEFAULT_STABLE_CHECKS = 3
DEFAULT_TIMEOUT = 300
DEFAULT_CONCURRENCY = 1
LOG_DIR = Path("/var/log/fireshare_processor")
LOG_FILE = LOG_DIR / f"processor_{datetime.utcnow().strftime('%Y%m%d')}.log"

QUALITY_PRESETS = {
    "low": {"crf": 28, "preset": "fast", "audio_bitrate": "128k"},
    "medium": {"crf": 23, "preset": "medium", "audio_bitrate": "192k"},
    "high": {"crf": 20, "preset": "slow", "audio_bitrate": "256k"},
}

RESOLUTION_MAP = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "auto": None,
}

VIDEO_SUFFIXES = (".mp4.processing", ".mp4.processed")

# ---------- Logging ----------
def setup_logging(debug: bool = False):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE)]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(threadName)s %(message)s",
        handlers=handlers,
    )


logger = logging.getLogger("fireshare_processor")

# ---------- Helpers ----------
def safe_strip_suffix(name: str, suffix: str) -> str:
    if name.lower().endswith(suffix.lower()):
        return name[: -len(suffix)]
    return name


def wait_for_stable(
    path: Path,
    timeout: int = DEFAULT_TIMEOUT,
    interval: float = DEFAULT_CHECK_INTERVAL,
    stable_checks: int = DEFAULT_STABLE_CHECKS,
) -> bool:
    start = time.time()
    last = -1
    stable = 0
    while time.time() - start < timeout:
        if not path.exists():
            logger.warning("File disappeared while waiting: %s", path)
            return False
        try:
            size = path.stat().st_size
        except Exception as exc:
            logger.debug("stat failed: %s", exc)
            size = -1
        if size == last and size > 0:
            stable += 1
            if stable >= stable_checks:
                logger.debug("File stable: %s (%d bytes)", path, size)
                return True
        else:
            stable = 0
        last = size
        time.sleep(interval)
    logger.error("Timeout waiting for stable file: %s", path)
    return False


def unique_final_name(output_dir: Path, base_name: str, preserve_name: bool = True) -> Path:
    base = Path(base_name)
    if preserve_name:
        candidate = output_dir / base.name
        counter = 1
        while candidate.exists():
            candidate = output_dir / f"{base.stem}_{counter}{base.suffix}"
            counter += 1
        return candidate
    else:
        unique = f"{int(time.time())}_{uuid.uuid4().hex}_{base.name}"
        return output_dir / unique


def ensure_ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def is_processing_file(p: Path) -> bool:
    return any(p.name.lower().endswith(s) for s in VIDEO_SUFFIXES)


# ---------- File processor ----------
class FileProcessor:
    def __init__(
        self,
        temp_dir: Path,
        output_dir: Path,
        quality: str,
        resolution: Optional[Tuple[int, int]],
        scan_cmd_template: Optional[str],
        preserve_name: bool,
        dry_run: bool = False,
    ):
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.quality = quality
        self.resolution = resolution
        self.scan_cmd_template = scan_cmd_template
        self.preserve_name = preserve_name
        self.dry_run = dry_run

        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["medium"])
        self.vcodec = "libx264"
        self.crf = preset["crf"]
        self.ff_preset = preset["preset"]
        self.audio_bitrate = preset["audio_bitrate"]

    def _create_workdir(self) -> Path:
        w = self.temp_dir / f"work_{uuid.uuid4().hex}"
        w.mkdir(parents=True, exist_ok=False)
        return w

    def _cleanup_workdir(self, wd: Path):
        try:
            shutil.rmtree(wd)
        except Exception as exc:
            logger.debug("Failed to cleanup workdir %s: %s", wd, exc)

    def _run_ffmpeg(self, src: Path, dst: Path) -> bool:
        if src.suffix.lower() != ".mp4":
            logger.error("ffmpeg input must be .mp4, got: %s", src)
            return False
        if dst.suffix.lower() != ".mp4":
            logger.error("ffmpeg output must be .mp4, got: %s", dst)
            return False

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(src),
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
        ]

        if self.resolution:
            w, h = self.resolution
            vf = f"scale=w={w}:h={h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
            cmd += ["-vf", vf]

        cmd += [
            "-c:v",
            self.vcodec,
            "-preset",
            self.ff_preset,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            self.audio_bitrate,
            "-movflags",
            "+faststart",
            str(dst),
        ]

        logger.info("Running ffmpeg: %s -> %s (quality=%s, resolution=%s)", src.name, dst.name, self.quality, ("auto" if not self.resolution else f"{self.resolution[1]}p"))
        if self.dry_run:
            logger.info("dry-run: ffmpeg cmd: %s", " ".join(cmd))
            return True

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            logger.error("ffmpeg failed for %s: %s", src.name, proc.stderr[-1000:])
            return False
        if not dst.exists() or dst.stat().st_size == 0:
            logger.error("ffmpeg produced invalid output for %s", src.name)
            return False
        return True

    def _generate_poster(self, final_mp4: Path, poster_png: Path, time_position: float = 3.0) -> bool:
        """
        Use ffmpeg to extract a poster image at `time_position` seconds.
        """
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(final_mp4),
            "-ss",
            str(time_position),
            "-vframes",
            "1",
            "-vf",
            "scale=320:-1",
            str(poster_png),
        ]

        logger.info("Generating poster: %s", poster_png.name)
        if self.dry_run:
            logger.info("dry-run: poster cmd: %s", " ".join(cmd))
            return True

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            logger.error("poster generation failed for %s: %s", final_mp4.name, proc.stderr[-500:])
            return False
        if not poster_png.exists() or poster_png.stat().st_size == 0:
            logger.error("poster file missing: %s", poster_png)
            return False
        return True

    def _run_scan(self, final_path: Path, max_retries: int = 3, initial_backoff: float = 1.0) -> bool:
        if not self.scan_cmd_template:
            logger.debug("No scan command configured; skipping scan for %s", final_path)
            return True

        cmd_str = self.scan_cmd_template.replace("{filename}", final_path.name)
        cmd = shlex.split(cmd_str)

        logger.info("Running scan command: %s", cmd_str)
        if self.dry_run:
            logger.info("dry-run: scan cmd: %s", cmd_str)
            return True

        backoff = initial_backoff
        for attempt in range(1, max_retries + 1):
            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                logger.info("scan attempt=%d exit=%s stdout=%s stderr=%s", attempt, res.returncode, (res.stdout or "").strip(), (res.stderr or "").strip())
                if res.returncode == 0:
                    return True
            except Exception as exc:
                logger.warning("Scan execution failed on attempt %d: %s", attempt, exc)

            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2.0

        logger.error("Scan failed after %d attempts for %s", max_retries, final_path)
        return False

    def process(self, src_path: Path) -> bool:
        logger.info("Starting processing: %s", src_path)
        if not wait_for_stable(src_path):
            logger.error("File not stable: %s", src_path)
            return False

        workdir: Optional[Path] = None
        try:
            workdir = self._create_workdir()
            temp_raw = workdir / src_path.name
            try:
                shutil.move(str(src_path), str(temp_raw))
            except Exception as exc:
                logger.exception("Failed to move to workdir: %s", exc)
                return False

            clean_name = safe_strip_suffix(temp_raw.name, ".processing")
            clean_name = safe_strip_suffix(clean_name, ".processed")
            if not clean_name.lower().endswith(".mp4"):
                logger.error("Resulting name is not .mp4 after strip: %s", clean_name)
                return False

            temp_mp4 = workdir / clean_name
            temp_raw.rename(temp_mp4)

            compressed_temp = workdir / "compressed_output.mp4"
            if not self._run_ffmpeg(temp_mp4, compressed_temp):
                logger.error("Compression failed for %s", temp_mp4)
                return False

            final = unique_final_name(self.output_dir, temp_mp4.name, preserve_name=self.preserve_name)
            try:
                shutil.move(str(compressed_temp), str(final))
            except Exception as exc:
                logger.exception("Failed to move final to output: %s", exc)
                return False

            logger.info("Moved final video to: %s", final)

            # Generate poster alongside the final video
            poster_file = final.with_suffix(".poster.png")
            self._generate_poster(final, poster_file)

            # Run container scan
            scan_ok = self._run_scan(final)
            if not scan_ok:
                logger.warning("Scan reported failure for %s", final)

            return True
        finally:
            if workdir:
                self._cleanup_workdir(workdir)

# ---------- Watcher / Coordinator ----------
class IncomingWatcher(FileSystemEventHandler):
    def __init__(self, processor: FileProcessor, watch_dir: Path, executor: ThreadPoolExecutor):
        self.processor = processor
        self.watch_dir = watch_dir
        self.executor = executor
        self._processing = set()
        self._lock = threading.Lock()

    def on_created(self, event):
        if event.is_directory:
            return
        self._maybe_submit(Path(event.src_path))

    def on_moved(self, event):
        if event.is_directory:
            return
        self._maybe_submit(Path(event.dest_path))

    def _maybe_submit(self, path: Path):
        if not is_processing_file(path):
            return
        key = str(path.resolve())
        with self._lock:
            if key in self._processing:
                logger.debug("Already queued: %s", key)
                return
            self._processing.add(key)
        logger.info("Queueing file for processing: %s", path)
        fut = self.executor.submit(self._run_task, path)
        fut.add_done_callback(lambda f, k=key: self._done_callback(k, f))

    def _run_task(self, path: Path) -> bool:
        try:
            return self.processor.process(path)
        except Exception as exc:
            logger.exception("Unhandled error processing %s: %s", path, exc)
            return False

    def _done_callback(self, key: str, fut):
        with self._lock:
            try:
                self._processing.discard(key)
            except Exception:
                pass

# ---------- Graceful shutdown ----------
shutdown_event = threading.Event()

def install_signal_handlers(observer: Observer, executor: ThreadPoolExecutor):
    def _signal(signum, frame):
        logger.info("Signal received: %s; shutting down...", signum)
        shutdown_event.set()
        try:
            observer.stop()
        except Exception:
            pass
        try:
            executor.shutdown(wait=False)
        except Exception:
            pass

    signal.signal(signal.SIGINT, _signal)
    signal.signal(signal.SIGTERM, _signal)

# ---------- CLI / main ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Production Fireshare processing daemon")
    p.add_argument("--watch-dir", default="~/fireshare/clips/uploads", help="Directory to watch for incoming .mp4.processing files")
    p.add_argument("--temp-dir", default="~/fireshare/clips/temp", help="Temporary working directory")
    p.add_argument("--output-dir", default="~/fireshare/clips/videos", help="Fireshare watched output directory (host path mounted to /videos in container)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Number of workers")
    p.add_argument("--quality", choices=["low", "medium", "high"], default="medium", help="Quality preset")
    p.add_argument("--resolution", choices=["1080p", "720p", "auto"], default="1080p")
    p.add_argument("--preserve-name", action="store_true", default=True)
    p.add_argument("--no-preserve-name", action="store_true")
    p.add_argument("--scan-cmd", default="docker exec fireshare fireshare scan-video --path /videos/{filename}", help="Scan command template")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p

def main():
    args = build_parser().parse_args()
    setup_logging(debug=args.debug)

    watch_dir = Path(os.path.expanduser(args.watch-dir)).resolve()
    temp_dir = Path(os.path.expanduser(args.temp-dir)).resolve()
    output_dir = Path(os.path.expanduser(args.output-dir)).resolve()

    for d in (watch_dir, temp_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    if watch_dir == output_dir:
        logger.error("watch-dir and output-dir must differ")
        sys.exit(2)

    if not ensure_ffmpeg_available() and not args.dry-run:
        logger.error("ffmpeg/ffprobe not available")
        sys.exit(3)

    resolution = RESOLUTION_MAP.get(args.resolution)
    preserve = args.preserve-name and not args.no-preserve-name

    processor = FileProcessor(
        temp_dir=temp_dir,
        output_dir=output_dir,
        quality=args.quality,
        resolution=resolution,
        scan_cmd_template=args.scan-cmd,
        preserve_name=preserve,
        dry_run=args.dry-run,
    )

    executor = ThreadPoolExecutor(max_workers=max(1, args.concurrency))
    observer = Observer()

    watcher = IncomingWatcher(processor=processor, watch_dir=watch_dir, executor=executor)
    observer.schedule(watcher, str(watch_dir), recursive=False)
    observer.start()

    install_signal_handlers(observer, executor) 

    logger.info("Started processor: watch %s -> output %s (temp %s)", watch_dir, output_dir, temp_dir)

    try:
        while not shutdown_event.is_set():
            time.sleep(1)
    finally:
        observer.stop()
        observer.join(timeout=5)
        executor.shutdown(wait=True)
        logger.info("Shutdown complete")

if __name__ == "__main__":
    main()
