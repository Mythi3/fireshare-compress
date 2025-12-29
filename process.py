#!/usr/bin/env python3
"""
Production-grade Fireshare processing daemon

Workflow:
  watch_temp (incoming uploads with .mp4.processing) -> temp processing -> compress -> write final mp4 into Fireshare /videos dir -> run scan

Features:
- Safe handling of .mp4.processing suffix (rename in temp before ffmpeg)
- Wait for file size stability to avoid partial uploads
- Per-file isolated temp working directory to avoid collisions
- Configurable concurrency (ThreadPoolExecutor)
- Robust logging with timestamped log file
- CLI flags for ffmpeg parameters, directories, dry-run, preserve-name
- Validations (ffmpeg/ffprobe availability, directories, avoiding watch==output)
- Graceful shutdown (SIGINT/SIGTERM)
- Optional custom scan-command template (default: "fireshare scan-video --path {path}")

Dependencies:
- Python 3.8+
- watchdog (pip install watchdog)
- ffmpeg/ffprobe in PATH

Example:
  ./process_production.py \
    --watch-dir ~/fireshare/clips/uploads \
    --temp-dir ~/fireshare/clips/temp \
    --output-dir ~/fireshare/clips/videos \
    --concurrency 2

"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception as exc:
    print("Missing dependency: watchdog. Install with `pip install watchdog`.")
    raise

# ---------- Configurable defaults ----------
DEFAULT_CHECK_INTERVAL = 2.0
DEFAULT_STABLE_CHECKS = 3
DEFAULT_TIMEOUT = 300
DEFAULT_CONCURRENCY = 1
LOG_DIR = Path("/var/log/fireshare_processor")
LOG_FILE = LOG_DIR / f"processor_{datetime.utcnow().strftime('%Y%m%d')}.log"

# ---------- Logging setup ----------
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

# ---------- Utilities ----------

def safe_strip_suffix(name: str, suffix: str) -> str:
    if name.lower().endswith(suffix.lower()):
        return name[: -len(suffix)]
    return name


def wait_for_stable(path: Path, timeout: int = DEFAULT_TIMEOUT, interval: float = DEFAULT_CHECK_INTERVAL, stable_checks: int = DEFAULT_STABLE_CHECKS) -> bool:
    """Wait until file size is stable for `stable_checks` checks.
    Returns True when stable, False on timeout or file disappearance.
    """
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


def unique_final_name(output_dir: Path, base_name: str, preserve_name: bool = False) -> Path:
    """Return a unique final filename. If preserve_name True, attempt to keep original filename and append suffix when conflict.
    Otherwise return timestamp_uuid_original.mp4
    """
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

# ---------- Worker: single-file processing ----------
class FileProcessor:
    def __init__(self, temp_dir: Path, output_dir: Path, ffmpeg_opts: dict, scan_cmd_template: Optional[str], preserve_name: bool, dry_run: bool = False):
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.ffmpeg_opts = ffmpeg_opts
        self.scan_cmd_template = scan_cmd_template
        self.preserve_name = preserve_name
        self.dry_run = dry_run
        self._lock = threading.Lock()

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
            "-i", str(src),
            "-map_metadata", "-1",
            "-map_chapters", "-1",
            "-c:v", self.ffmpeg_opts.get("vcodec", "libx264"),
            "-preset", self.ffmpeg_opts.get("preset", "medium"),
            "-crf", str(self.ffmpeg_opts.get("crf", 23)),
            "-pix_fmt", "yuv420p",
            "-c:a", self.ffmpeg_opts.get("acodec", "aac"),
            "-b:a", self.ffmpeg_opts.get("audio_bitrate", "192k"),
            "-movflags", "+faststart",
            str(dst),
        ]

        logger.info("Running ffmpeg: %s -> %s", src.name, dst.name)
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
        logger.info("ffmpeg success: %s -> %s", src.name, dst.name)
        return True

        def _run_scan(self, final_path: Path) -> bool:
        if not self.scan_cmd_template:
            logger.debug("No scan command configured; skipping scan for %s", final_path)
            return True
        cmd_str = self.scan_cmd_template.format(path=str(final_path))
        import shlex
        cmd = shlex.split(cmd_str)
        logger.info("Running scan command: %s", cmd_str)
        if self.dry_run:
            logger.info("dry-run: scan cmd: %s", cmd_str)
            return True
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logger.info("scan exit=%s stdout=%s stderr=%s", res.returncode, (res.stdout or "").strip(), (res.stderr or "").strip())
            return res.returncode == 0
        except Exception as exc:
            logger.warning("Scan command failed: %s", exc)
            return False

    def process(self, src_path: Path) -> bool:
        """Process a single .mp4.processing file end-to-end.
        Returns True on success.
        """
        logger.info("Starting processing: %s", src_path)

        # Wait until stable
        if not wait_for_stable(src_path):
            logger.error("File not stable: %s", src_path)
            return False

        # Create isolated workdir
        workdir = None
        try:
            workdir = self._create_workdir()
            # Move file to workdir (still with original name)
            temp_raw = workdir / src_path.name
            try:
                shutil.move(str(src_path), str(temp_raw))
            except Exception as exc:
                logger.exception("Failed to move to workdir: %s", exc)
                return False

            # Strip .processing / .processed suffix to target_mp4 name
            clean_name = safe_strip_suffix(temp_raw.name, ".processing")
            clean_name = safe_strip_suffix(clean_name, ".processed")
            if not clean_name.lower().endswith(".mp4"):
                logger.error("Resulting name is not .mp4 after strip: %s", clean_name)
                return False

            temp_mp4 = workdir / clean_name
            # Rename raw->mp4 (atomic within same fs)
            temp_raw.rename(temp_mp4)

            # Prepare compressed temp output name (guaranteed .mp4)
            compressed_temp = workdir / "compressed_output.mp4"

            # Run ffmpeg (compress + strip metadata)
            if not self._run_ffmpeg(temp_mp4, compressed_temp):
                logger.error("Compression failed for %s", temp_mp4)
                return False

            # Determine final name in output_dir
            final = unique_final_name(self.output_dir, temp_mp4.name, preserve_name=self.preserve_name)

            # Move compressed into final location (atomic move when possible)
            try:
                shutil.move(str(compressed_temp), str(final))
            except Exception as exc:
                logger.exception("Failed to move final to output: %s", exc)
                return False

            logger.info("Moved final video to: %s", final)

            # Run scan command (if configured)
            scan_ok = self._run_scan(final)
            if not scan_ok:
                logger.warning("Scan reported failure for %s", final)

            return True

        finally:
            if workdir:
                self._cleanup_workdir(workdir)

# ---------- Watcher / Coordinator ----------
class IncomingWatcher(FileSystemEventHandler):
    def __init__(self, processor: FileProcessor, watch_dir: Path, executor: ThreadPoolExecutor, seen_delay: float = 0.1):
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
        # Accept only .mp4.processing or .mp4.processed
        if not (path.name.lower().endswith('.mp4.processing') or path.name.lower().endswith('.mp4.processed')):
            return
        key = str(path.resolve())
        with self._lock:
            if key in self._processing:
                logger.debug("Already queued: %s", key)
                return
            self._processing.add(key)
        logger.info("Queueing file for processing: %s", path)
        # submit to executor
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
    p.add_argument("--output-dir", default="~/fireshare/clips/videos", help="Fireshare watched output directory (mounted to /videos in container)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Number of concurrent processing workers")
    p.add_argument("--crf", type=int, default=23, help="ffmpeg CRF (lower better quality)")
    p.add_argument("--preset", default="medium", help="ffmpeg preset")
    p.add_argument("--audio-bitrate", default="192k", help="audio bitrate for ffmpeg")
    p.add_argument("--scan-cmd", default="docker exec fireshare fireshare scan-video --path {path}", help="Command template to run after moving final file. Use {path} to interpolate the file path.")
    p.add_argument("--preserve-name", action="store_true", help="Attempt to preserve original filename in output directory (appends suffix on conflict)")
    p.add_argument("--dry-run", action="store_true", help="Do everything except actually run ffmpeg/move files (useful for testing)")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p


def main():
    args = build_parser().parse_args()
    setup_logging(debug=args.debug)

    watch_dir = Path(os.path.expanduser(args.watch_dir)).resolve()
    temp_dir = Path(os.path.expanduser(args.temp_dir)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()

    for d in (watch_dir, temp_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    if watch_dir == output_dir:
        logger.error("watch-dir and output-dir must be different to ensure Fireshare treats files as new. Exiting.")
        sys.exit(2)

    if not ensure_ffmpeg_available() and not args.dry_run:
        logger.error("ffmpeg/ffprobe not found in PATH. Install ffmpeg or run with --dry-run for testing.")
        sys.exit(3)

    ffmpeg_opts = {"vcodec": "libx264", "crf": args.crf, "preset": args.preset, "audio_bitrate": args.audio_bitrate}

    processor = FileProcessor(temp_dir=temp_dir, output_dir=output_dir, ffmpeg_opts=ffmpeg_opts, scan_cmd_template=args.scan_cmd, preserve_name=args.preserve_name, dry_run=args.dry_run)

    executor = ThreadPoolExecutor(max_workers=max(1, args.concurrency))
    observer = Observer()

    watcher = IncomingWatcher(processor=processor, watch_dir=watch_dir, executor=executor)
    observer.schedule(watcher, str(watch_dir), recursive=False)
    observer.start()

    install_signal_handlers(observer, executor)

    logger.info("Started processor. Watching %s -> output %s (temp %s). concurrency=%d", watch_dir, output_dir, temp_dir, args.concurrency)

    try:
        while not shutdown_event.is_set():
            time.sleep(1)
    finally:
        logger.info("Shutting down observer and executor...")
        try:
            observer.stop()
            observer.join(timeout=5)
        except Exception:
            pass
        try:
            executor.shutdown(wait=True)
        except Exception:
            pass
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
