#!/usr/bin/env python3
"""
Fireshare Video Processor with Watchdog
Monitors for .processing files, compresses them, and triggers Fireshare scan
"""

import os
import sys
import time
import shutil
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fireshare_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Quality presets for video compression
QUALITY_PRESETS = {
    'low': {
        '1080p': {
            'resolution': '1920x1080',
            'video_bitrate': '2M',
            'audio_bitrate': '128k',
            'crf': '28',
            'preset': 'fast'
        },
        '720p': {
            'resolution': '1280x720',
            'video_bitrate': '1.5M',
            'audio_bitrate': '128k',
            'crf': '28',
            'preset': 'fast'
        }
    },
    'medium': {
        '1080p': {
            'resolution': '1920x1080',
            'video_bitrate': '4M',
            'audio_bitrate': '192k',
            'crf': '23',
            'preset': 'medium'
        },
        '720p': {
            'resolution': '1280x720',
            'video_bitrate': '2.5M',
            'audio_bitrate': '192k',
            'crf': '23',
            'preset': 'medium'
        }
    },
    'high': {
        '1080p': {
            'resolution': '1920x1080',
            'video_bitrate': '6M',
            'audio_bitrate': '256k',
            'crf': '20',
            'preset': 'slow'
        },
        '720p': {
            'resolution': '1280x720',
            'video_bitrate': '4M',
            'audio_bitrate': '256k',
            'crf': '20',
            'preset': 'slow'
        }
    }
}


class VideoProcessor:
    """Handles video processing operations"""
    
    def __init__(self, quality='medium', resolution='1080p', temp_dir=None):
        self.quality = quality
        self.resolution = resolution
        self.temp_dir = Path(temp_dir or os.path.expanduser('~/fireshare/temp'))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        if quality not in QUALITY_PRESETS:
            raise ValueError(f"Invalid quality: {quality}. Must be one of {list(QUALITY_PRESETS.keys())}")
        if resolution not in QUALITY_PRESETS[quality]:
            raise ValueError(f"Invalid resolution: {resolution}. Must be one of {list(QUALITY_PRESETS[quality].keys())}")
        
        self.preset = QUALITY_PRESETS[quality][resolution]
        logger.info(f"VideoProcessor initialized: {quality} quality, {resolution} resolution")
    
    def get_video_info(self, video_path):
        """Get video information using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,duration',
                '-of', 'json',
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            data = json.loads(result.stdout)
            if 'streams' in data and len(data['streams']) > 0:
                return data['streams'][0]
            return None
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return None
    
    def compress_video_with_metadata_strip(self, input_path, output_path):
        """Compress video and strip metadata in a single pass"""
        try:
            # Build ffmpeg command - strip metadata during compression
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-map_metadata', '-1',  # Strip all metadata
                '-map_chapters', '-1',   # Strip chapters
                '-fflags', '+bitexact',  # Remove metadata timestamps
                '-vf', f"scale={self.preset['resolution']}:force_original_aspect_ratio=decrease,pad={self.preset['resolution']}:(ow-iw)/2:(oh-ih)/2",
                '-c:v', 'libx264',
                '-preset', self.preset['preset'],
                '-crf', self.preset['crf'],
                '-b:v', self.preset['video_bitrate'],
                '-maxrate', self.preset['video_bitrate'],
                '-bufsize', str(int(float(self.preset['video_bitrate'].rstrip('Mk')) * 2)) + self.preset['video_bitrate'][-1],
                '-c:a', 'aac',
                '-b:a', self.preset['audio_bitrate'],
                '-movflags', '+faststart',  # Enable streaming
                '-pix_fmt', 'yuv420p',      # Ensure compatibility
                '-y',
                str(output_path)
            ]
            
            logger.info(f"Starting compression with metadata stripping: {input_path.name}")
            logger.info(f"Quality: {self.quality}, Resolution: {self.resolution}")
            
            # Run compression with progress logging
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor progress
            stderr_output = []
            for line in process.stderr:
                stderr_output.append(line)
                if 'frame=' in line or 'time=' in line:
                    # Log progress periodically
                    if 'time=' in line:
                        logger.debug(line.strip())
            
            process.wait()
            
            if process.returncode != 0:
                logger.error(f"Compression failed: {''.join(stderr_output[-20:])}")
                return False
            
            # Verify output file exists and has size
            if output_path.exists() and output_path.stat().st_size > 0:
                original_size = input_path.stat().st_size
                compressed_size = output_path.stat().st_size
                reduction = ((original_size - compressed_size) / original_size) * 100
                logger.info(f"Compression complete: {output_path.name}")
                logger.info(f"Size reduction: {reduction:.1f}% ({original_size:,} -> {compressed_size:,} bytes)")
                return True
            else:
                logger.error(f"Output file is empty or doesn't exist: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return False
    
    def process_video(self, input_path, output_path):
        """Full processing pipeline: strip metadata and compress in one pass"""
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            if not input_path.exists():
                logger.error(f"Input file doesn't exist: {input_path}")
                return False
            
            logger.info(f"Processing video: {input_path.name}")
            logger.info("Stripping metadata and compressing video...")
            
            # Process video (metadata stripping + compression in one pass)
            if not self.compress_video_with_metadata_strip(input_path, output_path):
                return False
            
            logger.info(f"Processing complete: {output_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return False


class ProcessingFileHandler(FileSystemEventHandler):
    """Handles file system events for .processing files"""
    
    def __init__(self, processor, watch_dir, output_dir):
        self.processor = processor
        self.watch_dir = Path(watch_dir)
        self.output_dir = Path(output_dir)
        self.processing_files = set()  # Track files being processed
        
    def on_created(self, event):
        """Handle new file creation"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if it's a .processing file
        if file_path.suffix == '.processing' and file_path.stem.endswith('.mp4'):
            logger.info(f"Detected new file: {file_path.name}")
            
            # Wait for file to be fully written (check file size stability)
            if self._wait_for_file_complete(file_path):
                self._process_file(file_path)
    
    def on_moved(self, event):
        """Handle file rename/move events"""
        if event.is_directory:
            return
        
        dest_path = Path(event.dest_path)
        
        # Check if file was renamed to .processing
        if dest_path.suffix == '.processing' and dest_path.stem.endswith('.mp4'):
            logger.info(f"Detected renamed file: {dest_path.name}")
            
            if self._wait_for_file_complete(dest_path):
                self._process_file(dest_path)
    
    def _wait_for_file_complete(self, file_path, timeout=300, check_interval=2):
        """Wait for file to finish being written"""
        logger.info(f"Waiting for file to complete: {file_path.name}")
        
        start_time = time.time()
        last_size = -1
        stable_count = 0
        required_stable_checks = 3  # File size must be stable for 3 consecutive checks
        
        while time.time() - start_time < timeout:
            try:
                if not file_path.exists():
                    logger.warning(f"File disappeared: {file_path.name}")
                    return False
                
                current_size = file_path.stat().st_size
                
                if current_size == last_size and current_size > 0:
                    stable_count += 1
                    if stable_count >= required_stable_checks:
                        logger.info(f"File is complete: {file_path.name} ({current_size:,} bytes)")
                        return True
                else:
                    stable_count = 0
                
                last_size = current_size
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error checking file: {e}")
                time.sleep(check_interval)
        
        logger.error(f"Timeout waiting for file to complete: {file_path.name}")
        return False
    
    def _process_file(self, processing_file):
        """Process a .processing file"""
        try:
            # Prevent duplicate processing
            if str(processing_file) in self.processing_files:
                logger.warning(f"File already being processed: {processing_file.name}")
                return
            
            self.processing_files.add(str(processing_file))
            
            # Generate output filename (remove .processing extension)
            original_name = processing_file.stem  # e.g., "video.mp4"
            output_file = self.output_dir / original_name
            
            # Ensure output filename is unique
            counter = 1
            while output_file.exists():
                name_parts = original_name.rsplit('.', 1)
                if len(name_parts) == 2:
                    output_file = self.output_dir / f"{name_parts[0]}_{counter}.{name_parts[1]}"
                else:
                    output_file = self.output_dir / f"{original_name}_{counter}"
                counter += 1
            
            logger.info(f"Starting processing pipeline for: {processing_file.name}")
            
            # Process the video
            success = self.processor.process_video(processing_file, output_file)
            
            if success:
                logger.info(f"Successfully processed: {output_file.name}")
                
                # Trigger Fireshare scan
                self._trigger_fireshare_scan(output_file)
                
                # Delete the original .processing file
                try:
                    processing_file.unlink()
                    logger.info(f"Deleted original file: {processing_file.name}")
                except Exception as e:
                    logger.error(f"Failed to delete original file: {e}")
            else:
                logger.error(f"Failed to process: {processing_file.name}")
            
            # Remove from processing set
            self.processing_files.discard(str(processing_file))
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            self.processing_files.discard(str(processing_file))
    
    def _trigger_fireshare_scan(self, video_path):
        """Trigger Fireshare to scan the processed video"""
        try:
            logger.info(f"Triggering Fireshare scan for: {video_path.name}")
            
            subprocess.Popen(
                ["fireshare", "scan-video", f"--path={video_path}"],
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info("Fireshare scan triggered successfully")
            
        except Exception as e:
            logger.error(f"Failed to trigger Fireshare scan: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Fireshare Video Processor - Monitors and processes videos with compression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quality Presets:
  low     - Fast encoding, smaller files, lower quality
  medium  - Balanced encoding speed and quality (default)
  high    - Slower encoding, larger files, higher quality

Resolution Options:
  720p   - 1280x720 output
  1080p  - 1920x1080 output (default)

Examples:
  %(prog)s --quality medium
  %(prog)s --quality high --resolution 1080p
  %(prog)s --watch-dir ~/fireshare/uploads --output-dir ~/fireshare/processed
        """
    )
    
    parser.add_argument(
        '--quality',
        choices=['low', 'medium', 'high'],
        default='medium',
        help='Compression quality preset (default: medium)'
    )
    
    parser.add_argument(
        '--resolution',
        choices=['720p', '1080p'],
        default='1080p',
        help='Output resolution (default: 1080p)'
    )
    
    parser.add_argument(
        '--watch-dir',
        type=str,
        default='~/fireshare/uploads',
        help='Directory to monitor for .processing files (default: ~/fireshare/uploads)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='~/fireshare/uploads',
        help='Directory to output processed files (default: same as watch-dir)'
    )
    
    parser.add_argument(
        '--temp-dir',
        type=str,
        default='~/fireshare/temp',
        help='Temporary directory for processing (default: ~/fireshare/temp)'
    )
    
    args = parser.parse_args()
    
    # Expand paths
    watch_dir = Path(os.path.expanduser(args.watch_dir))
    output_dir = Path(os.path.expanduser(args.output_dir))
    temp_dir = Path(os.path.expanduser(args.temp_dir))
    
    # Create directories if they don't exist
    watch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Fireshare Video Processor Starting")
    logger.info("=" * 80)
    logger.info(f"Quality: {args.quality}")
    logger.info(f"Resolution: {args.resolution}")
    logger.info(f"Watch Directory: {watch_dir}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Temp Directory: {temp_dir}")
    logger.info("=" * 80)
    
    # Check for ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg is not installed or not in PATH. Please install FFmpeg.")
        sys.exit(1)
    
    # Check for fireshare command
    try:
        subprocess.run(['fireshare', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Fireshare command not found. Scan triggers will fail.")
    
    # Initialize processor
    processor = VideoProcessor(
        quality=args.quality,
        resolution=args.resolution,
        temp_dir=temp_dir
    )
    
    # Initialize file handler
    event_handler = ProcessingFileHandler(processor, watch_dir, output_dir)
    
    # Setup observer
    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)
    observer.start()
    
    logger.info("Monitoring started. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping monitor...")
        observer.stop()
        observer.join()
        logger.info("Monitor stopped.")


if __name__ == '__main__':
    main()
