"""Video composition and assembly component for VibeFrame 2.0."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
import logging
import numpy as np
import cv2
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips, CompositeVideoClip, vfx
import ffmpeg

from .models import Storyboard, SceneDescription
from .exceptions import VideoProcessingError, VideoGenerationError
from .utils import setup_logging

logger = setup_logging()

class VideoCompositor:
    """
    Video composition and assembly component.
    
    This component handles the final assembly of generated video clips into
    a complete music video, including audio synchronization, transitions,
    and format optimization.
    """
    
    def __init__(self):
        """Initialize VideoCompositor."""
        self.temp_dir = None
        self.supported_codecs = ["libx264", "libx265", "libvpx-vp9"]
        self.quality_presets = {
            "draft": {"crf": 28, "preset": "ultrafast"},
            "standard": {"crf": 23, "preset": "medium"},
            "high": {"crf": 18, "preset": "slow"},
            "maximum": {"crf": 15, "preset": "veryslow"}
        }
        
        logger.info("VideoCompositor initialized")
    
    def concatenate_clips(
        self,
        clip_paths: List[str],
        transition_type: str = "cut",
        transition_duration: float = 0.5
    ) -> str:
        """
        Concatenate video clips with transitions.
        
        Args:
            clip_paths: List of paths to video clips
            transition_type: Type of transition ("cut", "fade", "dissolve", "wipe")
            transition_duration: Duration of transition in seconds
            
        Returns:
            Path to concatenated video file
            
        Raises:
            VideoProcessingError: If concatenation fails
        """
        if not clip_paths:
            raise VideoProcessingError("No clips provided for concatenation")
        
        # Validate all clip files exist
        for clip_path in clip_paths:
            if not Path(clip_path).exists():
                raise VideoProcessingError(f"Clip file not found: {clip_path}")
        
        try:
            logger.info(f"Concatenating {len(clip_paths)} clips with {transition_type} transitions")
            
            if transition_type == "cut":
                # Simple concatenation without transitions
                return self._concatenate_simple(clip_paths)
            else:
                # Concatenation with transitions
                return self._concatenate_with_transitions(
                    clip_paths, transition_type, transition_duration
                )
                
        except Exception as e:
            error_msg = f"Video concatenation failed: {str(e)}"
            logger.error(error_msg)
            raise VideoProcessingError(error_msg) from e
    
    def _concatenate_simple(self, clip_paths: List[str]) -> str:
        """Simple concatenation without transitions using FFmpeg."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            output_path = temp_file.name
        
        try:
            # Create input list for FFmpeg
            input_list_path = output_path.replace(".mp4", "_inputs.txt")
            with open(input_list_path, "w") as f:
                for clip_path in clip_paths:
                    f.write(f"file '{os.path.abspath(clip_path)}'\n")
            
            # Use FFmpeg concat demuxer for fast concatenation
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", input_list_path,
                "-c", "copy",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up input list file
            os.unlink(input_list_path)
            
            if result.returncode != 0:
                raise VideoProcessingError(f"FFmpeg concatenation failed: {result.stderr}")
            
            logger.info(f"Simple concatenation completed: {output_path}")
            return output_path
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise
    
    def _concatenate_with_transitions(
        self,
        clip_paths: List[str],
        transition_type: str,
        transition_duration: float
    ) -> str:
        """Concatenate clips with transitions using MoviePy."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            output_path = temp_file.name
        
        try:
            # Load all clips
            clips = []
            for clip_path in clip_paths:
                clip = VideoFileClip(clip_path)
                clips.append(clip)
            
            if transition_type == "fade":
                # Apply fade transitions
                processed_clips = []
                for i, clip in enumerate(clips):
                    if i == 0:
                        # First clip: fade in only
                        processed_clip = clip.with_effects([vfx.FadeIn(transition_duration)])
                    elif i == len(clips) - 1:
                        # Last clip: fade out only
                        processed_clip = clip.with_effects([vfx.FadeOut(transition_duration)])
                    else:
                        # Middle clips: fade in and out
                        processed_clip = clip.with_effects([vfx.FadeIn(transition_duration), vfx.FadeOut(transition_duration)])
                    
                    processed_clips.append(processed_clip)
                
                # Concatenate with overlap
                final_clip = self._concatenate_with_overlap(processed_clips, transition_duration)
                
            elif transition_type == "dissolve":
                # Cross-dissolve transitions
                final_clip = self._create_dissolve_transitions(clips, transition_duration)
                
            else:
                # Default to simple concatenation for unsupported transitions
                final_clip = concatenate_videoclips(clips)
            
            # Write final video
            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Clean up clips
            for clip in clips:
                clip.close()
            final_clip.close()
            
            logger.info(f"Transition concatenation completed: {output_path}")
            return output_path
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise
    
    def _concatenate_with_overlap(self, clips: List, overlap_duration: float):
        """Concatenate clips with overlap for smooth transitions."""
        if len(clips) == 1:
            return clips[0]
        
        # Calculate start times with overlap
        start_times = [0]
        for i in range(1, len(clips)):
            prev_end = start_times[i-1] + clips[i-1].duration
            start_times.append(prev_end - overlap_duration)
        
        # Create composite with overlapping clips
        composite_clips = []
        for i, clip in enumerate(clips):
            clip_with_start = clip.set_start(start_times[i])
            composite_clips.append(clip_with_start)
        
        return CompositeVideoClip(composite_clips)
    
    def _create_dissolve_transitions(self, clips: List, transition_duration: float):
        """Create cross-dissolve transitions between clips."""
        if len(clips) == 1:
            return clips[0]
        
        result_clips = [clips[0]]
        
        for i in range(1, len(clips)):
            # Create dissolve transition
            prev_clip = result_clips[-1]
            curr_clip = clips[i]
            
            # Fade out previous clip
            prev_fade_out = prev_clip.with_effects([vfx.FadeOut(transition_duration)])
            
            # Fade in current clip
            curr_fade_in = curr_clip.with_effects([vfx.FadeIn(transition_duration)])
            
            # Set start time for current clip to overlap
            overlap_start = prev_clip.duration - transition_duration
            curr_fade_in = curr_fade_in.set_start(overlap_start)
            
            # Replace last clip with faded version and add current clip
            result_clips[-1] = prev_fade_out
            result_clips.append(curr_fade_in)
        
        return CompositeVideoClip(result_clips)
    
    def synchronize_audio(
        self,
        video_path: str,
        audio_path: str,
        sync_method: str = "stretch"
    ) -> str:
        """
        Synchronize video with audio track.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            sync_method: Synchronization method ("stretch", "trim", "loop")
            
        Returns:
            Path to synchronized video file
            
        Raises:
            VideoProcessingError: If synchronization fails
        """
        if not Path(video_path).exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")
        if not Path(audio_path).exists():
            raise VideoProcessingError(f"Audio file not found: {audio_path}")
        
        try:
            logger.info(f"Synchronizing video with audio using {sync_method} method")
            
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                output_path = temp_file.name
            
            # Load video and audio
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)
            
            video_duration = video_clip.duration
            audio_duration = audio_clip.duration
            
            logger.info(f"Video duration: {video_duration:.2f}s, Audio duration: {audio_duration:.2f}s")
            
            if sync_method == "stretch":
                # Stretch/compress video to match audio duration
                if abs(video_duration - audio_duration) > 0.1:  # Only if significant difference
                    speed_factor = video_duration / audio_duration
                    video_clip = video_clip.with_effects([vfx.MultiplySpeed(factor=speed_factor)])
                    logger.info(f"Video speed adjusted by factor: {speed_factor:.3f}")
                
            elif sync_method == "trim":
                # Trim longer track to match shorter one
                target_duration = min(video_duration, audio_duration)
                video_clip = video_clip.subclip(0, target_duration)
                audio_clip = audio_clip.subclip(0, target_duration)
                logger.info(f"Both tracks trimmed to {target_duration:.2f}s")
                
            elif sync_method == "loop":
                # Loop shorter track to match longer one
                target_duration = max(video_duration, audio_duration)
                
                if video_duration < target_duration:
                    # Loop video
                    loops_needed = int(np.ceil(target_duration / video_duration))
                    video_clips = [video_clip] * loops_needed
                    video_clip = concatenate_videoclips(video_clips).subclip(0, target_duration)
                    logger.info(f"Video looped {loops_needed} times")
                
                if audio_duration < target_duration:
                    # Loop audio
                    loops_needed = int(np.ceil(target_duration / audio_duration))
                    audio_clips = [audio_clip] * loops_needed
                    audio_clip = concatenate_audioclips(audio_clips).subclip(0, target_duration)
                    logger.info(f"Audio looped {loops_needed} times")
            
            # Set audio track
            final_clip = video_clip.set_audio(audio_clip)
            
            # Write synchronized video
            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Clean up
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
            logger.info(f"Audio synchronization completed: {output_path}")
            return output_path
            
        except Exception as e:
            error_msg = f"Audio synchronization failed: {str(e)}"
            logger.error(error_msg)
            raise VideoProcessingError(error_msg) from e
    
    def apply_transitions(
        self,
        clip1_path: str,
        clip2_path: str,
        transition_duration: float,
        transition_type: str
    ) -> str:
        """
        Apply transition between two clips.
        
        Args:
            clip1_path: Path to first clip
            clip2_path: Path to second clip
            transition_duration: Duration of transition in seconds
            transition_type: Type of transition
            
        Returns:
            Path to video with transition applied
            
        Raises:
            VideoProcessingError: If transition application fails
        """
        return self.concatenate_clips(
            [clip1_path, clip2_path],
            transition_type=transition_type,
            transition_duration=transition_duration
        )
    
    def normalize_clips(
        self,
        clip_paths: List[str],
        target_resolution: Tuple[int, int],
        target_fps: int
    ) -> List[str]:
        """
        Normalize clips to consistent resolution and frame rate.
        
        Args:
            clip_paths: List of paths to video clips
            target_resolution: Target resolution (width, height)
            target_fps: Target frame rate
            
        Returns:
            List of paths to normalized clips
            
        Raises:
            VideoProcessingError: If normalization fails
        """
        if not clip_paths:
            return []
        
        try:
            logger.info(f"Normalizing {len(clip_paths)} clips to {target_resolution} @ {target_fps}fps")
            
            normalized_paths = []
            
            for i, clip_path in enumerate(clip_paths):
                if not Path(clip_path).exists():
                    raise VideoProcessingError(f"Clip file not found: {clip_path}")
                
                # Create output path
                clip_name = Path(clip_path).stem
                with tempfile.NamedTemporaryFile(suffix=f"_normalized_{i}.mp4", delete=False) as temp_file:
                    output_path = temp_file.name
                
                # Use FFmpeg for efficient normalization
                width, height = target_resolution
                
                cmd = [
                    "ffmpeg", "-y",
                    "-i", clip_path,
                    "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                    "-r", str(target_fps),
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-preset", "medium",
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise VideoProcessingError(f"FFmpeg normalization failed for {clip_path}: {result.stderr}")
                
                normalized_paths.append(output_path)
                logger.debug(f"Normalized clip {i+1}/{len(clip_paths)}: {output_path}")
            
            logger.info(f"Clip normalization completed: {len(normalized_paths)} clips")
            return normalized_paths
            
        except Exception as e:
            error_msg = f"Clip normalization failed: {str(e)}"
            logger.error(error_msg)
            raise VideoProcessingError(error_msg) from e
    
    def export_final_video(
        self,
        video_path: str,
        output_path: str,
        codec: str = "libx264",
        quality: str = "standard",
        resolution: Optional[Tuple[int, int]] = None,
        fps: Optional[int] = None
    ) -> str:
        """
        Export final video with specified codec and quality.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video
            codec: Video codec to use
            quality: Quality preset ("draft", "standard", "high", "maximum")
            resolution: Optional target resolution
            fps: Optional target frame rate
            
        Returns:
            Path to exported video
            
        Raises:
            VideoProcessingError: If export fails
        """
        if not Path(video_path).exists():
            raise VideoProcessingError(f"Input video not found: {video_path}")
        
        if codec not in self.supported_codecs:
            logger.warning(f"Codec {codec} not in supported list, using libx264")
            codec = "libx264"
        
        if quality not in self.quality_presets:
            logger.warning(f"Quality {quality} not recognized, using standard")
            quality = "standard"
        
        try:
            logger.info(f"Exporting final video with {codec} codec, {quality} quality")
            
            # Get quality settings
            quality_settings = self.quality_presets[quality]
            
            # Build FFmpeg command
            cmd = ["ffmpeg", "-y", "-i", video_path]
            
            # Video encoding settings
            cmd.extend(["-c:v", codec])
            cmd.extend(["-crf", str(quality_settings["crf"])])
            cmd.extend(["-preset", quality_settings["preset"]])
            
            # Resolution scaling if specified
            if resolution:
                width, height = resolution
                cmd.extend(["-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"])
            
            # Frame rate if specified
            if fps:
                cmd.extend(["-r", str(fps)])
            
            # Audio encoding
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])
            
            # Output
            cmd.append(output_path)
            
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise VideoProcessingError(f"FFmpeg export failed: {result.stderr}")
            
            # Verify output file was created
            if not Path(output_path).exists():
                raise VideoProcessingError("Output file was not created")
            
            output_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Video export completed: {output_path} ({output_size:.1f} MB)")
            
            return output_path
            
        except Exception as e:
            error_msg = f"Video export failed: {str(e)}"
            logger.error(error_msg)
            raise VideoProcessingError(error_msg) from e
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
            
        Raises:
            VideoProcessingError: If unable to get video info
        """
        if not Path(video_path).exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")
        
        try:
            # Use ffprobe to get video information
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise VideoProcessingError(f"ffprobe failed: {result.stderr}")
            
            import json
            info = json.loads(result.stdout)
            
            # Extract relevant information
            video_stream = None
            audio_stream = None
            
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video" and video_stream is None:
                    video_stream = stream
                elif stream.get("codec_type") == "audio" and audio_stream is None:
                    audio_stream = stream
            
            video_info = {
                "duration": float(info.get("format", {}).get("duration", 0)),
                "size_bytes": int(info.get("format", {}).get("size", 0)),
                "format_name": info.get("format", {}).get("format_name", "unknown"),
            }
            
            if video_stream:
                video_info.update({
                    "width": int(video_stream.get("width", 0)),
                    "height": int(video_stream.get("height", 0)),
                    "fps": eval(video_stream.get("r_frame_rate", "0/1")),
                    "video_codec": video_stream.get("codec_name", "unknown"),
                    "pixel_format": video_stream.get("pix_fmt", "unknown"),
                })
            
            if audio_stream:
                video_info.update({
                    "audio_codec": audio_stream.get("codec_name", "unknown"),
                    "sample_rate": int(audio_stream.get("sample_rate", 0)),
                    "channels": int(audio_stream.get("channels", 0)),
                })
            
            return video_info
            
        except Exception as e:
            error_msg = f"Failed to get video info: {str(e)}"
            logger.error(error_msg)
            raise VideoProcessingError(error_msg) from e
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        if self.temp_dir and Path(self.temp_dir).exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info("Temporary files cleaned up")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup_temp_files()
        except:
            pass  # Ignore cleanup errors during destruction