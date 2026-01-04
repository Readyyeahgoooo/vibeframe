"""Video generation component for VibeFrame 2.0."""

import os
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Union
import logging
from pathlib import Path
from PIL import Image
import tempfile
import cv2

from .models import SceneDescription, CharacterDescriptor
from .exceptions import VideoGenerationError, ModelNotFoundError, InsufficientGPUMemoryError
from .utils import setup_logging

logger = setup_logging()

class LongCatVideoBackend:
    """
    LongCat-Video backend for text-to-video and image-to-video generation.
    
    This backend provides video generation capabilities using the LongCat-Video model,
    supporting text-to-video, image-to-video, and video continuation modes.
    """
    
    def __init__(self, checkpoint_dir: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize LongCat-Video backend.
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.checkpoint_dir = checkpoint_dir or self._get_default_checkpoint_dir()
        self.device = device or self._detect_device()
        self.model = None
        self.pipeline = None
        self.is_loaded = False
        
        # Model configuration
        self.max_frames = 16  # Maximum frames per generation
        self.default_fps = 8  # Default FPS for generated videos
        self.default_resolution = (512, 512)  # Default resolution
        
        logger.info(f"LongCatVideoBackend initialized with device: {self.device}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _get_default_checkpoint_dir(self) -> str:
        """Get default checkpoint directory."""
        home_dir = Path.home()
        return str(home_dir / ".cache" / "vibeframe" / "longcat-video")
    
    def _detect_device(self) -> str:
        """Detect optimal device for video generation."""
        if torch.cuda.is_available():
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if gpu_memory >= 8.0:  # Minimum 8GB for video generation
                logger.info(f"Using CUDA with {gpu_memory:.1f}GB GPU memory")
                return "cuda"
            else:
                logger.warning(f"GPU has only {gpu_memory:.1f}GB memory, using CPU")
                return "cpu"
        else:
            logger.info("CUDA not available, using CPU")
            return "cpu"
    
    def load_model(self) -> None:
        """
        Load LongCat-Video model weights and initialize pipeline.
        
        Raises:
            ModelNotFoundError: If model weights are not found
            InsufficientGPUMemoryError: If insufficient GPU memory
            VideoGenerationError: If model loading fails
        """
        if self.is_loaded:
            logger.info("Model already loaded")
            return
        
        try:
            logger.info("Loading LongCat-Video model...")
            
            # For now, we'll use a mock implementation since LongCat-Video
            # is not yet publicly available. In production, this would check
            # for actual model weights and load them.
            self._load_mock_model()
            
            self.is_loaded = True
            logger.info("LongCat-Video model loaded successfully")
            
        except torch.cuda.OutOfMemoryError as e:
            available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            raise InsufficientGPUMemoryError(
                f"Insufficient GPU memory for LongCat-Video model",
                required_gb=8.0,
                available_gb=available_memory
            ) from e
        except Exception as e:
            error_msg = f"Failed to load LongCat-Video model: {str(e)}"
            logger.error(error_msg)
            raise VideoGenerationError(error_msg) from e
    
    def _load_mock_model(self) -> None:
        """Load mock model for testing purposes."""
        logger.warning("Using mock LongCat-Video model for testing")
        
        # Create mock model components
        self.model = MockLongCatModel(device=self.device)
        self.pipeline = MockLongCatPipeline(model=self.model, device=self.device)
    
    def text_to_video(
        self,
        prompt: str,
        num_frames: int = 16,
        fps: int = 8,
        resolution: tuple = (512, 512),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate video from text prompt.
        
        Args:
            prompt: Text description of the video
            num_frames: Number of frames to generate (max 16)
            fps: Frames per second
            resolution: Video resolution (width, height)
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            seed: Random seed for reproducibility
            
        Returns:
            Generated video as numpy array (frames, height, width, channels)
            
        Raises:
            VideoGenerationError: If generation fails
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            logger.info(f"Generating video from text: '{prompt[:50]}...'")
            logger.info(f"Parameters: {num_frames} frames, {fps} FPS, {resolution}")
            
            # Validate parameters
            num_frames = min(num_frames, self.max_frames)
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Generate video using pipeline
            video_frames = self.pipeline.text_to_video(
                prompt=prompt,
                num_frames=num_frames,
                resolution=resolution,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
            logger.info(f"Generated video with shape: {video_frames.shape}")
            return video_frames
            
        except Exception as e:
            error_msg = f"Text-to-video generation failed: {str(e)}"
            logger.error(error_msg)
            raise VideoGenerationError(error_msg) from e
    
    def image_to_video(
        self,
        image: Union[np.ndarray, Image.Image, str],
        prompt: str,
        num_frames: int = 16,
        fps: int = 8,
        motion_strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate video from input image and text prompt.
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            prompt: Text description of the motion/animation
            num_frames: Number of frames to generate
            fps: Frames per second
            motion_strength: Strength of motion (0.0 to 1.0)
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            seed: Random seed for reproducibility
            
        Returns:
            Generated video as numpy array (frames, height, width, channels)
            
        Raises:
            VideoGenerationError: If generation fails
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            logger.info(f"Generating video from image with prompt: '{prompt[:50]}...'")
            
            # Process input image
            processed_image = self._process_input_image(image)
            
            # Validate parameters
            num_frames = min(num_frames, self.max_frames)
            motion_strength = max(0.0, min(1.0, motion_strength))
            
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Generate video using pipeline
            video_frames = self.pipeline.image_to_video(
                image=processed_image,
                prompt=prompt,
                num_frames=num_frames,
                motion_strength=motion_strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
            logger.info(f"Generated video with shape: {video_frames.shape}")
            return video_frames
            
        except Exception as e:
            error_msg = f"Image-to-video generation failed: {str(e)}"
            logger.error(error_msg)
            raise VideoGenerationError(error_msg) from e
    
    def video_continuation(
        self,
        video: Union[np.ndarray, str],
        prompt: str,
        num_frames: int = 16,
        fps: int = 8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Continue/extend existing video with new frames.
        
        Args:
            video: Input video (numpy array or file path)
            prompt: Text description for continuation
            num_frames: Number of new frames to generate
            fps: Frames per second
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            seed: Random seed for reproducibility
            
        Returns:
            Extended video as numpy array (frames, height, width, channels)
            
        Raises:
            VideoGenerationError: If generation fails
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            logger.info(f"Continuing video with prompt: '{prompt[:50]}...'")
            
            # Process input video
            processed_video = self._process_input_video(video)
            
            # Validate parameters
            num_frames = min(num_frames, self.max_frames)
            
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Generate continuation using pipeline
            continued_video = self.pipeline.video_continuation(
                video=processed_video,
                prompt=prompt,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
            logger.info(f"Generated continuation with shape: {continued_video.shape}")
            return continued_video
            
        except Exception as e:
            error_msg = f"Video continuation failed: {str(e)}"
            logger.error(error_msg)
            raise VideoGenerationError(error_msg) from e
    
    def _process_input_image(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """Process input image to standard format."""
        if isinstance(image, str):
            # Load from file path
            if not Path(image).exists():
                raise VideoGenerationError(f"Image file not found: {image}")
            image = Image.open(image).convert("RGB")
        
        if isinstance(image, Image.Image):
            # Convert PIL to numpy
            image = np.array(image)
        
        if not isinstance(image, np.ndarray):
            raise VideoGenerationError(f"Unsupported image type: {type(image)}")
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Convert RGBA to RGB
            image = image[:, :, :3]
        elif len(image.shape) == 2:
            # Convert grayscale to RGB
            image = np.stack([image] * 3, axis=2)
        
        # Normalize to 0-255 range
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        
        return image
    
    def _process_input_video(self, video: Union[np.ndarray, str]) -> np.ndarray:
        """Process input video to standard format."""
        if isinstance(video, str):
            # Load from file path
            if not Path(video).exists():
                raise VideoGenerationError(f"Video file not found: {video}")
            
            # Load video using OpenCV
            cap = cv2.VideoCapture(video)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            
            if not frames:
                raise VideoGenerationError(f"Could not load video: {video}")
            
            video = np.array(frames)
        
        if not isinstance(video, np.ndarray):
            raise VideoGenerationError(f"Unsupported video type: {type(video)}")
        
        # Ensure correct shape (frames, height, width, channels)
        if len(video.shape) != 4:
            raise VideoGenerationError(f"Video must have 4 dimensions, got {len(video.shape)}")
        
        return video
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved
            }
        else:
            return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0, "free_gb": 0}
    
    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.is_loaded:
            self.model = None
            self.pipeline = None
            self.is_loaded = False
            self.clear_memory()
            logger.info("LongCat-Video model unloaded")


class MockLongCatModel:
    """Mock LongCat-Video model for testing."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def to(self, device: str):
        self.device = device
        return self


class MockLongCatPipeline:
    """Mock LongCat-Video pipeline for testing."""
    
    def __init__(self, model: MockLongCatModel, device: str = "cpu"):
        self.model = model
        self.device = device
    
    def text_to_video(
        self,
        prompt: str,
        num_frames: int = 16,
        resolution: tuple = (512, 512),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50
    ) -> np.ndarray:
        """Generate mock video from text."""
        width, height = resolution
        
        # Generate colorful noise pattern that changes over time
        frames = []
        for i in range(num_frames):
            # Create a frame with moving patterns
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add some patterns based on the prompt
            if "dance" in prompt.lower():
                # Dancing pattern - oscillating colors
                t = i / num_frames * 2 * np.pi
                frame[:, :, 0] = (np.sin(t) * 127 + 128).astype(np.uint8)
                frame[:, :, 1] = (np.cos(t) * 127 + 128).astype(np.uint8)
            elif "nature" in prompt.lower():
                # Nature pattern - green gradient
                frame[:, :, 1] = 100 + i * 5  # Green channel
                frame[:, :, 0] = 50  # Red channel
                frame[:, :, 2] = 30  # Blue channel
            else:
                # Default pattern - moving gradient
                for y in range(height):
                    for x in range(width):
                        frame[y, x, 0] = (x + i * 10) % 256
                        frame[y, x, 1] = (y + i * 5) % 256
                        frame[y, x, 2] = ((x + y + i * 15) // 2) % 256
            
            frames.append(frame)
        
        return np.array(frames)
    
    def image_to_video(
        self,
        image: np.ndarray,
        prompt: str,
        num_frames: int = 16,
        motion_strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50
    ) -> np.ndarray:
        """Generate mock video from image."""
        height, width = image.shape[:2]
        
        # Generate frames by slightly modifying the input image
        frames = []
        for i in range(num_frames):
            # Apply slight transformations to simulate motion
            frame = image.copy()
            
            # Add motion based on motion_strength
            if motion_strength > 0.5:
                # Add some movement/distortion
                shift_x = int(np.sin(i / num_frames * 2 * np.pi) * 5 * motion_strength)
                shift_y = int(np.cos(i / num_frames * 2 * np.pi) * 3 * motion_strength)
                
                # Apply shift
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                frame = cv2.warpAffine(frame, M, (width, height))
            
            # Add slight color variations
            brightness_factor = 1.0 + 0.1 * np.sin(i / num_frames * 2 * np.pi)
            frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)
            
            frames.append(frame)
        
        return np.array(frames)
    
    def video_continuation(
        self,
        video: np.ndarray,
        prompt: str,
        num_frames: int = 16,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50
    ) -> np.ndarray:
        """Generate mock video continuation."""
        # Take the last frame as reference
        last_frame = video[-1]
        height, width = last_frame.shape[:2]
        
        # Generate continuation frames
        new_frames = []
        for i in range(num_frames):
            # Gradually modify the last frame
            frame = last_frame.copy()
            
            # Add progressive changes
            change_factor = (i + 1) / num_frames
            
            # Apply gradual color shift
            frame = frame.astype(np.float32)
            frame[:, :, 0] = np.clip(frame[:, :, 0] + change_factor * 20, 0, 255)
            frame[:, :, 1] = np.clip(frame[:, :, 1] - change_factor * 10, 0, 255)
            frame = frame.astype(np.uint8)
            
            new_frames.append(frame)
        
        # Concatenate original video with new frames
        return np.concatenate([video, np.array(new_frames)], axis=0)


class SHARPBackend:
    """
    SHARP backend for 2D-to-3D conversion and camera path animation.
    
    This backend provides 3D scene generation from 2D images using SHARP
    (Single-view 3D Human Avatar Reconstruction and Pose estimation) and
    renders camera paths through the 3D scene.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize SHARP backend.
        
        Args:
            checkpoint_path: Path to SHARP model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.checkpoint_path = checkpoint_path or self._get_default_checkpoint_path()
        self.device = device or self._detect_device()
        self.model = None
        self.renderer = None
        self.is_loaded = False
        
        # Model configuration
        self.default_resolution = (512, 512)
        self.default_fps = 24
        
        logger.info(f"SHARPBackend initialized with device: {self.device}")
        logger.info(f"Checkpoint path: {self.checkpoint_path}")
    
    def _get_default_checkpoint_path(self) -> str:
        """Get default checkpoint path."""
        home_dir = Path.home()
        return str(home_dir / ".cache" / "vibeframe" / "sharp" / "model.pth")
    
    def _detect_device(self) -> str:
        """Detect optimal device for 3D processing."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if gpu_memory >= 6.0:  # Minimum 6GB for 3D processing
                logger.info(f"Using CUDA with {gpu_memory:.1f}GB GPU memory")
                return "cuda"
            else:
                logger.warning(f"GPU has only {gpu_memory:.1f}GB memory, using CPU")
                return "cpu"
        else:
            logger.info("CUDA not available, using CPU")
            return "cpu"
    
    def load_model(self) -> None:
        """
        Load SHARP model weights and initialize renderer.
        
        Raises:
            ModelNotFoundError: If model weights are not found
            InsufficientGPUMemoryError: If insufficient GPU memory
            VideoGenerationError: If model loading fails
        """
        if self.is_loaded:
            logger.info("SHARP model already loaded")
            return
        
        try:
            logger.info("Loading SHARP model...")
            
            # For now, we'll use a mock implementation since SHARP
            # is not yet publicly available. In production, this would load
            # the actual model weights.
            self._load_mock_model()
            
            self.is_loaded = True
            logger.info("SHARP model loaded successfully")
            
        except torch.cuda.OutOfMemoryError as e:
            available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            raise InsufficientGPUMemoryError(
                f"Insufficient GPU memory for SHARP model",
                required_gb=6.0,
                available_gb=available_memory
            ) from e
        except Exception as e:
            error_msg = f"Failed to load SHARP model: {str(e)}"
            logger.error(error_msg)
            raise VideoGenerationError(error_msg) from e
    
    def _load_mock_model(self) -> None:
        """Load mock model for testing purposes."""
        logger.warning("Using mock SHARP model for testing")
        
        # Create mock model components
        self.model = MockSHARPModel(device=self.device)
        self.renderer = MockGaussianRenderer(device=self.device)
    
    def image_to_3d(self, image: Union[np.ndarray, Image.Image, str]) -> Dict[str, Any]:
        """
        Convert 2D image to 3D Gaussian splat representation.
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            
        Returns:
            Dictionary containing 3D scene representation
            
        Raises:
            VideoGenerationError: If 3D conversion fails
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            logger.info("Converting 2D image to 3D representation")
            
            # Process input image
            processed_image = self._process_input_image(image)
            
            # Generate 3D representation using model
            gaussian_splat = self.model.image_to_3d(processed_image)
            
            logger.info("3D conversion completed successfully")
            return gaussian_splat
            
        except Exception as e:
            error_msg = f"2D-to-3D conversion failed: {str(e)}"
            logger.error(error_msg)
            raise VideoGenerationError(error_msg) from e
    
    def render_camera_path(
        self,
        gaussian_splat: Dict[str, Any],
        camera_path: List[Dict[str, Any]],
        fps: int = 24,
        resolution: tuple = (512, 512)
    ) -> np.ndarray:
        """
        Render video from camera path through 3D scene.
        
        Args:
            gaussian_splat: 3D scene representation
            camera_path: List of camera keyframes
            fps: Frames per second
            resolution: Video resolution (width, height)
            
        Returns:
            Rendered video as numpy array (frames, height, width, channels)
            
        Raises:
            VideoGenerationError: If rendering fails
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            logger.info(f"Rendering camera path with {len(camera_path)} keyframes")
            logger.info(f"Parameters: {fps} FPS, {resolution}")
            
            # Render video using Gaussian splatting renderer
            video_frames = self.renderer.render_path(
                gaussian_splat=gaussian_splat,
                camera_path=camera_path,
                fps=fps,
                resolution=resolution
            )
            
            logger.info(f"Rendered video with shape: {video_frames.shape}")
            return video_frames
            
        except Exception as e:
            error_msg = f"Camera path rendering failed: {str(e)}"
            logger.error(error_msg)
            raise VideoGenerationError(error_msg) from e
    
    def generate_camera_path(
        self,
        start_position: tuple,
        end_position: tuple,
        num_frames: int,
        interpolation: str = "linear"
    ) -> List[Dict[str, Any]]:
        """
        Generate smooth camera path between two positions.
        
        Args:
            start_position: Starting camera position (x, y, z, pitch, yaw, roll, fov)
            end_position: Ending camera position (x, y, z, pitch, yaw, roll, fov)
            num_frames: Number of frames in the path
            interpolation: Interpolation method ("linear", "bezier", "catmull-rom")
            
        Returns:
            List of camera keyframes
            
        Raises:
            VideoGenerationError: If path generation fails
        """
        try:
            logger.info(f"Generating camera path with {num_frames} frames using {interpolation} interpolation")
            
            if interpolation == "linear":
                # Linear interpolation between start and end
                camera_path = []
                for i in range(num_frames):
                    t = i / (num_frames - 1) if num_frames > 1 else 0
                    
                    # Interpolate each parameter
                    position = tuple(
                        start_position[j] + t * (end_position[j] - start_position[j])
                        for j in range(len(start_position))
                    )
                    
                    camera_keyframe = {
                        "position": position[:3],
                        "rotation": position[3:6],
                        "fov": position[6] if len(position) > 6 else 60.0,
                        "timestamp": i / fps if 'fps' in locals() else i / 24.0
                    }
                    camera_path.append(camera_keyframe)
                
                return camera_path
            
            elif interpolation == "bezier":
                # Bezier curve interpolation (simplified)
                # In production, this would use proper Bezier curve math
                return self.generate_camera_path(start_position, end_position, num_frames, "linear")
            
            elif interpolation == "catmull-rom":
                # Catmull-Rom spline interpolation (simplified)
                # In production, this would use proper spline math
                return self.generate_camera_path(start_position, end_position, num_frames, "linear")
            
            else:
                raise VideoGenerationError(f"Unknown interpolation method: {interpolation}")
                
        except Exception as e:
            error_msg = f"Camera path generation failed: {str(e)}"
            logger.error(error_msg)
            raise VideoGenerationError(error_msg) from e
    
    def _process_input_image(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """Process input image to standard format."""
        if isinstance(image, str):
            # Load from file path
            if not Path(image).exists():
                raise VideoGenerationError(f"Image file not found: {image}")
            image = Image.open(image).convert("RGB")
        
        if isinstance(image, Image.Image):
            # Convert PIL to numpy
            image = np.array(image)
        
        if not isinstance(image, np.ndarray):
            raise VideoGenerationError(f"Unsupported image type: {type(image)}")
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Convert RGBA to RGB
            image = image[:, :, :3]
        elif len(image.shape) == 2:
            # Convert grayscale to RGB
            image = np.stack([image] * 3, axis=2)
        
        # Normalize to 0-255 range
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        
        return image
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved
            }
        else:
            return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0, "free_gb": 0}
    
    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.is_loaded:
            self.model = None
            self.renderer = None
            self.is_loaded = False
            self.clear_memory()
            logger.info("SHARP model unloaded")


class MockSHARPModel:
    """Mock SHARP model for testing."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def image_to_3d(self, image: np.ndarray) -> Dict[str, Any]:
        """Generate mock 3D representation from image."""
        height, width = image.shape[:2]
        
        # Create mock Gaussian splat data
        num_gaussians = 1000  # Number of 3D Gaussians
        
        gaussian_splat = {
            "positions": np.random.randn(num_gaussians, 3).astype(np.float32),  # 3D positions
            "colors": np.random.rand(num_gaussians, 3).astype(np.float32),      # RGB colors
            "opacities": np.random.rand(num_gaussians, 1).astype(np.float32),   # Alpha values
            "scales": np.random.rand(num_gaussians, 3).astype(np.float32) * 0.1, # Gaussian scales
            "rotations": np.random.randn(num_gaussians, 4).astype(np.float32),  # Quaternions
            "source_image": image,
            "image_dimensions": (height, width)
        }
        
        # Normalize quaternions
        gaussian_splat["rotations"] = gaussian_splat["rotations"] / np.linalg.norm(
            gaussian_splat["rotations"], axis=1, keepdims=True
        )
        
        return gaussian_splat


class MockGaussianRenderer:
    """Mock Gaussian splatting renderer for testing."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def render_path(
        self,
        gaussian_splat: Dict[str, Any],
        camera_path: List[Dict[str, Any]],
        fps: int = 24,
        resolution: tuple = (512, 512)
    ) -> np.ndarray:
        """Render mock video from camera path."""
        width, height = resolution
        num_frames = len(camera_path)
        
        # Get source image for reference
        source_image = gaussian_splat.get("source_image")
        if source_image is not None:
            # Resize source image to target resolution
            source_image = cv2.resize(source_image, (width, height))
        else:
            # Create default image
            source_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        frames = []
        for i, keyframe in enumerate(camera_path):
            # Create frame based on camera position
            frame = source_image.copy()
            
            # Apply camera-based transformations
            position = keyframe.get("position", (0, 0, 0))
            rotation = keyframe.get("rotation", (0, 0, 0))
            fov = keyframe.get("fov", 60.0)
            
            # Simulate camera movement effects
            # Zoom effect based on Z position
            zoom_factor = 1.0 + position[2] * 0.1
            if zoom_factor != 1.0:
                center_x, center_y = width // 2, height // 2
                M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_factor)
                frame = cv2.warpAffine(frame, M, (width, height))
            
            # Rotation effect
            if any(r != 0 for r in rotation):
                center_x, center_y = width // 2, height // 2
                angle = rotation[2] * 10  # Use roll for 2D rotation
                M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
                frame = cv2.warpAffine(frame, M, (width, height))
            
            # Pan effect based on X, Y position
            shift_x = int(position[0] * 20)
            shift_y = int(position[1] * 20)
            if shift_x != 0 or shift_y != 0:
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                frame = cv2.warpAffine(frame, M, (width, height))
            
            # Add slight variations to simulate 3D depth
            brightness_factor = 1.0 + np.sin(i / num_frames * 2 * np.pi) * 0.1
            frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)
            
            frames.append(frame)
        
        return np.array(frames)


class VideoGenerator:
    """
    Main video generator orchestrator that manages multiple backends.
    
    This class provides a unified interface for video generation using different
    backends (LongCat-Video, HunyuanVideo, SHARP) with automatic fallback logic.
    """
    
    def __init__(self, preferred_backend: str = "longcat", device: Optional[str] = None):
        """
        Initialize VideoGenerator with preferred backend.
        
        Args:
            preferred_backend: Preferred backend ("longcat", "hunyuan", "sharp")
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.preferred_backend = preferred_backend
        self.device = device
        self.backends = {}
        self.current_backend = None
        
        # Backend priority for fallback
        self.backend_priority = ["longcat", "hunyuan", "sharp"]
        if preferred_backend in self.backend_priority:
            # Move preferred backend to front
            self.backend_priority.remove(preferred_backend)
            self.backend_priority.insert(0, preferred_backend)
        
        logger.info(f"VideoGenerator initialized with preferred backend: {preferred_backend}")
        logger.info(f"Backend fallback order: {self.backend_priority}")
    
    def _get_backend(self, backend_name: str):
        """Get or create backend instance."""
        if backend_name not in self.backends:
            if backend_name == "longcat":
                self.backends[backend_name] = LongCatVideoBackend(device=self.device)
            elif backend_name == "sharp":
                self.backends[backend_name] = SHARPBackend(device=self.device)
            elif backend_name == "hunyuan":
                # HunyuanVideo backend would be implemented here
                logger.warning("HunyuanVideo backend not yet implemented, using LongCat fallback")
                self.backends[backend_name] = LongCatVideoBackend(device=self.device)
            else:
                raise VideoGenerationError(f"Unknown backend: {backend_name}")
        
        return self.backends[backend_name]
    
    def generate_text_to_video(
        self,
        prompt: str,
        num_frames: int = 16,
        fps: int = 8,
        resolution: tuple = (512, 512),
        backend: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate video from text prompt with automatic fallback.
        
        Args:
            prompt: Text description of the video
            num_frames: Number of frames to generate
            fps: Frames per second
            resolution: Video resolution (width, height)
            backend: Specific backend to use (None for auto-selection)
            **kwargs: Additional backend-specific parameters
            
        Returns:
            Generated video as numpy array
            
        Raises:
            VideoGenerationError: If all backends fail
        """
        backends_to_try = [backend] if backend else self.backend_priority
        
        for backend_name in backends_to_try:
            if backend_name == "sharp":
                continue  # SHARP doesn't support text-to-video directly
            
            try:
                logger.info(f"Attempting text-to-video with {backend_name} backend")
                backend_instance = self._get_backend(backend_name)
                
                result = backend_instance.text_to_video(
                    prompt=prompt,
                    num_frames=num_frames,
                    fps=fps,
                    resolution=resolution,
                    **kwargs
                )
                
                self.current_backend = backend_name
                logger.info(f"Text-to-video successful with {backend_name} backend")
                return result
                
            except Exception as e:
                logger.warning(f"{backend_name} backend failed: {str(e)}")
                if backend:  # If specific backend requested, don't fallback
                    raise
                continue
        
        raise VideoGenerationError("All backends failed for text-to-video generation")
    
    def generate_image_to_video(
        self,
        image: Union[np.ndarray, Image.Image, str],
        prompt: str,
        num_frames: int = 16,
        fps: int = 8,
        backend: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate video from image with automatic fallback.
        
        Args:
            image: Input image
            prompt: Text description of the motion/animation
            num_frames: Number of frames to generate
            fps: Frames per second
            backend: Specific backend to use (None for auto-selection)
            **kwargs: Additional backend-specific parameters
            
        Returns:
            Generated video as numpy array
            
        Raises:
            VideoGenerationError: If all backends fail
        """
        backends_to_try = [backend] if backend else self.backend_priority
        
        for backend_name in backends_to_try:
            try:
                logger.info(f"Attempting image-to-video with {backend_name} backend")
                backend_instance = self._get_backend(backend_name)
                
                if backend_name == "sharp":
                    # Use SHARP for 3D camera path animation
                    return self._generate_with_sharp(image, prompt, num_frames, fps, **kwargs)
                else:
                    # Use standard image-to-video
                    result = backend_instance.image_to_video(
                        image=image,
                        prompt=prompt,
                        num_frames=num_frames,
                        fps=fps,
                        **kwargs
                    )
                
                self.current_backend = backend_name
                logger.info(f"Image-to-video successful with {backend_name} backend")
                return result
                
            except Exception as e:
                logger.warning(f"{backend_name} backend failed: {str(e)}")
                if backend:  # If specific backend requested, don't fallback
                    raise
                continue
        
        raise VideoGenerationError("All backends failed for image-to-video generation")
    
    def continue_video(
        self,
        video: Union[np.ndarray, str],
        prompt: str,
        num_frames: int = 16,
        fps: int = 8,
        backend: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Continue/extend existing video with automatic fallback.
        
        Args:
            video: Input video
            prompt: Text description for continuation
            num_frames: Number of new frames to generate
            fps: Frames per second
            backend: Specific backend to use (None for auto-selection)
            **kwargs: Additional backend-specific parameters
            
        Returns:
            Extended video as numpy array
            
        Raises:
            VideoGenerationError: If all backends fail
        """
        backends_to_try = [backend] if backend else self.backend_priority
        
        for backend_name in backends_to_try:
            if backend_name == "sharp":
                continue  # SHARP doesn't support video continuation directly
            
            try:
                logger.info(f"Attempting video continuation with {backend_name} backend")
                backend_instance = self._get_backend(backend_name)
                
                result = backend_instance.video_continuation(
                    video=video,
                    prompt=prompt,
                    num_frames=num_frames,
                    fps=fps,
                    **kwargs
                )
                
                self.current_backend = backend_name
                logger.info(f"Video continuation successful with {backend_name} backend")
                return result
                
            except Exception as e:
                logger.warning(f"{backend_name} backend failed: {str(e)}")
                if backend:  # If specific backend requested, don't fallback
                    raise
                continue
        
        raise VideoGenerationError("All backends failed for video continuation")
    
    def _generate_with_sharp(
        self,
        image: Union[np.ndarray, Image.Image, str],
        prompt: str,
        num_frames: int,
        fps: int,
        camera_movement: str = "orbit",
        **kwargs
    ) -> np.ndarray:
        """
        Generate video using SHARP backend with 3D camera animation.
        
        Args:
            image: Input image
            prompt: Text description (used to determine camera movement)
            num_frames: Number of frames
            fps: Frames per second
            camera_movement: Type of camera movement ("orbit", "zoom", "pan")
            **kwargs: Additional parameters
            
        Returns:
            Generated video as numpy array
        """
        sharp_backend = self._get_backend("sharp")
        
        # Convert image to 3D
        gaussian_splat = sharp_backend.image_to_3d(image)
        
        # Generate camera path based on prompt and movement type
        if "orbit" in prompt.lower() or camera_movement == "orbit":
            # Orbital camera movement
            start_pos = (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 60.0)
            end_pos = (2.0, 0.0, 0.0, 0.0, 360.0, 0.0, 60.0)
        elif "zoom" in prompt.lower() or camera_movement == "zoom":
            # Zoom in/out movement
            start_pos = (3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 60.0)
            end_pos = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 60.0)
        else:
            # Default pan movement
            start_pos = (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 60.0)
            end_pos = (1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 60.0)
        
        camera_path = sharp_backend.generate_camera_path(
            start_position=start_pos,
            end_position=end_pos,
            num_frames=num_frames,
            interpolation="linear"
        )
        
        # Render video
        video = sharp_backend.render_camera_path(
            gaussian_splat=gaussian_splat,
            camera_path=camera_path,
            fps=fps,
            resolution=kwargs.get("resolution", (512, 512))
        )
        
        return video
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backends."""
        return self.backend_priority.copy()
    
    def get_current_backend(self) -> Optional[str]:
        """Get currently active backend."""
        return self.current_backend
    
    def clear_all_memory(self) -> None:
        """Clear memory for all loaded backends."""
        for backend in self.backends.values():
            if hasattr(backend, 'clear_memory'):
                backend.clear_memory()
        logger.info("Cleared memory for all backends")
    
    def unload_all_backends(self) -> None:
        """Unload all backends to free memory."""
        for backend in self.backends.values():
            if hasattr(backend, 'unload_model'):
                backend.unload_model()
        self.backends.clear()
        self.current_backend = None
        logger.info("Unloaded all backends")