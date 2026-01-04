"""Tests for video generation component."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import cv2
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

from vibeframe.video_generator import LongCatVideoBackend
from vibeframe.exceptions import VideoGenerationError, ModelNotFoundError


class TestLongCatVideoBackend:
    """Test LongCat-Video backend functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = LongCatVideoBackend(device="cpu")  # Force CPU for testing
    
    def test_initialization(self):
        """Test backend initialization."""
        backend = LongCatVideoBackend()
        assert backend.device in ["cpu", "cuda"]
        assert backend.max_frames == 16
        assert backend.default_fps == 8
        assert backend.default_resolution == (512, 512)
        assert not backend.is_loaded
    
    def test_device_detection(self):
        """Test device detection logic."""
        backend = LongCatVideoBackend()
        detected_device = backend._detect_device()
        assert detected_device in ["cpu", "cuda"]
    
    def test_model_loading(self):
        """Test model loading."""
        backend = LongCatVideoBackend(device="cpu")
        backend.load_model()
        assert backend.is_loaded
        assert backend.model is not None
        assert backend.pipeline is not None
    
    def test_text_to_video_basic(self):
        """Test basic text-to-video generation."""
        self.backend.load_model()
        
        prompt = "A person dancing in a colorful room"
        video = self.backend.text_to_video(prompt, num_frames=8)
        
        assert isinstance(video, np.ndarray)
        assert len(video.shape) == 4  # (frames, height, width, channels)
        assert video.shape[0] == 8  # Number of frames
        assert video.shape[3] == 3  # RGB channels
        assert video.dtype == np.uint8
    
    def test_image_to_video_basic(self):
        """Test basic image-to-video generation."""
        self.backend.load_model()
        
        # Create test image
        test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        prompt = "The image comes to life with gentle movement"
        
        video = self.backend.image_to_video(test_image, prompt, num_frames=8)
        
        assert isinstance(video, np.ndarray)
        assert len(video.shape) == 4
        assert video.shape[0] == 8
        assert video.shape[3] == 3
        assert video.dtype == np.uint8
    
    def test_video_continuation_basic(self):
        """Test basic video continuation."""
        self.backend.load_model()
        
        # Create test video
        test_video = np.random.randint(0, 256, (4, 256, 256, 3), dtype=np.uint8)
        prompt = "The scene continues with more action"
        
        continued_video = self.backend.video_continuation(test_video, prompt, num_frames=4)
        
        assert isinstance(continued_video, np.ndarray)
        assert len(continued_video.shape) == 4
        assert continued_video.shape[0] == 8  # Original 4 + new 4 frames
        assert continued_video.shape[3] == 3
        assert continued_video.dtype == np.uint8
    
    def test_memory_usage_tracking(self):
        """Test GPU memory usage tracking."""
        memory_info = self.backend.get_memory_usage()
        
        assert isinstance(memory_info, dict)
        assert "allocated_gb" in memory_info
        assert "reserved_gb" in memory_info
        assert "total_gb" in memory_info
        assert "free_gb" in memory_info
        
        for key, value in memory_info.items():
            assert isinstance(value, (int, float))
            assert value >= 0
    
    def test_model_unloading(self):
        """Test model unloading."""
        self.backend.load_model()
        assert self.backend.is_loaded
        
        self.backend.unload_model()
        assert not self.backend.is_loaded
        assert self.backend.model is None
        assert self.backend.pipeline is None


class TestLongCatVideoBackendEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = LongCatVideoBackend(device="cpu")
    
    def test_text_to_video_without_loading(self):
        """Test text-to-video without loading model first."""
        prompt = "A simple scene"
        # Should auto-load model
        video = self.backend.text_to_video(prompt, num_frames=4)
        assert isinstance(video, np.ndarray)
        assert self.backend.is_loaded
    
    def test_text_to_video_empty_prompt(self):
        """Test text-to-video with empty prompt."""
        self.backend.load_model()
        
        video = self.backend.text_to_video("", num_frames=4)
        assert isinstance(video, np.ndarray)
        assert video.shape[0] == 4
    
    def test_text_to_video_max_frames_limit(self):
        """Test that frame count is limited to maximum."""
        self.backend.load_model()
        
        # Request more than max frames
        video = self.backend.text_to_video("test", num_frames=32)
        assert video.shape[0] == self.backend.max_frames  # Should be capped at 16
    
    def test_image_to_video_pil_image(self):
        """Test image-to-video with PIL Image input."""
        self.backend.load_model()
        
        # Create PIL image
        pil_image = Image.new("RGB", (256, 256), color="red")
        video = self.backend.image_to_video(pil_image, "test", num_frames=4)
        
        assert isinstance(video, np.ndarray)
        assert video.shape[0] == 4
    
    def test_image_to_video_grayscale(self):
        """Test image-to-video with grayscale image."""
        self.backend.load_model()
        
        # Create grayscale image
        gray_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        video = self.backend.image_to_video(gray_image, "test", num_frames=4)
        
        assert isinstance(video, np.ndarray)
        assert video.shape[3] == 3  # Should be converted to RGB
    
    def test_image_to_video_rgba(self):
        """Test image-to-video with RGBA image."""
        self.backend.load_model()
        
        # Create RGBA image
        rgba_image = np.random.randint(0, 256, (256, 256, 4), dtype=np.uint8)
        video = self.backend.image_to_video(rgba_image, "test", num_frames=4)
        
        assert isinstance(video, np.ndarray)
        assert video.shape[3] == 3  # Should be converted to RGB
    
    def test_image_to_video_float_image(self):
        """Test image-to-video with float image."""
        self.backend.load_model()
        
        # Create float image (0-1 range)
        float_image = np.random.rand(256, 256, 3).astype(np.float32)
        video = self.backend.image_to_video(float_image, "test", num_frames=4)
        
        assert isinstance(video, np.ndarray)
        assert video.dtype == np.uint8
    
    def test_image_to_video_invalid_image(self):
        """Test image-to-video with invalid image."""
        self.backend.load_model()
        
        with pytest.raises(VideoGenerationError):
            self.backend.image_to_video("not an image", "test", num_frames=4)
    
    def test_image_to_video_nonexistent_file(self):
        """Test image-to-video with nonexistent file."""
        self.backend.load_model()
        
        with pytest.raises(VideoGenerationError):
            self.backend.image_to_video("/nonexistent/image.jpg", "test", num_frames=4)
    
    def test_video_continuation_invalid_video(self):
        """Test video continuation with invalid video."""
        self.backend.load_model()
        
        with pytest.raises(VideoGenerationError):
            self.backend.video_continuation("not a video", "test", num_frames=4)
    
    def test_video_continuation_wrong_dimensions(self):
        """Test video continuation with wrong video dimensions."""
        self.backend.load_model()
        
        # Create video with wrong dimensions
        wrong_video = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)  # Missing frame dimension
        
        with pytest.raises(VideoGenerationError):
            self.backend.video_continuation(wrong_video, "test", num_frames=4)
    
    def test_motion_strength_clamping(self):
        """Test that motion strength is clamped to valid range."""
        self.backend.load_model()
        
        test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Test with motion strength > 1.0
        video1 = self.backend.image_to_video(test_image, "test", motion_strength=1.5)
        assert isinstance(video1, np.ndarray)
        
        # Test with motion strength < 0.0
        video2 = self.backend.image_to_video(test_image, "test", motion_strength=-0.5)
        assert isinstance(video2, np.ndarray)
    
    def test_seed_reproducibility(self):
        """Test that using the same seed produces consistent results."""
        self.backend.load_model()
        
        prompt = "A consistent scene"
        seed = 42
        
        video1 = self.backend.text_to_video(prompt, num_frames=4, seed=seed)
        video2 = self.backend.text_to_video(prompt, num_frames=4, seed=seed)
        
        # Results should be identical with same seed
        np.testing.assert_array_equal(video1, video2)
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        self.backend.load_model()
        
        prompt = "A variable scene"
        
        video1 = self.backend.text_to_video(prompt, num_frames=4, seed=42)
        video2 = self.backend.text_to_video(prompt, num_frames=4, seed=123)
        
        # Results should be different with different seeds
        assert not np.array_equal(video1, video2)


# Property-based tests
class TestLongCatVideoBackendProperties:
    """Property-based tests for video generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = LongCatVideoBackend(device="cpu")
        self.backend.load_model()
    
    @given(
        prompt=st.text(min_size=1, max_size=200),
        num_frames=st.integers(min_value=1, max_value=32),
        fps=st.integers(min_value=1, max_value=60)
    )
    @settings(max_examples=10, deadline=30000)  # Reduced examples for speed
    def test_property_text_to_video_output_validity(self, prompt, num_frames, fps):
        """Property: Text-to-video always produces valid video output."""
        assume(len(prompt.strip()) > 0)  # Non-empty prompt
        
        video = self.backend.text_to_video(prompt, num_frames=num_frames, fps=fps)
        
        # Video should be valid numpy array
        assert isinstance(video, np.ndarray)
        assert len(video.shape) == 4
        assert video.shape[0] <= self.backend.max_frames  # Frames capped at max
        assert video.shape[0] > 0  # At least one frame
        assert video.shape[1] > 0  # Height > 0
        assert video.shape[2] > 0  # Width > 0
        assert video.shape[3] == 3  # RGB channels
        assert video.dtype == np.uint8
        assert np.all(video >= 0) and np.all(video <= 255)  # Valid pixel values
    
    @given(
        image_shape=st.tuples(
            st.integers(min_value=64, max_value=512),  # height
            st.integers(min_value=64, max_value=512),  # width
            st.integers(min_value=1, max_value=4)      # channels
        ),
        prompt=st.text(min_size=1, max_size=100),
        num_frames=st.integers(min_value=1, max_value=16),
        motion_strength=st.floats(min_value=-1.0, max_value=2.0)
    )
    @settings(max_examples=10, deadline=30000)
    def test_property_image_to_video_output_validity(self, image_shape, prompt, num_frames, motion_strength):
        """Property: Image-to-video always produces valid video output."""
        assume(len(prompt.strip()) > 0)
        
        height, width, channels = image_shape
        
        # Create test image
        if channels == 1:
            test_image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        else:
            test_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        
        video = self.backend.image_to_video(
            test_image, prompt, num_frames=num_frames, motion_strength=motion_strength
        )
        
        # Video should be valid
        assert isinstance(video, np.ndarray)
        assert len(video.shape) == 4
        assert video.shape[0] == min(num_frames, self.backend.max_frames)
        assert video.shape[3] == 3  # Always RGB output
        assert video.dtype == np.uint8
        assert np.all(video >= 0) and np.all(video <= 255)
    
    @given(
        video_shape=st.tuples(
            st.integers(min_value=1, max_value=8),     # frames
            st.integers(min_value=64, max_value=256),  # height
            st.integers(min_value=64, max_value=256),  # width
            st.just(3)                                 # RGB channels
        ),
        prompt=st.text(min_size=1, max_size=100),
        num_frames=st.integers(min_value=1, max_value=16)
    )
    @settings(max_examples=10, deadline=30000)
    def test_property_video_continuation_output_validity(self, video_shape, prompt, num_frames):
        """Property: Video continuation always produces valid extended video."""
        assume(len(prompt.strip()) > 0)
        
        frames, height, width, channels = video_shape
        
        # Create test video
        test_video = np.random.randint(0, 256, video_shape, dtype=np.uint8)
        
        continued_video = self.backend.video_continuation(test_video, prompt, num_frames=num_frames)
        
        # Extended video should be valid
        assert isinstance(continued_video, np.ndarray)
        assert len(continued_video.shape) == 4
        expected_frames = frames + min(num_frames, self.backend.max_frames)
        assert continued_video.shape[0] == expected_frames
        assert continued_video.shape[1] == height
        assert continued_video.shape[2] == width
        assert continued_video.shape[3] == 3
        assert continued_video.dtype == np.uint8
        assert np.all(continued_video >= 0) and np.all(continued_video <= 255)
        
        # Original frames should be preserved
        np.testing.assert_array_equal(continued_video[:frames], test_video)
    
    @given(
        prompt=st.text(min_size=1, max_size=100),
        seed=st.integers(min_value=0, max_value=2**31 - 1)
    )
    @settings(max_examples=5, deadline=30000)
    def test_property_seed_determinism(self, prompt, seed):
        """Property: Same seed produces identical results."""
        assume(len(prompt.strip()) > 0)
        
        video1 = self.backend.text_to_video(prompt, num_frames=4, seed=seed)
        video2 = self.backend.text_to_video(prompt, num_frames=4, seed=seed)
        
        # Should be identical
        np.testing.assert_array_equal(video1, video2)


class TestVideoGeneratorIntegration:
    """Integration tests for video generation with file I/O."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = LongCatVideoBackend(device="cpu")
        self.backend.load_model()
    
    def test_image_to_video_from_file(self):
        """Test image-to-video with image file input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image file
            image_path = Path(temp_dir) / "test_image.png"
            test_image = Image.new("RGB", (256, 256), color="blue")
            test_image.save(image_path)
            
            # Generate video from file
            video = self.backend.image_to_video(str(image_path), "test animation")
            
            assert isinstance(video, np.ndarray)
            assert len(video.shape) == 4
    
    def test_video_continuation_from_file(self):
        """Test video continuation with video file input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test video file
            video_path = Path(temp_dir) / "test_video.mp4"
            
            # Create simple video using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (256, 256))
            
            for i in range(5):
                frame = np.full((256, 256, 3), i * 50, dtype=np.uint8)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            out.release()
            
            # Generate continuation from file
            continued_video = self.backend.video_continuation(str(video_path), "continue the scene")
            
            assert isinstance(continued_video, np.ndarray)
            assert continued_video.shape[0] > 5  # Should have more frames than original


class TestSHARPBackend:
    """Test SHARP backend functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = SHARPBackend(device="cpu")
    
    def test_initialization(self):
        """Test SHARP backend initialization."""
        backend = SHARPBackend()
        assert backend.device in ["cpu", "cuda"]
        assert backend.default_resolution == (512, 512)
        assert backend.default_fps == 24
        assert not backend.is_loaded
    
    def test_model_loading(self):
        """Test SHARP model loading."""
        self.backend.load_model()
        assert self.backend.is_loaded
        assert self.backend.model is not None
        assert self.backend.renderer is not None
    
    def test_image_to_3d(self):
        """Test 2D to 3D conversion."""
        self.backend.load_model()
        
        # Create test image
        test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        gaussian_splat = self.backend.image_to_3d(test_image)
        
        assert isinstance(gaussian_splat, dict)
        assert "positions" in gaussian_splat
        assert "colors" in gaussian_splat
        assert "opacities" in gaussian_splat
        assert "scales" in gaussian_splat
        assert "rotations" in gaussian_splat
        assert "source_image" in gaussian_splat
        
        # Check shapes
        num_gaussians = gaussian_splat["positions"].shape[0]
        assert gaussian_splat["colors"].shape == (num_gaussians, 3)
        assert gaussian_splat["opacities"].shape == (num_gaussians, 1)
        assert gaussian_splat["scales"].shape == (num_gaussians, 3)
        assert gaussian_splat["rotations"].shape == (num_gaussians, 4)
    
    def test_camera_path_generation(self):
        """Test camera path generation."""
        start_pos = (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 60.0)
        end_pos = (1.0, 0.0, 2.0, 0.0, 45.0, 0.0, 60.0)
        num_frames = 10
        
        camera_path = self.backend.generate_camera_path(
            start_position=start_pos,
            end_position=end_pos,
            num_frames=num_frames,
            interpolation="linear"
        )
        
        assert len(camera_path) == num_frames
        
        for keyframe in camera_path:
            assert "position" in keyframe
            assert "rotation" in keyframe
            assert "fov" in keyframe
            assert "timestamp" in keyframe
            
            assert len(keyframe["position"]) == 3
            assert len(keyframe["rotation"]) == 3
            assert isinstance(keyframe["fov"], (int, float))
            assert isinstance(keyframe["timestamp"], (int, float))
    
    def test_render_camera_path(self):
        """Test camera path rendering."""
        self.backend.load_model()
        
        # Create test 3D scene
        test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        gaussian_splat = self.backend.image_to_3d(test_image)
        
        # Create test camera path
        camera_path = [
            {"position": (0, 0, 2), "rotation": (0, 0, 0), "fov": 60.0, "timestamp": 0.0},
            {"position": (1, 0, 2), "rotation": (0, 0, 0), "fov": 60.0, "timestamp": 0.5},
            {"position": (0, 1, 2), "rotation": (0, 0, 0), "fov": 60.0, "timestamp": 1.0},
        ]
        
        video = self.backend.render_camera_path(
            gaussian_splat=gaussian_splat,
            camera_path=camera_path,
            fps=24,
            resolution=(256, 256)
        )
        
        assert isinstance(video, np.ndarray)
        assert len(video.shape) == 4
        assert video.shape[0] == len(camera_path)  # Number of frames
        assert video.shape[1] == 256  # Height
        assert video.shape[2] == 256  # Width
        assert video.shape[3] == 3    # RGB channels
        assert video.dtype == np.uint8
    
    def test_different_interpolation_methods(self):
        """Test different camera path interpolation methods."""
        start_pos = (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 60.0)
        end_pos = (1.0, 0.0, 2.0, 0.0, 45.0, 0.0, 60.0)
        num_frames = 5
        
        for interpolation in ["linear", "bezier", "catmull-rom"]:
            camera_path = self.backend.generate_camera_path(
                start_position=start_pos,
                end_position=end_pos,
                num_frames=num_frames,
                interpolation=interpolation
            )
            
            assert len(camera_path) == num_frames
            # First and last frames should match start and end positions
            assert camera_path[0]["position"] == start_pos[:3]
            assert camera_path[-1]["position"] == end_pos[:3]


class TestVideoGenerator:
    """Test VideoGenerator orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = VideoGenerator(preferred_backend="longcat", device="cpu")
    
    def test_initialization(self):
        """Test VideoGenerator initialization."""
        generator = VideoGenerator()
        assert generator.preferred_backend == "longcat"
        assert "longcat" in generator.backend_priority
        assert generator.current_backend is None
    
    def test_backend_priority_ordering(self):
        """Test that preferred backend is moved to front of priority list."""
        generator = VideoGenerator(preferred_backend="sharp")
        assert generator.backend_priority[0] == "sharp"
        assert "longcat" in generator.backend_priority
        assert "hunyuan" in generator.backend_priority
    
    def test_text_to_video_with_longcat(self):
        """Test text-to-video generation with LongCat backend."""
        video = self.generator.generate_text_to_video(
            prompt="A person dancing",
            num_frames=4,
            backend="longcat"
        )
        
        assert isinstance(video, np.ndarray)
        assert len(video.shape) == 4
        assert video.shape[0] == 4
        assert self.generator.current_backend == "longcat"
    
    def test_image_to_video_with_longcat(self):
        """Test image-to-video generation with LongCat backend."""
        test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        video = self.generator.generate_image_to_video(
            image=test_image,
            prompt="Animation",
            num_frames=4,
            backend="longcat"
        )
        
        assert isinstance(video, np.ndarray)
        assert len(video.shape) == 4
        assert video.shape[0] == 4
        assert self.generator.current_backend == "longcat"
    
    def test_image_to_video_with_sharp(self):
        """Test image-to-video generation with SHARP backend."""
        test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        video = self.generator.generate_image_to_video(
            image=test_image,
            prompt="Orbit around the subject",
            num_frames=4,
            backend="sharp"
        )
        
        assert isinstance(video, np.ndarray)
        assert len(video.shape) == 4
        assert video.shape[0] == 4
        assert self.generator.current_backend == "sharp"
    
    def test_video_continuation(self):
        """Test video continuation."""
        test_video = np.random.randint(0, 256, (4, 256, 256, 3), dtype=np.uint8)
        
        continued_video = self.generator.continue_video(
            video=test_video,
            prompt="Continue the scene",
            num_frames=4,
            backend="longcat"
        )
        
        assert isinstance(continued_video, np.ndarray)
        assert len(continued_video.shape) == 4
        assert continued_video.shape[0] == 8  # Original 4 + new 4 frames
        assert self.generator.current_backend == "longcat"
    
    def test_backend_fallback(self):
        """Test automatic backend fallback."""
        # This would test fallback in a real scenario where one backend fails
        # For now, we just test that the fallback mechanism exists
        available_backends = self.generator.get_available_backends()
        assert len(available_backends) > 0
        assert "longcat" in available_backends
    
    def test_sharp_camera_movements(self):
        """Test different camera movements with SHARP."""
        test_image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        for movement in ["orbit", "zoom", "pan"]:
            video = self.generator.generate_image_to_video(
                image=test_image,
                prompt=f"{movement} movement",
                num_frames=3,
                backend="sharp",
                camera_movement=movement
            )
            
            assert isinstance(video, np.ndarray)
            assert video.shape[0] == 3
    
    def test_memory_management(self):
        """Test memory management functions."""
        # Load a backend
        self.generator.generate_text_to_video("test", num_frames=2)
        
        # Test memory clearing
        self.generator.clear_all_memory()  # Should not raise
        
        # Test backend unloading
        self.generator.unload_all_backends()
        assert len(self.generator.backends) == 0
        assert self.generator.current_backend is None


class TestVideoGeneratorEdgeCases:
    """Test edge cases for VideoGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = VideoGenerator(device="cpu")
    
    def test_invalid_backend_request(self):
        """Test requesting invalid backend."""
        with pytest.raises(VideoGenerationError):
            self.generator.generate_text_to_video(
                prompt="test",
                backend="nonexistent"
            )
    
    def test_sharp_text_to_video_skipped(self):
        """Test that SHARP is skipped for text-to-video."""
        # Should fallback to other backends since SHARP doesn't support T2V
        video = self.generator.generate_text_to_video(
            prompt="test",
            num_frames=2
        )
        
        assert isinstance(video, np.ndarray)
        # Should not use SHARP backend
        assert self.generator.current_backend != "sharp"
    
    def test_empty_prompt_handling(self):
        """Test handling of empty prompts."""
        video = self.generator.generate_text_to_video(
            prompt="",
            num_frames=2
        )
        
        assert isinstance(video, np.ndarray)
        assert video.shape[0] == 2


# Property-based tests for SHARP backend
class TestSHARPBackendProperties:
    """Property-based tests for SHARP backend."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = SHARPBackend(device="cpu")
        self.backend.load_model()
    
    @given(
        image_shape=st.tuples(
            st.integers(min_value=64, max_value=256),  # height
            st.integers(min_value=64, max_value=256),  # width
            st.just(3)                                 # RGB channels
        )
    )
    @settings(max_examples=5, deadline=30000)
    def test_property_image_to_3d_output_validity(self, image_shape):
        """Property: Image-to-3D always produces valid 3D representation."""
        height, width, channels = image_shape
        
        # Create test image
        test_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        
        gaussian_splat = self.backend.image_to_3d(test_image)
        
        # Validate 3D representation
        assert isinstance(gaussian_splat, dict)
        required_keys = ["positions", "colors", "opacities", "scales", "rotations"]
        for key in required_keys:
            assert key in gaussian_splat
            assert isinstance(gaussian_splat[key], np.ndarray)
        
        # Check consistency of Gaussian count
        num_gaussians = gaussian_splat["positions"].shape[0]
        assert gaussian_splat["colors"].shape[0] == num_gaussians
        assert gaussian_splat["opacities"].shape[0] == num_gaussians
        assert gaussian_splat["scales"].shape[0] == num_gaussians
        assert gaussian_splat["rotations"].shape[0] == num_gaussians
        
        # Check data types and ranges
        assert gaussian_splat["positions"].dtype == np.float32
        assert gaussian_splat["colors"].dtype == np.float32
        assert np.all(gaussian_splat["colors"] >= 0) and np.all(gaussian_splat["colors"] <= 1)
        assert np.all(gaussian_splat["opacities"] >= 0) and np.all(gaussian_splat["opacities"] <= 1)
    
    @given(
        num_frames=st.integers(min_value=2, max_value=10),
        start_pos=st.tuples(*[st.floats(min_value=-5.0, max_value=5.0) for _ in range(7)]),
        end_pos=st.tuples(*[st.floats(min_value=-5.0, max_value=5.0) for _ in range(7)])
    )
    @settings(max_examples=5, deadline=30000)
    def test_property_camera_path_generation_validity(self, num_frames, start_pos, end_pos):
        """Property: Camera path generation always produces valid paths."""
        assume(num_frames >= 2)
        assume(all(not np.isnan(x) and not np.isinf(x) for x in start_pos))
        assume(all(not np.isnan(x) and not np.isinf(x) for x in end_pos))
        
        camera_path = self.backend.generate_camera_path(
            start_position=start_pos,
            end_position=end_pos,
            num_frames=num_frames,
            interpolation="linear"
        )
        
        # Validate path structure
        assert len(camera_path) == num_frames
        
        for i, keyframe in enumerate(camera_path):
            assert isinstance(keyframe, dict)
            assert "position" in keyframe
            assert "rotation" in keyframe
            assert "fov" in keyframe
            assert "timestamp" in keyframe
            
            # Check data types and structure
            assert len(keyframe["position"]) == 3
            assert len(keyframe["rotation"]) == 3
            assert isinstance(keyframe["fov"], (int, float))
            assert isinstance(keyframe["timestamp"], (int, float))
            
            # Check that values are finite
            assert all(np.isfinite(x) for x in keyframe["position"])
            assert all(np.isfinite(x) for x in keyframe["rotation"])
            assert np.isfinite(keyframe["fov"])
            assert np.isfinite(keyframe["timestamp"])
        
        # Check interpolation correctness
        if num_frames > 1:
            # First frame should match start position
            first_pos = camera_path[0]["position"]
            expected_start = start_pos[:3]
            for i in range(3):
                assert abs(first_pos[i] - expected_start[i]) < 1e-6
            
            # Last frame should match end position
            last_pos = camera_path[-1]["position"]
            expected_end = end_pos[:3]
            for i in range(3):
                assert abs(last_pos[i] - expected_end[i]) < 1e-6