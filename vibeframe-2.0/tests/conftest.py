"""Pytest configuration and fixtures for VibeFrame tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List

from vibeframe.models import (
    CutPoint, CharacterDescriptor, CameraKeyframe, CameraPath,
    SceneDescription, Storyboard, AudioFeatures
)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_cut_points() -> List[CutPoint]:
    """Create sample cut points for testing."""
    return [
        CutPoint(timestamp=0.0, confidence=0.9, beat_strength=0.8, section="intro"),
        CutPoint(timestamp=5.2, confidence=0.85, beat_strength=0.7, section="verse"),
        CutPoint(timestamp=12.1, confidence=0.92, beat_strength=0.9, section="chorus"),
        CutPoint(timestamp=20.5, confidence=0.88, beat_strength=0.75, section="verse"),
        CutPoint(timestamp=28.3, confidence=0.95, beat_strength=0.95, section="chorus"),
    ]

@pytest.fixture
def sample_character() -> CharacterDescriptor:
    """Create a sample character descriptor."""
    return CharacterDescriptor(
        appearance="young woman with long black hair and blue eyes",
        clothing="wearing a red leather jacket and dark jeans",
        style="anime style, detailed, vibrant colors",
        age="25",
        ethnicity="Asian",
        distinctive_features="small scar above left eyebrow"
    )

@pytest.fixture
def sample_camera_path() -> CameraPath:
    """Create a sample camera path."""
    keyframes = [
        CameraKeyframe(0.0, (0, 0, 5), (0, 0, 0), 45),
        CameraKeyframe(2.5, (2, 1, 4), (10, 15, 0), 50),
        CameraKeyframe(5.0, (-1, 2, 3), (-5, -10, 5), 40)
    ]
    return CameraPath(keyframes=keyframes, interpolation="linear")

@pytest.fixture
def sample_scenes(sample_character, sample_camera_path) -> List[SceneDescription]:
    """Create sample scene descriptions."""
    return [
        SceneDescription(
            id=1,
            start_time=0.0,
            end_time=5.2,
            duration=5.2,
            description="Opening scene with character introduction",
            character_action="walking towards camera",
            camera_angle="medium shot, eye level",
            lighting="golden hour, warm lighting",
            environment="urban street with neon signs",
            video_prompt="A young woman walks confidently down a neon-lit street",
            character_descriptor=sample_character,
            camera_path=sample_camera_path
        ),
        SceneDescription(
            id=2,
            start_time=5.2,
            end_time=12.1,
            duration=6.9,
            description="Character in action sequence",
            character_action="dancing to the beat",
            camera_angle="wide shot, low angle",
            lighting="dynamic colored lights",
            environment="nightclub interior",
            video_prompt="The same woman dances energetically in a vibrant nightclub",
            character_descriptor=sample_character
        )
    ]

@pytest.fixture
def sample_storyboard(sample_scenes) -> Storyboard:
    """Create a sample storyboard."""
    return Storyboard(
        project_name="Test Music Video",
        audio_file="test_song.mp3",
        audio_duration=180.0,
        global_style="cyberpunk anime",
        theme="urban adventure",
        scenes=sample_scenes,
        fps=24,
        resolution=(1920, 1080),
        model="longcat",
        created_at=datetime(2024, 1, 1, 12, 0, 0)
    )

@pytest.fixture
def sample_audio_features() -> AudioFeatures:
    """Create sample audio features."""
    return AudioFeatures(
        tempo=120.0,
        energy=0.75,
        spectral_centroid=2500.0,
        zero_crossing_rate=0.1,
        mfcc=[1.2, -0.5, 0.8, -0.3, 0.6, -0.1, 0.4, -0.2, 0.3, -0.1, 0.2, -0.05, 0.1],
        chroma=[0.8, 0.2, 0.1, 0.3, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.3, 0.2]
    )