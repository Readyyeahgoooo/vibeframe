"""Property-based tests for VibeFrame data models."""

import pytest
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck
from datetime import datetime
from typing import List

from vibeframe.models import (
    CutPoint, CharacterDescriptor, CameraKeyframe, CameraPath,
    SceneDescription, Storyboard, AudioFeatures, ProjectStatus,
    validate_audio_file, validate_image_file, validate_video_file
)

# Hypothesis strategies for generating test data
@st.composite
def cut_point_strategy(draw):
    """Generate valid CutPoint instances."""
    return CutPoint(
        timestamp=draw(st.floats(min_value=0.0, max_value=300.0)),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        beat_strength=draw(st.floats(min_value=0.0, max_value=1.0)),
        section=draw(st.sampled_from(["intro", "verse", "chorus", "bridge", "outro"]))
    )

@st.composite
def character_descriptor_strategy(draw):
    """Generate valid CharacterDescriptor instances."""
    return CharacterDescriptor(
        appearance=draw(st.text(min_size=10, max_size=100)),
        clothing=draw(st.text(min_size=10, max_size=100)),
        style=draw(st.text(min_size=5, max_size=50)),
        age=draw(st.one_of(st.none(), st.text(min_size=1, max_size=20))),
        ethnicity=draw(st.one_of(st.none(), st.text(min_size=1, max_size=30))),
        distinctive_features=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100)))
    )

@st.composite
def camera_keyframe_strategy(draw):
    """Generate valid CameraKeyframe instances."""
    return CameraKeyframe(
        timestamp=draw(st.floats(min_value=0.0, max_value=60.0)),
        position=draw(st.tuples(
            st.floats(min_value=-10.0, max_value=10.0),
            st.floats(min_value=-10.0, max_value=10.0),
            st.floats(min_value=0.1, max_value=20.0)
        )),
        rotation=draw(st.tuples(
            st.floats(min_value=-180.0, max_value=180.0),
            st.floats(min_value=-180.0, max_value=180.0),
            st.floats(min_value=-180.0, max_value=180.0)
        )),
        fov=draw(st.floats(min_value=10.0, max_value=120.0))
    )

@st.composite
def camera_path_strategy(draw):
    """Generate valid CameraPath instances."""
    num_keyframes = draw(st.integers(min_value=2, max_value=10))
    timestamps = draw(st.lists(
        st.floats(min_value=0.0, max_value=60.0), 
        min_size=num_keyframes, 
        max_size=num_keyframes,
        unique=True
    ))
    timestamps.sort()
    
    keyframes = []
    for timestamp in timestamps:
        keyframe = CameraKeyframe(
            timestamp=timestamp,
            position=draw(st.tuples(
                st.floats(min_value=-10.0, max_value=10.0),
                st.floats(min_value=-10.0, max_value=10.0),
                st.floats(min_value=0.1, max_value=20.0)
            )),
            rotation=draw(st.tuples(
                st.floats(min_value=-180.0, max_value=180.0),
                st.floats(min_value=-180.0, max_value=180.0),
                st.floats(min_value=-180.0, max_value=180.0)
            )),
            fov=draw(st.floats(min_value=10.0, max_value=120.0))
        )
        keyframes.append(keyframe)
    
    return CameraPath(
        keyframes=keyframes,
        interpolation=draw(st.sampled_from(["linear", "bezier", "catmull-rom"]))
    )

@st.composite
def scene_description_strategy(draw):
    """Generate valid SceneDescription instances."""
    start_time = draw(st.floats(min_value=0.0, max_value=250.0))
    duration = draw(st.floats(min_value=1.0, max_value=30.0))
    end_time = start_time + duration
    
    return SceneDescription(
        id=draw(st.integers(min_value=1, max_value=100)),
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        description=draw(st.text(min_size=10, max_size=200)),
        character_action=draw(st.text(min_size=5, max_size=100)),
        camera_angle=draw(st.text(min_size=5, max_size=100)),
        lighting=draw(st.text(min_size=5, max_size=100)),
        environment=draw(st.text(min_size=5, max_size=100)),
        video_prompt=draw(st.text(min_size=10, max_size=300)),
        character_descriptor=draw(st.one_of(st.none(), character_descriptor_strategy())),
        reference_image=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
        generated_video_path=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
        camera_path=draw(st.one_of(st.none(), camera_path_strategy()))
    )

@st.composite
def storyboard_strategy(draw):
    """Generate valid Storyboard instances."""
    scenes = draw(st.lists(scene_description_strategy(), min_size=1, max_size=20))
    
    return Storyboard(
        project_name=draw(st.text(min_size=5, max_size=50)),
        audio_file=draw(st.text(min_size=5, max_size=100)),
        audio_duration=draw(st.floats(min_value=10.0, max_value=600.0)),
        global_style=draw(st.text(min_size=5, max_size=100)),
        theme=draw(st.one_of(st.none(), st.text(min_size=5, max_size=100))),
        scenes=scenes,
        fps=draw(st.sampled_from([24, 30, 60])),
        resolution=draw(st.sampled_from([(1280, 720), (1920, 1080), (3840, 2160)])),
        model=draw(st.sampled_from(["longcat", "hunyuan", "sharp"])),
        created_at=draw(st.one_of(st.none(), st.datetimes()))
    )

# Feature: ai-music-video-generator, Property 24: Project Persistence Round-Trip
@given(storyboard=storyboard_strategy())
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])  # Reduced examples and suppress slow check
def test_property_project_persistence_round_trip(storyboard):
    """
    For any created project with storyboard and configuration, saving and then 
    loading the project should restore all data exactly, allowing continuation 
    from any step.
    """
    # Serialize to JSON
    json_str = storyboard.to_json()
    
    # Deserialize from JSON
    loaded_storyboard = Storyboard.from_json(json_str)
    
    # Property: All basic fields should be identical
    assert loaded_storyboard.project_name == storyboard.project_name
    assert loaded_storyboard.audio_file == storyboard.audio_file
    assert loaded_storyboard.audio_duration == storyboard.audio_duration
    assert loaded_storyboard.global_style == storyboard.global_style
    assert loaded_storyboard.theme == storyboard.theme
    assert loaded_storyboard.fps == storyboard.fps
    assert loaded_storyboard.resolution == storyboard.resolution
    assert loaded_storyboard.model == storyboard.model
    
    # Property: Scene count should be identical
    assert len(loaded_storyboard.scenes) == len(storyboard.scenes)
    
    # Property: Each scene should be identical
    for orig_scene, loaded_scene in zip(storyboard.scenes, loaded_storyboard.scenes):
        assert loaded_scene.id == orig_scene.id
        assert loaded_scene.start_time == orig_scene.start_time
        assert loaded_scene.end_time == orig_scene.end_time
        assert loaded_scene.duration == orig_scene.duration
        assert loaded_scene.description == orig_scene.description
        assert loaded_scene.character_action == orig_scene.character_action
        assert loaded_scene.camera_angle == orig_scene.camera_angle
        assert loaded_scene.lighting == orig_scene.lighting
        assert loaded_scene.environment == orig_scene.environment
        assert loaded_scene.video_prompt == orig_scene.video_prompt
        assert loaded_scene.reference_image == orig_scene.reference_image
        assert loaded_scene.generated_video_path == orig_scene.generated_video_path
        
        # Property: Character descriptors should be identical
        if orig_scene.character_descriptor is not None:
            assert loaded_scene.character_descriptor is not None
            assert loaded_scene.character_descriptor.appearance == orig_scene.character_descriptor.appearance
            assert loaded_scene.character_descriptor.clothing == orig_scene.character_descriptor.clothing
            assert loaded_scene.character_descriptor.style == orig_scene.character_descriptor.style
            assert loaded_scene.character_descriptor.age == orig_scene.character_descriptor.age
            assert loaded_scene.character_descriptor.ethnicity == orig_scene.character_descriptor.ethnicity
            assert loaded_scene.character_descriptor.distinctive_features == orig_scene.character_descriptor.distinctive_features
        else:
            assert loaded_scene.character_descriptor is None
        
        # Property: Camera paths should be identical
        if orig_scene.camera_path is not None:
            assert loaded_scene.camera_path is not None
            assert loaded_scene.camera_path.interpolation == orig_scene.camera_path.interpolation
            assert len(loaded_scene.camera_path.keyframes) == len(orig_scene.camera_path.keyframes)
            
            for orig_kf, loaded_kf in zip(orig_scene.camera_path.keyframes, loaded_scene.camera_path.keyframes):
                assert loaded_kf.timestamp == orig_kf.timestamp
                assert loaded_kf.position == orig_kf.position
                assert loaded_kf.rotation == orig_kf.rotation
                assert loaded_kf.fov == orig_kf.fov
        else:
            assert loaded_scene.camera_path is None
    
    # Property: Created timestamp should be preserved (if set)
    if storyboard.created_at is not None:
        assert loaded_storyboard.created_at == storyboard.created_at

# Additional property tests for individual components
@given(character=character_descriptor_strategy())
@settings(max_examples=100)
def test_character_descriptor_prompt_string_property(character):
    """
    For any character descriptor, the prompt string should contain 
    all non-None fields.
    """
    prompt_string = character.to_prompt_string()
    
    # Property: All non-None fields should be in the prompt string
    assert character.appearance in prompt_string
    assert character.clothing in prompt_string
    assert character.style in prompt_string
    
    if character.age is not None:
        assert character.age in prompt_string
    if character.ethnicity is not None:
        assert character.ethnicity in prompt_string
    if character.distinctive_features is not None:
        assert character.distinctive_features in prompt_string

@given(camera_path=camera_path_strategy())
@settings(max_examples=100)
def test_camera_path_interpolation_property(camera_path):
    """
    For any camera path, interpolating at keyframe timestamps should 
    return the exact keyframe values.
    """
    for keyframe in camera_path.keyframes:
        interpolated = camera_path.interpolate(keyframe.timestamp)
        
        # Property: Interpolating at keyframe timestamp returns exact values (within tolerance)
        assert interpolated.timestamp == keyframe.timestamp
        
        # Use approximate equality for floating point values
        for i in range(3):
            assert abs(interpolated.position[i] - keyframe.position[i]) < 1e-10
            assert abs(interpolated.rotation[i] - keyframe.rotation[i]) < 1e-10
        
        assert abs(interpolated.fov - keyframe.fov) < 1e-10

@given(
    storyboard=storyboard_strategy(),
    scene_id=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=100)
def test_storyboard_scene_lookup_property(storyboard, scene_id):
    """
    For any storyboard and scene ID, get_scene_by_id should return the scene 
    with that ID if it exists, or None if it doesn't.
    """
    scene = storyboard.get_scene_by_id(scene_id)
    
    # Property: If scene is found, it should have the correct ID
    if scene is not None:
        assert scene.id == scene_id
        # Property: The scene should be in the storyboard's scenes list
        assert scene in storyboard.scenes
    
    # Property: If scene is not found, no scene in the list should have that ID
    if scene is None:
        assert all(s.id != scene_id for s in storyboard.scenes)

# Unit tests for validation functions
class TestValidationFunctions:
    """Unit tests for file validation functions."""
    
    def test_validate_audio_file(self):
        """Test audio file validation."""
        # Valid audio files
        assert validate_audio_file("song.mp3")
        assert validate_audio_file("track.wav")
        assert validate_audio_file("music.flac")
        assert validate_audio_file("audio.ogg")
        assert validate_audio_file("sound.m4a")
        assert validate_audio_file("SONG.MP3")  # Case insensitive
        
        # Invalid audio files
        assert not validate_audio_file("video.mp4")
        assert not validate_audio_file("image.jpg")
        assert not validate_audio_file("document.txt")
        assert not validate_audio_file("song")  # No extension
    
    def test_validate_image_file(self):
        """Test image file validation."""
        # Valid image files
        assert validate_image_file("photo.jpg")
        assert validate_image_file("image.jpeg")
        assert validate_image_file("picture.png")
        assert validate_image_file("bitmap.bmp")
        assert validate_image_file("scan.tiff")
        assert validate_image_file("PHOTO.JPG")  # Case insensitive
        
        # Invalid image files
        assert not validate_image_file("video.mp4")
        assert not validate_image_file("audio.mp3")
        assert not validate_image_file("document.txt")
        assert not validate_image_file("image")  # No extension
    
    def test_validate_video_file(self):
        """Test video file validation."""
        # Valid video files
        assert validate_video_file("movie.mp4")
        assert validate_video_file("clip.avi")
        assert validate_video_file("video.mov")
        assert validate_video_file("film.mkv")
        assert validate_video_file("web.webm")
        assert validate_video_file("MOVIE.MP4")  # Case insensitive
        
        # Invalid video files
        assert not validate_video_file("audio.mp3")
        assert not validate_video_file("image.jpg")
        assert not validate_video_file("document.txt")
        assert not validate_video_file("video")  # No extension

# Unit tests for edge cases and error conditions
class TestModelValidation:
    """Unit tests for model validation."""
    
    def test_cut_point_validation(self):
        """Test CutPoint validation."""
        # Valid cut point
        cp = CutPoint(0.0, 0.5, 0.8, "verse")
        assert cp.timestamp == 0.0
        
        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            CutPoint(0.0, 1.5, 0.8, "verse")
        
        # Invalid beat strength
        with pytest.raises(ValueError, match="Beat strength must be between 0 and 1"):
            CutPoint(0.0, 0.5, -0.1, "verse")
        
        # Invalid section
        with pytest.raises(ValueError, match="Invalid section"):
            CutPoint(0.0, 0.5, 0.8, "invalid")
    
    def test_camera_keyframe_validation(self):
        """Test CameraKeyframe validation."""
        # Valid keyframe
        kf = CameraKeyframe(0.0, (0, 0, 5), (0, 0, 0), 45)
        assert kf.fov == 45
        
        # Invalid FOV
        with pytest.raises(ValueError, match="FOV must be between 10 and 120 degrees"):
            CameraKeyframe(0.0, (0, 0, 5), (0, 0, 0), 5)
        
        with pytest.raises(ValueError, match="FOV must be between 10 and 120 degrees"):
            CameraKeyframe(0.0, (0, 0, 5), (0, 0, 0), 150)
    
    def test_camera_path_validation(self):
        """Test CameraPath validation."""
        # Valid path
        kf1 = CameraKeyframe(0.0, (0, 0, 5), (0, 0, 0), 45)
        kf2 = CameraKeyframe(1.0, (1, 0, 5), (0, 0, 0), 45)
        path = CameraPath([kf1, kf2])
        assert len(path.keyframes) == 2
        
        # Invalid - too few keyframes
        with pytest.raises(ValueError, match="Camera path must have at least 2 keyframes"):
            CameraPath([kf1])
        
        # Invalid interpolation
        with pytest.raises(ValueError, match="Invalid interpolation"):
            CameraPath([kf1, kf2], interpolation="invalid")
    
    def test_scene_description_validation(self):
        """Test SceneDescription validation."""
        # Valid scene
        scene = SceneDescription(
            id=1, start_time=0.0, end_time=5.0, duration=5.0,
            description="test", character_action="test", camera_angle="test",
            lighting="test", environment="test", video_prompt="test"
        )
        assert scene.duration == 5.0
        
        # Invalid - start >= end
        with pytest.raises(ValueError, match="Start time must be before end time"):
            SceneDescription(
                id=1, start_time=5.0, end_time=5.0, duration=0.0,
                description="test", character_action="test", camera_angle="test",
                lighting="test", environment="test", video_prompt="test"
            )
        
        # Invalid - duration mismatch
        with pytest.raises(ValueError, match="Duration must match end_time - start_time"):
            SceneDescription(
                id=1, start_time=0.0, end_time=5.0, duration=3.0,
                description="test", character_action="test", camera_angle="test",
                lighting="test", environment="test", video_prompt="test"
            )
    
    def test_storyboard_validation(self):
        """Test Storyboard validation."""
        scene = SceneDescription(
            id=1, start_time=0.0, end_time=5.0, duration=5.0,
            description="test", character_action="test", camera_angle="test",
            lighting="test", environment="test", video_prompt="test"
        )
        
        # Valid storyboard
        storyboard = Storyboard(
            project_name="test", audio_file="test.mp3", audio_duration=60.0,
            global_style="test", theme="test", scenes=[scene]
        )
        assert len(storyboard.scenes) == 1
        
        # Invalid - no scenes
        with pytest.raises(ValueError, match="Storyboard must have at least one scene"):
            Storyboard(
                project_name="test", audio_file="test.mp3", audio_duration=60.0,
                global_style="test", theme="test", scenes=[]
            )
        
        # Invalid FPS
        with pytest.raises(ValueError, match="FPS must be 24, 30, or 60"):
            Storyboard(
                project_name="test", audio_file="test.mp3", audio_duration=60.0,
                global_style="test", theme="test", scenes=[scene], fps=25
            )
        
        # Invalid model
        with pytest.raises(ValueError, match="Model must be 'longcat', 'hunyuan', or 'sharp'"):
            Storyboard(
                project_name="test", audio_file="test.mp3", audio_duration=60.0,
                global_style="test", theme="test", scenes=[scene], model="invalid"
            )

# Audio analysis property tests
@st.composite
def audio_features_strategy(draw):
    """Generate valid AudioFeatures instances."""
    return AudioFeatures(
        tempo=draw(st.floats(min_value=60.0, max_value=200.0)),
        energy=draw(st.floats(min_value=0.0, max_value=1.0)),
        spectral_centroid=draw(st.floats(min_value=100.0, max_value=8000.0)),
        zero_crossing_rate=draw(st.floats(min_value=0.0, max_value=1.0)),
        mfcc=draw(st.lists(st.floats(min_value=-50.0, max_value=50.0), min_size=13, max_size=13)),
        chroma=draw(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=12, max_size=12))
    )

@given(features=audio_features_strategy())
@settings(max_examples=50)
def test_audio_features_to_dict_property(features):
    """
    For any AudioFeatures object, to_dict() should return a dictionary 
    with all expected keys and valid values.
    """
    feature_dict = features.to_dict()
    
    # Property: All expected keys should be present
    expected_keys = {"tempo", "energy", "spectral_centroid", "zero_crossing_rate", "mfcc_mean", "chroma_mean"}
    assert set(feature_dict.keys()) == expected_keys
    
    # Property: Values should be within expected ranges
    assert 60.0 <= feature_dict["tempo"] <= 200.0
    assert 0.0 <= feature_dict["energy"] <= 1.0
    assert 100.0 <= feature_dict["spectral_centroid"] <= 8000.0
    assert 0.0 <= feature_dict["zero_crossing_rate"] <= 1.0
    assert -50.0 <= feature_dict["mfcc_mean"] <= 50.0
    assert 0.0 <= feature_dict["chroma_mean"] <= 1.0
    
    # Property: Mean values should be computed correctly
    if features.mfcc:
        expected_mfcc_mean = sum(features.mfcc) / len(features.mfcc)
        assert abs(feature_dict["mfcc_mean"] - expected_mfcc_mean) < 1e-10
    
    if features.chroma:
        expected_chroma_mean = sum(features.chroma) / len(features.chroma)
        assert abs(feature_dict["chroma_mean"] - expected_chroma_mean) < 1e-10