"""Tests for ScenePlanner component."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings
import requests

from vibeframe.scene_planner import ScenePlanner
from vibeframe.models import CutPoint, SceneDescription, AudioFeatures
from vibeframe.exceptions import APIError, ScenePlanningError

class TestScenePlanner:
    """Test ScenePlanner functionality."""
    
    def test_initialization_with_api_key(self):
        """Test ScenePlanner initialization with API key."""
        planner = ScenePlanner("test-api-key")
        assert planner.api_key == "test-api-key"
        assert planner.model == "google/gemini-2.0-flash-exp:free"
        assert planner.base_url == "https://openrouter.ai/api/v1"
    
    def test_initialization_without_api_key(self):
        """Test ScenePlanner initialization without API key (template mode)."""
        planner = ScenePlanner()
        assert planner.api_key is None
        assert len(planner.scene_templates) > 0
    
    def test_analyze_mood(self):
        """Test mood analysis from audio features."""
        planner = ScenePlanner()
        
        # High energy, fast tempo
        features = {"energy": 0.8, "tempo": 150, "spectral_centroid": 3000}
        mood = planner.analyze_mood(features)
        assert "energetic" in mood.lower()
        
        # Low energy, slow tempo
        features = {"energy": 0.2, "tempo": 70, "spectral_centroid": 1000}
        mood = planner.analyze_mood(features)
        assert "calm" in mood.lower() or "peaceful" in mood.lower()
        
        # Medium energy
        features = {"energy": 0.5, "tempo": 120, "spectral_centroid": 2000}
        mood = planner.analyze_mood(features)
        assert isinstance(mood, str) and len(mood) > 0
    
    def test_enhance_prompt(self):
        """Test prompt enhancement functionality."""
        planner = ScenePlanner()
        
        description = "Character walking in a field"
        style = "cinematic"
        
        enhanced = planner.enhance_prompt(description, style)
        
        assert style in enhanced
        assert description in enhanced
        assert "high quality" in enhanced
        assert "Avoid:" in enhanced
    
    def test_enhance_prompt_with_previous_scene(self):
        """Test prompt enhancement with previous scene context."""
        planner = ScenePlanner()
        
        previous_scene = SceneDescription(
            id=1, start_time=0.0, end_time=5.0, duration=5.0,
            description="Previous scene", character_action="walking",
            camera_angle="wide", lighting="bright", environment="forest",
            video_prompt="test prompt"
        )
        
        enhanced = planner.enhance_prompt("New scene", "anime", previous_scene)
        
        assert "anime" in enhanced
        assert "continuity" in enhanced
    
    def test_template_generation(self):
        """Test template-based scene generation."""
        planner = ScenePlanner()  # No API key
        
        audio_features = {"energy": 0.7, "tempo": 140}
        scene_data = planner._generate_with_template("chorus", audio_features, "cinematic")
        
        assert "description" in scene_data
        assert "character_action" in scene_data
        assert "camera_angle" in scene_data
        assert "lighting" in scene_data
        assert "environment" in scene_data
        
        assert "cinematic" in scene_data["description"]
        assert "chorus" in scene_data["character_action"]
    
    def test_fallback_scene_creation(self):
        """Test fallback scene creation."""
        planner = ScenePlanner()
        
        audio_features = {"energy": 0.5, "tempo": 120}
        scene = planner._create_fallback_scene(
            scene_id=1,
            start_time=0.0,
            end_time=5.0,
            section="verse",
            audio_features=audio_features,
            global_style="anime"
        )
        
        assert scene.id == 1
        assert scene.start_time == 0.0
        assert scene.end_time == 5.0
        assert scene.duration == 5.0
        assert "verse" in scene.description
        assert "anime" in scene.description
    
    def test_action_sequence_decomposition_template(self):
        """Test action sequence decomposition using templates."""
        planner = ScenePlanner()  # No API key
        
        action = "throws a ball"
        character = "young athlete"
        
        # Test 2 shots
        shots = planner.generate_action_sequence(action, 2, character)
        assert len(shots) == 2
        assert all(character in shot for shot in shots)
        assert all(action in shot for shot in shots)
        
        # Test 3 shots
        shots = planner.generate_action_sequence(action, 3, character)
        assert len(shots) == 3
        
        # Test more shots
        shots = planner.generate_action_sequence(action, 5, character)
        assert len(shots) == 5
    
    def test_generate_scene_descriptions_no_cut_points(self):
        """Test error handling when no cut points provided."""
        planner = ScenePlanner()
        
        with pytest.raises(ScenePlanningError, match="No cut points provided"):
            planner.generate_scene_descriptions([], [])
    
    def test_generate_scene_descriptions_mismatched_features(self):
        """Test error handling when audio features don't match cut points."""
        planner = ScenePlanner()
        
        cut_points = [
            CutPoint(0.0, 0.8, 0.9, "intro"),
            CutPoint(10.0, 0.7, 0.8, "verse"),
            CutPoint(20.0, 0.9, 0.9, "chorus")
        ]
        
        # Wrong number of features
        audio_features = [{"energy": 0.5, "tempo": 120}]  # Should be 2 features for 3 cut points
        
        with pytest.raises(ScenePlanningError, match="Audio features count"):
            planner.generate_scene_descriptions(cut_points, audio_features)
    
    def test_generate_scene_descriptions_template_mode(self):
        """Test scene generation using templates (no API key)."""
        planner = ScenePlanner()  # No API key
        
        cut_points = [
            CutPoint(0.0, 0.8, 0.9, "intro"),
            CutPoint(10.0, 0.7, 0.8, "verse"),
            CutPoint(20.0, 0.9, 0.9, "chorus")
        ]
        
        audio_features = [
            {"energy": 0.6, "tempo": 130},
            {"energy": 0.8, "tempo": 140}
        ]
        
        scenes = planner.generate_scene_descriptions(
            cut_points, audio_features, "cinematic", "adventure"
        )
        
        assert len(scenes) == 2  # n-1 scenes for n cut points
        
        for i, scene in enumerate(scenes):
            assert scene.id == i + 1
            assert scene.start_time == cut_points[i].timestamp
            assert scene.end_time == cut_points[i + 1].timestamp
            assert scene.duration == scene.end_time - scene.start_time
            assert len(scene.description) > 0
            assert len(scene.character_action) > 0
            assert len(scene.camera_angle) > 0
            assert len(scene.lighting) > 0
            assert len(scene.environment) > 0
            assert len(scene.video_prompt) > 0
    
    @patch('requests.post')
    def test_api_generation_success(self, mock_post):
        """Test successful API-based scene generation."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "description": "Character dancing in a vibrant club",
                        "character_action": "Dancing energetically to the beat",
                        "camera_angle": "Dynamic rotating camera around character",
                        "lighting": "Colorful strobe lights and neon",
                        "environment": "Modern nightclub with crowd"
                    })
                }
            }]
        }
        mock_post.return_value = mock_response
        
        planner = ScenePlanner("test-api-key")
        
        scene_data = planner._generate_with_api(
            section="chorus",
            audio_features={"energy": 0.8, "tempo": 140},
            duration=10.0,
            global_style="cinematic",
            user_theme="party",
            previous_scene=None
        )
        
        assert scene_data["description"] == "Character dancing in a vibrant club"
        assert scene_data["character_action"] == "Dancing energetically to the beat"
        assert "camera_angle" in scene_data
        assert "lighting" in scene_data
        assert "environment" in scene_data
        
        # Verify API was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "openrouter.ai" in call_args[0][0]
        assert call_args[1]["json"]["model"] == "google/gemini-2.0-flash-exp:free"
    
    @patch('requests.post')
    def test_api_generation_rate_limit(self, mock_post):
        """Test API rate limit handling."""
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "1"}
        mock_post.return_value = mock_response
        
        planner = ScenePlanner("test-api-key")
        
        with pytest.raises(APIError, match="API request failed after"):
            planner._generate_with_api(
                section="verse",
                audio_features={"energy": 0.5, "tempo": 120},
                duration=8.0,
                global_style="anime",
                user_theme=None,
                previous_scene=None
            )
    
    @patch('requests.post')
    def test_api_generation_invalid_json(self, mock_post):
        """Test handling of invalid JSON response from API."""
        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "This is not valid JSON"
                }
            }]
        }
        mock_post.return_value = mock_response
        
        planner = ScenePlanner("test-api-key")
        
        with pytest.raises(APIError, match="Failed to parse API response"):
            planner._generate_with_api(
                section="bridge",
                audio_features={"energy": 0.4, "tempo": 100},
                duration=6.0,
                global_style="realistic",
                user_theme="mystery",
                previous_scene=None
            )
    
    @patch('requests.post')
    def test_api_generation_missing_fields(self, mock_post):
        """Test handling of API response with missing required fields."""
        # Mock response with incomplete data
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "description": "Incomplete scene",
                        "character_action": "Some action"
                        # Missing camera_angle, lighting, environment
                    })
                }
            }]
        }
        mock_post.return_value = mock_response
        
        planner = ScenePlanner("test-api-key")
        
        with pytest.raises(APIError, match="Failed to parse API response"):
            planner._generate_with_api(
                section="outro",
                audio_features={"energy": 0.3, "tempo": 90},
                duration=12.0,
                global_style="dreamy",
                user_theme="farewell",
                previous_scene=None
            )
    
    @patch('requests.post')
    def test_api_action_decomposition_success(self, mock_post):
        """Test successful API-based action decomposition."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps([
                        "Wide shot: Character winds up to throw the ball",
                        "Medium shot: Character releases the ball with force",
                        "Close-up: Character watches the ball's trajectory"
                    ])
                }
            }]
        }
        mock_post.return_value = mock_response
        
        planner = ScenePlanner("test-api-key")
        
        shots = planner.generate_action_sequence("throws a ball", 3, "young athlete")
        
        assert len(shots) == 3
        assert "Wide shot" in shots[0]
        assert "Medium shot" in shots[1]
        assert "Close-up" in shots[2]
        assert all("ball" in shot for shot in shots)
    
    @patch('requests.get')
    def test_get_api_status_success(self, mock_get):
        """Test API status check when API is available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        planner = ScenePlanner("test-api-key")
        status = planner.get_api_status()
        
        assert status["api_available"] is True
        assert status["fallback_mode"] is False
        assert status["model"] == "google/gemini-2.0-flash-exp:free"
    
    @patch('requests.get')
    def test_get_api_status_invalid_key(self, mock_get):
        """Test API status check with invalid API key."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        planner = ScenePlanner("test-api-key")
        status = planner.get_api_status()
        
        assert status["api_available"] is False
        assert status["fallback_mode"] is True
        assert "Invalid API key" in status["reason"]
    
    def test_get_api_status_no_key(self):
        """Test API status check without API key."""
        planner = ScenePlanner()  # No API key
        status = planner.get_api_status()
        
        assert status["api_available"] is False
        assert status["fallback_mode"] is True
        assert "No API key" in status["reason"]
    
    @patch('requests.get')
    def test_get_api_status_rate_limit(self, mock_get):
        """Test API status check when rate limited."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response
        
        planner = ScenePlanner("test-api-key")
        status = planner.get_api_status()
        
        assert status["api_available"] is False
        assert status["fallback_mode"] is True
        assert "Rate limit" in status["reason"]

    @patch('requests.post')
    def test_api_fallback_on_failure(self, mock_post):
        """Test fallback to templates when API fails."""
        # Mock API failure
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")
        
        planner = ScenePlanner("test-api-key")
        
        cut_points = [
            CutPoint(0.0, 0.8, 0.9, "intro"),
            CutPoint(10.0, 0.7, 0.8, "verse")
        ]
        
        audio_features = [{"energy": 0.6, "tempo": 130}]
        
        # Should not raise exception, should fall back to templates
        scenes = planner.generate_scene_descriptions(
            cut_points, audio_features, "cinematic", "adventure"
        )
        
        assert len(scenes) == 1
        assert "cinematic" in scenes[0].description
        assert len(scenes[0].character_action) > 0
        
        # Verify API was attempted
        mock_post.assert_called()

    @patch('requests.post')
    def test_api_fallback_on_rate_limit(self, mock_post):
        """Test fallback to templates when API rate limit is exceeded."""
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "1"}
        mock_post.return_value = mock_response
        
        planner = ScenePlanner("test-api-key")
        
        cut_points = [
            CutPoint(0.0, 0.8, 0.9, "chorus"),
            CutPoint(15.0, 0.9, 0.9, "outro")
        ]
        
        audio_features = [{"energy": 0.8, "tempo": 140}]
        
        # Should fall back to templates after rate limit
        scenes = planner.generate_scene_descriptions(
            cut_points, audio_features, "anime", "party"
        )
        
        assert len(scenes) == 1
        assert "anime" in scenes[0].description
        assert "chorus" in scenes[0].character_action
        
        # Verify API was attempted multiple times
        assert mock_post.call_count == 3  # max_retries

    @patch('requests.post')
    def test_api_fallback_on_invalid_response(self, mock_post):
        """Test fallback to templates when API returns invalid response."""
        # Mock invalid JSON response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "This is not valid JSON for scene generation"
                }
            }]
        }
        mock_post.return_value = mock_response
        
        planner = ScenePlanner("test-api-key")
        
        cut_points = [
            CutPoint(0.0, 0.5, 0.6, "bridge"),
            CutPoint(8.0, 0.4, 0.5, "outro")
        ]
        
        audio_features = [{"energy": 0.4, "tempo": 100}]
        
        # Should fall back to templates after invalid response
        scenes = planner.generate_scene_descriptions(
            cut_points, audio_features, "realistic", "mystery"
        )
        
        assert len(scenes) == 1
        assert "realistic" in scenes[0].description
        assert "bridge" in scenes[0].character_action
        
        # Verify API was attempted
        mock_post.assert_called()

    def test_template_fallback_without_api_key(self):
        """Test that template fallback works correctly without API key."""
        planner = ScenePlanner()  # No API key
        
        cut_points = [
            CutPoint(0.0, 0.6, 0.7, "verse"),
            CutPoint(12.0, 0.8, 0.9, "chorus"),
            CutPoint(25.0, 0.3, 0.4, "outro")
        ]
        
        audio_features = [
            {"energy": 0.6, "tempo": 125},
            {"energy": 0.8, "tempo": 140}
        ]
        
        scenes = planner.generate_scene_descriptions(
            cut_points, audio_features, "dreamy", "romance"
        )
        
        assert len(scenes) == 2
        
        # First scene (verse)
        assert "dreamy" in scenes[0].description
        assert "verse" in scenes[0].character_action
        assert scenes[0].start_time == 0.0
        assert scenes[0].end_time == 12.0
        
        # Second scene (chorus)
        assert "dreamy" in scenes[1].description
        assert "chorus" in scenes[1].character_action
        assert scenes[1].start_time == 12.0
        assert scenes[1].end_time == 25.0

# Property-based tests
@st.composite
def cut_points_strategy(draw):
    """Generate valid cut points for testing."""
    num_points = draw(st.integers(min_value=2, max_value=10))
    timestamps = draw(st.lists(
        st.floats(min_value=0.0, max_value=180.0),
        min_size=num_points,
        max_size=num_points,
        unique=True
    ))
    timestamps.sort()
    
    sections = ["intro", "verse", "chorus", "bridge", "outro"]
    
    return [
        CutPoint(
            timestamp=timestamp,
            confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
            beat_strength=draw(st.floats(min_value=0.0, max_value=1.0)),
            section=draw(st.sampled_from(sections))
        )
        for timestamp in timestamps
    ]

@st.composite
def audio_features_strategy(draw):
    """Generate valid audio features for testing."""
    return {
        "energy": draw(st.floats(min_value=0.0, max_value=1.0)),
        "tempo": draw(st.floats(min_value=60.0, max_value=200.0)),
        "spectral_centroid": draw(st.floats(min_value=500.0, max_value=8000.0)),
        "zero_crossing_rate": draw(st.floats(min_value=0.0, max_value=1.0))
    }

# Feature: ai-music-video-generator, Property 4: Scene Description Completeness
@given(
    cut_points=cut_points_strategy(),
    global_style=st.one_of(st.none(), st.sampled_from(["cinematic", "anime", "realistic", "dreamy"])),
    user_theme=st.one_of(st.none(), st.sampled_from(["adventure", "romance", "mystery", "party"]))
)
@settings(max_examples=20)
def test_property_scene_description_completeness(cut_points, global_style, user_theme):
    """
    For any generated scene description, it should contain all required fields:
    description, character_action, camera_angle, lighting, and environment.
    """
    if len(cut_points) < 2:
        return  # Skip if not enough cut points
    
    planner = ScenePlanner()  # Template mode for reliable testing
    
    # Generate audio features for each segment
    audio_features = []
    for i in range(len(cut_points) - 1):
        features = {
            "energy": 0.5,
            "tempo": 120.0,
            "spectral_centroid": 2000.0,
            "zero_crossing_rate": 0.1
        }
        audio_features.append(features)
    
    scenes = planner.generate_scene_descriptions(
        cut_points, audio_features, global_style, user_theme
    )
    
    # Property: All scenes should have complete descriptions
    for scene in scenes:
        assert isinstance(scene.description, str) and len(scene.description) > 0
        assert isinstance(scene.character_action, str) and len(scene.character_action) > 0
        assert isinstance(scene.camera_angle, str) and len(scene.camera_angle) > 0
        assert isinstance(scene.lighting, str) and len(scene.lighting) > 0
        assert isinstance(scene.environment, str) and len(scene.environment) > 0
        assert isinstance(scene.video_prompt, str) and len(scene.video_prompt) > 0

# Feature: ai-music-video-generator, Property 5: Global Style Consistency
@given(
    cut_points=cut_points_strategy(),
    global_style=st.sampled_from(["cinematic", "anime", "realistic", "dreamy", "cyberpunk"])
)
@settings(max_examples=15)
def test_property_global_style_consistency(cut_points, global_style):
    """
    For any user-specified global style and any set of generated scene descriptions,
    all descriptions should contain or reference the specified style.
    """
    if len(cut_points) < 2:
        return  # Skip if not enough cut points
    
    planner = ScenePlanner()  # Template mode
    
    # Generate audio features
    audio_features = []
    for i in range(len(cut_points) - 1):
        features = {
            "energy": 0.6,
            "tempo": 130.0,
            "spectral_centroid": 2500.0,
            "zero_crossing_rate": 0.15
        }
        audio_features.append(features)
    
    scenes = planner.generate_scene_descriptions(
        cut_points, audio_features, global_style, None
    )
    
    # Property: All scenes should reference the global style
    for scene in scenes:
        # Check if style appears in description or video prompt
        style_in_description = global_style.lower() in scene.description.lower()
        style_in_prompt = global_style.lower() in scene.video_prompt.lower()
        
        assert style_in_description or style_in_prompt, \
            f"Style '{global_style}' not found in scene: {scene.description}"

# Feature: ai-music-video-generator, Property 14: Action Sequence Decomposition
@given(
    action=st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Ps', 'Pe', 'Po'))),
    character=st.text(min_size=5, max_size=30, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Ps', 'Pe', 'Po'))),
    num_shots=st.integers(min_value=2, max_value=8)
)
@settings(max_examples=20)
def test_property_action_sequence_decomposition(action, character, num_shots):
    """
    For any action description, the Scene_Planner should generate at least 2
    connected shot prompts that together describe the complete action.
    """
    planner = ScenePlanner()  # Template mode
    
    shots = planner.generate_action_sequence(action, num_shots, character)
    
    # Property: Should generate exactly the requested number of shots
    assert len(shots) == num_shots
    
    # Property: Each shot should be a non-empty string
    assert all(isinstance(shot, str) and len(shot) > 0 for shot in shots)
    
    # Property: Character should appear in all shots
    assert all(character.lower() in shot.lower() for shot in shots)
    
    # Property: Action should appear in most shots (allowing for some variation)
    action_appearances = sum(1 for shot in shots if action.lower() in shot.lower())
    assert action_appearances >= max(1, num_shots // 2), \
        f"Action '{action}' should appear in at least half the shots"