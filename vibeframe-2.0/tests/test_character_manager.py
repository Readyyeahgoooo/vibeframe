"""Tests for CharacterManager component."""

import pytest
import tempfile
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck

from vibeframe.character_manager import CharacterManager
from vibeframe.models import CharacterDescriptor, SceneDescription
from vibeframe.exceptions import CharacterError

class TestCharacterManager:
    """Test CharacterManager functionality."""
    
    def test_initialization(self):
        """Test CharacterManager initialization."""
        manager = CharacterManager()
        assert len(manager.characters) == 0
        assert manager.character_counter == 0
        assert len(manager.character_patterns) > 0
    
    def test_extract_character_description_basic(self):
        """Test basic character extraction."""
        manager = CharacterManager()
        
        # Test with clear character description
        prompt = "A young woman with long blonde hair walking in a field"
        character = manager.extract_character_description(prompt)
        
        assert character is not None
        assert "young woman" in character.distinctive_features.lower()
        assert "blonde hair" in character.appearance.lower()
        assert "young" in (character.age or "").lower() or "woman" in (character.age or "").lower()
    
    def test_extract_character_description_no_character(self):
        """Test extraction with no character present."""
        manager = CharacterManager()
        
        # Test with no character keywords
        prompt = "Beautiful landscape with mountains and trees"
        character = manager.extract_character_description(prompt)
        
        assert character is None
    
    def test_extract_character_description_empty_input(self):
        """Test extraction with empty/invalid input."""
        manager = CharacterManager()
        
        assert manager.extract_character_description("") is None
        assert manager.extract_character_description(None) is None
        assert manager.extract_character_description(123) is None
    
    def test_extract_character_description_complex(self):
        """Test extraction with complex character description."""
        manager = CharacterManager()
        
        prompt = "A tall, athletic man in his 30s wearing a blue suit and tie, standing confidently"
        character = manager.extract_character_description(prompt)
        
        assert character is not None
        assert "tall" in character.appearance.lower()
        assert "athletic" in character.appearance.lower()
        assert "30s" in (character.age or "").lower() or "man" in (character.age or "").lower()
        assert "suit" in character.clothing.lower()
    
    def test_extract_multiple_characters(self):
        """Test extracting multiple characters."""
        manager = CharacterManager()
        
        prompt1 = "A young girl with curly red hair"
        prompt2 = "An elderly man with a beard"
        
        char1 = manager.extract_character_description(prompt1)
        char2 = manager.extract_character_description(prompt2)
        
        assert char1 is not None
        assert char2 is not None
        assert len(manager.characters) == 2
    
    def test_inject_character_basic(self):
        """Test basic character injection."""
        manager = CharacterManager()
        
        character = CharacterDescriptor(
            appearance="blonde hair, blue eyes",
            clothing="casual dress",
            style="cinematic",
            age="young woman",
            ethnicity=None,
            distinctive_features="young woman with blonde hair"
        )
        
        prompt = "Character walking in a park"
        enhanced = manager.inject_character(prompt, character)
        
        assert "blonde hair" in enhanced
        assert "consistency" in enhanced.lower()
    
    def test_inject_character_no_character_keyword(self):
        """Test character injection when no character keyword present."""
        manager = CharacterManager()
        
        character = CharacterDescriptor(
            appearance="blonde hair",
            clothing="dress",
            style="cinematic",
            age="young woman",
            ethnicity=None,
            distinctive_features="young woman"
        )
        
        prompt = "Walking in a park"
        enhanced = manager.inject_character(prompt, character)
        
        assert enhanced.startswith("blonde hair")
        assert "consistency" in enhanced.lower()
    
    def test_inject_character_empty_inputs(self):
        """Test character injection with empty inputs."""
        manager = CharacterManager()
        
        character = CharacterDescriptor(
            appearance="test",
            clothing="test",
            style="test",
            age="test",
            ethnicity=None,
            distinctive_features="test character"
        )
        
        # Empty prompt
        assert manager.inject_character("", character) == ""
        
        # None character
        assert manager.inject_character("test prompt", None) == "test prompt"
    
    @patch('cv2.VideoCapture')
    def test_get_reference_frame_success(self, mock_video_capture):
        """Test successful reference frame extraction."""
        manager = CharacterManager()
        
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "test_video.mp4"
            video_path.touch()  # Create empty file
            
            with patch('cv2.imwrite', return_value=True):
                frame_path = manager.get_reference_frame(str(video_path), 0.5)
                
                assert frame_path is not None
                assert "reference_frames" in frame_path
                assert frame_path.endswith(".jpg")
    
    def test_get_reference_frame_invalid_video(self):
        """Test reference frame extraction with invalid video."""
        manager = CharacterManager()
        
        # Non-existent file
        result = manager.get_reference_frame("nonexistent.mp4")
        assert result is None
        
        # Empty path
        result = manager.get_reference_frame("")
        assert result is None
    
    @patch('cv2.VideoCapture')
    def test_get_reference_frame_video_open_failure(self, mock_video_capture):
        """Test reference frame extraction when video cannot be opened."""
        manager = CharacterManager()
        
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "test_video.mp4"
            video_path.touch()
            
            result = manager.get_reference_frame(str(video_path))
            assert result is None
    
    def test_validate_consistency_identical(self):
        """Test consistency validation with identical characters."""
        manager = CharacterManager()
        
        char1 = CharacterDescriptor(
            id="char1", name="Test 1", description="young woman",
            appearance="blonde hair, blue eyes", clothing="red dress",
            age_gender="young woman", reference_frame_path=None,
            consistency_prompt="test"
        )
        
        char2 = CharacterDescriptor(
            id="char2", name="Test 2", description="young woman",
            appearance="blonde hair, blue eyes", clothing="red dress",
            age_gender="young woman", reference_frame_path=None,
            consistency_prompt="test"
        )
        
        result = manager.validate_consistency(char1, char2)
        
        assert result["consistent"] is True
        assert result["confidence"] > 0.8
        assert len(result["issues"]) == 0
        assert result["similarity_score"] > 0.8
    
    def test_validate_consistency_different(self):
        """Test consistency validation with different characters."""
        manager = CharacterManager()
        
        char1 = CharacterDescriptor(
            id="char1", name="Test 1", description="young woman",
            appearance="blonde hair, blue eyes", clothing="red dress",
            age_gender="young woman", reference_frame_path=None,
            consistency_prompt="test"
        )
        
        char2 = CharacterDescriptor(
            id="char2", name="Test 2", description="old man",
            appearance="gray hair, brown eyes", clothing="black suit",
            age_gender="elderly man", reference_frame_path=None,
            consistency_prompt="test"
        )
        
        result = manager.validate_consistency(char1, char2)
        
        assert result["consistent"] is False
        assert result["confidence"] < 0.6
        assert len(result["issues"]) > 0
    
    def test_validate_consistency_missing_characters(self):
        """Test consistency validation with missing characters."""
        manager = CharacterManager()
        
        char1 = CharacterDescriptor(
            id="char1", name="Test", description="test",
            appearance="test", clothing="test", age_gender="test",
            reference_frame_path=None, consistency_prompt="test"
        )
        
        result = manager.validate_consistency(char1, None)
        
        assert result["consistent"] is False
        assert result["confidence"] == 0.0
        assert "Missing character data" in result["issues"]
    
    def test_get_character_by_id(self):
        """Test getting character by ID."""
        manager = CharacterManager()
        
        # Add a character
        prompt = "A young man with dark hair"
        character = manager.extract_character_description(prompt)
        
        # Retrieve by ID
        retrieved = manager.get_character_by_id(character.id)
        assert retrieved is not None
        assert retrieved.id == character.id
        
        # Non-existent ID
        assert manager.get_character_by_id("nonexistent") is None
    
    def test_get_all_characters(self):
        """Test getting all characters."""
        manager = CharacterManager()
        
        # Initially empty
        assert len(manager.get_all_characters()) == 0
        
        # Add characters
        manager.extract_character_description("A young woman")
        manager.extract_character_description("An old man")
        
        all_chars = manager.get_all_characters()
        assert len(all_chars) == 2
    
    def test_update_character_reference(self):
        """Test updating character reference frame."""
        manager = CharacterManager()
        
        # Add a character
        character = manager.extract_character_description("A young woman")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            ref_path = Path(temp_dir) / "reference.jpg"
            ref_path.touch()
            
            # Update reference
            success = manager.update_character_reference(character.id, str(ref_path))
            assert success is True
            
            # Check it was updated
            updated_char = manager.get_character_by_id(character.id)
            assert updated_char.reference_frame_path == str(ref_path)
    
    def test_update_character_reference_invalid(self):
        """Test updating character reference with invalid inputs."""
        manager = CharacterManager()
        
        # Non-existent character
        success = manager.update_character_reference("nonexistent", "path.jpg")
        assert success is False
        
        # Add a character
        character = manager.extract_character_description("A young woman")
        
        # Non-existent file
        success = manager.update_character_reference(character.id, "nonexistent.jpg")
        assert success is False
    
    def test_merge_characters_success(self):
        """Test successful character merging."""
        manager = CharacterManager()
        
        # Create similar characters
        char1 = manager.extract_character_description("A young woman with blonde hair")
        char2 = manager.extract_character_description("A young woman with long blonde hair")
        
        # Make them more similar for merging
        manager.characters[char1.id].appearance = "blonde hair"
        manager.characters[char2.id].appearance = "blonde hair"
        manager.characters[char1.id].age_gender = "young woman"
        manager.characters[char2.id].age_gender = "young woman"
        
        merged = manager.merge_characters(char1.id, char2.id)
        
        assert merged is not None
        assert merged.id == char1.id
        assert len(manager.characters) == 1
        assert char2.id not in manager.characters
    
    def test_merge_characters_too_different(self):
        """Test merging characters that are too different."""
        manager = CharacterManager()
        
        char1 = manager.extract_character_description("A young woman")
        char2 = manager.extract_character_description("An old man")
        
        merged = manager.merge_characters(char1.id, char2.id)
        
        assert merged is None
        assert len(manager.characters) == 2  # Both still exist
    
    def test_merge_characters_invalid_ids(self):
        """Test merging with invalid character IDs."""
        manager = CharacterManager()
        
        char1 = manager.extract_character_description("A young woman")
        
        # Non-existent second character
        merged = manager.merge_characters(char1.id, "nonexistent")
        assert merged is None
        
        # Non-existent first character
        merged = manager.merge_characters("nonexistent", char1.id)
        assert merged is None
    
    def test_clear_characters(self):
        """Test clearing all characters."""
        manager = CharacterManager()
        
        # Add characters
        manager.extract_character_description("A young woman")
        manager.extract_character_description("An old man")
        
        assert len(manager.characters) == 2
        assert manager.character_counter == 2
        
        # Clear
        manager.clear_characters()
        
        assert len(manager.characters) == 0
        assert manager.character_counter == 0
    
    def test_export_import_characters(self):
        """Test exporting and importing characters."""
        manager = CharacterManager()
        
        # Add characters
        char1 = manager.extract_character_description("A young woman")
        char2 = manager.extract_character_description("An old man")
        
        # Export
        exported_data = manager.export_characters()
        
        assert "characters" in exported_data
        assert "character_counter" in exported_data
        assert len(exported_data["characters"]) == 2
        
        # Clear and import
        manager.clear_characters()
        assert len(manager.characters) == 0
        
        success = manager.import_characters(exported_data)
        assert success is True
        assert len(manager.characters) == 2
        assert manager.character_counter == 2
    
    def test_import_characters_invalid_data(self):
        """Test importing invalid character data."""
        manager = CharacterManager()
        
        # Invalid data structure
        success = manager.import_characters({"invalid": "data"})
        assert success is False
        
        # Malformed character data
        invalid_data = {
            "characters": {
                "char1": {"invalid": "character_data"}
            }
        }
        success = manager.import_characters(invalid_data)
        assert success is False
    
    def test_text_similarity_comparison(self):
        """Test text similarity comparison method."""
        manager = CharacterManager()
        
        # Identical text
        similarity = manager._compare_text_similarity("blonde hair", "blonde hair")
        assert similarity == 1.0
        
        # Completely different
        similarity = manager._compare_text_similarity("blonde hair", "dark eyes")
        assert similarity == 0.0
        
        # Partial overlap
        similarity = manager._compare_text_similarity("long blonde hair", "short blonde hair")
        assert 0.0 < similarity < 1.0
        
        # Empty strings
        similarity = manager._compare_text_similarity("", "")
        assert similarity == 1.0
        
        # One empty
        similarity = manager._compare_text_similarity("text", "")
        assert similarity == 0.0

class TestCharacterManagerEdgeCases:
    """Test edge cases for CharacterManager."""
    
    def test_extract_character_no_character_in_prompt(self):
        """Test with no character in prompt."""
        manager = CharacterManager()
        
        # Prompts without character keywords
        prompts = [
            "Beautiful landscape with mountains",
            "Abstract art with colors",
            "Empty room with furniture",
            "Car driving on highway",
            "Sunset over ocean"
        ]
        
        for prompt in prompts:
            character = manager.extract_character_description(prompt)
            assert character is None
    
    def test_extract_character_multiple_characters_in_prompt(self):
        """Test with multiple characters in one prompt."""
        manager = CharacterManager()
        
        prompt = "A young woman and an old man walking together in the park"
        character = manager.extract_character_description(prompt)
        
        # Should extract at least one character
        assert character is not None
        # Should contain information about at least one of the characters
        assert any(keyword in character.distinctive_features.lower() 
                  for keyword in ["woman", "man", "young", "old"])
    
    def test_extract_character_ambiguous_descriptions(self):
        """Test with ambiguous character descriptions."""
        manager = CharacterManager()
        
        ambiguous_prompts = [
            "Someone walking in the distance",
            "A figure in the shadows",
            "Person wearing something dark",
            "Individual with unclear features"
        ]
        
        for prompt in ambiguous_prompts:
            character = manager.extract_character_description(prompt)
            # Should still extract something, even if minimal
            if character:
                assert len(character.appearance) > 0
                assert len(character.clothing) > 0
    
    def test_extract_character_very_short_prompt(self):
        """Test with very short prompts."""
        manager = CharacterManager()
        
        short_prompts = [
            "man",
            "woman walking",
            "girl",
            "boy running"
        ]
        
        for prompt in short_prompts:
            character = manager.extract_character_description(prompt)
            if character:
                assert isinstance(character.appearance, str)
                assert isinstance(character.clothing, str)
    
    def test_extract_character_very_long_prompt(self):
        """Test with very long, detailed prompts."""
        manager = CharacterManager()
        
        long_prompt = """
        A tall, elegant young woman in her mid-twenties with long, flowing blonde hair 
        that catches the sunlight, piercing blue eyes that seem to hold ancient wisdom, 
        wearing a flowing red silk dress that moves gracefully in the wind, adorned with 
        delicate silver jewelry including a necklace with a mysterious pendant, standing 
        confidently on a cliff overlooking a vast ocean during golden hour, her expression 
        serene yet determined as she gazes into the distance contemplating her next adventure
        """
        
        character = manager.extract_character_description(long_prompt)
        
        assert character is not None
        assert "blonde hair" in character.appearance.lower()
        assert "dress" in character.clothing.lower()
        assert "young" in (character.age or "").lower() or "woman" in (character.age or "").lower()
    
    def test_inject_character_with_existing_character_references(self):
        """Test character injection when prompt already has character references."""
        manager = CharacterManager()
        
        character = CharacterDescriptor(
            appearance="blonde hair, blue eyes",
            clothing="red dress",
            style="cinematic",
            age="young woman",
            ethnicity=None,
            distinctive_features="elegant woman"
        )
        
        prompts_with_character = [
            "The character walks through the forest",
            "A person stands by the lake",
            "The figure moves gracefully",
            "An individual approaches the building"
        ]
        
        for prompt in prompts_with_character:
            enhanced = manager.inject_character(prompt, character)
            # Should replace the generic reference with specific character
            assert "blonde hair" in enhanced
            assert "red dress" in enhanced
            # Generic terms should be replaced
            assert not any(term in enhanced.lower() for term in ["character", "person", "figure", "individual"])
    
    def test_inject_character_with_conflicting_descriptions(self):
        """Test character injection with conflicting character descriptions."""
        manager = CharacterManager()
        
        character = CharacterDescriptor(
            appearance="blonde hair, blue eyes",
            clothing="red dress",
            style="cinematic",
            age="young woman",
            ethnicity=None,
            distinctive_features="elegant woman"
        )
        
        conflicting_prompt = "A dark-haired man in a black suit walks down the street"
        enhanced = manager.inject_character(conflicting_prompt, character)
        
        # Should add character description while preserving original prompt
        assert "blonde hair" in enhanced
        assert "red dress" in enhanced
        assert "dark-haired man" in enhanced  # Original should be preserved
    
    def test_validate_consistency_edge_cases(self):
        """Test consistency validation edge cases."""
        manager = CharacterManager()
        
        # Test with minimal character data
        char1 = CharacterDescriptor(
            appearance="",
            clothing="",
            style="",
            age=None,
            ethnicity=None,
            distinctive_features=""
        )
        
        char2 = CharacterDescriptor(
            appearance="blonde hair",
            clothing="dress",
            style="cinematic",
            age="young",
            ethnicity=None,
            distinctive_features="woman"
        )
        
        result = manager.validate_consistency(char1, char2)
        assert result["consistent"] is False
        assert result["confidence"] < 0.5
        
        # Test with identical empty characters
        char3 = CharacterDescriptor(
            appearance="",
            clothing="",
            style="",
            age=None,
            ethnicity=None,
            distinctive_features=""
        )
        
        result = manager.validate_consistency(char1, char3)
        # Empty characters should be considered consistent
        assert result["consistent"] is True
    
    def test_character_extraction_with_special_characters(self):
        """Test character extraction with special characters and unicode."""
        manager = CharacterManager()
        
        special_prompts = [
            "A young woman with café-au-lait skin",
            "Character wearing a tête-à-tête style dress",
            "Person with naïve expression",
            "Individual with résumé in hand",
            "Figure with piñata-colored clothing"
        ]
        
        for prompt in special_prompts:
            character = manager.extract_character_description(prompt)
            if character:
                # Should handle special characters gracefully
                assert isinstance(character.appearance, str)
                assert isinstance(character.clothing, str)
    
    def test_character_extraction_with_numbers_and_measurements(self):
        """Test character extraction with numeric descriptions."""
        manager = CharacterManager()
        
        numeric_prompts = [
            "A 25-year-old woman with 6-foot height",
            "Person wearing size 8 shoes",
            "Character with 20/20 vision",
            "Individual weighing 150 pounds",
            "Figure with 36-24-36 measurements"
        ]
        
        for prompt in numeric_prompts:
            character = manager.extract_character_description(prompt)
            if character:
                # Should extract meaningful information despite numbers
                assert len(character.distinctive_features) > 0
    
    def test_get_reference_frame_edge_cases(self):
        """Test reference frame extraction edge cases."""
        manager = CharacterManager()
        
        # Test with invalid timestamps
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "test.mp4"
            video_path.touch()  # Create empty file
            
            # Negative timestamp
            result = manager.get_reference_frame(str(video_path), -0.5)
            assert result is None
            
            # Timestamp > 1.0
            result = manager.get_reference_frame(str(video_path), 1.5)
            assert result is None
            
            # Timestamp = 0.0 (edge case)
            result = manager.get_reference_frame(str(video_path), 0.0)
            assert result is None  # Will fail due to empty file
            
            # Timestamp = 1.0 (edge case)
            result = manager.get_reference_frame(str(video_path), 1.0)
            assert result is None  # Will fail due to empty file
    
    def test_character_manager_memory_efficiency(self):
        """Test memory efficiency with many characters."""
        manager = CharacterManager()
        
        # Create many characters
        for i in range(100):
            prompt = f"Character {i} with unique feature {i}"
            character = manager.extract_character_description(prompt)
            if character:
                pass  # Just ensure it doesn't crash
        
        # Should handle many characters without issues
        all_chars = manager.get_all_characters()
        assert len(all_chars) <= 100  # Some prompts might not extract characters
        
        # Clear should work efficiently
        manager.clear_characters()
        assert len(manager.characters) == 0
        assert len(manager.get_all_characters()) == 0
    
    def test_character_export_import_edge_cases(self):
        """Test export/import with edge cases."""
        manager = CharacterManager()
        
        # Test export with no characters
        empty_export = manager.export_characters()
        assert "characters" in empty_export
        assert len(empty_export["characters"]) == 0
        
        # Test import of empty data
        success = manager.import_characters(empty_export)
        assert success is True
        assert len(manager.characters) == 0
        
        # Test import with corrupted data
        corrupted_data = {
            "characters": {
                "char1": {
                    "character": {
                        "appearance": "test",
                        # Missing required fields
                    }
                }
            }
        }
        success = manager.import_characters(corrupted_data)
        assert success is False

# Property-based tests
@st.composite
def character_prompt_strategy(draw):
    """Generate character prompts for testing."""
    age_terms = ["young", "old", "teenage", "middle-aged", "elderly"]
    gender_terms = ["man", "woman", "boy", "girl", "person"]
    hair_colors = ["blonde", "brown", "black", "red", "gray"]
    hair_lengths = ["long", "short", "curly", "straight"]
    clothing = ["dress", "suit", "shirt", "jacket", "casual clothes"]
    
    age = draw(st.sampled_from(age_terms))
    gender = draw(st.sampled_from(gender_terms))
    hair_color = draw(st.sampled_from(hair_colors))
    hair_length = draw(st.sampled_from(hair_lengths))
    cloth = draw(st.sampled_from(clothing))
    
    return f"A {age} {gender} with {hair_length} {hair_color} hair wearing {cloth}"

@st.composite
def character_descriptor_strategy(draw):
    """Generate CharacterDescriptor objects for testing."""
    return CharacterDescriptor(
        appearance=draw(st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Ps', 'Pe', 'Po', 'Zs')))),
        clothing=draw(st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Ps', 'Pe', 'Po', 'Zs')))),
        style=draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Ps', 'Pe', 'Po', 'Zs')))),
        age=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Ps', 'Pe', 'Po', 'Zs'))))),
        ethnicity=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Ps', 'Pe', 'Po', 'Zs'))))),
        distinctive_features=draw(st.one_of(st.none(), st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Ps', 'Pe', 'Po', 'Zs')))))
    )

# Feature: ai-music-video-generator, Property 6: Character Extraction and Injection
@given(prompt=character_prompt_strategy())
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_property_character_extraction_and_injection(prompt):
    """
    For any character prompt, the Character_Manager should extract character
    information and be able to inject it back into video prompts consistently.
    """
    manager = CharacterManager()
    
    # Extract character
    character = manager.extract_character_description(prompt)
    
    # Property: Should extract character from valid prompts
    assert character is not None
    assert isinstance(character.appearance, str) and len(character.appearance) > 0
    assert isinstance(character.clothing, str) and len(character.clothing) > 0
    assert isinstance(character.style, str) and len(character.style) > 0
    
    # Property: Injection should preserve character information
    test_prompt = "Character walking in a scene"
    injected = manager.inject_character(test_prompt, character)
    
    assert isinstance(injected, str) and len(injected) > 0
    assert len(injected) >= len(test_prompt)  # Should be enhanced, not shortened

# Feature: ai-music-video-generator, Property 7: Video Continuation Frame Extraction
@given(timestamp=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
def test_property_video_continuation_frame_extraction(timestamp):
    """
    For any valid timestamp, the Character_Manager should handle frame extraction
    gracefully, either succeeding with valid video or failing safely with invalid input.
    """
    manager = CharacterManager()
    
    # Test with non-existent file (should fail gracefully)
    result = manager.get_reference_frame("nonexistent_video.mp4", timestamp)
    
    # Property: Should handle invalid input gracefully
    assert result is None  # Should not crash, should return None for invalid input

# Feature: ai-music-video-generator, Property 8: Multiple Character Distinctness
@given(
    prompt1=character_prompt_strategy(),
    prompt2=character_prompt_strategy()
)
@settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
def test_property_multiple_character_distinctness(prompt1, prompt2):
    """
    For any two different character prompts, the Character_Manager should create
    distinct character descriptors with unique IDs and maintain them separately.
    """
    manager = CharacterManager()
    
    # Extract two characters
    char1 = manager.extract_character_description(prompt1)
    char2 = manager.extract_character_description(prompt2)
    
    # Property: Both extractions should succeed
    assert char1 is not None
    assert char2 is not None
    
    # Property: Manager should track both characters
    assert len(manager.characters) == 2
    
    # Property: Characters should be retrievable
    all_chars = manager.get_all_characters()
    assert len(all_chars) == 2
    assert char1 in all_chars
    assert char2 in all_chars