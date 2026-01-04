"""Property-based and unit tests for AudioAnalyzer component."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck, assume
from pathlib import Path
import tempfile
import soundfile as sf
from unittest.mock import Mock, patch

from vibeframe.audio_analyzer import AudioAnalyzer
from vibeframe.models import CutPoint, AudioFeatures
from vibeframe.exceptions import AudioLoadError, AudioAnalysisError

# Test data generation strategies
@st.composite
def audio_data_strategy(draw):
    """Generate synthetic audio data for testing."""
    duration = draw(st.floats(min_value=1.0, max_value=60.0))
    sample_rate = draw(st.sampled_from([22050, 44100, 48000]))
    
    # Generate simple sine wave with some noise
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    frequency = draw(st.floats(min_value=100.0, max_value=1000.0))
    
    # Create audio signal
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some noise for realism
    noise_level = draw(st.floats(min_value=0.0, max_value=0.1))
    noise = np.random.normal(0, noise_level, num_samples)
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio.astype(np.float32), sample_rate, duration

def create_temp_audio_file(audio_data, sample_rate):
    """Create a temporary audio file for testing."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio_data, sample_rate)
    temp_file.close()
    return temp_file.name

# Feature: ai-music-video-generator, Property 1: Audio Analysis Output Validity
@given(audio_data=audio_data_strategy())
@settings(max_examples=10, deadline=60000, suppress_health_check=[HealthCheck.too_slow])  # Increased deadline and reduced examples
def test_property_audio_analysis_output_validity(audio_data):
    """
    For any valid audio input, the audio analysis should produce valid output 
    with beats, drums, structure, and features within expected ranges.
    """
    audio, sample_rate, duration = audio_data
    
    # Create temporary audio file
    temp_file = create_temp_audio_file(audio, sample_rate)
    
    try:
        analyzer = AudioAnalyzer(temp_file)
        analyzer.load_audio()
        
        # Property: Beat detection should return valid timestamps
        beats = analyzer.detect_beats(min_interval=2.0)
        assert isinstance(beats, list)
        assert all(isinstance(beat, float) for beat in beats)
        assert all(0.0 <= beat <= duration for beat in beats)
        assert beats == sorted(beats)  # Should be sorted
        
        # Property: Drum detection should return valid (timestamp, strength) pairs
        drums = analyzer.detect_drums()
        assert isinstance(drums, list)
        assert all(isinstance(drum, tuple) and len(drum) == 2 for drum in drums)
        assert all(isinstance(drum[0], float) and isinstance(drum[1], float) for drum in drums)
        assert all(0.0 <= drum[0] <= duration for drum in drums)
        assert all(0.0 <= drum[1] <= 1.0 for drum in drums)  # Normalized strength
        
        # Property: Structure detection should return valid sections
        structure = analyzer.detect_structure()
        assert isinstance(structure, list)
        assert len(structure) > 0  # Should have at least one section
        
        for section in structure:
            assert isinstance(section, dict)
            assert 'start_time' in section
            assert 'end_time' in section
            assert 'duration' in section
            assert 'label' in section
            assert 'confidence' in section
            
            assert 0.0 <= section['start_time'] <= duration
            assert 0.0 <= section['end_time'] <= duration
            assert section['start_time'] < section['end_time']
            assert abs(section['duration'] - (section['end_time'] - section['start_time'])) < 0.1
            assert section['label'] in ["intro", "verse", "chorus", "bridge", "outro"]
            assert 0.0 <= section['confidence'] <= 1.0
        
        # Property: Feature analysis should return valid features for any time range
        if duration > 2.0:  # Only test if audio is long enough
            mid_point = duration / 2
            features = analyzer.analyze_features(0.0, min(2.0, duration))
            
            assert isinstance(features, dict)
            assert 'tempo' in features
            assert 'energy' in features
            assert 'spectral_centroid' in features
            assert 'zero_crossing_rate' in features
            assert 'mfcc' in features
            assert 'chroma' in features
            
            assert 40.0 <= features['tempo'] <= 300.0  # Reasonable tempo range
            assert 0.0 <= features['energy'] <= 1.0
            assert features['spectral_centroid'] > 0
            assert 0.0 <= features['zero_crossing_rate'] <= 1.0
            assert len(features['mfcc']) == 13
            assert len(features['chroma']) == 12
        
        # Property: Cut point generation should return valid CutPoint objects
        cut_points = analyzer.generate_cut_points(strategy="auto")
        assert isinstance(cut_points, list)
        assert all(isinstance(cp, CutPoint) for cp in cut_points)
        assert all(0.0 <= cp.timestamp <= duration for cp in cut_points)
        assert all(0.0 <= cp.confidence <= 1.0 for cp in cut_points)
        assert all(0.0 <= cp.beat_strength <= 1.0 for cp in cut_points)
        assert all(cp.section in ["intro", "verse", "chorus", "bridge", "outro"] for cp in cut_points)
        
        # Property: Cut points should be sorted by timestamp
        timestamps = [cp.timestamp for cp in cut_points]
        assert timestamps == sorted(timestamps)
        
    finally:
        # Clean up temporary file
        Path(temp_file).unlink(missing_ok=True)

# Feature: ai-music-video-generator, Property 2: Audio Loading Robustness
@given(
    duration=st.floats(min_value=0.1, max_value=10.0),
    sample_rate=st.sampled_from([8000, 22050, 44100, 48000, 96000])
)
@settings(max_examples=10, deadline=15000)
def test_property_audio_loading_robustness(duration, sample_rate):
    """
    For any valid audio file format and parameters, the audio loading should 
    succeed and return consistent metadata.
    """
    # Generate simple audio data
    num_samples = int(duration * sample_rate)
    audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples))
    audio_data = audio_data.astype(np.float32)
    
    temp_file = create_temp_audio_file(audio_data, sample_rate)
    
    try:
        analyzer = AudioAnalyzer(temp_file)
        
        # Property: Loading should succeed
        y, sr = analyzer.load_audio()
        
        # Property: Returned data should be consistent
        assert isinstance(y, np.ndarray)
        assert isinstance(sr, int)
        assert len(y) > 0
        assert sr > 0
        
        # Property: Duration should be approximately correct
        loaded_duration = len(y) / sr
        assert abs(loaded_duration - duration) < 0.1  # Allow small tolerance
        
        # Property: Analyzer state should be updated
        assert analyzer.y is not None
        assert analyzer.sr is not None
        assert analyzer.duration > 0
        assert abs(analyzer.duration - duration) < 0.1
        
    finally:
        Path(temp_file).unlink(missing_ok=True)

# Feature: ai-music-video-generator, Property 3: Error Handling for Invalid Audio
@given(
    invalid_path=st.text(min_size=1, max_size=50).filter(lambda x: not x.endswith('.wav'))
)
@settings(max_examples=20)
def test_property_error_handling_invalid_audio(invalid_path):
    """
    For any invalid audio file path or corrupted audio data, the system should 
    raise appropriate exceptions with descriptive messages.
    """
    # Property: Non-existent file should raise AudioLoadError
    with pytest.raises(AudioLoadError, match="Audio file not found"):
        AudioAnalyzer(invalid_path)
    
    # Test with empty file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()
    
    try:
        analyzer = AudioAnalyzer(temp_file.name)
        
        # Property: Empty file should raise AudioLoadError
        with pytest.raises(AudioLoadError, match="Failed to load audio file"):
            analyzer.load_audio()
            
    finally:
        Path(temp_file.name).unlink(missing_ok=True)

# Unit tests for audio analysis edge cases
class TestAudioAnalyzerEdgeCases:
    """Unit tests for AudioAnalyzer edge cases."""
    
    def test_very_short_audio(self):
        """Test with very short audio (<5 seconds)."""
        # Create 2-second audio
        duration = 2.0
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples))
        
        temp_file = create_temp_audio_file(audio_data.astype(np.float32), sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            analyzer.load_audio()
            
            # Should still work with short audio
            beats = analyzer.detect_beats(min_interval=0.5)
            assert isinstance(beats, list)
            
            drums = analyzer.detect_drums()
            assert isinstance(drums, list)
            
            structure = analyzer.detect_structure()
            assert isinstance(structure, list)
            assert len(structure) >= 1
            
            # Feature analysis should work for the entire duration
            features = analyzer.analyze_features(0.0, duration)
            assert isinstance(features, dict)
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_very_long_audio(self):
        """Test with very long audio (>10 minutes) - simulated."""
        # We'll simulate long audio by mocking the duration
        duration = 2.0  # Actual short audio
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples))
        
        temp_file = create_temp_audio_file(audio_data.astype(np.float32), sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            analyzer.load_audio()
            
            # Mock the duration to simulate long audio
            analyzer.duration = 720.0  # 12 minutes
            
            # Should handle long audio gracefully
            beats = analyzer.detect_beats(min_interval=4.0)
            assert isinstance(beats, list)
            
            # Cut points should respect the simulated duration
            cut_points = analyzer.generate_cut_points()
            assert all(cp.timestamp <= analyzer.duration for cp in cut_points)
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_silent_audio(self):
        """Test with silent audio."""
        duration = 5.0
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio_data = np.zeros(num_samples, dtype=np.float32)  # Silent audio
        
        temp_file = create_temp_audio_file(audio_data, sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            analyzer.load_audio()
            
            # Should handle silent audio without crashing
            beats = analyzer.detect_beats()
            assert isinstance(beats, list)
            
            drums = analyzer.detect_drums()
            assert isinstance(drums, list)
            
            structure = analyzer.detect_structure()
            assert isinstance(structure, list)
            
            features = analyzer.analyze_features(0.0, 2.0)
            assert isinstance(features, dict)
            assert features['energy'] == 0.0  # Silent audio should have zero energy
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_single_beat_audio(self):
        """Test with audio that has only one clear beat."""
        duration = 3.0
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        
        # Create audio with a single strong beat at the beginning
        t = np.linspace(0, duration, num_samples)
        audio_data = np.zeros(num_samples)
        
        # Add a strong impulse at the beginning
        impulse_samples = int(0.1 * sample_rate)  # 0.1 second impulse
        audio_data[:impulse_samples] = np.sin(2 * np.pi * 440 * t[:impulse_samples])
        
        temp_file = create_temp_audio_file(audio_data.astype(np.float32), sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            analyzer.load_audio()
            
            # Should detect at least the beginning as a beat
            beats = analyzer.detect_beats(min_interval=1.0)
            assert isinstance(beats, list)
            assert len(beats) >= 1
            assert 0.0 in beats  # Should include the start
            
            drums = analyzer.detect_drums()
            assert isinstance(drums, list)
            
            cut_points = analyzer.generate_cut_points()
            assert isinstance(cut_points, list)
            assert len(cut_points) >= 1
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_analysis_without_loading(self):
        """Test that analysis methods fail gracefully when audio isn't loaded."""
        # Create a valid file path but don't load the audio
        duration = 2.0
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples))
        
        temp_file = create_temp_audio_file(audio_data.astype(np.float32), sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            # Don't call load_audio()
            
            # All analysis methods should raise AudioAnalysisError
            with pytest.raises(AudioAnalysisError, match="Audio must be loaded"):
                analyzer.detect_beats()
            
            with pytest.raises(AudioAnalysisError, match="Audio must be loaded"):
                analyzer.detect_drums()
            
            with pytest.raises(AudioAnalysisError, match="Audio must be loaded"):
                analyzer.detect_structure()
            
            with pytest.raises(AudioAnalysisError, match="Audio must be loaded"):
                analyzer.analyze_features(0.0, 1.0)
            
            with pytest.raises(AudioAnalysisError, match="Audio must be loaded"):
                analyzer.generate_cut_points()
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_invalid_feature_analysis_range(self):
        """Test feature analysis with invalid time ranges."""
        duration = 5.0
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples))
        
        temp_file = create_temp_audio_file(audio_data.astype(np.float32), sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            analyzer.load_audio()
            
            # Invalid range: start >= end
            with pytest.raises(AudioAnalysisError, match="Invalid time range"):
                analyzer.analyze_features(3.0, 3.0)
            
            # Invalid range: beyond audio duration
            with pytest.raises(AudioAnalysisError, match="Invalid time range"):
                analyzer.analyze_features(10.0, 15.0)
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_get_summary(self):
        """Test the get_summary method."""
        duration = 3.0
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples))
        
        temp_file = create_temp_audio_file(audio_data.astype(np.float32), sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            
            # Before loading
            summary = analyzer.get_summary()
            assert summary["status"] == "not_loaded"
            
            # After loading
            analyzer.load_audio()
            analyzer.detect_beats()  # This sets tempo and beat_times
            
            summary = analyzer.get_summary()
            assert summary["status"] == "loaded"
            assert summary["duration"] == analyzer.duration
            assert summary["sample_rate"] == analyzer.sr
            assert summary["tempo"] == analyzer.tempo
            assert summary["num_beats"] == len(analyzer.beat_times)
            assert summary["audio_path"] == str(analyzer.audio_path)
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_get_audio_features_object(self):
        """Test the get_audio_features_object method."""
        duration = 3.0
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples))
        
        temp_file = create_temp_audio_file(audio_data.astype(np.float32), sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            analyzer.load_audio()
            
            # Get AudioFeatures object
            features = analyzer.get_audio_features_object(0.0, 2.0)
            
            assert isinstance(features, AudioFeatures)
            assert hasattr(features, 'tempo')
            assert hasattr(features, 'energy')
            assert hasattr(features, 'spectral_centroid')
            assert hasattr(features, 'zero_crossing_rate')
            assert hasattr(features, 'mfcc')
            assert hasattr(features, 'chroma')
            
            # Test to_dict method
            feature_dict = features.to_dict()
            assert isinstance(feature_dict, dict)
            assert "tempo" in feature_dict
            assert "energy" in feature_dict
            
        finally:
            Path(temp_file).unlink(missing_ok=True)