"""Simplified tests for AudioAnalyzer component."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf

from vibeframe.audio_analyzer import AudioAnalyzer
from vibeframe.models import CutPoint, AudioFeatures
from vibeframe.exceptions import AudioLoadError, AudioAnalysisError

def create_test_audio(duration=3.0, sample_rate=22050, frequency=440.0):
    """Create simple test audio data."""
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    audio_data = np.sin(2 * np.pi * frequency * t)
    return audio_data.astype(np.float32), sample_rate

def create_temp_audio_file(audio_data, sample_rate):
    """Create a temporary audio file for testing."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio_data, sample_rate)
    temp_file.close()
    return temp_file.name

class TestAudioAnalyzer:
    """Test AudioAnalyzer functionality."""
    
    def test_audio_loading(self):
        """Test basic audio loading functionality."""
        audio_data, sample_rate = create_test_audio(duration=2.0)
        temp_file = create_temp_audio_file(audio_data, sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            y, sr = analyzer.load_audio()
            
            assert isinstance(y, np.ndarray)
            assert isinstance(sr, int)
            assert len(y) > 0
            assert sr == sample_rate
            assert abs(analyzer.duration - 2.0) < 0.1
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_beat_detection(self):
        """Test beat detection functionality."""
        audio_data, sample_rate = create_test_audio(duration=5.0)
        temp_file = create_temp_audio_file(audio_data, sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            analyzer.load_audio()
            
            beats = analyzer.detect_beats(min_interval=1.0)
            
            assert isinstance(beats, list)
            assert all(isinstance(beat, float) for beat in beats)
            assert all(0.0 <= beat <= analyzer.duration for beat in beats)
            assert beats == sorted(beats)  # Should be sorted
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_drum_detection(self):
        """Test drum detection functionality."""
        audio_data, sample_rate = create_test_audio(duration=3.0)
        temp_file = create_temp_audio_file(audio_data, sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            analyzer.load_audio()
            
            drums = analyzer.detect_drums()
            
            assert isinstance(drums, list)
            assert all(isinstance(drum, tuple) and len(drum) == 2 for drum in drums)
            assert all(isinstance(drum[0], float) and isinstance(drum[1], float) for drum in drums)
            assert all(0.0 <= drum[0] <= analyzer.duration for drum in drums)
            assert all(0.0 <= drum[1] <= 1.0 for drum in drums)  # Normalized strength
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_structure_detection(self):
        """Test musical structure detection."""
        audio_data, sample_rate = create_test_audio(duration=4.0)
        temp_file = create_temp_audio_file(audio_data, sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            analyzer.load_audio()
            
            structure = analyzer.detect_structure()
            
            assert isinstance(structure, list)
            assert len(structure) > 0
            
            for section in structure:
                assert isinstance(section, dict)
                assert 'start_time' in section
                assert 'end_time' in section
                assert 'duration' in section
                assert 'label' in section
                assert 'confidence' in section
                
                assert 0.0 <= section['start_time'] <= analyzer.duration
                assert 0.0 <= section['end_time'] <= analyzer.duration
                assert section['start_time'] < section['end_time']
                assert section['label'] in ["intro", "verse", "chorus", "bridge", "outro"]
                assert 0.0 <= section['confidence'] <= 1.0
                
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_feature_analysis(self):
        """Test audio feature extraction."""
        audio_data, sample_rate = create_test_audio(duration=3.0)
        temp_file = create_temp_audio_file(audio_data, sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            analyzer.load_audio()
            
            features = analyzer.analyze_features(0.0, 2.0)
            
            assert isinstance(features, dict)
            assert 'tempo' in features
            assert 'energy' in features
            assert 'spectral_centroid' in features
            assert 'zero_crossing_rate' in features
            assert 'mfcc' in features
            assert 'chroma' in features
            
            assert 40.0 <= features['tempo'] <= 300.0
            assert 0.0 <= features['energy'] <= 1.0
            assert features['spectral_centroid'] > 0
            assert 0.0 <= features['zero_crossing_rate'] <= 1.0
            assert len(features['mfcc']) == 13
            assert len(features['chroma']) == 12
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_cut_point_generation(self):
        """Test cut point generation."""
        audio_data, sample_rate = create_test_audio(duration=4.0)
        temp_file = create_temp_audio_file(audio_data, sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            analyzer.load_audio()
            
            cut_points = analyzer.generate_cut_points(strategy="auto")
            
            assert isinstance(cut_points, list)
            assert all(isinstance(cp, CutPoint) for cp in cut_points)
            assert all(0.0 <= cp.timestamp <= analyzer.duration for cp in cut_points)
            assert all(0.0 <= cp.confidence <= 1.0 for cp in cut_points)
            assert all(0.0 <= cp.beat_strength <= 1.0 for cp in cut_points)
            assert all(cp.section in ["intro", "verse", "chorus", "bridge", "outro"] for cp in cut_points)
            
            # Cut points should be sorted by timestamp
            timestamps = [cp.timestamp for cp in cut_points]
            assert timestamps == sorted(timestamps)
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_audio_features_object(self):
        """Test AudioFeatures object creation."""
        audio_data, sample_rate = create_test_audio(duration=3.0)
        temp_file = create_temp_audio_file(audio_data, sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            analyzer.load_audio()
            
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
    
    def test_get_summary(self):
        """Test the get_summary method."""
        audio_data, sample_rate = create_test_audio(duration=3.0)
        temp_file = create_temp_audio_file(audio_data, sample_rate)
        
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
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Non-existent file
        with pytest.raises(AudioLoadError, match="Audio file not found"):
            AudioAnalyzer("nonexistent.wav")
        
        # Analysis without loading
        audio_data, sample_rate = create_test_audio(duration=2.0)
        temp_file = create_temp_audio_file(audio_data, sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            # Don't call load_audio()
            
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
    
    def test_invalid_feature_range(self):
        """Test feature analysis with invalid time ranges."""
        audio_data, sample_rate = create_test_audio(duration=3.0)
        temp_file = create_temp_audio_file(audio_data, sample_rate)
        
        try:
            analyzer = AudioAnalyzer(temp_file)
            analyzer.load_audio()
            
            # Invalid range: start >= end
            with pytest.raises(AudioAnalysisError, match="Invalid time range"):
                analyzer.analyze_features(2.0, 2.0)
            
            # Invalid range: beyond audio duration
            with pytest.raises(AudioAnalysisError, match="Invalid time range"):
                analyzer.analyze_features(5.0, 10.0)
                
        finally:
            Path(temp_file).unlink(missing_ok=True)