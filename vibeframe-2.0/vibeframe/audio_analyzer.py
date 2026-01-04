"""Audio analysis component for VibeFrame 2.0."""

import librosa
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path

from .models import CutPoint, AudioFeatures
from .exceptions import AudioLoadError, AudioAnalysisError
from .utils import setup_logging

logger = setup_logging()

class AudioAnalyzer:
    """
    Analyzes audio files to detect beats, drums, musical structure, and optimal cut points.
    
    This component is responsible for the first stage of the music video generation pipeline,
    extracting musical information that will drive scene timing and transitions.
    """
    
    def __init__(self, audio_path: str):
        """
        Initialize AudioAnalyzer with path to audio file.
        
        Args:
            audio_path: Path to the audio file to analyze
            
        Raises:
            AudioLoadError: If the audio file cannot be found or is invalid
        """
        self.audio_path = Path(audio_path)
        if not self.audio_path.exists():
            raise AudioLoadError(f"Audio file not found: {audio_path}")
        
        self.y: Optional[np.ndarray] = None
        self.sr: Optional[int] = None
        self.duration: float = 0.0
        self.tempo: Optional[float] = None
        self.beat_frames: Optional[np.ndarray] = None
        self.beat_times: Optional[np.ndarray] = None
        
        logger.info(f"AudioAnalyzer initialized for: {self.audio_path}")
    
    def load_audio(self) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return waveform and sample rate.
        
        Returns:
            Tuple of (waveform, sample_rate)
            
        Raises:
            AudioLoadError: If the audio file cannot be loaded
        """
        try:
            logger.info(f"Loading audio file: {self.audio_path}")
            
            # Load audio with librosa (automatically handles various formats)
            self.y, self.sr = librosa.load(str(self.audio_path), sr=None)
            self.duration = librosa.get_duration(y=self.y, sr=self.sr)
            
            logger.info(f"Audio loaded successfully:")
            logger.info(f"  Duration: {self.duration:.2f} seconds")
            logger.info(f"  Sample rate: {self.sr} Hz")
            logger.info(f"  Samples: {len(self.y)}")
            
            return self.y, self.sr
            
        except Exception as e:
            error_msg = f"Failed to load audio file {self.audio_path}: {str(e)}"
            logger.error(error_msg)
            raise AudioLoadError(error_msg) from e
    
    def detect_beats(self, min_interval: float = 4.0) -> List[float]:
        """
        Detect beat timestamps with configurable minimum interval between cuts.
        
        Args:
            min_interval: Minimum time (in seconds) between cuts to avoid frenetic editing
            
        Returns:
            List of beat timestamps in seconds
            
        Raises:
            AudioAnalysisError: If beat detection fails
        """
        if self.y is None or self.sr is None:
            raise AudioAnalysisError("Audio must be loaded before beat detection")
        
        try:
            logger.info(f"Detecting beats with minimum interval: {min_interval}s")
            
            # Use librosa's beat tracking
            self.tempo, self.beat_frames = librosa.beat.beat_track(
                y=self.y, 
                sr=self.sr,
                hop_length=512,
                start_bpm=120.0,
                tightness=100
            )
            
            # Convert frame indices to time
            self.beat_times = librosa.frames_to_time(self.beat_frames, sr=self.sr)
            
            # Filter beats to respect minimum interval
            filtered_beats = [0.0]  # Always start at beginning
            last_beat = 0.0
            
            for beat_time in self.beat_times:
                if beat_time - last_beat >= min_interval:
                    filtered_beats.append(float(beat_time))
                    last_beat = beat_time
            
            # Ensure we don't exceed audio duration
            filtered_beats = [b for b in filtered_beats if b < self.duration]
            
            logger.info(f"Detected {len(self.beat_times)} total beats")
            logger.info(f"Filtered to {len(filtered_beats)} beats with min interval {min_interval}s")
            logger.info(f"Estimated tempo: {float(self.tempo):.1f} BPM")
            
            return filtered_beats
            
        except Exception as e:
            error_msg = f"Beat detection failed: {str(e)}"
            logger.error(error_msg)
            raise AudioAnalysisError(error_msg) from e
    
    def detect_drums(self) -> List[Tuple[float, float]]:
        """
        Detect drum hits and percussion events using onset strength detection.
        
        Returns:
            List of (timestamp, strength) pairs for drum hits
            
        Raises:
            AudioAnalysisError: If drum detection fails
        """
        if self.y is None or self.sr is None:
            raise AudioAnalysisError("Audio must be loaded before drum detection")
        
        try:
            logger.info("Detecting drum hits and percussion events")
            
            # Separate harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(self.y)
            
            # Detect onsets in the percussive component
            onset_frames = librosa.onset.onset_detect(
                y=y_percussive,
                sr=self.sr,
                hop_length=512,
                backtrack=True,
                units='frames'
            )
            
            # Convert to time and get onset strengths
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)
            onset_strengths = librosa.onset.onset_strength(
                y=y_percussive,
                sr=self.sr,
                hop_length=512
            )
            
            # Get strength values at onset times
            drum_hits = []
            for onset_frame, onset_time in zip(onset_frames, onset_times):
                if onset_frame < len(onset_strengths):
                    strength = float(onset_strengths[onset_frame])
                    drum_hits.append((float(onset_time), strength))
            
            # Normalize strengths to 0-1 range
            if drum_hits:
                max_strength = max(hit[1] for hit in drum_hits)
                if max_strength > 0:
                    drum_hits = [(time, strength / max_strength) for time, strength in drum_hits]
            
            logger.info(f"Detected {len(drum_hits)} drum hits")
            
            return drum_hits
            
        except Exception as e:
            error_msg = f"Drum detection failed: {str(e)}"
            logger.error(error_msg)
            raise AudioAnalysisError(error_msg) from e
    
    def detect_structure(self) -> List[Dict[str, Any]]:
        """
        Detect musical sections (intro, verse, chorus, bridge, outro) using structural segmentation.
        
        Returns:
            List of dictionaries with section information
            
        Raises:
            AudioAnalysisError: If structure detection fails
        """
        if self.y is None or self.sr is None:
            raise AudioAnalysisError("Audio must be loaded before structure detection")
        
        try:
            logger.info("Detecting musical structure")
            
            # Extract chroma features for structural analysis
            chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr, hop_length=512)
            
            # Use recurrence matrix for segmentation
            R = librosa.segment.recurrence_matrix(
                chroma,
                mode='affinity',
                metric='cosine'
            )
            
            # Detect segment boundaries - use a fixed number of segments
            # For short audio, use fewer segments
            num_segments = min(5, max(2, int(self.duration / 30)))  # 1 segment per 30 seconds
            boundaries = librosa.segment.agglomerative(R, k=num_segments)
            boundary_times = librosa.frames_to_time(boundaries, sr=self.sr)
            
            # Create sections with estimated labels
            sections = []
            section_labels = self._estimate_section_labels(boundary_times)
            
            for i, (start_time, label) in enumerate(zip(boundary_times, section_labels)):
                end_time = boundary_times[i + 1] if i + 1 < len(boundary_times) else self.duration
                
                sections.append({
                    'start_time': float(start_time),
                    'end_time': float(end_time),
                    'duration': float(end_time - start_time),
                    'label': label,
                    'confidence': 0.8  # Placeholder confidence score
                })
            
            logger.info(f"Detected {len(sections)} musical sections:")
            for section in sections:
                logger.info(f"  {section['label']}: {section['start_time']:.1f}s - {section['end_time']:.1f}s")
            
            return sections
            
        except Exception as e:
            error_msg = f"Structure detection failed: {str(e)}"
            logger.error(error_msg)
            raise AudioAnalysisError(error_msg) from e
    
    def _estimate_section_labels(self, boundary_times: np.ndarray) -> List[str]:
        """
        Estimate section labels based on position and duration.
        
        This is a heuristic approach - in practice, more sophisticated
        machine learning models would be used for accurate labeling.
        
        Args:
            boundary_times: Array of section boundary times
            
        Returns:
            List of section labels
        """
        num_sections = len(boundary_times)
        labels = []
        
        if num_sections == 0:
            return []
        
        # Simple heuristic based on song structure patterns
        if num_sections == 1:
            labels = ["verse"]
        elif num_sections == 2:
            labels = ["verse", "chorus"]
        elif num_sections == 3:
            labels = ["intro", "verse", "chorus"]
        elif num_sections == 4:
            labels = ["intro", "verse", "chorus", "outro"]
        elif num_sections == 5:
            labels = ["intro", "verse", "chorus", "verse", "outro"]
        else:
            # For longer songs, use a pattern
            labels = ["intro"]
            remaining = num_sections - 2  # Subtract intro and outro
            
            # Alternate between verse and chorus
            for i in range(remaining):
                if i % 2 == 0:
                    labels.append("verse")
                else:
                    labels.append("chorus")
            
            labels.append("outro")
        
        return labels[:num_sections]
    
    def analyze_features(self, start_time: float, end_time: float) -> Dict[str, float]:
        """
        Extract audio features (tempo, energy, mood) for a time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Dictionary of audio features
            
        Raises:
            AudioAnalysisError: If feature extraction fails
        """
        if self.y is None or self.sr is None:
            raise AudioAnalysisError("Audio must be loaded before feature analysis")
        
        try:
            # Convert time to sample indices
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            
            # Extract segment
            y_segment = self.y[start_sample:end_sample]
            
            if len(y_segment) == 0:
                raise AudioAnalysisError(f"Invalid time range: {start_time}-{end_time}")
            
            # Extract features
            features = {}
            
            # Tempo (use global tempo if available, otherwise compute local)
            if self.tempo is not None:
                features['tempo'] = float(self.tempo)
            else:
                try:
                    tempo, _ = librosa.beat.beat_track(y=y_segment, sr=self.sr)
                    features['tempo'] = float(tempo) if tempo > 0 else 120.0  # Default tempo
                except:
                    features['tempo'] = 120.0  # Fallback tempo
            
            # RMS Energy
            rms = librosa.feature.rms(y=y_segment, hop_length=512)
            features['energy'] = float(np.mean(rms))
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=y_segment, sr=self.sr)
            features['spectral_centroid'] = float(np.mean(spectral_centroid))
            
            # Zero crossing rate (roughness)
            zcr = librosa.feature.zero_crossing_rate(y_segment, hop_length=512)
            features['zero_crossing_rate'] = float(np.mean(zcr))
            
            # MFCC (timbre)
            mfcc = librosa.feature.mfcc(y=y_segment, sr=self.sr, n_mfcc=13)
            features['mfcc'] = [float(np.mean(mfcc[i])) for i in range(13)]
            
            # Chroma (harmony)
            chroma = librosa.feature.chroma_stft(y=y_segment, sr=self.sr)
            features['chroma'] = [float(np.mean(chroma[i])) for i in range(12)]
            
            return features
            
        except Exception as e:
            error_msg = f"Feature analysis failed for range {start_time}-{end_time}: {str(e)}"
            logger.error(error_msg)
            raise AudioAnalysisError(error_msg) from e
    
    def generate_cut_points(self, strategy: str = "auto") -> List[CutPoint]:
        """
        Generate optimal cut points using specified strategy.
        
        Args:
            strategy: Cut point generation strategy ("auto", "beats", "structure")
            
        Returns:
            List of CutPoint objects with timestamps and metadata
            
        Raises:
            AudioAnalysisError: If cut point generation fails
        """
        if self.y is None or self.sr is None:
            raise AudioAnalysisError("Audio must be loaded before generating cut points")
        
        try:
            logger.info(f"Generating cut points using strategy: {strategy}")
            
            cut_points = []
            
            if strategy == "auto" or strategy == "beats":
                # Use beat-based cut points
                beat_times = self.detect_beats()
                drum_hits = self.detect_drums()
                sections = self.detect_structure()
                
                # Create cut points from beats
                for i, beat_time in enumerate(beat_times):
                    # Find corresponding section
                    section_label = "verse"  # Default
                    for section in sections:
                        if section['start_time'] <= beat_time < section['end_time']:
                            section_label = section['label']
                            break
                    
                    # Calculate confidence based on nearby drum hits
                    confidence = 0.7  # Base confidence
                    for drum_time, drum_strength in drum_hits:
                        if abs(drum_time - beat_time) < 0.5:  # Within 0.5 seconds
                            confidence = min(1.0, confidence + drum_strength * 0.3)
                            break
                    
                    # Calculate beat strength (normalized)
                    beat_strength = 0.8  # Default strength
                    if self.beat_times is not None and i < len(self.beat_times):
                        # Use onset strength at this beat
                        onset_strength = librosa.onset.onset_strength(
                            y=self.y, sr=self.sr, hop_length=512
                        )
                        beat_frame = librosa.time_to_frames(beat_time, sr=self.sr)
                        if beat_frame < len(onset_strength):
                            beat_strength = float(onset_strength[beat_frame])
                    
                    # Normalize beat strength
                    beat_strength = min(1.0, beat_strength / np.max(
                        librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=512)
                    ))
                    
                    cut_points.append(CutPoint(
                        timestamp=beat_time,
                        confidence=confidence,
                        beat_strength=beat_strength,
                        section=section_label
                    ))
            
            elif strategy == "structure":
                # Use structure-based cut points
                sections = self.detect_structure()
                
                for section in sections:
                    cut_points.append(CutPoint(
                        timestamp=section['start_time'],
                        confidence=section['confidence'],
                        beat_strength=0.9,  # Structure boundaries are strong
                        section=section['label']
                    ))
            
            else:
                raise AudioAnalysisError(f"Unknown cut point strategy: {strategy}")
            
            # Sort by timestamp
            cut_points.sort(key=lambda cp: cp.timestamp)
            
            logger.info(f"Generated {len(cut_points)} cut points")
            
            return cut_points
            
        except Exception as e:
            error_msg = f"Cut point generation failed: {str(e)}"
            logger.error(error_msg)
            raise AudioAnalysisError(error_msg) from e
    
    def get_audio_features_object(self, start_time: float, end_time: float) -> AudioFeatures:
        """
        Get AudioFeatures object for a time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            AudioFeatures object
        """
        features = self.analyze_features(start_time, end_time)
        
        return AudioFeatures(
            tempo=features['tempo'],
            energy=features['energy'],
            spectral_centroid=features['spectral_centroid'],
            zero_crossing_rate=features['zero_crossing_rate'],
            mfcc=features['mfcc'],
            chroma=features['chroma']
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the audio analysis results.
        
        Returns:
            Dictionary with analysis summary
        """
        if self.y is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "duration": self.duration,
            "sample_rate": self.sr,
            "tempo": self.tempo,
            "num_beats": len(self.beat_times) if self.beat_times is not None else 0,
            "audio_path": str(self.audio_path)
        }