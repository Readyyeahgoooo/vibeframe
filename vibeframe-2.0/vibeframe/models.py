"""Data models for VibeFrame 2.0."""

import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum

class VideoModel(Enum):
    """Supported video generation models."""
    LONGCAT = "longcat"
    HUNYUAN = "hunyuan"
    SHARP = "sharp"

class ProcessingStep(Enum):
    """Processing steps in the pipeline."""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    GENERATION = "generation"
    ASSEMBLY = "assembly"
    COMPLETE = "complete"

@dataclass
class CutPoint:
    """Represents a cut point in the audio for scene transitions."""
    timestamp: float
    confidence: float
    beat_strength: float
    section: str  # intro, verse, chorus, bridge, outro
    
    def __post_init__(self):
        """Validate cut point data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if not 0 <= self.beat_strength <= 1:
            raise ValueError("Beat strength must be between 0 and 1")
        if self.section not in ["intro", "verse", "chorus", "bridge", "outro"]:
            raise ValueError(f"Invalid section: {self.section}")

@dataclass
class CharacterDescriptor:
    """Describes a character for consistency across scenes."""
    appearance: str  # "young woman with long black hair, blue eyes"
    clothing: str    # "wearing a red leather jacket and jeans"
    style: str       # "anime style, detailed, vibrant colors"
    age: Optional[str] = None
    ethnicity: Optional[str] = None
    distinctive_features: Optional[str] = None
    
    def to_prompt_string(self) -> str:
        """Convert character descriptor to a prompt string."""
        parts = [self.appearance, self.clothing, self.style]
        if self.age:
            parts.append(f"age: {self.age}")
        if self.ethnicity:
            parts.append(f"ethnicity: {self.ethnicity}")
        if self.distinctive_features:
            parts.append(self.distinctive_features)
        return ", ".join(parts)

@dataclass
class CameraKeyframe:
    """Camera position and orientation at a specific time."""
    timestamp: float
    position: Tuple[float, float, float]  # x, y, z
    rotation: Tuple[float, float, float]  # pitch, yaw, roll
    fov: float  # field of view in degrees
    
    def __post_init__(self):
        """Validate camera keyframe data."""
        if not 10 <= self.fov <= 120:
            raise ValueError("FOV must be between 10 and 120 degrees")

@dataclass
class CameraPath:
    """Camera path for SHARP mode 3D animation."""
    keyframes: List[CameraKeyframe]
    interpolation: str = "linear"  # "linear", "bezier", "catmull-rom"
    
    def __post_init__(self):
        """Validate camera path data."""
        if len(self.keyframes) < 2:
            raise ValueError("Camera path must have at least 2 keyframes")
        if self.interpolation not in ["linear", "bezier", "catmull-rom"]:
            raise ValueError(f"Invalid interpolation: {self.interpolation}")
    
    def interpolate(self, timestamp: float) -> CameraKeyframe:
        """Interpolate camera position at given timestamp."""
        # Find surrounding keyframes
        if timestamp <= self.keyframes[0].timestamp:
            return self.keyframes[0]
        if timestamp >= self.keyframes[-1].timestamp:
            return self.keyframes[-1]
        
        # Find the two keyframes to interpolate between
        for i in range(len(self.keyframes) - 1):
            if self.keyframes[i].timestamp <= timestamp <= self.keyframes[i + 1].timestamp:
                kf1, kf2 = self.keyframes[i], self.keyframes[i + 1]
                
                # Linear interpolation
                t = (timestamp - kf1.timestamp) / (kf2.timestamp - kf1.timestamp)
                
                pos = tuple(
                    kf1.position[j] + t * (kf2.position[j] - kf1.position[j])
                    for j in range(3)
                )
                rot = tuple(
                    kf1.rotation[j] + t * (kf2.rotation[j] - kf1.rotation[j])
                    for j in range(3)
                )
                fov = kf1.fov + t * (kf2.fov - kf1.fov)
                
                return CameraKeyframe(timestamp, pos, rot, fov)
        
        raise ValueError(f"Could not interpolate timestamp {timestamp}")

@dataclass
class SceneDescription:
    """Describes a single scene in the music video."""
    id: int
    start_time: float
    end_time: float
    duration: float
    description: str
    character_action: str
    camera_angle: str
    lighting: str
    environment: str
    video_prompt: str
    character_descriptor: Optional[CharacterDescriptor] = None
    reference_image: Optional[str] = None
    generated_video_path: Optional[str] = None
    camera_path: Optional[CameraPath] = None
    
    def __post_init__(self):
        """Validate scene description data."""
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
        if abs(self.duration - (self.end_time - self.start_time)) > 0.1:
            raise ValueError("Duration must match end_time - start_time")

@dataclass
class Storyboard:
    """Complete storyboard for a music video project."""
    project_name: str
    audio_file: str
    audio_duration: float
    global_style: str
    theme: Optional[str]
    scenes: List[SceneDescription]
    fps: int = 24
    resolution: Tuple[int, int] = (1920, 1080)
    model: str = "longcat"
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate storyboard data and set defaults."""
        if not self.scenes:
            raise ValueError("Storyboard must have at least one scene")
        if self.fps not in [24, 30, 60]:
            raise ValueError("FPS must be 24, 30, or 60")
        if self.model not in ["longcat", "hunyuan", "sharp"]:
            raise ValueError("Model must be 'longcat', 'hunyuan', or 'sharp'")
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_json(self) -> str:
        """Serialize storyboard to JSON string."""
        data = asdict(self)
        # Convert datetime to ISO string
        if data['created_at']:
            data['created_at'] = self.created_at.isoformat()
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Storyboard':
        """Deserialize storyboard from JSON string."""
        data = json.loads(json_str)
        
        # Convert ISO string back to datetime
        if data.get('created_at'):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        # Convert resolution list back to tuple
        if isinstance(data.get('resolution'), list):
            data['resolution'] = tuple(data['resolution'])
        
        # Reconstruct nested objects
        scenes = []
        for scene_data in data['scenes']:
            # Reconstruct character descriptor
            if scene_data.get('character_descriptor'):
                scene_data['character_descriptor'] = CharacterDescriptor(
                    **scene_data['character_descriptor']
                )
            
            # Reconstruct camera path
            if scene_data.get('camera_path'):
                keyframes = []
                for kf_data in scene_data['camera_path']['keyframes']:
                    # Convert position and rotation lists back to tuples
                    if isinstance(kf_data.get('position'), list):
                        kf_data['position'] = tuple(kf_data['position'])
                    if isinstance(kf_data.get('rotation'), list):
                        kf_data['rotation'] = tuple(kf_data['rotation'])
                    keyframes.append(CameraKeyframe(**kf_data))
                
                scene_data['camera_path'] = CameraPath(
                    keyframes=keyframes,
                    interpolation=scene_data['camera_path']['interpolation']
                )
            
            scenes.append(SceneDescription(**scene_data))
        
        data['scenes'] = scenes
        return cls(**data)
    
    def get_total_duration(self) -> float:
        """Get total duration of all scenes."""
        return sum(scene.duration for scene in self.scenes)
    
    def get_scene_by_id(self, scene_id: int) -> Optional[SceneDescription]:
        """Get scene by ID."""
        for scene in self.scenes:
            if scene.id == scene_id:
                return scene
        return None

@dataclass
class ProjectStatus:
    """Status information for a music video project."""
    project_name: str
    audio_file: str
    created_at: datetime
    last_modified: datetime
    total_scenes: int
    scenes_generated: int
    current_step: str  # ProcessingStep enum value
    progress_percent: float
    estimated_time_remaining: Optional[float] = None
    errors: List[str] = None
    
    def __post_init__(self):
        """Set defaults and validate."""
        if self.errors is None:
            self.errors = []
        if not 0 <= self.progress_percent <= 100:
            raise ValueError("Progress percent must be between 0 and 100")
        if self.current_step not in [step.value for step in ProcessingStep]:
            raise ValueError(f"Invalid processing step: {self.current_step}")
    
    def add_error(self, error: str) -> None:
        """Add an error to the status."""
        self.errors.append(error)
        self.last_modified = datetime.now()
    
    def update_progress(self, step: ProcessingStep, progress: float) -> None:
        """Update the current processing step and progress."""
        self.current_step = step.value
        self.progress_percent = progress
        self.last_modified = datetime.now()

@dataclass
class AudioFeatures:
    """Audio features extracted from a time segment."""
    tempo: float
    energy: float  # 0-1 scale
    spectral_centroid: float
    zero_crossing_rate: float
    mfcc: List[float]  # Mel-frequency cepstral coefficients
    chroma: List[float]  # Chromagram
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "tempo": self.tempo,
            "energy": self.energy,
            "spectral_centroid": self.spectral_centroid,
            "zero_crossing_rate": self.zero_crossing_rate,
            "mfcc_mean": sum(self.mfcc) / len(self.mfcc) if self.mfcc else 0,
            "chroma_mean": sum(self.chroma) / len(self.chroma) if self.chroma else 0
        }

# Validation functions
def validate_audio_file(file_path: str) -> bool:
    """Validate that a file is a supported audio format."""
    supported_formats = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    return any(file_path.lower().endswith(fmt) for fmt in supported_formats)

def validate_image_file(file_path: str) -> bool:
    """Validate that a file is a supported image format."""
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return any(file_path.lower().endswith(fmt) for fmt in supported_formats)

def validate_video_file(file_path: str) -> bool:
    """Validate that a file is a supported video format."""
    supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    return any(file_path.lower().endswith(fmt) for fmt in supported_formats)