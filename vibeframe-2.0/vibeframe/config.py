"""Configuration management for VibeFrame 2.0."""

import os
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Platform-specific presets
PLATFORM_PRESETS = {
    "youtube": {
        "resolution": (1920, 1080),
        "fps": 30,
        "aspect_ratio": "16:9",
        "codec": "libx264",
        "quality": "high",
        "bitrate": "8M"
    },
    "instagram": {
        "resolution": (1080, 1080),
        "fps": 30,
        "aspect_ratio": "1:1",
        "codec": "libx264",
        "quality": "standard",
        "bitrate": "5M"
    },
    "instagram_story": {
        "resolution": (1080, 1920),
        "fps": 30,
        "aspect_ratio": "9:16",
        "codec": "libx264",
        "quality": "standard",
        "bitrate": "5M"
    },
    "tiktok": {
        "resolution": (1080, 1920),
        "fps": 30,
        "aspect_ratio": "9:16",
        "codec": "libx264",
        "quality": "standard",
        "bitrate": "5M"
    },
    "twitter": {
        "resolution": (1280, 720),
        "fps": 30,
        "aspect_ratio": "16:9",
        "codec": "libx264",
        "quality": "standard",
        "bitrate": "5M"
    },
    "facebook": {
        "resolution": (1280, 720),
        "fps": 30,
        "aspect_ratio": "16:9",
        "codec": "libx264",
        "quality": "standard",
        "bitrate": "4M"
    }
}


# Resolution presets
RESOLUTION_PRESETS = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4K": (3840, 2160),
    "8K": (7680, 4320)
}


# FPS options
FPS_OPTIONS = [24, 25, 30, 50, 60]


# Aspect ratio presets
ASPECT_RATIOS = {
    "16:9": (16, 9),
    "9:16": (9, 16),
    "1:1": (1, 1),
    "4:3": (4, 3),
    "21:9": (21, 9)
}


# Codec options
CODEC_OPTIONS = {
    "h264": "libx264",
    "h265": "libx265",
    "vp9": "libvpx-vp9",
    "av1": "libaom-av1"
}


# Quality presets
QUALITY_PRESETS = {
    "draft": {"crf": 28, "preset": "ultrafast", "description": "Fast encoding, lower quality"},
    "standard": {"crf": 23, "preset": "medium", "description": "Balanced quality and speed"},
    "high": {"crf": 18, "preset": "slow", "description": "High quality, slower encoding"},
    "maximum": {"crf": 15, "preset": "veryslow", "description": "Maximum quality, very slow"}
}


@dataclass
class Config:
    """Main configuration class for VibeFrame 2.0."""
    
    # API Keys
    openrouter_api_key: Optional[str] = None
    huggingface_api_token: Optional[str] = None
    
    # Model paths
    longcat_model_path: str = "models/longcat-video"
    sharp_model_path: str = "models/sharp"
    hunyuan_model_path: str = "models/hunyuan-video"
    
    # Default settings
    default_resolution: Tuple[int, int] = (1920, 1080)
    default_fps: int = 24
    default_model: str = "longcat"
    default_codec: str = "libx264"
    default_quality: str = "standard"
    default_aspect_ratio: str = "16:9"
    
    # Processing settings
    max_video_duration: float = 300.0  # 5 minutes max
    min_scene_duration: float = 2.0    # 2 seconds min per scene
    max_scenes: int = 50               # Maximum scenes per video
    
    # Cache settings
    cache_dir: str = "cache"
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    # Rate limiting
    openrouter_rate_limit: int = 100   # requests per hour
    huggingface_rate_limit: int = 1000 # requests per hour
    
    # Platform presets
    platform_presets: Dict = field(default_factory=lambda: PLATFORM_PRESETS)
    resolution_presets: Dict = field(default_factory=lambda: RESOLUTION_PRESETS)
    quality_presets: Dict = field(default_factory=lambda: QUALITY_PRESETS)
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", self.openrouter_api_key)
        self.huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN", self.huggingface_api_token)
        
        # Override defaults from environment
        if os.getenv("DEFAULT_MODEL"):
            self.default_model = os.getenv("DEFAULT_MODEL")
        
        if os.getenv("CACHE_DIR"):
            self.cache_dir = os.getenv("CACHE_DIR")
            
        if os.getenv("ENABLE_CACHING"):
            self.enable_caching = os.getenv("ENABLE_CACHING").lower() == "true"
    
    def get_platform_preset(self, platform: str) -> Dict:
        """
        Get configuration preset for a specific platform.
        
        Args:
            platform: Platform name (youtube, instagram, tiktok, etc.)
            
        Returns:
            Dictionary with platform-specific settings
        """
        if platform.lower() not in self.platform_presets:
            raise ValueError(f"Unknown platform: {platform}. Available: {list(self.platform_presets.keys())}")
        
        return self.platform_presets[platform.lower()].copy()
    
    def get_resolution(self, preset: str) -> Tuple[int, int]:
        """
        Get resolution from preset name.
        
        Args:
            preset: Resolution preset (720p, 1080p, 4K, etc.)
            
        Returns:
            Tuple of (width, height)
        """
        if preset in self.resolution_presets:
            return self.resolution_presets[preset]
        
        # Try parsing as WxH
        if 'x' in preset:
            try:
                width, height = preset.split('x')
                return (int(width), int(height))
            except:
                pass
        
        raise ValueError(f"Unknown resolution preset: {preset}")
    
    def get_quality_settings(self, quality: str) -> Dict:
        """
        Get quality settings.
        
        Args:
            quality: Quality preset name
            
        Returns:
            Dictionary with quality settings
        """
        if quality not in self.quality_presets:
            raise ValueError(f"Unknown quality preset: {quality}. Available: {list(self.quality_presets.keys())}")
        
        return self.quality_presets[quality].copy()
    
    def calculate_aspect_ratio_resolution(
        self,
        aspect_ratio: str,
        base_height: int = 1080
    ) -> Tuple[int, int]:
        """
        Calculate resolution from aspect ratio.
        
        Args:
            aspect_ratio: Aspect ratio (16:9, 9:16, 1:1, etc.)
            base_height: Base height to calculate from
            
        Returns:
            Tuple of (width, height)
        """
        if aspect_ratio not in ASPECT_RATIOS:
            raise ValueError(f"Unknown aspect ratio: {aspect_ratio}")
        
        ratio_w, ratio_h = ASPECT_RATIOS[aspect_ratio]
        width = int(base_height * ratio_w / ratio_h)
        
        # Ensure even dimensions (required for most codecs)
        width = width + (width % 2)
        height = base_height + (base_height % 2)
        
        return (width, height)

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config

def update_config(**kwargs) -> None:
    """Update configuration values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")