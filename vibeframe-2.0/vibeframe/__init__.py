"""
VibeFrame 2.0: AI Music Video Generator

A comprehensive AI music video generation system that transforms audio files 
into visually consistent, professionally edited music videos.
"""

from .audio_analyzer import AudioAnalyzer
from .scene_planner import ScenePlanner
from .character_manager import CharacterManager
from .models import *
from .exceptions import *
from .config import Config
from .utils import setup_logging

__version__ = "2.0.0"
__author__ = "VibeFrame Team"
__description__ = "AI Music Video Generator with Character Consistency"

__all__ = [
    "AudioAnalyzer",
    "ScenePlanner", 
    "CharacterManager",
    "Config",
    "setup_logging"
]