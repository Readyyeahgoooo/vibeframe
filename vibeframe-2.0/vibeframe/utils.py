"""Utility functions for VibeFrame 2.0."""

import os
import hashlib
import json
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging for VibeFrame."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('vibeframe.log')
        ]
    )
    return logging.getLogger('vibeframe')

def ensure_directory(path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_file_hash(file_path: str) -> str:
    """Get SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def get_cache_key(*args) -> str:
    """Generate a cache key from arguments."""
    key_string = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(key_string.encode()).hexdigest()

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"

def format_file_size(bytes_size: int) -> str:
    """Format file size in bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"

def validate_json_structure(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """Validate that a JSON object has all required fields."""
    for field in required_fields:
        if field not in data:
            return False
    return True

def safe_filename(filename: str) -> str:
    """Convert a string to a safe filename."""
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename

def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_cached": torch.cuda.memory_reserved()
            }
    except ImportError:
        pass
    
    return {"available": False}

def estimate_gpu_memory_usage(resolution: tuple, fps: int, duration: float, model: str) -> float:
    """Estimate GPU memory usage in GB for video generation."""
    width, height = resolution
    pixels = width * height
    frames = fps * duration
    
    # Base memory usage estimates (in GB)
    base_usage = {
        "longcat": 8.0,
        "hunyuan": 12.0,
        "sharp": 4.0
    }
    
    # Scale by resolution and duration
    resolution_factor = pixels / (1920 * 1080)  # Relative to 1080p
    duration_factor = min(duration / 10.0, 2.0)  # Cap at 2x for long videos
    
    estimated_usage = base_usage.get(model, 8.0) * resolution_factor * duration_factor
    return estimated_usage

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def interpolate_value(start: float, end: float, t: float) -> float:
    """Linear interpolation between two values."""
    return start + t * (end - start)

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(value, max_val))

def normalize_audio_features(features: Dict[str, float]) -> Dict[str, float]:
    """Normalize audio features to 0-1 range."""
    normalized = {}
    
    # Tempo normalization (assume 60-180 BPM range)
    if 'tempo' in features:
        normalized['tempo'] = clamp((features['tempo'] - 60) / 120, 0, 1)
    
    # Energy is already 0-1
    if 'energy' in features:
        normalized['energy'] = clamp(features['energy'], 0, 1)
    
    # Spectral centroid normalization (assume 0-8000 Hz range)
    if 'spectral_centroid' in features:
        normalized['spectral_centroid'] = clamp(features['spectral_centroid'] / 8000, 0, 1)
    
    # Zero crossing rate is already normalized
    if 'zero_crossing_rate' in features:
        normalized['zero_crossing_rate'] = clamp(features['zero_crossing_rate'], 0, 1)
    
    return normalized