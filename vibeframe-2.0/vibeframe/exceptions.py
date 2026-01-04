"""Custom exceptions for VibeFrame 2.0."""

class VibeFrameError(Exception):
    """Base exception for all VibeFrame errors."""
    pass

class AudioLoadError(VibeFrameError):
    """Raised when audio file cannot be loaded."""
    pass

class AudioAnalysisError(VibeFrameError):
    """Raised when audio analysis fails."""
    pass

class APIError(VibeFrameError):
    """Base class for API-related errors."""
    def __init__(self, message: str, service: str = None, status_code: int = None):
        super().__init__(message)
        self.service = service
        self.status_code = status_code

class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    def __init__(self, message: str, service: str, limit: int, period: str, retry_after: int = None):
        super().__init__(message, service)
        self.limit = limit
        self.period = period
        self.retry_after = retry_after

class APIAuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass

class ModelError(VibeFrameError):
    """Base class for model-related errors."""
    pass

class ModelNotFoundError(ModelError):
    """Raised when model weights are not found."""
    pass

class InsufficientGPUMemoryError(ModelError):
    """Raised when there's insufficient GPU memory."""
    def __init__(self, message: str, required_gb: float, available_gb: float):
        super().__init__(message)
        self.required_gb = required_gb
        self.available_gb = available_gb

class VideoGenerationError(VibeFrameError):
    """Raised when video generation fails."""
    pass

class VideoProcessingError(VibeFrameError):
    """Raised when video processing fails."""
    pass

class ScenePlanningError(VibeFrameError):
    """Raised when scene planning fails."""
    pass

class CharacterError(VibeFrameError):
    """Raised when character management operations fail."""
    pass

class StoryboardValidationError(VibeFrameError):
    """Raised when storyboard validation fails."""
    pass

class ProjectError(VibeFrameError):
    """Raised when project operations fail."""
    pass

class WorkflowError(VibeFrameError):
    """Raised when workflow orchestration fails."""
    pass

class ConfigurationError(VibeFrameError):
    """Raised when configuration is invalid."""
    pass