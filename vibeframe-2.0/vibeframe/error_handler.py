"""Error handling and user feedback for VibeFrame 2.0."""

import logging
import traceback
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from .exceptions import (
    AudioAnalysisError,
    ScenePlanningError,
    VideoGenerationError,
    VideoProcessingError,
    WorkflowError,
    CharacterConsistencyError
)
from .utils import setup_logging

logger = setup_logging()


class ErrorCategory(Enum):
    """Categories of errors for better user feedback."""
    INPUT = "input"
    API = "api"
    MODEL = "model"
    FILESYSTEM = "filesystem"
    PROCESSING = "processing"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ErrorHandler:
    """
    Centralized error handling with user-friendly messages.
    
    Provides context-aware error messages, categorization, and
    suggestions for resolution.
    """
    
    def __init__(self):
        """Initialize ErrorHandler."""
        self.error_history = []
        self.retry_counts = {}
        logger.info("ErrorHandler initialized")
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, ErrorCategory]:
        """
        Handle an error and generate user-friendly message.
        
        Args:
            error: The exception that occurred
            context: Optional context information
            
        Returns:
            Tuple of (user_message, technical_message, category)
        """
        context = context or {}
        
        # Categorize error
        category = self._categorize_error(error)
        
        # Generate messages
        user_message = self._generate_user_message(error, category, context)
        technical_message = self._generate_technical_message(error, context)
        
        # Log error
        self._log_error(error, category, context)
        
        # Store in history
        self.error_history.append({
            "error": str(error),
            "category": category.value,
            "context": context,
            "user_message": user_message
        })
        
        return user_message, technical_message, category
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on its type."""
        if isinstance(error, AudioAnalysisError):
            return ErrorCategory.PROCESSING
        elif isinstance(error, ScenePlanningError):
            return ErrorCategory.API
        elif isinstance(error, VideoGenerationError):
            return ErrorCategory.MODEL
        elif isinstance(error, VideoProcessingError):
            return ErrorCategory.PROCESSING
        elif isinstance(error, WorkflowError):
            return ErrorCategory.PROCESSING
        elif isinstance(error, CharacterConsistencyError):
            return ErrorCategory.PROCESSING
        elif isinstance(error, FileNotFoundError):
            return ErrorCategory.FILESYSTEM
        elif isinstance(error, PermissionError):
            return ErrorCategory.FILESYSTEM
        elif isinstance(error, ConnectionError):
            return ErrorCategory.NETWORK
        elif isinstance(error, TimeoutError):
            return ErrorCategory.NETWORK
        elif isinstance(error, ValueError):
            return ErrorCategory.INPUT
        elif isinstance(error, KeyError):
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.UNKNOWN
    
    def _generate_user_message(
        self,
        error: Exception,
        category: ErrorCategory,
        context: Dict[str, Any]
    ) -> str:
        """Generate user-friendly error message with suggestions."""
        error_str = str(error).lower()
        
        # Category-specific messages
        if category == ErrorCategory.INPUT:
            if "audio" in error_str or "file" in error_str:
                return (
                    "❌ Audio file issue detected.\n\n"
                    "💡 Suggestions:\n"
                    "• Ensure the file is a valid audio format (MP3, WAV, FLAC)\n"
                    "• Check that the file is not corrupted\n"
                    "• Try converting to WAV format\n"
                    "• Ensure the file is at least 5 seconds long"
                )
            else:
                return (
                    "❌ Invalid input provided.\n\n"
                    "💡 Please check your input and try again."
                )
        
        elif category == ErrorCategory.API:
            if "rate limit" in error_str or "429" in error_str:
                return (
                    "❌ API rate limit reached.\n\n"
                    "💡 Suggestions:\n"
                    "• Wait a few minutes before trying again\n"
                    "• Consider upgrading your API plan\n"
                    "• The system will use template-based fallback for now"
                )
            elif "api key" in error_str or "401" in error_str or "403" in error_str:
                return (
                    "❌ API authentication failed.\n\n"
                    "💡 Suggestions:\n"
                    "• Check that your API key is correct\n"
                    "• Ensure the API key has proper permissions\n"
                    "• The system will use template-based fallback"
                )
            elif "timeout" in error_str:
                return (
                    "❌ API request timed out.\n\n"
                    "💡 Suggestions:\n"
                    "• Check your internet connection\n"
                    "• Try again in a moment\n"
                    "• The system will retry automatically"
                )
            else:
                return (
                    "❌ API service error.\n\n"
                    "💡 The system will use template-based fallback for scene generation."
                )
        
        elif category == ErrorCategory.MODEL:
            if "memory" in error_str or "cuda" in error_str or "gpu" in error_str:
                return (
                    "❌ GPU/Memory issue detected.\n\n"
                    "💡 Suggestions:\n"
                    "• Try reducing video resolution\n"
                    "• Close other GPU-intensive applications\n"
                    "• Consider using CPU mode (slower but more stable)\n"
                    "• The system will try fallback models"
                )
            elif "model" in error_str and "not found" in error_str:
                return (
                    "❌ AI model not available.\n\n"
                    "💡 Suggestions:\n"
                    "• Check your internet connection for model download\n"
                    "• Ensure you have enough disk space\n"
                    "• The system will try alternative models"
                )
            else:
                return (
                    "❌ Video generation model error.\n\n"
                    "💡 The system will try alternative generation methods."
                )
        
        elif category == ErrorCategory.FILESYSTEM:
            if "permission" in error_str:
                return (
                    "❌ File permission error.\n\n"
                    "💡 Suggestions:\n"
                    "• Check file/folder permissions\n"
                    "• Ensure you have write access to the output directory\n"
                    "• Try running with appropriate permissions"
                )
            elif "not found" in error_str:
                return (
                    "❌ File not found.\n\n"
                    "💡 Suggestions:\n"
                    "• Verify the file path is correct\n"
                    "• Ensure the file hasn't been moved or deleted\n"
                    "• Check that previous steps completed successfully"
                )
            elif "disk" in error_str or "space" in error_str:
                return (
                    "❌ Insufficient disk space.\n\n"
                    "💡 Suggestions:\n"
                    "• Free up disk space\n"
                    "• Use a different output directory\n"
                    "• Reduce video quality/resolution"
                )
            else:
                return (
                    "❌ File system error.\n\n"
                    "💡 Please check file paths and permissions."
                )
        
        elif category == ErrorCategory.PROCESSING:
            if "audio" in error_str:
                return (
                    "❌ Audio processing failed.\n\n"
                    "💡 Suggestions:\n"
                    "• Ensure audio file is valid and not corrupted\n"
                    "• Try converting to a standard format (WAV, 44.1kHz)\n"
                    "• Check that the audio is at least 5 seconds long"
                )
            elif "video" in error_str:
                return (
                    "❌ Video processing failed.\n\n"
                    "💡 Suggestions:\n"
                    "• Check that FFmpeg is installed correctly\n"
                    "• Ensure sufficient disk space\n"
                    "• Try reducing video quality/resolution"
                )
            else:
                return (
                    "❌ Processing error occurred.\n\n"
                    "💡 Please try again or contact support if the issue persists."
                )
        
        elif category == ErrorCategory.NETWORK:
            return (
                "❌ Network connection error.\n\n"
                "💡 Suggestions:\n"
                "• Check your internet connection\n"
                "• Verify firewall settings\n"
                "• Try again in a moment\n"
                "• The system will retry automatically"
            )
        
        elif category == ErrorCategory.CONFIGURATION:
            return (
                "❌ Configuration error.\n\n"
                "💡 Suggestions:\n"
                "• Check your .env file settings\n"
                "• Verify API keys are set correctly\n"
                "• Review configuration documentation"
            )
        
        else:
            return (
                "❌ An unexpected error occurred.\n\n"
                "💡 Please try again or contact support with the error details."
            )
    
    def _generate_technical_message(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> str:
        """Generate technical error message for logging."""
        error_type = type(error).__name__
        error_msg = str(error)
        
        technical_msg = f"Error Type: {error_type}\n"
        technical_msg += f"Error Message: {error_msg}\n"
        
        if context:
            technical_msg += f"Context: {context}\n"
        
        # Add traceback
        tb = traceback.format_exc()
        technical_msg += f"\nTraceback:\n{tb}"
        
        return technical_msg
    
    def _log_error(
        self,
        error: Exception,
        category: ErrorCategory,
        context: Dict[str, Any]
    ) -> None:
        """Log error with appropriate level."""
        error_msg = f"[{category.value.upper()}] {type(error).__name__}: {str(error)}"
        
        if context:
            error_msg += f" | Context: {context}"
        
        if category in [ErrorCategory.NETWORK, ErrorCategory.API]:
            # Transient errors - warning level
            logger.warning(error_msg)
        else:
            # Serious errors - error level
            logger.error(error_msg)
    
    def should_retry(
        self,
        error: Exception,
        operation_id: str,
        max_retries: int = 3
    ) -> bool:
        """
        Determine if an operation should be retried.
        
        Args:
            error: The exception that occurred
            operation_id: Unique identifier for the operation
            max_retries: Maximum number of retries
            
        Returns:
            True if should retry, False otherwise
        """
        # Get current retry count
        retry_count = self.retry_counts.get(operation_id, 0)
        
        # Check if max retries reached
        if retry_count >= max_retries:
            logger.info(f"Max retries ({max_retries}) reached for {operation_id}")
            return False
        
        # Determine if error is retryable
        category = self._categorize_error(error)
        
        retryable_categories = [
            ErrorCategory.NETWORK,
            ErrorCategory.API,
        ]
        
        if category in retryable_categories:
            # Increment retry count
            self.retry_counts[operation_id] = retry_count + 1
            logger.info(f"Retry {retry_count + 1}/{max_retries} for {operation_id}")
            return True
        
        return False
    
    def reset_retry_count(self, operation_id: str) -> None:
        """Reset retry count for an operation."""
        if operation_id in self.retry_counts:
            del self.retry_counts[operation_id]
    
    def get_error_history(self, limit: int = 10) -> list:
        """Get recent error history."""
        return self.error_history[-limit:]
    
    def clear_error_history(self) -> None:
        """Clear error history."""
        self.error_history.clear()
        logger.info("Error history cleared")


# Global error handler instance
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler
