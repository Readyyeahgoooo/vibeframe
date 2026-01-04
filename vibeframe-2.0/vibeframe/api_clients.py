"""API clients with rate limiting and caching for VibeFrame 2.0."""

import time
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import logging

import requests

from .utils import setup_logging

logger = setup_logging()


class RateLimiter:
    """Rate limiter with exponential backoff."""
    
    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.request_times = []
        self.backoff_until = None
        self.backoff_multiplier = 1.0
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Check if in backoff period
        if self.backoff_until and now < self.backoff_until:
            wait_time = self.backoff_until - now
            logger.info(f"Rate limit backoff: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            return
        
        # Remove old request times (older than 1 minute)
        cutoff = now - 60.0
        self.request_times = [t for t in self.request_times if t > cutoff]
        
        # Check if at limit
        if len(self.request_times) >= self.requests_per_minute:
            # Wait until oldest request is > 1 minute old
            wait_time = 60.0 - (now - self.request_times[0])
            if wait_time > 0:
                logger.debug(f"Rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        
        # Record this request
        self.request_times.append(time.time())
    
    def trigger_backoff(self, retry_after: Optional[int] = None) -> None:
        """
        Trigger exponential backoff.
        
        Args:
            retry_after: Seconds to wait (from API response)
        """
        if retry_after:
            backoff_time = retry_after
        else:
            backoff_time = min(60.0 * self.backoff_multiplier, 300.0)  # Max 5 minutes
            self.backoff_multiplier *= 2.0
        
        self.backoff_until = time.time() + backoff_time
        logger.warning(f"Rate limit triggered: backing off for {backoff_time:.1f}s")
    
    def reset_backoff(self) -> None:
        """Reset backoff multiplier after successful request."""
        self.backoff_multiplier = 1.0
        self.backoff_until = None


class RequestCache:
    """Simple file-based cache for API responses."""
    
    def __init__(self, cache_dir: Optional[str] = None, ttl_hours: int = 24):
        """
        Initialize request cache.
        
        Args:
            cache_dir: Directory for cache files
            ttl_hours: Time-to-live for cache entries in hours
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".vibeframe" / "cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        logger.info(f"Request cache initialized: {self.cache_dir}")
    
    def _get_cache_key(self, url: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> str:
        """Generate cache key from request parameters."""
        key_data = {
            "url": url,
            "params": params or {},
            "data": data or {}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, url: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Get cached response.
        
        Args:
            url: Request URL
            params: Query parameters
            data: Request data
            
        Returns:
            Cached response or None if not found/expired
        """
        cache_key = self._get_cache_key(url, params, data)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            
            # Check if expired
            cached_time = datetime.fromisoformat(cached['timestamp'])
            if datetime.now() - cached_time > self.ttl:
                logger.debug(f"Cache expired for {url}")
                cache_file.unlink()
                return None
            
            logger.debug(f"Cache hit for {url}")
            return cached['response']
            
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def set(self, url: str, response: Dict, params: Optional[Dict] = None, data: Optional[Dict] = None) -> None:
        """
        Cache a response.
        
        Args:
            url: Request URL
            response: Response data to cache
            params: Query parameters
            data: Request data
        """
        cache_key = self._get_cache_key(url, params, data)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cached_data = {
                "timestamp": datetime.now().isoformat(),
                "url": url,
                "response": response
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cached_data, f)
            
            logger.debug(f"Cached response for {url}")
            
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def clear(self) -> None:
        """Clear all cached responses."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("Cache cleared")


class OpenRouterClient:
    """OpenRouter API client with rate limiting and caching."""
    
    def __init__(self, api_key: Optional[str] = None, requests_per_minute: int = 20):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
            requests_per_minute: Rate limit
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.cache = RequestCache(ttl_hours=24)
        logger.info("OpenRouterClient initialized")
    
    def chat_completion(
        self,
        messages: list,
        model: str = "openai/gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make a chat completion request.
        
        Args:
            messages: List of message dicts
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use caching
            
        Returns:
            API response dict
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not set")
        
        url = f"{self.base_url}/chat/completions"
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Check cache
        if use_cache:
            cached = self.cache.get(url, data=data)
            if cached:
                return cached
        
        # Rate limit
        self.rate_limiter.wait_if_needed()
        
        # Make request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                self.rate_limiter.trigger_backoff(retry_after)
                raise Exception(f"Rate limit exceeded. Retry after {retry_after}s")
            
            response.raise_for_status()
            result = response.json()
            
            # Cache successful response
            if use_cache:
                self.cache.set(url, result, data=data)
            
            # Reset backoff on success
            self.rate_limiter.reset_backoff()
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API error: {e}")
            raise


class HuggingFaceClient:
    """HuggingFace API client with rate limiting and caching."""
    
    def __init__(self, token: Optional[str] = None, requests_per_minute: int = 30):
        """
        Initialize HuggingFace client.
        
        Args:
            token: HuggingFace token
            requests_per_minute: Rate limit
        """
        self.token = token
        self.base_url = "https://api-inference.huggingface.co"
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.cache = RequestCache(ttl_hours=168)  # 1 week for model outputs
        logger.info("HuggingFaceClient initialized")
    
    def text_generation(
        self,
        prompt: str,
        model: str = "gpt2",
        max_length: int = 100,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text using HuggingFace model.
        
        Args:
            prompt: Input prompt
            model: Model to use
            max_length: Maximum length
            use_cache: Whether to use caching
            
        Returns:
            API response dict
        """
        url = f"{self.base_url}/models/{model}"
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_length": max_length
            }
        }
        
        # Check cache
        if use_cache:
            cached = self.cache.get(url, data=data)
            if cached:
                return cached
        
        # Rate limit
        self.rate_limiter.wait_if_needed()
        
        # Make request
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=60)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                self.rate_limiter.trigger_backoff(retry_after)
                raise Exception(f"Rate limit exceeded. Retry after {retry_after}s")
            
            response.raise_for_status()
            result = response.json()
            
            # Cache successful response
            if use_cache:
                self.cache.set(url, result, data=data)
            
            # Reset backoff on success
            self.rate_limiter.reset_backoff()
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HuggingFace API error: {e}")
            raise
    
    def check_model_status(self, model: str) -> Dict[str, Any]:
        """
        Check if a model is loaded and ready.
        
        Args:
            model: Model name
            
        Returns:
            Status information
        """
        url = f"{self.base_url}/models/{model}"
        
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Model status check error: {e}")
            return {"error": str(e)}
