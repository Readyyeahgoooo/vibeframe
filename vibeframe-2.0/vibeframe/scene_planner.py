"""Scene planning component for VibeFrame 2.0."""

import json
import requests
import time
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import asdict

from .models import CutPoint, SceneDescription, CharacterDescriptor, AudioFeatures
from .exceptions import APIError, ScenePlanningError
from .utils import setup_logging

logger = setup_logging()

class ScenePlanner:
    """
    Generates creative scene descriptions and detailed video prompts using LLM.
    
    This component is responsible for the second stage of the music video generation pipeline,
    converting audio analysis results into creative scene descriptions that will drive video generation.
    """
    
    def __init__(self, openrouter_api_key: Optional[str] = None):
        """
        Initialize ScenePlanner with OpenRouter API key.
        
        Args:
            openrouter_api_key: API key for OpenRouter service. If None, will use template fallback.
        """
        self.api_key = openrouter_api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "google/gemini-2.0-flash-exp:free"  # Free tier model
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Template fallbacks for when API is unavailable
        self.scene_templates = {
            "intro": {
                "high_energy": "Dynamic opening shot with character in motion, bright lighting, wide camera angle",
                "medium_energy": "Character introduction scene, balanced lighting, medium camera angle", 
                "low_energy": "Atmospheric opening, soft lighting, close-up camera angle"
            },
            "verse": {
                "high_energy": "Character performing energetic actions, dynamic lighting, moving camera",
                "medium_energy": "Character in narrative scene, natural lighting, steady camera",
                "low_energy": "Contemplative character moment, moody lighting, static camera"
            },
            "chorus": {
                "high_energy": "Climactic performance scene, dramatic lighting, multiple camera angles",
                "medium_energy": "Character in central action, bright lighting, medium shots",
                "low_energy": "Emotional peak moment, warm lighting, intimate camera work"
            },
            "bridge": {
                "high_energy": "Transition scene with movement, changing lighting, camera transitions",
                "medium_energy": "Character in different setting, contrasting lighting, new angle",
                "low_energy": "Reflective transition, subtle lighting changes, slow camera movement"
            },
            "outro": {
                "high_energy": "Energetic conclusion, bright lighting, wide establishing shot",
                "medium_energy": "Resolution scene, balanced lighting, medium camera angle",
                "low_energy": "Peaceful ending, soft lighting, fade-out camera work"
            }
        }
        
        logger.info(f"ScenePlanner initialized with API key: {'Yes' if self.api_key else 'No (template fallback)'}")
    
    def generate_scene_descriptions(
        self,
        cut_points: List[CutPoint],
        audio_features: List[Dict[str, float]],
        global_style: Optional[str] = None,
        user_theme: Optional[str] = None
    ) -> List[SceneDescription]:
        """
        Generate scene descriptions for all cut points.
        
        Args:
            cut_points: List of cut points from audio analysis
            audio_features: Audio features for each time segment
            global_style: User-specified global style (e.g., "cinematic", "anime", "realistic")
            user_theme: User-specified theme (e.g., "adventure", "romance", "mystery")
            
        Returns:
            List of SceneDescription objects
            
        Raises:
            ScenePlanningError: If scene generation fails
        """
        if not cut_points:
            raise ScenePlanningError("No cut points provided for scene generation")
        
        if len(audio_features) != len(cut_points) - 1:
            # Adjust audio features to match cut points (we need n-1 features for n cut points)
            if len(audio_features) == len(cut_points):
                audio_features = audio_features[:-1]  # Remove last feature
            else:
                raise ScenePlanningError(f"Audio features count ({len(audio_features)}) doesn't match cut points ({len(cut_points)})")
        
        logger.info(f"Generating scene descriptions for {len(cut_points)} cut points")
        logger.info(f"Global style: {global_style}, Theme: {user_theme}")
        
        scenes = []
        previous_scene = None
        
        for i in range(len(cut_points) - 1):
            start_cut = cut_points[i]
            end_cut = cut_points[i + 1]
            features = audio_features[i]
            
            try:
                scene = self._generate_single_scene(
                    scene_id=i + 1,
                    start_time=start_cut.timestamp,
                    end_time=end_cut.timestamp,
                    section=start_cut.section,
                    audio_features=features,
                    global_style=global_style,
                    user_theme=user_theme,
                    previous_scene=previous_scene
                )
                
                scenes.append(scene)
                previous_scene = scene
                
                logger.info(f"Generated scene {i + 1}: {scene.description[:50]}...")
                
            except Exception as e:
                logger.error(f"Failed to generate scene {i + 1}: {str(e)}")
                # Create fallback scene
                fallback_scene = self._create_fallback_scene(
                    scene_id=i + 1,
                    start_time=start_cut.timestamp,
                    end_time=end_cut.timestamp,
                    section=start_cut.section,
                    audio_features=features,
                    global_style=global_style
                )
                scenes.append(fallback_scene)
                previous_scene = fallback_scene
        
        logger.info(f"Successfully generated {len(scenes)} scene descriptions")
        return scenes
    
    def _generate_single_scene(
        self,
        scene_id: int,
        start_time: float,
        end_time: float,
        section: str,
        audio_features: Dict[str, float],
        global_style: Optional[str],
        user_theme: Optional[str],
        previous_scene: Optional[SceneDescription]
    ) -> SceneDescription:
        """Generate a single scene description."""
        duration = end_time - start_time
        
        # Try API first, fall back to templates if needed
        if self.api_key:
            try:
                scene_data = self._generate_with_api(
                    section=section,
                    audio_features=audio_features,
                    duration=duration,
                    global_style=global_style,
                    user_theme=user_theme,
                    previous_scene=previous_scene
                )
            except Exception as e:
                logger.warning(f"API generation failed: {str(e)}, falling back to templates")
                scene_data = self._generate_with_template(
                    section=section,
                    audio_features=audio_features,
                    global_style=global_style
                )
        else:
            scene_data = self._generate_with_template(
                section=section,
                audio_features=audio_features,
                global_style=global_style
            )
        
        # Create enhanced video prompt
        video_prompt = self.enhance_prompt(
            description=scene_data["description"],
            style=global_style or "cinematic",
            previous_scene=previous_scene
        )
        
        return SceneDescription(
            id=scene_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            description=scene_data["description"],
            character_action=scene_data["character_action"],
            camera_angle=scene_data["camera_angle"],
            lighting=scene_data["lighting"],
            environment=scene_data["environment"],
            video_prompt=video_prompt
        )
    
    def _generate_with_api(
        self,
        section: str,
        audio_features: Dict[str, float],
        duration: float,
        global_style: Optional[str],
        user_theme: Optional[str],
        previous_scene: Optional[SceneDescription]
    ) -> Dict[str, str]:
        """Generate scene using OpenRouter API."""
        mood = self.analyze_mood(audio_features)
        
        # Build context from previous scene
        previous_context = ""
        if previous_scene:
            previous_context = f"Previous scene: {previous_scene.description}. Character was {previous_scene.character_action}."
        
        prompt = f"""You are a creative director for music videos. Generate a detailed scene description based on this musical analysis:

Musical Context:
- Section: {section} ({duration:.1f} seconds)
- Tempo: {audio_features.get('tempo', 120):.0f} BPM
- Energy: {audio_features.get('energy', 0.5):.2f} (0-1 scale)
- Mood: {mood}
- Global Style: {global_style or 'cinematic'}
- Theme: {user_theme or 'general'}

{previous_context}

Generate a scene that:
1. Matches the musical mood and energy level
2. Maintains narrative continuity with the previous scene
3. Includes specific character actions, camera work, and lighting
4. Follows the global style and theme

Respond with valid JSON in this exact format:
{{
  "description": "Brief scene summary (1-2 sentences)",
  "character_action": "What the character is doing",
  "camera_angle": "Camera position and movement",
  "lighting": "Lighting setup and mood",
  "environment": "Setting and background details"
}}"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vibeframe/ai-music-video-generator",
            "X-Title": "VibeFrame AI Music Video Generator"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.8,
            "max_tokens": 500
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 429:
                    # Rate limit hit
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit hit, waiting {retry_after} seconds")
                    if attempt == self.max_retries - 1:
                        raise APIError(f"API request failed after {self.max_retries} attempts: Rate limit exceeded")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                content = result["choices"][0]["message"]["content"].strip()
                
                # Parse JSON response
                try:
                    # Clean up the response (remove markdown code blocks if present)
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    
                    scene_data = json.loads(content)
                    
                    # Validate required fields
                    required_fields = ["description", "character_action", "camera_angle", "lighting", "environment"]
                    if all(field in scene_data for field in required_fields):
                        return scene_data
                    else:
                        raise ValueError(f"Missing required fields: {set(required_fields) - set(scene_data.keys())}")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse API response: {str(e)}")
                    logger.debug(f"Raw response: {content}")
                    if attempt == self.max_retries - 1:
                        raise APIError(f"Failed to parse API response after {self.max_retries} attempts")
                    continue
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    raise APIError(f"API request failed after {self.max_retries} attempts: {str(e)}")
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        raise APIError("Failed to generate scene with API")
    
    def _generate_with_template(
        self,
        section: str,
        audio_features: Dict[str, float],
        global_style: Optional[str]
    ) -> Dict[str, str]:
        """Generate scene using template fallback."""
        energy = audio_features.get('energy', 0.5)
        
        # Determine energy level
        if energy > 0.7:
            energy_level = "high_energy"
        elif energy > 0.3:
            energy_level = "medium_energy"
        else:
            energy_level = "low_energy"
        
        # Get base template
        base_description = self.scene_templates.get(section, self.scene_templates["verse"])[energy_level]
        
        # Customize with style
        style_prefix = ""
        if global_style:
            style_prefix = f"{global_style} style, "
        
        return {
            "description": f"{style_prefix}{base_description}",
            "character_action": f"Character performing {energy_level.replace('_', ' ')} actions matching the {section} section",
            "camera_angle": f"Camera work appropriate for {energy_level.replace('_', ' ')} scene",
            "lighting": f"Lighting setup for {energy_level.replace('_', ' ')} mood",
            "environment": f"Environment matching {section} section with {global_style or 'cinematic'} aesthetic"
        }
    
    def _create_fallback_scene(
        self,
        scene_id: int,
        start_time: float,
        end_time: float,
        section: str,
        audio_features: Dict[str, float],
        global_style: Optional[str]
    ) -> SceneDescription:
        """Create a basic fallback scene when all generation methods fail."""
        duration = end_time - start_time
        
        return SceneDescription(
            id=scene_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            description=f"Scene {scene_id}: {section} section with {global_style or 'cinematic'} style",
            character_action=f"Character in {section} scene",
            camera_angle="Medium shot",
            lighting="Balanced lighting",
            environment=f"{section.title()} scene environment",
            video_prompt=f"{global_style or 'Cinematic'} {section} scene with character, medium shot, balanced lighting"
        )
    
    def enhance_prompt(
        self,
        description: str,
        style: str,
        previous_scene: Optional[SceneDescription] = None
    ) -> str:
        """
        Convert scene description into detailed video generation prompt.
        
        Args:
            description: Basic scene description
            style: Visual style (e.g., "cinematic", "anime", "realistic")
            previous_scene: Previous scene for continuity
            
        Returns:
            Enhanced prompt suitable for video generation
        """
        # Base prompt components
        prompt_parts = []
        
        # Add style prefix
        prompt_parts.append(f"{style} style")
        
        # Add main description
        prompt_parts.append(description)
        
        # Add quality and technical specifications
        quality_terms = [
            "high quality",
            "detailed",
            "professional cinematography",
            "smooth motion",
            "good lighting"
        ]
        prompt_parts.extend(quality_terms)
        
        # Add continuity hints if previous scene exists
        if previous_scene:
            prompt_parts.append("maintaining visual continuity")
        
        # Join with commas
        enhanced_prompt = ", ".join(prompt_parts)
        
        # Add negative prompt elements
        negative_elements = [
            "blurry",
            "low quality", 
            "distorted",
            "text",
            "watermark"
        ]
        
        enhanced_prompt += f". Avoid: {', '.join(negative_elements)}"
        
        return enhanced_prompt
    
    def generate_action_sequence(
        self,
        action: str,
        num_shots: int,
        character: str
    ) -> List[str]:
        """
        Break down an action into multiple connected shot prompts.
        
        Args:
            action: Action to decompose (e.g., "character throws a ball")
            num_shots: Number of shots to create
            character: Character description
            
        Returns:
            List of shot prompts that together describe the complete action
        """
        if num_shots < 2:
            return [f"{character} {action}"]
        
        # Try API first for complex action decomposition
        if self.api_key:
            try:
                return self._decompose_action_with_api(action, num_shots, character)
            except Exception as e:
                logger.warning(f"API action decomposition failed: {str(e)}, using template approach")
        
        # Template-based decomposition
        return self._decompose_action_with_template(action, num_shots, character)
    
    def _decompose_action_with_api(self, action: str, num_shots: int, character: str) -> List[str]:
        """Decompose action using API."""
        prompt = f"""Break down this action into {num_shots} connected video shots:

Action: {character} {action}

Create {num_shots} shot descriptions that:
1. Show the action from different angles/perspectives
2. Create smooth visual flow between shots
3. Include specific camera movements and framing
4. Maintain character consistency

Respond with a JSON array of shot descriptions:
["shot 1 description", "shot 2 description", ...]"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 300
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        # Parse JSON array
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        shots = json.loads(content)
        
        if isinstance(shots, list) and len(shots) == num_shots:
            return shots
        else:
            raise ValueError("Invalid API response format")
    
    def _decompose_action_with_template(self, action: str, num_shots: int, character: str) -> List[str]:
        """Decompose action using templates."""
        base_action = f"{character} {action}"
        
        if num_shots == 2:
            return [
                f"Wide shot: {base_action}, establishing the scene",
                f"Close-up: {base_action}, focusing on character expression"
            ]
        elif num_shots == 3:
            return [
                f"Wide shot: {base_action}, establishing the scene",
                f"Medium shot: {base_action}, showing the action clearly", 
                f"Close-up: {base_action}, capturing the result/emotion"
            ]
        else:
            # For more shots, create variations
            shots = [f"Wide shot: {base_action}, establishing the scene"]
            
            for i in range(1, num_shots - 1):
                angle = ["medium shot", "close-up", "over-shoulder", "low angle", "high angle"][i % 5]
                shots.append(f"{angle.title()}: {base_action}, shot {i + 1}")
            
            shots.append(f"Final shot: {base_action}, concluding the sequence")
            
            return shots
    
    def analyze_mood(self, audio_features: Dict[str, float]) -> str:
        """
        Determine mood/emotion from audio features.
        
        Args:
            audio_features: Dictionary of audio features
            
        Returns:
            Mood description string
        """
        energy = audio_features.get('energy', 0.5)
        tempo = audio_features.get('tempo', 120)
        spectral_centroid = audio_features.get('spectral_centroid', 2000)
        
        # Simple mood classification based on features
        if energy > 0.7 and tempo > 140:
            return "energetic and exciting"
        elif energy > 0.6 and tempo > 120:
            return "upbeat and positive"
        elif energy < 0.3 and tempo < 80:
            return "calm and peaceful"
        elif energy < 0.4 and spectral_centroid < 1500:
            return "melancholic and introspective"
        elif tempo > 160:
            return "intense and driving"
        elif energy > 0.5:
            return "dynamic and engaging"
        else:
            return "balanced and steady"
    
    def get_api_status(self) -> Dict[str, Any]:
        """
        Get current API status and usage information.
        
        Returns:
            Dictionary with API status information
        """
        if not self.api_key:
            return {
                "api_available": False,
                "reason": "No API key provided",
                "fallback_mode": True
            }
        
        try:
            # Test API with a simple request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    "api_available": True,
                    "model": self.model,
                    "fallback_mode": False
                }
            elif response.status_code == 401:
                return {
                    "api_available": False,
                    "reason": "Invalid API key",
                    "fallback_mode": True
                }
            elif response.status_code == 429:
                return {
                    "api_available": False,
                    "reason": "Rate limit exceeded",
                    "fallback_mode": True
                }
            else:
                return {
                    "api_available": False,
                    "reason": f"API error: {response.status_code}",
                    "fallback_mode": True
                }
                
        except Exception as e:
            return {
                "api_available": False,
                "reason": f"Connection error: {str(e)}",
                "fallback_mode": True
            }