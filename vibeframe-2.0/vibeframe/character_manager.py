"""Character management component for VibeFrame 2.0."""

import re
import json
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
from dataclasses import asdict

from .models import CharacterDescriptor, SceneDescription
from .exceptions import CharacterError
from .utils import setup_logging

logger = setup_logging()

class CharacterManager:
    """
    Manages character consistency across video scenes.
    
    This component extracts character descriptions from prompts, maintains character
    consistency across scenes, and provides reference frames for video continuation.
    """
    
    def __init__(self):
        """Initialize CharacterManager."""
        self.characters: Dict[str, Dict[str, Any]] = {}  # Store character with metadata
        self.character_counter = 0
        
        # Character extraction patterns
        self.character_patterns = [
            # Direct character descriptions
            r'(?:character|person|figure|individual|subject)\s+(?:is\s+)?([^,\.]+)',
            # Descriptive patterns
            r'(?:young|old|tall|short|beautiful|handsome)\s+([^,\.]+)',
            # Profession/role patterns
            r'(?:a|an)\s+([^,\.]+?)(?:\s+(?:character|person|figure))?',
            # Appearance patterns
            r'(?:with|having)\s+([^,\.]+?)(?:\s+(?:hair|eyes|skin))',
            # Action-based patterns
            r'([^,\.]+?)\s+(?:walking|running|dancing|singing|standing|sitting)',
        ]
        
        logger.info("CharacterManager initialized")
    
    def extract_character_description(self, prompt: str) -> Optional[Tuple[str, CharacterDescriptor]]:
        """
        Extract character description from a text prompt using regex and NLP.
        
        Args:
            prompt: Text prompt containing character description
            
        Returns:
            Tuple of (character_id, CharacterDescriptor) if character found, None otherwise
        """
        if not prompt or not isinstance(prompt, str):
            return None
        
        prompt_lower = prompt.lower().strip()
        
        # Skip if no character-related keywords
        character_keywords = ['character', 'person', 'figure', 'individual', 'subject', 
                            'man', 'woman', 'boy', 'girl', 'human', 'people']
        
        if not any(keyword in prompt_lower for keyword in character_keywords):
            return None
        
        # Extract character features
        extracted_features = []
        
        for pattern in self.character_patterns:
            matches = re.finditer(pattern, prompt_lower, re.IGNORECASE)
            for match in matches:
                feature = match.group(1).strip()
                if feature and len(feature) > 2:  # Filter out very short matches
                    extracted_features.append(feature)
        
        if not extracted_features:
            # Fallback: extract any descriptive text near character keywords
            for keyword in character_keywords:
                if keyword in prompt_lower:
                    # Extract surrounding context
                    start_idx = prompt_lower.find(keyword)
                    context_start = max(0, start_idx - 20)
                    context_end = min(len(prompt), start_idx + len(keyword) + 30)
                    context = prompt[context_start:context_end].strip()
                    
                    if context:
                        extracted_features.append(context)
                        break
        
        if not extracted_features:
            return None
        
        # Create character descriptor
        self.character_counter += 1
        character_id = f"char_{self.character_counter}"
        
        # Combine and clean features
        combined_description = " ".join(extracted_features)
        cleaned_description = re.sub(r'\s+', ' ', combined_description).strip()
        
        # Extract specific attributes
        appearance = self._extract_appearance(cleaned_description)
        clothing = self._extract_clothing(cleaned_description)
        age_gender = self._extract_age_gender(cleaned_description)
        
        character = CharacterDescriptor(
            appearance=appearance,
            clothing=clothing,
            style="cinematic",  # Default style
            age=age_gender,
            ethnicity=None,
            distinctive_features=cleaned_description
        )
        
        # Store character with metadata
        character_data = {
            "id": character_id,
            "name": f"Character {self.character_counter}",
            "description": cleaned_description,
            "character": character,
            "reference_frame_path": None,
            "consistency_prompt": self._build_consistency_prompt(cleaned_description, appearance, clothing, age_gender)
        }
        
        # Store character
        self.characters[character_id] = character_data
        
        logger.info(f"Extracted character: {character_data['name']} - {character_data['description'][:50]}...")
        return character_id, character
    
    def _extract_appearance(self, description: str) -> str:
        """Extract appearance-related features."""
        appearance_patterns = [
            r'((?:long|short|curly|straight|blonde|brown|black|red|dark|light)\s+hair)',
            r'((?:blue|brown|green|hazel|dark|bright)\s+eyes)',
            r'((?:pale|tan|dark|fair|olive)\s+skin)',
            r'((?:tall|short|slim|athletic|muscular|petite))',
        ]
        
        features = []
        for pattern in appearance_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            features.extend(matches)
        
        return ", ".join(features) if features else "general appearance"
    
    def _extract_clothing(self, description: str) -> str:
        """Extract clothing-related features."""
        clothing_patterns = [
            r'((?:wearing|dressed in|in)\s+[^,\.]+)',
            r'((?:shirt|dress|jacket|coat|pants|jeans|skirt|suit|uniform|outfit)\s*[^,\.]*)',
            r'((?:casual|formal|elegant|sporty|vintage|modern)\s+(?:clothing|attire|outfit))',
        ]
        
        features = []
        for pattern in clothing_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            features.extend(matches)
        
        return ", ".join(features) if features else "casual clothing"
    
    def _extract_age_gender(self, description: str) -> str:
        """Extract age and gender information."""
        age_gender_patterns = [
            r'(young|old|elderly|teenage|adult|middle-aged)',
            r'(man|woman|boy|girl|male|female)',
            r'((?:20|30|40|50|60)\s*(?:year|years)\s*old)',
        ]
        
        features = []
        for pattern in age_gender_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            features.extend(matches)
        
        return ", ".join(features) if features else "adult"
    
    def _build_consistency_prompt(self, description: str, appearance: str, clothing: str, age_gender: str) -> str:
        """Build a consistency prompt for character."""
        prompt_parts = []
        
        if age_gender and age_gender != "adult":
            prompt_parts.append(age_gender)
        
        if appearance and appearance != "general appearance":
            prompt_parts.append(appearance)
        
        if clothing and clothing != "casual clothing":
            prompt_parts.append(clothing)
        
        # Add the original description as fallback
        if not prompt_parts:
            prompt_parts.append(description)
        
        return ", ".join(prompt_parts)
    
    def inject_character(self, prompt: str, character: CharacterDescriptor) -> str:
        """
        Inject character description into a video generation prompt.
        
        Args:
            prompt: Original video prompt
            character: Character to inject
            
        Returns:
            Enhanced prompt with character consistency
        """
        if not prompt or not character:
            return prompt
        
        # Use the character's to_prompt_string method
        character_prompt = character.to_prompt_string()
        
        # Check if character is already mentioned in prompt
        if any(keyword in prompt.lower() for keyword in ['character', 'person', 'figure', 'individual']):
            # Replace generic character references with specific description
            enhanced_prompt = re.sub(
                r'\b(?:character|person|figure|individual)\b',
                character_prompt,
                prompt,
                flags=re.IGNORECASE
            )
        else:
            # Add character description to the beginning
            enhanced_prompt = f"{character_prompt}, {prompt}"
        
        # Ensure consistency keywords are present
        consistency_keywords = ["same character", "consistent appearance", "character continuity"]
        if not any(keyword in enhanced_prompt.lower() for keyword in consistency_keywords):
            enhanced_prompt += ", maintaining character consistency"
        
        logger.debug(f"Injected character into prompt: {enhanced_prompt[:100]}...")
        return enhanced_prompt
    
    def get_reference_frame(self, video_path: str, timestamp: float = 0.5) -> Optional[str]:
        """
        Extract a reference frame from a video for character consistency.
        
        Args:
            video_path: Path to video file
            timestamp: Timestamp to extract frame (as fraction of video length)
            
        Returns:
            Path to extracted frame image, None if extraction fails
        """
        if not video_path or not Path(video_path).exists():
            logger.warning(f"Video file not found: {video_path}")
            return None
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return None
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0 or fps == 0:
                logger.error(f"Invalid video properties: frames={total_frames}, fps={fps}")
                cap.release()
                return None
            
            # Calculate frame number
            target_frame = int(total_frames * timestamp)
            target_frame = max(0, min(target_frame, total_frames - 1))
            
            # Seek to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                logger.error(f"Could not read frame at position {target_frame}")
                return None
            
            # Save frame
            video_name = Path(video_path).stem
            frame_filename = f"{video_name}_frame_{target_frame}.jpg"
            frame_path = Path(video_path).parent / "reference_frames" / frame_filename
            
            # Create directory if it doesn't exist
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save frame
            success = cv2.imwrite(str(frame_path), frame)
            
            if success:
                logger.info(f"Extracted reference frame: {frame_path}")
                return str(frame_path)
            else:
                logger.error(f"Failed to save frame: {frame_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting reference frame: {str(e)}")
            return None
    
    def validate_consistency(self, character1: CharacterDescriptor, character2: CharacterDescriptor) -> Dict[str, Any]:
        """
        Validate consistency between two character descriptions.
        
        Args:
            character1: First character
            character2: Second character
            
        Returns:
            Dictionary with consistency validation results
        """
        if not character1 or not character2:
            return {
                "consistent": False,
                "confidence": 0.0,
                "issues": ["Missing character data"],
                "similarity_score": 0.0
            }
        
        issues = []
        similarity_scores = []
        
        # Compare appearance
        appearance_similarity = self._compare_text_similarity(
            character1.appearance, character2.appearance
        )
        similarity_scores.append(appearance_similarity)
        
        if appearance_similarity < 0.5:
            issues.append(f"Appearance mismatch: '{character1.appearance}' vs '{character2.appearance}'")
        
        # Compare clothing
        clothing_similarity = self._compare_text_similarity(
            character1.clothing, character2.clothing
        )
        similarity_scores.append(clothing_similarity)
        
        if clothing_similarity < 0.3:  # Clothing can change more than appearance
            issues.append(f"Clothing mismatch: '{character1.clothing}' vs '{character2.clothing}'")
        
        # Compare age
        age_similarity = self._compare_text_similarity(
            character1.age or "", character2.age or ""
        )
        similarity_scores.append(age_similarity)
        
        if age_similarity < 0.7:
            issues.append(f"Age mismatch: '{character1.age}' vs '{character2.age}'")
        
        # Calculate overall similarity
        overall_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        # Determine consistency
        consistent = overall_similarity >= 0.6 and len(issues) <= 1
        confidence = overall_similarity
        
        result = {
            "consistent": consistent,
            "confidence": confidence,
            "issues": issues,
            "similarity_score": overall_similarity,
            "appearance_similarity": appearance_similarity,
            "clothing_similarity": clothing_similarity,
            "age_similarity": age_similarity
        }
        
        logger.debug(f"Character consistency validation: {result}")
        return result
    
    def _compare_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compare similarity between two text descriptions.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 and not text2:
            return 1.0  # Both empty strings are identical
        
        if not text1 or not text2:
            return 0.0  # One empty, one not
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_character_id(self, character: CharacterDescriptor) -> Optional[str]:
        """Get character ID for a given CharacterDescriptor."""
        for char_id, data in self.characters.items():
            if data["character"] == character:
                return char_id
        return None
    
    def get_character_by_id(self, character_id: str) -> Optional[CharacterDescriptor]:
        """Get character by ID."""
        character_data = self.characters.get(character_id)
        return character_data["character"] if character_data else None
    
    def get_all_characters(self) -> List[CharacterDescriptor]:
        """Get all managed characters."""
        return [data["character"] for data in self.characters.values()]
    
    def update_character_reference(self, character_id: str, reference_frame_path: str) -> bool:
        """
        Update character's reference frame path.
        
        Args:
            character_id: Character ID
            reference_frame_path: Path to reference frame
            
        Returns:
            True if updated successfully, False otherwise
        """
        if character_id not in self.characters:
            logger.warning(f"Character not found: {character_id}")
            return False
        
        if not Path(reference_frame_path).exists():
            logger.warning(f"Reference frame not found: {reference_frame_path}")
            return False
        
        self.characters[character_id]["reference_frame_path"] = reference_frame_path
        logger.info(f"Updated reference frame for {character_id}: {reference_frame_path}")
        return True
    
    def merge_characters(self, character1_id: str, character2_id: str) -> Optional[CharacterDescriptor]:
        """
        Merge two similar characters into one.
        
        Args:
            character1_id: First character ID
            character2_id: Second character ID
            
        Returns:
            Merged character descriptor, None if merge fails
        """
        char1_data = self.characters.get(character1_id)
        char2_data = self.characters.get(character2_id)
        
        if not char1_data or not char2_data:
            logger.warning(f"Cannot merge: characters not found ({character1_id}, {character2_id})")
            return None
        
        char1 = char1_data["character"]
        char2 = char2_data["character"]
        
        # Validate they are similar enough to merge
        validation = self.validate_consistency(char1, char2)
        if not validation["consistent"]:
            logger.warning(f"Characters too different to merge: {validation['issues']}")
            return None
        
        # Create merged character
        merged_appearance = char1.appearance if len(char1.appearance) > len(char2.appearance) else char2.appearance
        merged_clothing = char1.clothing if len(char1.clothing) > len(char2.clothing) else char2.clothing
        merged_age = char1.age if char1.age else char2.age
        
        merged_character = CharacterDescriptor(
            appearance=merged_appearance,
            clothing=merged_clothing,
            style=char1.style,
            age=merged_age,
            ethnicity=char1.ethnicity or char2.ethnicity,
            distinctive_features=f"{char1.distinctive_features}; {char2.distinctive_features}"
        )
        
        # Update storage
        char1_data["character"] = merged_character
        char1_data["description"] = f"{char1_data['description']}; {char2_data['description']}"
        del self.characters[character2_id]
        
        logger.info(f"Merged characters {character1_id} and {character2_id}")
        return merged_character
    
    def clear_characters(self):
        """Clear all characters."""
        self.characters.clear()
        self.character_counter = 0
        logger.info("Cleared all characters")
    
    def export_characters(self) -> Dict[str, Any]:
        """
        Export all characters to a dictionary.
        
        Returns:
            Dictionary containing all character data
        """
        return {
            "characters": {char_id: {
                "id": data["id"],
                "name": data["name"],
                "description": data["description"],
                "character": asdict(data["character"]),
                "reference_frame_path": data["reference_frame_path"],
                "consistency_prompt": data["consistency_prompt"]
            } for char_id, data in self.characters.items()},
            "character_counter": self.character_counter
        }
    
    def import_characters(self, data: Dict[str, Any]) -> bool:
        """
        Import characters from a dictionary.
        
        Args:
            data: Dictionary containing character data
            
        Returns:
            True if import successful, False otherwise
        """
        try:
            if "characters" not in data:
                logger.error("Invalid character data: missing 'characters' key")
                return False
            
            # Clear existing characters
            self.clear_characters()
            
            # Import characters
            for char_id, char_data in data["characters"].items():
                character = CharacterDescriptor(**char_data["character"])
                self.characters[char_id] = {
                    "id": char_data["id"],
                    "name": char_data["name"],
                    "description": char_data["description"],
                    "character": character,
                    "reference_frame_path": char_data["reference_frame_path"],
                    "consistency_prompt": char_data["consistency_prompt"]
                }
            
            # Import counter
            self.character_counter = data.get("character_counter", len(self.characters))
            
            logger.info(f"Imported {len(self.characters)} characters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import characters: {str(e)}")
            return False