"""Project management component for VibeFrame 2.0."""

import os
import json
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from .models import Storyboard, ProjectStatus, ProcessingStep, SceneDescription
from .exceptions import ProjectError, StoryboardValidationError
from .utils import setup_logging

logger = setup_logging()

class ProjectManager:
    """
    Project management component for VibeFrame 2.0.
    
    This component handles project creation, storyboard management,
    file organization, and project status tracking.
    """
    
    def __init__(self, projects_root: Optional[str] = None):
        """
        Initialize ProjectManager.
        
        Args:
            projects_root: Root directory for all projects
        """
        self.projects_root = Path(projects_root) if projects_root else Path.home() / "VibeFrame" / "projects"
        self.projects_root.mkdir(parents=True, exist_ok=True)
        
        # Project structure
        self.project_structure = {
            "audio": "audio",
            "storyboard": "storyboard.json",
            "scenes": "scenes",
            "clips": "clips",
            "final": "final",
            "temp": "temp",
            "metadata": "project.json"
        }
        
        logger.info(f"ProjectManager initialized with root: {self.projects_root}")
    
    def create_project(self, audio_path: str, project_name: str) -> Dict[str, Any]:
        """
        Create new project with audio file.
        
        Args:
            audio_path: Path to audio file
            project_name: Name for the project
            
        Returns:
            Dictionary with project information
            
        Raises:
            ProjectError: If project creation fails
        """
        if not Path(audio_path).exists():
            raise ProjectError(f"Audio file not found: {audio_path}")
        
        # Sanitize project name
        safe_name = self._sanitize_filename(project_name)
        project_dir = self.projects_root / safe_name
        
        # Check if project already exists
        if project_dir.exists():
            raise ProjectError(f"Project '{safe_name}' already exists")
        
        try:
            logger.info(f"Creating project: {safe_name}")
            
            # Create project directory structure
            project_dir.mkdir(parents=True)
            
            for folder_name in ["scenes", "clips", "final", "temp"]:
                (project_dir / folder_name).mkdir()
            
            # Copy audio file
            audio_ext = Path(audio_path).suffix
            project_audio_path = project_dir / "audio" / f"original{audio_ext}"
            project_audio_path.parent.mkdir(exist_ok=True)
            shutil.copy2(audio_path, project_audio_path)
            
            # Create project metadata
            project_info = {
                "name": project_name,
                "safe_name": safe_name,
                "created_at": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "audio_file": str(project_audio_path),
                "original_audio_path": audio_path,
                "status": ProcessingStep.ANALYSIS.value,
                "progress_percent": 0.0,
                "total_scenes": 0,
                "scenes_generated": 0,
                "errors": []
            }
            
            # Save project metadata
            metadata_path = project_dir / self.project_structure["metadata"]
            with open(metadata_path, 'w') as f:
                json.dump(project_info, f, indent=2)
            
            logger.info(f"Project created successfully: {project_dir}")
            
            return {
                "project_dir": str(project_dir),
                "project_info": project_info,
                "audio_path": str(project_audio_path)
            }
            
        except Exception as e:
            # Clean up on error
            if project_dir.exists():
                shutil.rmtree(project_dir)
            
            error_msg = f"Failed to create project '{project_name}': {str(e)}"
            logger.error(error_msg)
            raise ProjectError(error_msg) from e
    
    def save_storyboard(self, project_name: str, storyboard: Storyboard) -> None:
        """
        Save storyboard to project directory.
        
        Args:
            project_name: Name of the project
            storyboard: Storyboard object to save
            
        Raises:
            ProjectError: If saving fails
        """
        project_dir = self._get_project_dir(project_name)
        
        try:
            logger.info(f"Saving storyboard for project: {project_name}")
            
            # Validate storyboard
            self._validate_storyboard(storyboard)
            
            # Save storyboard
            storyboard_path = project_dir / self.project_structure["storyboard"]
            storyboard_json = storyboard.to_json()
            
            with open(storyboard_path, 'w') as f:
                f.write(storyboard_json)
            
            # Update project metadata
            self._update_project_metadata(project_name, {
                "total_scenes": len(storyboard.scenes),
                "last_modified": datetime.now().isoformat(),
                "status": ProcessingStep.PLANNING.value,
                "progress_percent": 25.0
            })
            
            logger.info(f"Storyboard saved: {storyboard_path}")
            
        except Exception as e:
            error_msg = f"Failed to save storyboard for '{project_name}': {str(e)}"
            logger.error(error_msg)
            raise ProjectError(error_msg) from e
    
    def load_storyboard(self, project_name: str) -> Storyboard:
        """
        Load storyboard from project directory.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Loaded Storyboard object
            
        Raises:
            ProjectError: If loading fails
        """
        project_dir = self._get_project_dir(project_name)
        storyboard_path = project_dir / self.project_structure["storyboard"]
        
        if not storyboard_path.exists():
            raise ProjectError(f"Storyboard not found for project '{project_name}'")
        
        try:
            logger.info(f"Loading storyboard for project: {project_name}")
            
            with open(storyboard_path, 'r') as f:
                storyboard_json = f.read()
            
            storyboard = Storyboard.from_json(storyboard_json)
            
            # Validate loaded storyboard
            self._validate_storyboard(storyboard)
            
            logger.info(f"Storyboard loaded: {len(storyboard.scenes)} scenes")
            return storyboard
            
        except Exception as e:
            error_msg = f"Failed to load storyboard for '{project_name}': {str(e)}"
            logger.error(error_msg)
            raise ProjectError(error_msg) from e
    
    def save_generated_clip(self, project_name: str, scene_id: int, clip_path: str) -> str:
        """
        Save generated clip to project.
        
        Args:
            project_name: Name of the project
            scene_id: ID of the scene
            clip_path: Path to generated clip
            
        Returns:
            Path where clip was saved
            
        Raises:
            ProjectError: If saving fails
        """
        if not Path(clip_path).exists():
            raise ProjectError(f"Clip file not found: {clip_path}")
        
        project_dir = self._get_project_dir(project_name)
        clips_dir = project_dir / self.project_structure["clips"]
        
        try:
            logger.info(f"Saving clip for scene {scene_id} in project: {project_name}")
            
            # Create clip filename
            clip_ext = Path(clip_path).suffix
            clip_filename = f"scene_{scene_id:03d}{clip_ext}"
            project_clip_path = clips_dir / clip_filename
            
            # Copy clip to project directory
            shutil.copy2(clip_path, project_clip_path)
            
            # Update project metadata
            project_info = self._load_project_metadata(project_name)
            scenes_generated = project_info.get("scenes_generated", 0) + 1
            total_scenes = project_info.get("total_scenes", 1)
            
            # Avoid division by zero
            if total_scenes > 0:
                progress = min(50.0 + (scenes_generated / total_scenes) * 40.0, 90.0)
            else:
                progress = 50.0
            
            self._update_project_metadata(project_name, {
                "scenes_generated": scenes_generated,
                "last_modified": datetime.now().isoformat(),
                "status": ProcessingStep.GENERATION.value,
                "progress_percent": progress
            })
            
            logger.info(f"Clip saved: {project_clip_path}")
            return str(project_clip_path)
            
        except Exception as e:
            error_msg = f"Failed to save clip for scene {scene_id}: {str(e)}"
            logger.error(error_msg)
            raise ProjectError(error_msg) from e
    
    def get_project_status(self, project_name: str) -> ProjectStatus:
        """
        Get current project status and progress.
        
        Args:
            project_name: Name of the project
            
        Returns:
            ProjectStatus object
            
        Raises:
            ProjectError: If project not found
        """
        try:
            project_info = self._load_project_metadata(project_name)
            
            status = ProjectStatus(
                project_name=project_info["name"],
                audio_file=project_info["audio_file"],
                created_at=datetime.fromisoformat(project_info["created_at"]),
                last_modified=datetime.fromisoformat(project_info["last_modified"]),
                total_scenes=project_info.get("total_scenes", 0),
                scenes_generated=project_info.get("scenes_generated", 0),
                current_step=project_info.get("status", ProcessingStep.ANALYSIS.value),
                progress_percent=project_info.get("progress_percent", 0.0),
                errors=project_info.get("errors", [])
            )
            
            return status
            
        except Exception as e:
            error_msg = f"Failed to get status for project '{project_name}': {str(e)}"
            logger.error(error_msg)
            raise ProjectError(error_msg) from e
    
    def cleanup_project(self, project_name: str, keep_final_video: bool = True) -> None:
        """
        Clean up intermediate files.
        
        Args:
            project_name: Name of the project
            keep_final_video: Whether to keep the final video file
            
        Raises:
            ProjectError: If cleanup fails
        """
        project_dir = self._get_project_dir(project_name)
        
        try:
            logger.info(f"Cleaning up project: {project_name}")
            
            # Clean up temporary files
            temp_dir = project_dir / self.project_structure["temp"]
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                temp_dir.mkdir()
            
            # Clean up intermediate clips if final video exists
            final_dir = project_dir / self.project_structure["final"]
            clips_dir = project_dir / self.project_structure["clips"]
            
            if keep_final_video and final_dir.exists() and any(final_dir.iterdir()):
                # Final video exists, can clean up clips
                if clips_dir.exists():
                    shutil.rmtree(clips_dir)
                    clips_dir.mkdir()
                logger.info("Intermediate clips cleaned up")
            
            # Update metadata
            self._update_project_metadata(project_name, {
                "last_modified": datetime.now().isoformat()
            })
            
            logger.info(f"Project cleanup completed: {project_name}")
            
        except Exception as e:
            error_msg = f"Failed to cleanup project '{project_name}': {str(e)}"
            logger.error(error_msg)
            raise ProjectError(error_msg) from e
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """
        List all projects with their metadata.
        
        Returns:
            List of project information dictionaries
        """
        projects = []
        
        try:
            for project_dir in self.projects_root.iterdir():
                if not project_dir.is_dir():
                    continue
                
                metadata_path = project_dir / self.project_structure["metadata"]
                if not metadata_path.exists():
                    continue
                
                try:
                    with open(metadata_path, 'r') as f:
                        project_info = json.load(f)
                    
                    # Add computed fields
                    project_info["project_dir"] = str(project_dir)
                    project_info["size_mb"] = self._get_directory_size(project_dir) / (1024 * 1024)
                    
                    projects.append(project_info)
                    
                except Exception as e:
                    logger.warning(f"Failed to load project metadata from {project_dir}: {str(e)}")
                    continue
            
            # Sort by last modified (newest first)
            projects.sort(key=lambda p: p.get("last_modified", ""), reverse=True)
            
            logger.info(f"Found {len(projects)} projects")
            return projects
            
        except Exception as e:
            error_msg = f"Failed to list projects: {str(e)}"
            logger.error(error_msg)
            raise ProjectError(error_msg) from e
    
    def delete_project(self, project_name: str) -> None:
        """
        Delete project and all associated files.
        
        Args:
            project_name: Name of the project to delete
            
        Raises:
            ProjectError: If deletion fails
        """
        project_dir = self._get_project_dir(project_name)
        
        try:
            logger.info(f"Deleting project: {project_name}")
            
            # Remove entire project directory
            shutil.rmtree(project_dir)
            
            logger.info(f"Project deleted: {project_name}")
            
        except Exception as e:
            error_msg = f"Failed to delete project '{project_name}': {str(e)}"
            logger.error(error_msg)
            raise ProjectError(error_msg) from e
    
    def save_final_video(self, project_name: str, video_path: str) -> str:
        """
        Save final assembled video to project.
        
        Args:
            project_name: Name of the project
            video_path: Path to final video
            
        Returns:
            Path where video was saved
            
        Raises:
            ProjectError: If saving fails
        """
        if not Path(video_path).exists():
            raise ProjectError(f"Video file not found: {video_path}")
        
        project_dir = self._get_project_dir(project_name)
        final_dir = project_dir / self.project_structure["final"]
        
        try:
            logger.info(f"Saving final video for project: {project_name}")
            
            # Create final video filename
            video_ext = Path(video_path).suffix
            final_filename = f"{self._sanitize_filename(project_name)}_final{video_ext}"
            final_video_path = final_dir / final_filename
            
            # Copy video to project directory
            shutil.copy2(video_path, final_video_path)
            
            # Update project metadata
            self._update_project_metadata(project_name, {
                "last_modified": datetime.now().isoformat(),
                "status": ProcessingStep.COMPLETE.value,
                "progress_percent": 100.0,
                "final_video": str(final_video_path)
            })
            
            logger.info(f"Final video saved: {final_video_path}")
            return str(final_video_path)
            
        except Exception as e:
            error_msg = f"Failed to save final video: {str(e)}"
            logger.error(error_msg)
            raise ProjectError(error_msg) from e
    
    def update_scene_in_storyboard(
        self,
        project_name: str,
        scene_id: int,
        updates: Dict[str, Any]
    ) -> None:
        """
        Update specific scene in storyboard.
        
        Args:
            project_name: Name of the project
            scene_id: ID of scene to update
            updates: Dictionary of fields to update
            
        Raises:
            ProjectError: If update fails
        """
        try:
            # Load current storyboard
            storyboard = self.load_storyboard(project_name)
            
            # Find and update scene
            scene_found = False
            for scene in storyboard.scenes:
                if scene.id == scene_id:
                    # Update scene fields
                    for field, value in updates.items():
                        if hasattr(scene, field):
                            setattr(scene, field, value)
                    scene_found = True
                    break
            
            if not scene_found:
                raise ProjectError(f"Scene {scene_id} not found in storyboard")
            
            # Save updated storyboard
            self.save_storyboard(project_name, storyboard)
            
            logger.info(f"Scene {scene_id} updated in project {project_name}")
            
        except Exception as e:
            error_msg = f"Failed to update scene {scene_id}: {str(e)}"
            logger.error(error_msg)
            raise ProjectError(error_msg) from e
    
    def _get_project_dir(self, project_name: str) -> Path:
        """Get project directory path."""
        safe_name = self._sanitize_filename(project_name)
        project_dir = self.projects_root / safe_name
        
        if not project_dir.exists():
            raise ProjectError(f"Project '{project_name}' not found")
        
        return project_dir
    
    def _load_project_metadata(self, project_name: str) -> Dict[str, Any]:
        """Load project metadata."""
        project_dir = self._get_project_dir(project_name)
        metadata_path = project_dir / self.project_structure["metadata"]
        
        if not metadata_path.exists():
            raise ProjectError(f"Project metadata not found for '{project_name}'")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def _update_project_metadata(self, project_name: str, updates: Dict[str, Any]) -> None:
        """Update project metadata."""
        project_info = self._load_project_metadata(project_name)
        project_info.update(updates)
        
        project_dir = self._get_project_dir(project_name)
        metadata_path = project_dir / self.project_structure["metadata"]
        
        with open(metadata_path, 'w') as f:
            json.dump(project_info, f, indent=2)
    
    def _validate_storyboard(self, storyboard: Storyboard) -> None:
        """Validate storyboard structure and content."""
        if not storyboard.scenes:
            raise StoryboardValidationError("Storyboard must have at least one scene")
        
        # Check scene timing consistency
        for i, scene in enumerate(storyboard.scenes):
            if scene.start_time >= scene.end_time:
                raise StoryboardValidationError(f"Scene {scene.id}: start_time must be before end_time")
            
            if abs(scene.duration - (scene.end_time - scene.start_time)) > 0.1:
                raise StoryboardValidationError(f"Scene {scene.id}: duration doesn't match time range")
            
            # Check for overlaps with next scene
            if i < len(storyboard.scenes) - 1:
                next_scene = storyboard.scenes[i + 1]
                if scene.end_time > next_scene.start_time:
                    logger.warning(f"Scene {scene.id} overlaps with scene {next_scene.id}")
        
        # Check total duration consistency
        total_scene_duration = sum(scene.duration for scene in storyboard.scenes)
        if abs(total_scene_duration - storyboard.audio_duration) > 1.0:
            logger.warning(f"Total scene duration ({total_scene_duration:.1f}s) doesn't match audio duration ({storyboard.audio_duration:.1f}s)")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        import re
        import unicodedata
        
        # Normalize unicode characters
        filename = unicodedata.normalize('NFKD', filename)
        
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]', '_', filename)
        
        # Remove non-ASCII characters that might cause filesystem issues
        sanitized = ''.join(c for c in sanitized if ord(c) < 127 or c.isalnum())
        
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip(' .')
        
        # Replace multiple underscores with single
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        
        # Ensure not empty
        if not sanitized:
            sanitized = "untitled"
        
        return sanitized
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except (OSError, PermissionError):
            pass  # Ignore files we can't access
        
        return total_size